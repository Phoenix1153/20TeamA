import torch
from torch.autograd import Variable
import math
from .CW_attack_class import AttackCarliniWagnerL2
import numpy as np
import torchvision.transforms as transforms


#model in cpu
def fgsm(model, data_loader, criterion, epsilon, normalizer):
    CUDA_AVAILABLE = torch.cuda.is_available()
    model = model.cuda() if(CUDA_AVAILABLE) else model
    model.eval()
    correct_cln, correct_adv, total= 0, 0, 0
    for i, (images, labels) in enumerate(data_loader):
        x = Variable(images, requires_grad = True).cuda() if(CUDA_AVAILABLE) else Variable(images, requires_grad = True)
        y_true = Variable(labels, requires_grad = False).cuda() if(CUDA_AVAILABLE) else Variable(labels, requires_grad = False)
        x.retain_grad()
        h = model(normalizer(x))
        _, predictions = torch.max(h,1)
        correct_cln += (predictions == y_true).sum()
        loss = criterion(h, y_true)
        model.zero_grad()
        if x.grad is not None:
            x.grad.data.fill_(0)
        loss.backward()
        
        x_adv = x.detach() + epsilon * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv,0,1)
        
        h_adv = model(normalizer(x_adv))
        _, predictions_adv = torch.max(h_adv,1)
        correct_adv += (predictions_adv == y_true).sum()
        if i == 0:
            test_data_adv = x_adv.data.cpu()
            test_label = labels
        else:
            test_data_adv = torch.cat([test_data_adv, x_adv.data.cpu()],0)
            test_label = torch.cat([test_label, labels],0)
        total += len(predictions)
    model.train()
    
    print("Before FGSM the accuracy is", float(100*correct_cln)/total)
    print("After FGSM the accuracy is", float(100*correct_adv)/total)
    return test_data_adv, test_label

def pgd(model, data_loader, criterion, epsilon, normalizer, iteration):
    CUDA_AVAILABLE = torch.cuda.is_available()
    model = model.cuda() if(CUDA_AVAILABLE) else model
    alpha = epsilon / math.sqrt(float(iteration))
    model.eval()
    correct_cln, correct_adv, total= 0, 0, 0
    for i,(images,labels) in enumerate(data_loader):
        x = Variable(images, requires_grad = True).cuda() if(CUDA_AVAILABLE) else Variable(images, requires_grad = True)
        y_true = Variable(labels, requires_grad = False).cuda() if(CUDA_AVAILABLE) else Variable(labels, requires_grad = False)
        x.retain_grad()
        h = model(normalizer(x))
        _, predictions = torch.max(h,1)
        correct_cln += (predictions == y_true).sum()
        x_rand = x.detach()
        # PGD
        x_rand = x_rand + torch.zeros_like(x_rand).uniform_(-epsilon,epsilon)
        x_rand = torch.clamp(x_rand,0,1)
        x_adv = Variable(x_rand.data, requires_grad=True).cuda() if(CUDA_AVAILABLE) else Variable(x_rand.data, requires_grad=True)
        for j in range(0,iteration):
            #print('batch = {}, iter = {}'.format(i,j))
            h_adv = model(normalizer(x_adv))
            loss = criterion(h_adv, y_true)
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()
            
            x_adv = x_adv.detach() + alpha * torch.sign(x_adv.grad)
            # according to the paper of Kurakin:
            x_adv = torch.where(x_adv > x+epsilon, x+epsilon, x_adv)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = torch.where(x_adv < x-epsilon, x-epsilon, x_adv)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = Variable(x_adv.data, requires_grad=True).cuda() if(CUDA_AVAILABLE) else Variable(x_adv.data, requires_grad=True)
        h_adv = model(normalizer(x_adv))
        _, predictions_adv = torch.max(h_adv,1)
        correct_adv += (predictions_adv == y_true).sum()
        #print(x.data.size(),x_adv.data.size(),labels.size())
        if i == 0:
            test_data_adv = x_adv.data.cpu()
            test_label = labels
        else:
            test_data_adv = torch.cat([test_data_adv, x_adv.data.detach().cpu()],0)
            test_label = torch.cat([test_label, labels],0)

        total += len(predictions)
    model.train()

    #print("Error Rate is ",float(total-correct)*100/total)
    print('PGD alpha is ', alpha)
    print("Before PGD the accuracy is",float(100*correct_cln)/total)
    print("After PGD the accuracy is",float(100*correct_adv)/total)
    return test_data_adv, test_label 
    
def step_ll(model, data_loader, criterion, epsilon, normalizer):
    CUDA_AVAILABLE = torch.cuda.is_available()
    model = model.cuda() if(CUDA_AVAILABLE) else model
    model.eval()
    correct_cln, correct_adv, total= 0, 0, 0
    for i, (images, labels) in enumerate(data_loader):
        x = Variable(images, requires_grad = True).cuda() if(CUDA_AVAILABLE) else Variable(images, requires_grad = True)
        y_true = Variable(labels, requires_grad = False).cuda() if(CUDA_AVAILABLE) else Variable(labels, requires_grad = False)
        x.retain_grad()

        h = model(normalizer(x))
        _, predictions = torch.max(h,1)
        # Step-LL
        _, predictions_ll = torch.min(h,1)
        correct_cln += (predictions == y_true).sum()
        loss = criterion(h, predictions_ll)
        model.zero_grad()
        if x.grad is not None:
            x.grad.data.fill_(0)
        loss.backward()
        
        #x.grad.sign_()   # change the grad with sign ?
        x_adv = x.detach() - epsilon * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv,0,1)
        
        h_adv = model(normalizer(x_adv))
        _, predictions_adv = torch.max(h_adv,1)
        correct_adv += (predictions_adv == y_true).sum()
        #print(x.data.size(),x_adv.data.size(),labels.size())
        if i == 0:
            test_data_adv = x_adv.data.cpu()
            test_label = labels
        else:
            test_data_adv = torch.cat([test_data_adv, x_adv.data.detach().cpu()],0)
            test_label = torch.cat([test_label, labels],0)
        total += len(predictions)
    model.train()
    
    #print("Error Rate is ", float(total-correct)*100/total)
    print("Before Step-ll the accuracy is", float(100*correct_cln)/total)
    print("After Step-ll the accuracy is", float(100*correct_adv)/total)

    return test_data_adv, test_label
    
def momentum_ifgsm(model, data_loader, criterion, epsilon, normalizer, iteration, attack_momentum):
    CUDA_AVAILABLE = torch.cuda.is_available()
    model = model.cuda() if(CUDA_AVAILABLE) else model
    alpha = epsilon / math.sqrt(float(iteration))
    model.eval()
    correct_cln, correct_adv, total= 0, 0, 0
    for i,(images,labels) in enumerate(data_loader):
        x = Variable(images, requires_grad = True).cuda() if(CUDA_AVAILABLE) else Variable(images, requires_grad = True)
        y_true = Variable(labels, requires_grad = False).cuda() if(CUDA_AVAILABLE) else Variable(labels, requires_grad = False)
        x.retain_grad()
        x_adv = Variable(x.data, requires_grad=True).cuda() if(CUDA_AVAILABLE) else Variable(x.data, requires_grad=True)
        x_grad = torch.zeros(x.size()).cuda() if(CUDA_AVAILABLE) else torch.zeros(x.size())

        h = model(normalizer(x))
        _, predictions = torch.max(h,1)        
        correct_cln += (predictions == y_true).sum()#    

        for j in range(0,iteration):
            h_adv = model(normalizer(x_adv))

            loss = criterion(h_adv, y_true)
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()
            
            norm = x_adv.grad
            for k in range(1,4):
                norm = torch.norm(norm,p=1,dim=k).unsqueeze(dim=k)
            
            # Momentum on gradient noise
            x_grad = attack_momentum * x_grad + x_adv.grad / norm

            x_adv = x_adv.detach() + alpha * torch.sign(x_grad)
            # according to the paper of Kurakin:
            x_adv = torch.where(x_adv > x+epsilon, x+epsilon, x_adv)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = torch.where(x_adv < x-epsilon, x-epsilon, x_adv)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = Variable(x_adv.data, requires_grad=True).cuda() if(CUDA_AVAILABLE) else Variable(x_adv.data, requires_grad=True)

        h_adv = model(normalizer(x_adv))
        _, predictions_adv = torch.max(h_adv,1)
        correct_adv += (predictions_adv == y_true).sum()

        if i == 0:
            test_data_adv = x_adv.data.cpu()
            test_label = labels
        else:
            test_data_adv = torch.cat([test_data_adv, x_adv.data.detach().cpu()],0)
            test_label = torch.cat([test_label, labels],0)
        total += len(predictions)
    model.train()
    
    print('MI-FGSM alpha is ', alpha)
    print("Before Momentum IFGSM the accuracy is",float(100*correct_cln)/total)
    print("After Momentum IFGSM the accuracy is",float(100*correct_adv)/total)
    
    return test_data_adv, test_label

def CarliniWagnerL2(model, data_loader, steps, search_steps, normalizer, debug=False):
    CUDA_AVAILABLE = torch.cuda.is_available()
    model = model.cuda() if(CUDA_AVAILABLE) else model
    attack = AttackCarliniWagnerL2(
        normalizer=normalizer,
        targeted=False,
        max_steps=steps,
        search_steps=search_steps,
        cuda=CUDA_AVAILABLE,
        debug=debug)
    
    model.eval()
    for batch_idx, (input, target) in enumerate(data_loader):
        input = input.cuda() if(CUDA_AVAILABLE) else input
        target = target.cuda() if(CUDA_AVAILABLE) else target

        input_adv = attack.run(model, input, target, batch_idx)

        if batch_idx == 0:
            test_data_adv = torch.from_numpy(input_adv.transpose((0, 3, 1, 2)))
            test_label = target
        else:
            test_data_adv = torch.cat([test_data_adv, torch.from_numpy(input_adv.transpose((0, 3, 1, 2)))],0)
            test_label = torch.cat([test_label, target],0)
    print(test_data_adv.size())
    return test_data_adv, test_label

def Fourier_based_Corruption(dataset, imgsize, position):
    CUDA_AVAILABLE = torch.cuda.is_available()   
    N = imgsize
    origin_value = 1.0
    i, j = position[0], position[1]
    print("position({},{})".format(i,j))
    testset = dataset
    samples_size = dataset.__len__()
    samples = np.array(range(samples_size))
    loader = transforms.Compose([
    transforms.ToTensor()]) 
    
    F_base_vec = torch.zeros((N,N,2)).cuda() if(CUDA_AVAILABLE) else torch.zeros((N,N,2))
    F_base_vec[i][j][0] = F_base_vec[i][j][0] = origin_value   
    Uij = torch.ifft(F_base_vec,2)[:,:,0].cuda() if(CUDA_AVAILABLE) else torch.ifft(F_base_vec,2)[:,:,0]              
    Uij /= torch.norm(Uij,p=2)
    result = torch.zeros((samples_size,3,N,N)).cuda() if(CUDA_AVAILABLE) else torch.zeros((samples_size,3,N,N))

    for k in range(samples_size):
        img = testset[samples[k]][0]
        img_array = loader(img)
        img_new_array = torch.zeros((3,N,N)).cuda() if(CUDA_AVAILABLE) else torch.zeros((3,N,N))
        for channel in range(3):
            img_one_channel = torch.Tensor(img_array[channel,:,:])
            img_one_channel = img_one_channel.cuda() if(CUDA_AVAILABLE) else img_one_channel
            L2norm = torch.norm(img_one_channel,p=2) * 0.1  
            r = 1                                                 
            rvUij = Uij * L2norm * r
            img_one_channel += rvUij
            img_new_array[channel,:,:] = img_one_channel
        result[k] = img_new_array
    result = result.cpu()
    return result