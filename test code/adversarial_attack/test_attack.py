import torch 
import torch.nn as nn
import numpy as np 
import pickle
import os
import argparse
import BNN_resnet
import FP_resnet
import utils
import torchvision
import torchvision.transforms as transforms
import PytorchPlus.adversarial_attack.adversarial_attack as PP_attack

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

model_names = sorted(name for name in FP_resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(FP_resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='attack implementation')
parser.add_argument('--attack', default='fgsm', help='attack type to be used(fgsm, ifgsm, step_ll, pgd....)')
parser.add_argument('--generate', type=get_bool, default=False, help='whether to generate adv examples as .p files')
parser.add_argument('--model', default='resnet', help='target model or model generate adv(resnet, vgg,...)')
parser.add_argument('--modelpath', default="../model_path/naive_param.pkl", help='target model path')
parser.add_argument('--dataroot', default="../data/train/mnist/", help='training data path')
parser.add_argument('--attack_batchsize', type=int, default=128, help='batchsize in Attack')
parser.add_argument('--attack_epsilon', type=float, default=8.0, help='epsilon in Attack')
parser.add_argument('--attack_iter', type=int, default=10, help='iteration times in Attack')
parser.add_argument('--attack_momentum', type=float, default=1.0, help='momentum paramter in Attack')
parser.add_argument('--savepath', default="../save_path/test", help='saving path of clean and adv data')
parser.add_argument('--dataset', default='CIFAR-10', help='dataset used for attacking')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices = model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--GPU_ID', default='0', help='used GPU ID')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

def return_data():
    if args.dataset == 'MNIST':
        test_dataset = torchvision.datasets.MNIST(root=args.dataroot,train=False, transform=transforms.ToTensor())
    elif args.dataset == 'CIFAR-10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataroot,train=False, transform=transform_test)
    elif args.dataset == 'ImageNet':
        data_path = args.dataroot
        valdir = os.path.join(data_path, 'val')
        test_dataset = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # scale up to 256
                transforms.Scale(256),
                # center crop to 224
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.attack_batchsize,num_workers = 4,shuffle=False,drop_last=False)
    return test_loader

def main():
    CUDA_AVAILABLE = torch.cuda.is_available()
    if args.model == 'BNN_ResNet20':
        print('attack BNN_ResNet20 model')
        model = torch.nn.DataParallel(BNN_resnet.__dict__[args.arch]())
        #model.cuda()
    elif args.model == 'FP_ResNet20':
        print('attack FP_ResNet20 model')
        model = torch.nn.DataParallel(FP_resnet.__dict__[args.arch]())
        #model.cuda()
    else:
        print('model argument error, exit.')
    
    checkpoint = torch.load(args.modelpath, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module

    if args.dataset == 'MNIST':
        eps = args.attack_epsilon
        norm = lambda x:x
    else:
        eps = args.attack_epsilon / 255.0
        if args.dataset == 'CIFAR-10':
            mu = torch.Tensor((0.485, 0.456, 0.406)).unsqueeze(-1).unsqueeze(-1).cuda() if(CUDA_AVAILABLE) else torch.Tensor((0.485, 0.456, 0.406)).unsqueeze(-1).unsqueeze(-1)
            std = torch.Tensor((0.229, 0.224, 0.225)).unsqueeze(-1).unsqueeze(-1).cuda() if(CUDA_AVAILABLE) else torch.Tensor((0.229, 0.224, 0.225)).unsqueeze(-1).unsqueeze(-1)
            normalize = lambda x: (x-mu)/std
        else:
            print("data set error.")
        
    dataloader = return_data()
    print("epsilon : ",args.attack_epsilon)
    
    if args.attack == 'fgsm':
        test_data_adv, test_label = PP_attack.fgsm(model = model, data_loader = dataloader, criterion = nn.CrossEntropyLoss(), epsilon = eps, normalizer = normalize)
    elif args.attack == 'stepll':
        test_data_adv, test_label = PP_attack.step_ll(model = model, data_loader = dataloader, criterion = nn.CrossEntropyLoss(), epsilon = eps, normalizer = normalize)
    elif args.attack == 'pgd':
        test_data_adv, test_label = PP_attack.pgd(model = model, data_loader = dataloader, criterion = nn.CrossEntropyLoss(), epsilon = eps, normalizer = normalize, iteration=args.attack_iter)
    elif args.attack == 'momentum_ifgsm':
        test_data_adv, test_label = PP_attack.momentum_ifgsm(model = model, data_loader = dataloader, criterion = nn.CrossEntropyLoss(), epsilon = eps, normalizer = normalize, iteration=args.attack_iter, attack_momentum=args.attack_momentum)
    elif args.attack == 'CW_L2':
        test_data_adv, test_label = PP_attack.CarliniWagnerL2(model = model, data_loader = dataloader, steps=10, search_steps=2, normalizer = normalize, debug=False)
    
    
    if args.generate:
        utils.save_data_label(args.savepath+"{2:s}_{0:s}_{1:.0f}.pkl".format(args.attack,args.attack_epsilon,args.model), test_data_adv)
        utils.save_data_label(args.savepath+"{:s}_label.pkl".format(args.dataset), test_label)

if __name__ == "__main__":
    main()