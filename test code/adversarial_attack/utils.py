import os
import numpy as np
import pickle
import torch

def read_data_label(data_path, label_path):
    if not os.path.exists(data_path):
        print("path do not exists.")
        os._exit(0)
    with open(data_path, 'rb') as fr:
        test_data = pickle.load(fr)
        size = len(test_data)
    with open(label_path, 'rb') as fr:
        test_label = pickle.load(fr)
    return test_data, test_label, size

def save_data_label(path, data):
    with open(path,'wb') as f:
        pickle.dump(data.cpu(), f, pickle.HIGHEST_PROTOCOL)

'''
def save_img(imgpath,test_data_cln, test_data_adv, test_label, test_label_adv):
    #save adversarial example
    if Path(imgpath).exists()==False:
        Path(imgpath).mkdir(parents=True)
    toImg = transforms.ToPILImage()
    image = test_data_cln.cpu()
    image_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    tot = len(image)
    batch = 10
    for i in range(0, batch):
        #print(image[i].size())
        im = toImg(image[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_cln.jpg'.format(i,test_label[i])))
        im = toImg(image_adv[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_adv.jpg'.format(i,test_label_adv[i])))

def display(test_data_cln, test_data_adv, test_label, test_label_adv):
    # display a batch adv
    toPil = transforms.ToPILImage()
    curr = test_data_cln.cpu()
    curr_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    disp_batch = 10
    for a in range(disp_batch):
        b = a + disp_batch 
        plt.figure()
        plt.subplot(121)
        plt.title('Original Label: {}'.format(label[a].cpu().numpy()),loc ='left')
        plt.imshow(toPil(curr[a].cpu().clone().squeeze()))
        plt.subplot(122)
        plt.title('Adv Label : {}'.format(label_adv[a].cpu().numpy()),loc ='left')
        plt.imshow(toPil(curr_adv[a].cpu().clone().squeeze()))
        plt.show()
'''


