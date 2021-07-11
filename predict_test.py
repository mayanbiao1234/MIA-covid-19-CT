import torch
import numpy as np
import torchvision
import os
from torchvision import transforms,datasets,models
from torch.utils.data import DataLoader
import torch.nn as nn
from tools import save_model, show_accuracy, show_loss, show_img
from collections import Counter
import matplotlib.pyplot as plt
from datasets import LoadData
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
import pretrainedmodels
import evaluation as eva
from collections import OrderedDict
from sklearn.metrics import f1_score
from setting import parse_opts
from model import generate_model
import torch
import numpy as np
from torch import nn
from models import resnet
import nibabel as nib

USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda:0" if USE_CUDA else "cpu")
device = torch.device("cuda:0")

input_W = 448
input_H = 448
input_D = 56
resnet_shortcut = 'B'
no_cuda = 'False'
n_seg_classes = 1
pretrain_path = '/root/data/covid19zhongzhuan/shiyan10/7junyun3dresnet_10_23dataset.pt'
new_layer_names = ['conv_seg']

print('------Starting------')
#testloader, trainloader = LoadData(BATCH_SIZE)
model1 = resnet.resnet10(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_seg_classes=n_seg_classes)
model = nn.Sequential(
    model1,
    #nn.Linear(44957696, 2))
    nn.Linear(11239424, 2))
model = nn.DataParallel(model,device_ids = [0])
model.to(device)
state_dict = torch.load(pretrain_path)
model.load_state_dict(state_dict)
model.eval()
###########################################################111111111111111111#####################
pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset1/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset1.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')

######################################22222222222222222222222222222###################################
pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset2/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset2.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')

###########################################3333333333333333333333333333333################################

pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset3/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset3.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')

#######################################444444444444444444444444444444444444##################################
pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset4/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset4.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')

############################################55555555555555555555555555555555555555555#########################
pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset5/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset5.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')

#############################################6666666666666666666666666666###############################
pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset6/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset6.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')

##########################################77777777777777777777777777777##########################
pre_list = []
for i in tqdm(range(450)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset7/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset7.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')
############################################888888888888888888888##################################
pre_list = []
for i in tqdm(range(305)):
    data_path = '/root/data/covid19_test/test_avg_guiyihua1/subset8/' + str(i) + '.nii.gz'

    tf = transforms.Compose([
                lambda x: np.array(nib.load(x).get_fdata()),
                transforms.ToTensor(),
            ])
    img = tf(data_path)
    img = img.unsqueeze(dim = 0)
    img = img.unsqueeze(dim = 0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    #print(img.shape)
    label = np.argmax(model(img).detach().cpu().numpy())
    pre_list.extend(np.array([label]))

pre_list = np.array(pre_list).astype(np.int)
with open(r'/root/data/covid19_test/test_result/10_7_0.8482/subset8.txt',"w") as f:
    for i in pre_list:
        f.write(str(i))
        f.write('\n')
