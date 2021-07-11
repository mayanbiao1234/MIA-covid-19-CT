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
pretrain_path = '/root/data/covid19zhongzhuan/shiyan10/9junyun3dresnet_10_23dataset.pt'
new_layer_names = ['conv_seg']
testdataSize = 374
BATCH_SIZE = 4

def vaild(model, device, dataset):
    model.eval()
    correct = 0
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    prob_all = []
    label_all = [] #true
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            testdata = torch.cat((testdata, output.cpu()), 0)
            testlabel = torch.cat((testlabel, y.cpu()))
            # loss = nn.CrossEntropyLoss(output, y)
            loss = F.cross_entropy(output, y)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

            prob = output.cpu().detach().numpy()
            prob_all.extend(np.argmax(prob, axis=1))
            label_all.extend(y.cpu())

    print(
        "Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss, correct, testdataSize, 100. * correct / testdataSize))
    print("F1:{:.4f}".format(f1_score(label_all, prob_all, average='macro')))

    return label_all, prob_all


print('------Starting------')
testloader, trainloader = LoadData(BATCH_SIZE)
model1 = resnet.resnet10(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_seg_classes=n_seg_classes)

model = nn.Sequential(
    model1,
    nn.Linear(11239424, 2))
    #nn.Linear(44957696, 2))
model = nn.DataParallel(model,device_ids = [0])
model.to(device)
state_dict = torch.load(pretrain_path)
model.load_state_dict(state_dict)
label_True, pre_label = vaild(model, device, testloader)
label_True = np.array(label_True).astype(np.int)
pre_label = np.array(pre_label).astype(np.int)
with open(r'/root/data/covid_prediction_zhongzhuan/10.8301.txt',"w") as f:
    for i in pre_label:
        f.write(str(i))
        f.write('\n')
with open(r'/root/Desktop/True_label.txt',"w") as f:
    for i in label_True:
        f.write(str(i))
        f.write('\n')


