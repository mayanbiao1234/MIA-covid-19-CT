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

List_true = []
with open(r"/root/Desktop/True_label.txt","r") as f:  ## do not touch
    for line in f.readlines():
        line = line.strip('\n')
        List_true.append(int(line))
List_true = np.array(List_true)

List_preb = []
with open(r"/root/Desktop/pre_label.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_preb.append(int(line))
List_preb = np.array(List_preb)
print("F1:{:.4f}".format(f1_score(List_true, List_preb, average='macro')))
