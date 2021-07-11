#from nets import *
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
from pytorch_metric_learning import losses
#if torch.cuda.is_available():
#    device = 'cuda'
#else:
#    device = 'cpu'
#device = 'cuda'
USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda:0" if USE_CUDA else "cpu")
device = torch.device("cuda:0")
LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 32

traindataSize = 6240
testdataSize = 374

input_W = 448
input_H = 448
input_D = 56
resnet_shortcut = 'B'
no_cuda = 'False'
n_seg_classes = 1
pretrain_path = '/root/Desktop/covid19/pretrain/resnet_10_23dataset.pth'
new_layer_names = ['conv_seg']
# model_save_path = "./The_neuro_ANN/cifar3.h5"

# 训练和验证
criteration = nn.CrossEntropyLoss()
def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    prob_all = []
    label_all = []
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        #print("output size:",output.size())
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

        prob = output.cpu().detach().numpy()
        prob_all.extend(np.argmax(prob, axis=1))
        label_all.extend(y.cpu())

        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch, loss, correct, traindataSize,
                                                                 100 * correct / traindataSize))
    print("F1:{:.4f}".format(f1_score(label_all, prob_all, average='macro')))



def vaild(model, device, dataset):
    model.eval()
    correct = 0
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    prob_all = []
    label_all = []
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


if __name__ == '__main__':
    print('------Starting------')
    # 确定是否使用GPU

    testloader, trainloader = LoadData(BATCH_SIZE)
    # print(train_dataset.shape)
    print('------Dataset initialized------')
    # model = pretrainedmodels.resnet18(num_classes=1000, pretrained='imagenet')
    # #model.load_state_dict(torch.load("/root/Desktop/premodel/resnet18-5c106cde.pth"))
    # model.fc = nn.Sequential(
    #     #model1,
    #     nn.Linear(512,2)
    ##########################################################################
    model1 = resnet.resnet10(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_seg_classes=n_seg_classes)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])
    net_dict = model1.state_dict()
    #model = nn.DataParallel(model,device_ids = [0])
    #model.load_state_dict(torch.load("/root/.cache/torch/checkpoints/se_resnet50-ce0d4300.pth")
    #model = torch.nn.DataParallel(model)
    #model.to(device)
    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

    net_dict.update(pretrain_dict)
    model1.load_state_dict(net_dict)
    model = nn.Sequential(
        model1,
        nn.Linear(11239424, 2)
        #nn.Linear(512,2)
    )
    # new_parameters = []
    # for pname, p in model.named_parameters():
    #     for layer_name in new_layer_names:
    #         if pname.find(layer_name) >= 0:
    #             new_parameters.append(p)
    #             break
    #
    # new_parameters_id = list(map(id, new_parameters))
    # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    # parameters = {'base_parameters': base_parameters,
    #               'new_parameters': new_parameters}
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)
    summary(model, (1, 56, 448, 448))
    #optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.09)
    #optimizer1 = torch.optim.Adam(model.parameters(), lr = 0.0001)
    #optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, trainloader, optimizer, epoch)
        vaild(model, device, testloader)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        #if epoch % 2 == 0:
        save_model(model.state_dict(),'MODEL','/root/data/covid19zhongzhuan/shiyan10/'+ str(epoch) + 'junyun3dresnet_10_23dataset.pt')


    # print('------Network load successfully------')
    # # trainset 第一个维度控制第几张图片，第二个维度控制是data还是label
    #
    #
    # Loss_crossEntropy = nn.CrossEntropyLoss().to(device)
    # print('------Loss created------')
    #
    # optimizer= torch.optim.RMSprop(newmodel.parameters(), lr=1e-3)
    # train_loss_history = []
    # train_acc_history = []
    # test_loss_history = []
    # test_acc_history = []
    # MAX = 0.0
    # tag = 0
    # for epoch in range(EPOCHS):
    #     epoch_train_loss = 0.0
    #     epoch_train_acc = 0.0
    #     epoch_test_loss = 0.0
    #     epoch_test_acc = 0.0
    #     # train 阶段
    #     train_step = 0
    #     for i, data in enumerate(trainloader):
    #         correct = 0
    #         train_step += 1
    #         images, labels = data
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         output = newmodel(images)
    #         # 梯度清零
    #         optimizer.zero_grad()
    #         # print(output)
    #         # 加上正则化的loss
    #         loss = Loss_crossEntropy(output, labels.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         _, predicted = torch.max(output.data, 1)
    #         correct += (predicted == labels.to(device)).sum().item()
    #         acc = correct / BATCH_SIZE
    #
    #         epoch_train_loss += loss.item()
    #         epoch_train_acc += acc
    #
    #         print('Train-%d-%d, loss: %.3f, acc: %.3f' % (epoch, i, loss, acc))
    #     train_loss_history.append(epoch_train_loss / train_step)
    #     train_acc_history.append(epoch_train_acc / train_step)
    #     # test 阶段
    #     # 不跟踪梯度
    #     with torch.no_grad():
    #         test_step = 0
    #         for test_data in testloader:
    #             test_step += 1
    #             test_correct = 0
    #             test_images, test_labels = test_data
    #
    #             test_output = newmodel(test_images.to(device))
    #             test_loss = Loss_crossEntropy(test_output, test_labels.to(device))
    #             _, test_predicted = torch.max(test_output.data, 1)
    #             test_correct += (test_predicted == test_labels.to(device)).sum().item()
    #             test_acc = test_correct / 64
    #
    #             epoch_test_loss += test_loss.item()
    #             epoch_test_acc += test_acc
    #             print('Test epoch: %d, loss: %.3f, acc: %.3f' % (epoch, test_loss, test_acc))
    #     test_loss_history.append(epoch_test_loss / test_step)
    #     test_acc_history.append(epoch_test_acc / test_step)
    # save_model(newmodel.state_dict(),'SE_MobileNet','model.pt')
    # show_loss(train_loss_history, test_loss_history)
    # show_accuracy(train_acc_history, test_acc_history)
