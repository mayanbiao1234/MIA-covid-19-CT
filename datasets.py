# coding=utf-8

import torch
import os, glob
import random, csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#import tifffile as tiff
import cv2
import nibabel as nib
from skimage import transform,data

class Pokemon(Dataset):

    def __init__(self, root, resize, mode):

        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize

        self.name2label = {}
        # 返回指定目录下的文件列表，并对文件列表进行排序，
        # os.listdir每次返回目录下的文件列表顺序会不一致，
        # 排序是为了每次返回文件列表顺序一致
        for name in sorted(os.listdir(os.path.join(root))):
            # 过滤掉非目录文件
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 构建字典，名字：0~12数字
            self.name2label[name] = len(self.name2label.keys())

        # eg: {'squirtle': 4, 'bulbasaur': 0, 'pikachu': 3, 'mewtwo': 2, 'charmander': 1}
        print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv("imagesfile.csv")

        # # 对数据集进行划分
        # if mode == "train":  # 60%
        #     self.images = self.images[:int(0.8 * len(self.images))]
        #     self.labels = self.labels[:int(0.8 * len(self.labels))]
        # # elif mode == "val":  # 20% = 60%~80%
        # #     self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
        # #     self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        # else:  # 20% = 80%~100%
        #     self.images = self.images[int(0.8 * len(self.images)):]
        #     self.labels = self.labels[int(0.8 * len(self.labels)):]

    # 将目录下的图片路径与其对应的标签写入csv文件，
    # 并将csv文件写入的内容读出，返回图片名与其标签
    def load_csv(self, filename):
        """
        :param filename:
        :return:
        """
        # 是否已经存在了cvs文件
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 获取指定目录下所有的满足后缀的图像名
                # pokemon/mewtwo/00001.png
                #images += glob.glob(os.path.join(self.root, name, "*.png"))
                images += glob.glob(os.path.join(self.root, name, "*.nii.gz"))
                #images += glob.glob(os.path.join(self.root, name, "*.tif"))

            # 1165 'pokemon/pikachu/00000058.png'
            print(len(images), images)

            # 将元素打乱
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:  # 'pokemon/pikachu/00000058.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 将图片路径以及对应的标签写入到csv文件中
                    # 'pokemon/pikachu/00000058.png', 0
                    writer.writerow([img, label])
                print("writen into csv file: ", filename)

        # 如果已经存在了csv文件，则读取csv文件
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon/pikachu/00000058.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon/bulbasaur/00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            #lambda x: transform.resize(np.array(Image.open(x).convert("RGB")), (self.resize, self.resize)),  # string path => image data
            #lambda x: np.resize(np.array(nib.load(x).get_fdata()), (self.resize, self.resize, 56)),
            lambda x: np.array(nib.load(x).get_fdata()),
            #对数据进行放大
            #transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            #对数据今天旋转
            #transforms.RandomRotation(15),
            #对数据进行中心裁剪，大小为224*224
            #transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        img = img.unsqueeze(dim = 0)
        #print(img.shape)
        img = img.type(torch.FloatTensor)
        label = torch.tensor(label)
        return img, label


def LoadData(BATCH_SIZE):

    traindata = Pokemon("/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train", 448, "train")
    testdata = Pokemon("/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_val", 448, "test")
    print(len(testdata),len(traindata))
    trainloader = DataLoader(traindata,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    testloader = DataLoader(testdata,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    return testloader,trainloader
#/root/data/covid19/covid19_train_junyun/nii_train
#/root/data/covid19/covid19_train_junyun/nii_val

# LoadData()