import os
import random
import shutil
import sys
from PIL import Image
from PIL import ImageDraw
import numpy as np
from skimage import io

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        #print("文件夹已创建成功")
    else:
        pass#print("文件夹已存在")

new_path1 = '/root/Desktop/train1_angumentation/covid_rot15/'
new_path2 = '/root/Desktop/train1_angumentation/noncovid_rot15/'
new_path3 = '/root/Desktop/train1_angumentation/covid_rot345/'
new_path4 = '/root/Desktop/train1_angumentation/noncovid_rot345/'
new_path5 = '/root/Desktop/train1_angumentation/covid_shui/'
new_path6 = '/root/Desktop/train1_angumentation/noncovid_shui/'
mkdir(new_path1)
mkdir(new_path2)
mkdir(new_path3)
mkdir(new_path4)
mkdir(new_path5)
mkdir(new_path6)

#************************************开始数据增强********************************
path1 = '/root/Desktop/train1/covid'
path2 = '/root/Desktop/train1/noncovid'


for i in range(0, 689):
    print(i)
    path_ori = path1 + '/' + 'ct_scan_' + str(i) +'/'
    for j in os.listdir(path_ori):
        #print(j)
        img = Image.open(path_ori + j)
        img1 = img.rotate(15)
        img2 = img.rotate(345)
        img3 = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转

        mkdir(new_path1 + 'ct_scan_' + str(i))
        mkdir(new_path3 + 'ct_scan_' + str(i))
        mkdir(new_path5 + 'ct_scan_' + str(i))

        img1.save(new_path1 + 'ct_scan_' +str(i) + '/' + j, quality = 95)
        img2.save(new_path3 + 'ct_scan_' +str(i) + '/' + j, quality = 95)
        img3.save(new_path5 + 'ct_scan_' +str(i) + '/' + j, quality = 95)

for i in range(0, 871):
    print(i)
    path_ori = path2 + '/' + 'ct_scan_' + str(i) +'/'
    for j in os.listdir(path_ori):
        #print(j)
        img = Image.open(path_ori + j)
        img1 = img.rotate(15)
        img2 = img.rotate(345)
        img3 = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转

        mkdir(new_path2 + 'ct_scan_' + str(i))
        mkdir(new_path4 + 'ct_scan_' + str(i))
        mkdir(new_path6 + 'ct_scan_' + str(i))

        img1.save(new_path2 + 'ct_scan_' +str(i) + '/' + j, quality = 95)
        img2.save(new_path4 + 'ct_scan_' +str(i) + '/' + j, quality = 95)
        img3.save(new_path6 + 'ct_scan_' +str(i) + '/' + j, quality = 95)







#     img = Image.open(path1 + '/' + 'ct_scan_' + str(i))
#     img1 = img.rotate(15)
#     img2 = img.rotate(350)
#     #img3 = img.rotate(270)
#     img4 = img.transpose(Image.FLIP_LEFT_RIGHT)#水平翻转
#     #img5 = img.transpose(Image.FLIP_TOP_BOTTOM)#垂直翻转
#
#     img1.save('/root/Desktop/448_new_enhance/train_90' + '/' + str(j) + '.JPG')
#     img2.save('/root/Desktop/448_new_enhance/train_180' + '/' + str(j+1448) + '.JPG')
#     #img3.save('/root/Desktop/448_new_enhance/train_270' + '/' + str(j+2896) + '.JPG')
#     img4.save('/root/Desktop/448_new_enhance/train_shui' + '/' + str(j+4344) + '.JPG')
#     #img5.save('/root/Desktop/448_new_enhance/train_chui' + '/' + str(j+5792) + '.JPG')
#     j = j + 1
#
#
#
# #对val进行增强
# j = 30001
# #for i in range(8366):
# for i in test_list:
#     print(i)
#     img = Image.open(path2 + '/' + str(i))
#     img1 = img.rotate(90)
#     img2 = img.rotate(180)
#     img3 = img.rotate(270)
#     img4 = img.transpose(Image.FLIP_LEFT_RIGHT)
#     img5 = img.transpose(Image.FLIP_TOP_BOTTOM)
#
#     img1.save('/root/Desktop/448_new_enhance/test_90' + '/' + str(j) + '.JPG')
#     img2.save('/root/Desktop/448_new_enhance/test_180' + '/' + str(j+450) + '.JPG')
#     img3.save('/root/Desktop/448_new_enhance/test_270' + '/' + str(j+900) + '.JPG')
#     img4.save('/root/Desktop/448_new_enhance/test_shui' + '/' + str(j+1350) + '.JPG')
#     img5.save('/root/Desktop/448_new_enhance/test_chui' + '/' + str(j+1800) + '.JPG')
#     j = j + 1


