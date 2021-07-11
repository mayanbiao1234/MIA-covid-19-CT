import os
import numpy as np
import shutil

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        #print("文件夹已创建成功")
    else:
        pass#print("文件夹已存在")

new_path = '/root/Desktop/val1/noncovid/'
mkdir(new_path)

for i in range(113, 114): #train_covid(0,689) train_noncovid(0,871)
    length = 0
    path = '/root/Desktop/val/non-covid/ct_scan_'+str(i)+'/'
    for j in os.listdir(path):
        length = length + 1
    arr_list = np.linspace(0,length-1,56).astype(dtype=int)
    # print('==================================')
    # print(length)
    # print(arr_list)
    flag = 0
    for j in arr_list:
        #print(j)
        mkdir(new_path + 'ct_scan_' + str(i) + '/')
        shutil.copy(path + str(j) + '.jpg', new_path + 'ct_scan_'+ str(i) +'/' + str(flag) + '.jpg')
        flag = flag + 1
    print(i)