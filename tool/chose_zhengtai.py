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

new_path = '/root/Desktop/val2/covid/'
mkdir(new_path)

for i in range(0, 165): #train_covid(0,689) train_noncovid(0,871)
    length = 0
    path = '/root/Desktop/val/covid/ct_scan_'+str(i)+'/'
    for j in os.listdir(path):
        length = length + 1
    # print(length)
    jiange = int(length / 56)
    # print(jiange)
    # print(int(length*0.25))
    arr_list1 = np.linspace(0, int(length*0.25), 7).astype(dtype=int)
    arr_list2 = np.linspace(int(length*0.25) + jiange, int(length*0.75) - jiange, 42).astype(dtype=int)
    arr_list3 = np.linspace(int(length*0.75), length - 1, 7).astype(dtype=int)
    # print(arr_list1)
    # print(arr_list2)
    # print(arr_list3)
    arr_list = np.hstack([arr_list1, arr_list2, arr_list3])
    print(arr_list)
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