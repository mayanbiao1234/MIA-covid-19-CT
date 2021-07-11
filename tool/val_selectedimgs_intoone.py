import os
import shutil

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("文件夹已创建成功")
    else:
        print("文件夹已存在")

mkdir('/root/Desktop/val_covid_onlyjiequ')  ############文件夹路径要改
mkdir('/root/Desktop/val_noncovid_onlyjiequ')  ############文件夹路径要改

flag = 1
for i in os.listdir('/root/Desktop/val_onlyjiequ/val_noncovid_onlyjieduan/'):  ############文件夹路径要改
    path = '/root/Desktop/val_onlyjiequ/val_noncovid_onlyjieduan/' + i +'/'   ############文件夹路径要改
    for j in os.listdir(path):
        address = path + j
        new_address = '/root/Desktop/val_noncovid_onlyjiequ/' + str(flag) + '.jpg'  ############文件夹路径要改
        shutil.copy(address, new_address)
        flag = flag + 1
        print(flag)
