import os
import shutil

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("文件夹已创建成功")
    else:
        print("文件夹已存在")


## 将选中的图片单独拿出来
flag_see = 1
mkdir('/root/Desktop/small_train/train_covid')
for i in range(0,687):   ############文件夹路径要改 cobid(0,687) non-covid(0,865)
    path = '/root/Desktop/data/train/covid/ct_scan_' + str(i) + '/' ############文件夹路径要改
    file_len = len(os.listdir(path))
    chose_rgb = int(file_len*0.53)

    mkdir('/root/Desktop/small_train/train_covid/' + str(i))

    if file_len < 3:
        for name in range(1, 4):
            address = path + str(chose_rgb) + '.jpg'
            new_address = '/root/Desktop/small_train/train_covid/' + str(i) + '/' + str(name) + '.jpg'  ############文件夹路径要改
            new_see = '/root/Desktop/see1/' + str(flag_see) + '.jpg'
            shutil.copy(address, new_address)
            shutil.copy(address, new_see)
            flag_see = flag_see + 1

    if file_len >= 3:
        for name in range(1, 4):
            address = path + str(chose_rgb - 2 + name) + '.jpg'
            new_address = '/root/Desktop/small_train/train_covid/' + str(i) + '/' + str(name) + '.jpg'  ############文件夹路径要改
            shutil.copy(address, new_address)
            new_see = '/root/Desktop/see1/' + str(flag_see) + '.jpg'
            shutil.copy(address, new_see)
            flag_see = flag_see + 1

