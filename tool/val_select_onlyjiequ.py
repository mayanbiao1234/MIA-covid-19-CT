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
mkdir('/root/Desktop/val_onlyjiequ/val_noncovid_onlyjieduan')
for i in range(0,209):   ############文件夹路径要改 cobid(0,165) non-covid(0,209)
    path = '/root/Desktop/data/val/non-covid/ct_scan_' + str(i) + '/' ############文件夹路径要改
    file_len = len(os.listdir(path))
    print(file_len)
    path2 = '/root/Desktop/val_onlyjiequ/val_noncovid_onlyjieduan/new_ct_scan_' + str(i)  ############文件夹路径要改
    mkdir(path2)
    if file_len < 10:
        for value in os.listdir(path):
            address = path + value
            new_address = '/root/Desktop/val_onlyjiequ/val_noncovid_onlyjieduan/new_ct_scan_' + str(i) + '/' + value  ############文件夹路径要改
            shutil.copy(address, new_address)
    if file_len >= 10 and file_len < 50:
        file_min = int(file_len * 0.35)
        file_max = file_len - int(file_len * 0.35)
        print("长度为%d,删掉%d及之前的,并删掉%d及之后的" % (file_len, file_min, file_max))
    if file_len >= 50:
        file_min = int(file_len * 0.30)
        file_max = file_len - int(file_len * 0.30)
        print("长度为%d,删掉%d及之前的,并删掉%d及之后的" % (file_len, file_min, file_max))
    if file_len >= 10:
        for value in os.listdir(path):
            #print(value)
            value_number = os.path.splitext(value)
            #print(value_number)
            if int(value_number[0]) > file_min and int(value_number[0]) < file_max:
                address = path + value
                new_address = '/root/Desktop/val_onlyjiequ/val_noncovid_onlyjieduan/new_ct_scan_' + str(i) + '/' + value  ############文件夹路径要改
                shutil.copy(address, new_address)
print("finished!")

