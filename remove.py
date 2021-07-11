import os
import shutil

determination = '/root/Desktop/train/non-covid/'
if not os.path.exists(determination):
    os.makedirs(determination)

path = r'/root/Desktop/data/train/non-covid'
folders = os.listdir(path)
print(folders)
for folder in folders:
    dir = path + '/' + str(folder)  #'/root/Desktop/data/train/covid/ct_scan_23'
    files = os.listdir(dir)
    for file in files:
        source = dir + '/' + str(file)
        deter = determination + str(str(folder) + '_' + str(file))
        shutil.copyfile(source, deter)