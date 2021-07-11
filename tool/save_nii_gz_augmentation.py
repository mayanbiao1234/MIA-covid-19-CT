import SimpleITK as sitk
from PIL import Image
import os
import numpy as np
from skimage import transform

## read
# itk_img = sitk.ReadImage('/root/Desktop/14.nii.gz')
# img = sitk.GetArrayFromImage(itk_img)
# print("img shape:", img.shape)

## save
for i in range(0, 689): #train_covid(0,689)
    path = '/root/Desktop/train1_angumentation/covid_rot15/ct_scan_'+str(i)+'/'
    IMG = np.zeros((56, 448, 448)).astype(np.float32)  ## 3d size
    #print(IMG.dtype)
    for j in range(0, 56):
        img = np.array(Image.open(path + str(j) +'.jpg'))
        if j==0: print(img.shape)
        if img.shape != (448, 448):
            img = transform.resize(img, (int(448), int(448)))

            if j == 0: print(img.shape)
        IMG[j,:,:] = img
    mean_1 = np.mean(IMG)
    std_1 = np.std(IMG)
    # print(IMG)
    IMG = (IMG - mean_1) / std_1
    # print(IMG)
    # print(IMG.max())
    # print(IMG.min())
    # print(IMG.mean())

    #IMG = IMG.astype(np.float32)
    #print(IMG.dtype)
    out_img = sitk.GetImageFromArray(IMG)
    sitk.WriteImage(out_img, '/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train/nii_covid/' + str(i+689) + '.nii.gz')

    print(IMG.shape)

    print(i+689)
    print('')

for i in range(0, 689): #train_covid(0,689)
    path = '/root/Desktop/train1_angumentation/covid_rot345/ct_scan_'+str(i)+'/'
    IMG = np.zeros((56, 448, 448)).astype(np.float32)  ## 3d size
    #print(IMG.dtype)
    for j in range(0, 56):
        img = np.array(Image.open(path + str(j) +'.jpg'))
        if j==0: print(img.shape)
        if img.shape != (448, 448):
            img = transform.resize(img, (int(448), int(448)))

            if j == 0: print(img.shape)
        IMG[j,:,:] = img
    mean_1 = np.mean(IMG)
    std_1 = np.std(IMG)
    # print(IMG)
    IMG = (IMG - mean_1) / std_1
    # print(IMG)
    # print(IMG.max())
    # print(IMG.min())
    # print(IMG.mean())

    #IMG = IMG.astype(np.float32)
    #print(IMG.dtype)
    out_img = sitk.GetImageFromArray(IMG)
    sitk.WriteImage(out_img, '/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train/nii_covid/' + str(i+689*2) + '.nii.gz')

    print(IMG.shape)

    print(i+689*2)
    print('')


for i in range(0, 689): #train_covid(0,689)
    path = '/root/Desktop/train1_angumentation/covid_shui/ct_scan_'+str(i)+'/'
    IMG = np.zeros((56, 448, 448)).astype(np.float32)  ## 3d size
    #print(IMG.dtype)
    for j in range(0, 56):
        img = np.array(Image.open(path + str(j) +'.jpg'))
        if j==0: print(img.shape)
        if img.shape != (448, 448):
            img = transform.resize(img, (int(448), int(448)))

            if j == 0: print(img.shape)
        IMG[j,:,:] = img
    mean_1 = np.mean(IMG)
    std_1 = np.std(IMG)
    # print(IMG)
    IMG = (IMG - mean_1) / std_1
    # print(IMG)
    # print(IMG.max())
    # print(IMG.min())
    # print(IMG.mean())

    #IMG = IMG.astype(np.float32)
    #print(IMG.dtype)
    out_img = sitk.GetImageFromArray(IMG)
    sitk.WriteImage(out_img, '/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train/nii_covid/' + str(i+689*3) + '.nii.gz')

    print(IMG.shape)

    print(i+689*3)
    print('')


for i in range(0, 871): #train_covid(0,689)
    path = '/root/Desktop/train1_angumentation/noncovid_rot15/ct_scan_'+str(i)+'/'
    IMG = np.zeros((56, 448, 448)).astype(np.float32)  ## 3d size
    #print(IMG.dtype)
    for j in range(0, 56):
        img = np.array(Image.open(path + str(j) +'.jpg'))
        if j==0: print(img.shape)
        if img.shape != (448, 448):
            img = transform.resize(img, (int(448), int(448)))

            if j == 0: print(img.shape)
        IMG[j,:,:] = img
    mean_1 = np.mean(IMG)
    std_1 = np.std(IMG)
    # print(IMG)
    IMG = (IMG - mean_1) / std_1
    # print(IMG)
    # print(IMG.max())
    # print(IMG.min())
    # print(IMG.mean())

    #IMG = IMG.astype(np.float32)
    #print(IMG.dtype)
    out_img = sitk.GetImageFromArray(IMG)
    sitk.WriteImage(out_img, '/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train/nii_noncovid/' + str(i+871) + '.nii.gz')

    print(IMG.shape)

    print(i+871)
    print('')


for i in range(0, 871): #train_covid(0,689)
    path = '/root/Desktop/train1_angumentation/noncovid_rot345/ct_scan_'+str(i)+'/'
    IMG = np.zeros((56, 448, 448)).astype(np.float32)  ## 3d size
    #print(IMG.dtype)
    for j in range(0, 56):
        img = np.array(Image.open(path + str(j) +'.jpg'))
        if j==0: print(img.shape)
        if img.shape != (448, 448):
            img = transform.resize(img, (int(448), int(448)))

            if j == 0: print(img.shape)
        IMG[j,:,:] = img
    mean_1 = np.mean(IMG)
    std_1 = np.std(IMG)
    # print(IMG)
    IMG = (IMG - mean_1) / std_1
    # print(IMG)
    # print(IMG.max())
    # print(IMG.min())
    # print(IMG.mean())

    #IMG = IMG.astype(np.float32)
    #print(IMG.dtype)
    out_img = sitk.GetImageFromArray(IMG)
    sitk.WriteImage(out_img, '/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train/nii_noncovid/' + str(i+871*2) + '.nii.gz')

    print(IMG.shape)

    print(i+871*2)
    print('')


for i in range(0, 871): #train_covid(0,689)
    path = '/root/Desktop/train1_angumentation/noncovid_shui/ct_scan_'+str(i)+'/'
    IMG = np.zeros((56, 448, 448)).astype(np.float32)  ## 3d size
    #print(IMG.dtype)
    for j in range(0, 56):
        img = np.array(Image.open(path + str(j) +'.jpg'))
        if j==0: print(img.shape)
        if img.shape != (448, 448):
            img = transform.resize(img, (int(448), int(448)))

            if j == 0: print(img.shape)
        IMG[j,:,:] = img
    mean_1 = np.mean(IMG)
    std_1 = np.std(IMG)
    # print(IMG)
    IMG = (IMG - mean_1) / std_1
    # print(IMG)
    # print(IMG.max())
    # print(IMG.min())
    # print(IMG.mean())

    #IMG = IMG.astype(np.float32)
    #print(IMG.dtype)
    out_img = sitk.GetImageFromArray(IMG)
    sitk.WriteImage(out_img, '/root/data/covid19_dataaugment/covid19_junyun_augmentation/nii_train/nii_noncovid/' + str(i+871*3) + '.nii.gz')

    print(IMG.shape)

    print(i+871*3)
    print('')

