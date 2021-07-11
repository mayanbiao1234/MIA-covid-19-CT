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
for i in range(0, 209): #train_covid(0,689)
    path = '/root/Desktop/val2/noncovid/ct_scan_'+str(i)+'/'
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
    sitk.WriteImage(out_img, '/root/data/covid19/covid19_gaosi/nii_val/nii_noncovid/' + str(i) + '.nii.gz')

    print(IMG.shape)

    print(i)
    print('')
