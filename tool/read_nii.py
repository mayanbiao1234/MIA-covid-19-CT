import SimpleITK as sitk
import nibabel
from PIL import Image
import os
import numpy as np

## read
itk_img = sitk.ReadImage('/root/data/covid19/nii_train/nii_covid/0.nii.gz')
img = sitk.GetArrayFromImage(itk_img)
print("img shape:", img.shape)

img_other = np.array(nibabel.load('/root/data/covid19/nii_train/nii_covid/0.nii.gz').get_fdata())
print(img_other.shape)