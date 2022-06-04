#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:18:11 2020

@author: ssamahkh@bm.technion.ac.il
This script performs affine registration to all datasets and saves the new 
processed images in a path determined by the user
"""
import sys
sys.path.append('/tcmldrive/Samah/ISBI21')
import nibabel as nib
from MulSliceView import multi_slice_viewer
from Preprocessing import crop_center
import os
import SimpleITK as sitk
from Preprocessing import Int_normalize
from Preprocessing import Crop_Images
from Preprocessing import Pad_Img
from Preprocessing import Resample_Size
from Preprocessing import Resample_uniform
import numpy as np
import matplotlib.pylab as plt
from AffineReg import CallAffineReg
from dipy.viz import regtools


#%%%%%%% Load Data 
#Dataname = 'MGH10/MGH10'
#Dataname = 'ISBR18'
#Dataname = 'CUMC12'
Dataname = 'LPBA40'
affine_ctrl = True
headPath4 = '/Database/'+Dataname+'/Heads'
AtlasPath4= '/Database/'+Dataname+'/Atlases'
MasksPath4 =  '/Database/'+Dataname+'/Masks'
Images_all_4 = list()
Atlases_all_4 = list()
Masks_all_4 = list()

for files in os.listdir(headPath4):
    image = nib.load(os.path.join(headPath4,files))
    Images_all_4.append(Pad_Img(image.get_fdata().squeeze()))
    if  Dataname=='LPBA40' :
        num = int(files[1:3])
        atlas=nib.load(os.path.join(AtlasPath4,'l'+str(num)+'.img'))
        #atlas = sitk.GetArrayFromImage(atlas)
    else :
        atlas=nib.load(os.path.join(AtlasPath4,files))
    Atlases_all_4.append(Pad_Img(atlas.get_fdata().squeeze()))
    mask = nib.load(os.path.join(MasksPath4,files))
    mask = Pad_Img(mask.get_fdata())
    mask = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(mask),1)
    Masks_all_4.append(sitk.GetArrayFromImage(mask))

Images_all_4  = [Images_all_4[i]*Masks_all_4[i] for i in range(0,len(Images_all_4))]
######## Flip Axis (Transpose for the two first dims)
  

Images_all_4 = [ np.rollaxis(np.rollaxis(Ii,1),2)  for Ii in Images_all_4]
Atlases_all_4 =  [np.rollaxis(np.rollaxis(Ii,1),2) for Ii in Atlases_all_4]

    
multi_slice_viewer(Images_all_4[0].squeeze()) 
multi_slice_viewer(Atlases_all_4[0].squeeze())   

#%%%%%%% Preprocessing 
Images_all_p_4 = list()
Atlases_all_p_4 = list()
output_spacing = [1.,1.,1.]
for i in range(0,len(Images_all_4)):
    I = Images_all_4[i]
    #Is = sitk.GetImageFromArray(I)
    #print(Is.GetSpacing())
    Is = Pad_Img(I.squeeze())
    Is = Resample_uniform(sitk.GetImageFromArray(Is),output_spacing)
    Is = sitk.GetArrayFromImage(Is).squeeze()
    Images_all_p_4.append(Is)
    A = Atlases_all_4[i]
    As = Pad_Img(A.squeeze())
    As = Resample_uniform(sitk.GetImageFromArray(As),output_spacing)
    As = sitk.GetArrayFromImage(As).squeeze()
    Atlases_all_p_4.append(As)
    
Images_all_p_4 = [crop_center(Ii,(160,192,224)) for Ii in Images_all_p_4]  
Atlases_all_p_4 = [crop_center(Ai,(160,192,224)) for Ai in Atlases_all_p_4]    
multi_slice_viewer(Images_all_p_4[10])
multi_slice_viewer(Atlases_all_p_4[10])


#%%%%%%%%% Affine Reg 
if affine_ctrl == True :
    Images_all_p_4 = [Int_normalize(Ii) for Ii in Images_all_p_4]  
    #Atlases_all_p_4 = [Ai for Ai in Atlases_all_4] 
    regtools.overlay_slices(Images_all_p_4[0], Images_all_p_4[1], None, 0,
                             "Template", "Transformed")
    for i in range(0,2): #len(Images_all_p_4)
        print('------------- Strating : iteration '+str(i)+'------------')
        trans_image, fixed_image,trans_atlas = CallAffineReg(Images_all_p_4[i], Images_all_p_4[0],Atlases_all_p_4[i])
        #trans_atlas, fixed_atlas = CallAffineReg(Atlases_all_p_4[i], Atlases_all_p_4[0])
    
        Images_all_p_4[i] = trans_image
        Atlases_all_p_4[i] = trans_atlas
    
    regtools.overlay_slices(Images_all_p_4[0], Images_all_p_4[1], None, 0,
                             "Template", "Transformed") 
    regtools.overlay_slices(Atlases_all_p_4[0], Atlases_all_p_4[1], None, 0,
                         "Template", "Transformed")  

#%%%%%%%%% Imshow Example  
s = 41
img0 = Images_all_p_4[0][s,...].squeeze()
img1 = Images_all_p_4[1][s,...].squeeze()
img0 = 255 * ((img0 - img0.min()) / (img0.max() - img0.min()))
img1 = 255 * ((img1 - img1.min()) / (img1.max() - img1.min()))
img0_red = np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
img1_green = np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
overlay = np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)

img0_red[..., 0] = img0
img1_green[..., 1] = img1
overlay[..., 0] = img0
overlay[..., 1] = img1

fig = plt.figure(figsize=(6, 1.9))
ax = fig.add_subplot(131)
ax.imshow(img0,cmap='gray')
plt.axis('off')
ax = fig.add_subplot(132)
ax.imshow(overlay)
plt.axis('off')
ax = fig.add_subplot(133)
ax.imshow(img1,cmap='gray')
plt.axis('off')

#%%%%%%%% Save New Images Data_PostPro

PathImgs = '/Data_PostPro/'+Dataname+'/Heads'
PathAtls = '/Data_PostPro/'+Dataname+'/Atlases'
for i in range(0,len(Images_all_p_4)):
    img = nib.Nifti1Image(Images_all_p_4[i],image.affine)
    nib.save(img, os.path.join(PathImgs,'Proc_img_'+str(i)+'.nii.gz'))
    atlas = nib.Nifti1Image(Atlases_all_p_4[i],image.affine)
    nib.save(atlas, os.path.join(PathAtls,'Proc_Atlas_'+str(i)+'.nii.gz'))