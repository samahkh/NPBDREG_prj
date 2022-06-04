#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:18:31 2020

@author: ssamahkh@bm.technion.ac.il
"""
import sys

import nibabel as nib
import os
from Preprocessing import Int_normalize
from Preprocessing import crop_center
import sklearn.model_selection
import numpy as np

#%%%%%%% Load LPBA40 Data 
#Dataname = 'MGH10'
#Dataname = 'ISBR18'
#Dataname = 'CUMC12'
Dataname = 'LPBA40'
SaveAllData = True
SavePerData = False
affine_ctrl = True
headPath4 = '/Data_PostPro/'+Dataname+'/Heads'
AtlasPath4= '/Data_PostPro/'+Dataname+'/Atlases'

Images_all_4 = list()
Atlases_all_4 = list()
for files in os.listdir(headPath4):
    image = nib.load(os.path.join(headPath4,files))
    Images_all_4.append(image.get_fdata())
for files in os.listdir(AtlasPath4):  
    atlas=nib.load(os.path.join(AtlasPath4,files))
    Atlases_all_4.append(atlas.get_fdata()) 
          

multi_slice_viewer(Images_all_4[1].squeeze()) 
multi_slice_viewer(Atlases_all_4[1].squeeze())  


#%%%% Process Atlases
def Segment_atlas(atlas,n_mean):
    A = np.round(atlas)
    labels_all = np.unique(A)
    #n_all = [np.sum(A==i) for i in labels_all]
    #labels = np.where(np.array(n_all)>10000)[0]
    labels = np.array(n_mean).argsort()[-8:]
    Anew = np.round(np.zeros(A.shape))
    for l in labels:
        Anew[np.where(A==labels_all[l])] = int(labels_all[l])
    return Anew
def Filter_atlas(atlas,labels):
    A = np.round(atlas)
    Anew = np.round(np.zeros(A.shape))
    for l in labels:
        Anew[np.where(A==int(l))] = int(l)
    return Anew
def inter_atlas(a,ref):
    lint = np.intersect1d(np.unique(ref),np.unique(a)) 
    Anew = np.round(np.zeros(a.shape))
    for l in lint :
        Anew[np.where(a==l)] = int(l)
    return Anew
    
#%%%%%%%%%

for i in range(len(Atlases_all_4)):
    for j in range(len(Atlases_all_4)):
        Atlases_all_4[i] = inter_atlas(Atlases_all_4[i],Atlases_all_4[j])

  
#%%%%%%% Split into Training, Validation and Test 


Images_all_f = [Int_normalize(Ii) for Ii in Images_all_4]
          
#Atlases_all_4 = [inter_atlas(Ai,Atlases_all_4[1]) for Ai in Atlases_all_4]

#Atlases_all_4 = [ Filter_atlas(Ai,labels) for Ai in Atlases_all_4]
                      
X_train, X_test, y_train, y_test  = sklearn.model_selection.train_test_split(
        Images_all_f, Atlases_all_4, test_size=0.3, random_state=1)

X_train, X_val, y_train, y_val  = sklearn.model_selection.train_test_split(
        X_train, y_train, test_size=0.2, random_state=1)


#%%%%%%%%%
'''
if SavePerData == True:
    PathImgs = '/tcmldrive/Samah/FullDataReg/Data/Train/'+Dataname+'/Heads'
    PathAtls = '/tcmldrive/Samah/FullDataReg/Data/Train/'+Dataname+'/Atlases'
    for i in range(0,len(X_train)):
        img = nib.Nifti1Image(X_train[i],image.affine)
        nib.save(img, os.path.join(PathImgs,'img_'+str(i)+'.nii.gz'))
        atlas = nib.Nifti1Image(y_train[i],image.affine)
        nib.save(atlas, os.path.join(PathAtls,'Atlas_'+str(i)+'.nii.gz'))
    
    PathImgs = '/tcmldrive/Samah/FullDataReg/Data/Eval/'+Dataname+'/Heads'
    PathAtls = '/tcmldrive/Samah/FullDataReg/Data/Eval/'+Dataname+'/Atlases'
    for i in range(0,len(X_val)):
        img = nib.Nifti1Image(X_val[i],image.affine)
        nib.save(img, os.path.join(PathImgs,'img_'+str(i)+'.nii.gz'))
        atlas = nib.Nifti1Image(y_val[i],image.affine)
        nib.save(atlas, os.path.join(PathAtls,'Atlas_'+str(i)+'.nii.gz'))
    
    PathImgs = '/tcmldrive/Samah/FullDataReg/Data/Test/'+Dataname+'/Heads'
    PathAtls = '/tcmldrive/Samah/FullDataReg/Data/Test/'+Dataname+'/Atlases'
    for i in range(0,len(X_test)):
        img = nib.Nifti1Image(X_test[i],image.affine)
        nib.save(img, os.path.join(PathImgs,'img_'+str(i)+'.nii.gz'))
        atlas = nib.Nifti1Image(y_test[i],image.affine)
        nib.save(atlas, os.path.join(PathAtls,'Atlas_'+str(i)+'.nii.gz'))
        
 '''       
#%%%%%%%%%
if SaveAllData == True:
    PathImgs = '/tcmldrive/Samah/FullDataReg/AllData/Train/Heads'
    PathAtls = '/tcmldrive/Samah/FullDataReg/AllData/Train/Atlases'
    for i in range(0,len(X_train)):
        img = nib.Nifti1Image(X_train[i],image.affine)
        nib.save(img, os.path.join(PathImgs,Dataname+'_img_'+str(i)+'.nii.gz'))
        atlas = nib.Nifti1Image(y_train[i],image.affine)
        nib.save(atlas, os.path.join(PathAtls,Dataname+'_Atlas_'+str(i)+'.nii.gz'))
    
    PathImgs = '/tcmldrive/Samah/FullDataReg/AllData/Eval/Heads'
    PathAtls = '/tcmldrive/Samah/FullDataReg/AllData/Eval/Atlases'
    for i in range(0,len(X_val)):
        img = nib.Nifti1Image(X_val[i],image.affine)
        nib.save(img, os.path.join(PathImgs,Dataname+'_img_'+str(i)+'.nii.gz'))
        atlas = nib.Nifti1Image(y_val[i],image.affine)
        nib.save(atlas, os.path.join(PathAtls,Dataname+'_Atlas_'+str(i)+'.nii.gz'))
    
    PathImgs = '/tcmldrive/Samah/FullDataReg/AllData/Test/Heads'
    PathAtls = '/tcmldrive/Samah/FullDataReg/AllData/Test/Atlases'
    for i in range(0,len(X_test)):
        img = nib.Nifti1Image(X_test[i],image.affine)
        nib.save(img, os.path.join(PathImgs,Dataname+'_img_'+str(i)+'.nii.gz'))
        atlas = nib.Nifti1Image(y_test[i],image.affine)
        nib.save(atlas, os.path.join(PathAtls,Dataname+'_Atlas_'+str(i)+'.nii.gz'))
