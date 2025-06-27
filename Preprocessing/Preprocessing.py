
import SimpleITK as sitk
import numpy as np



def Resample_uniform(image,output_spacing):

    nx = image.GetSize()[0]
    ny = image.GetSize()[1]
    nz = image.GetSize()[2]
    IT  = sitk.Transform(3, sitk.sitkIdentity) 
    extreme_points = [image.TransformIndexToPhysicalPoint((0,0,0)),
    image.TransformIndexToPhysicalPoint((nx,0,0)),
    image.TransformIndexToPhysicalPoint((nx,ny,0)),
    image.TransformIndexToPhysicalPoint((nx,ny,nz)),
    image.TransformIndexToPhysicalPoint((nx,0,nz)),
    image.TransformIndexToPhysicalPoint((0,ny,nz)),
    image.TransformIndexToPhysicalPoint((0,ny,0)),
    image.TransformIndexToPhysicalPoint((0,0,nz))]
    
    extreme_points_transformed = [IT.TransformPoint(pnt) 
        for pnt in extreme_points]
    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
    max_x = max(extreme_points_transformed)[0]
    max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    output_direction = [1.0, 0.0, 0.0,0.0, 1.0,0.0,0.0,0.0,1.0]
    # Minimal x,y coordinates are the new origin.
    output_origin = [min_x, min_y,min_z]
    # Compute grid size based on the physical size and spacing.
    output_size = [int((max_x-min_x)/output_spacing[0]), 
                   int((max_y-min_y)/output_spacing[1]),
                   int((max_z-min_z)/output_spacing[2])]
   
    
    resized_img_2 = sitk.Resample(image , output_size, IT, 
                                    sitk.sitkNearestNeighbor, output_origin, 
                                    output_spacing, output_direction)
   

    
    return resized_img_2 


def Resample_Size(image,S):
    newSize = [S,S,S]
    m = image.GetWidth()
    n = image.GetHeight()
    nz = image.GetDepth()

    s = image.GetSpacing()
    output_spacing = [s[0]*1*m/S,s[1]*n/S,s[2]*nz/S]
    #output_spacing = [1.7,2.0,2.0] 
    resized_img_1 = sitk.Resample(image, newSize, sitk.Transform(),
                              sitk.sitkNearestNeighbor, image.GetOrigin(),output_spacing,
                          image.GetDirection(), 0.0, image.GetPixelID())
    
    return resized_img_1

def Int_normalize(img):
    img  = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    return img

def Crop_Images(Images_all, Masks_all):
    min_box_all = []
    max_box_all = []
    for i in range(0,len(Images_all)):
        M =   Masks_all[i]
        a = np.where(M> 0.1)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]),np.min(a[2]), np.max(a[2])
        ids = np.where(M<0.01)
        I = Images_all[i]
        I[ids[0],ids[1],ids[2]] = 0
        min_box_all.append(bbox[0])
        max_box_all.append(bbox[1])
    
    mb = np.min(np.array(min_box_all))
    Mb = np.min(np.array(max_box_all))
    if ((Mb-mb)%2):
        Mb = Mb +1
    return mb,Mb

def Pad_Img(I):
    nx = 256-I.shape[0]
    ny = 256-I.shape[1]
    nz = 256-I.shape[2]
    I_new = np.pad(I, ((round(nx/2),round(nx/2)),(round(ny/2),round(ny/2)),(round(nz/2),round(nz/2))), 'constant')
    return I_new


def crop_center(I,S_vec):
    sx = round(S_vec[0]/2)
    sy = round(S_vec[1]/2)
    sz = round(S_vec[2]/2)
     
    nx = round(I.shape[0]/2)
    ny = round(I.shape[1]/2)
    nz = round(I.shape[2]/2)
    
    I_c = I[nx-sx:nx+sx,ny-sy:ny+sy,nz-sz:nz+sz]
    return I_c
 
    
