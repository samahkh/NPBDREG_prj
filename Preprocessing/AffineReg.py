#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:08:07 2020

@author: ssamahkh@bm.technion.ac.il
"""

#from dipy.viz import regtools
from dipy.align.metrics import CCMetric
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                  AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                    AffineTransform3D)

def CallAffineReg(I_moving, I_fixed,atlas_moving):
    
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    
    # The optimization strategy
    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 1.0]
    factors = [4, 2, 1]
    
    affreg = AffineRegistration(metric=metric,
                                 level_iters=level_iters,
                                sigmas=sigmas,
                                 factors=factors)
    
    
    
    transform = AffineTransform3D()
    params0 = None
    template_data = I_fixed
    moving_data = I_moving
    translation = affreg.optimize(template_data, moving_data, transform, params0)
    
    
    transformed = translation.transform(moving_data,interp='nearest')
    transformed_atlas = translation.transform(atlas_moving,interp='nearest')
    # If you with to show results
    '''
    regtools.overlay_slices(template_data[...], transformed[...], None, 0,
                             "Template", "Transformed")
    
    regtools.overlay_slices(template_data[...], moving_data[...], None, 0,
                             "Template", "Transformed")

    '''
    return transformed, template_data,transformed_atlas