#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting a noisy hsSRS image using a trained network
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import time

from tifffile import imread
from Misc_and_data_IO import plot_some, save_tiff_imagej_compatible , limit_gpu_memory
from Model import Unet_Denoising_hyperspectral

limit_gpu_memory(fraction=1/2)
        
"""
Load trained model (located in base directory models with name my_model).
The configuration was saved during training and is automatically loaded 
when DLSRS is initialized with config=None.
"""

model = Unet_Denoising_hyperspectral (config=None, name='miapaca2_dataset', basedir='models')
axes = 'ZYX'

x = imread('./Data/Testing_set/low_SNR/MiaPaCa2_fixed_control_889nm_30mW_1040nm_200mW_nframe_10_6rods_1.tif')

print('image size =', x.shape)
print('image axes =', axes)

start = time.time()
restored = model.predict(x, axes,n_tiles=16)
end = time.time()
elapsed = end - start
print('Prediction time =', elapsed, ' Seconds')

save_tiff_imagej_compatible('Results/%s_MiaPaCa2_fixed_control_889nm_30mW_1040nm_200mW_nframe_10_6rods_1.tif' % model.name, restored, axes )

plt.figure(figsize=(16,10))
plot_some(np.stack([x,restored]),
          title_list=[['low SNR (maximum projection)','U-net denoising (maximum projection)']], 
          pmin=2,pmax=99.8);