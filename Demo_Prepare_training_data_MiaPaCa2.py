#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import Miapaca2 fingerprint SRS images for 3D denoising using Unet_Denoising_hyperspectral network

Haonan Lin
11/14/2018
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

from Misc_and_data_IO import plot_some, RawData, create_patches, no_background_patches

# Load raw data from folder. Specify source and target image dir, as well as image axes

raw_data = RawData.from_folder (
    basepath    = './Data/Training_set/',
    source_dirs = ['low_SNR'],
    target_dir  = 'GT',
    axes        = 'ZYX',
)

# Generate 3D training patches 
X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (40,64,64),
    n_patches_per_image = 256,
    patch_filter  = no_background_patches(threshold=0.95, percentile=99.9),
    save_file           = './Data/Training_set/Training_set.npz',
)

print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)


# shows the maximum projection of some of the generated patch pairs 
for i in range(2):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1)), 0
    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
    plt.show()
None;
