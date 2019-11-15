#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training a 3D Unet for hyperspectral image denoising
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import matplotlib.pyplot as plt

from Misc_and_data_IO import load_training_data, axes_dict, plot_some, plot_history, limit_gpu_memory
from Model import Config, Unet_Denoising_hyperspectral

limit_gpu_memory(fraction=1/2)

# Load training data and validation data
(X,Y), (X_val,Y_val), axes = load_training_data('./Data/Training_set/Training_set.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('Example validation patches (top: low SNR, bottom: GT)');

# Config a Unet model for denoising. Note traing_steps_per_epoch needs to be ~400 to get promising results
config = Config(axes, n_channel_in, n_channel_out, residual = True, unet_n_depth=2, train_steps_per_epoch=10)
print(config)
vars(config)

# We now create a Unet model with the chosen configuration:
model = Unet_Denoising_hyperspectral (config, 'miapaca2_dataset', basedir='./Models')

# Train the DLSRS model
history = model.train(X,Y, validation_data=(X_val,Y_val))

#Plot final training history
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);

# Example results for validation images
plt.figure(figsize=(12,7))
_P = model.keras_model.predict(X_val[0:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[0:5],Y_val[0:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');
             
# Export trained model for prediction in the future
model.export_TF()
