#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:30:51 2019

@author: haonan
"""
from keras.layers import Input, Conv3D, Activation, Lambda, BatchNormalization
from keras.layers import MaxPooling3D, UpSampling3D, Dropout
from keras.models import Model
from keras.layers.merge import Add, Concatenate
from keras.callbacks import Callback, TerminateOnNaN
from keras.utils import Sequence
from keras import backend as K
from six import string_types
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

from Misc_and_data_IO import backend_channels_last, _raise, move_channel_for_backend, tile_iterator 
from Misc_and_data_IO import Resizer, consume, is_tf_backend, axes_check_and_normalize, axes_dict, load_json, save_json
from Misc_and_data_IO import export_SavedModel, PercentileNormalizer, PadAndCropResizer, move_image_axes, NoNormalizer

import numpy as np
import tensorflow as tf
import os
import datetime
import argparse
import warnings

"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Internal functions using Keras to build network
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
"""    



def unet(n_dim=3, n_depth=1, kern_size=3, n_first=16, n_channel_out=1, residual=True, prob_out=False, last_activation='linear'):
    """Construct a deeep neural network with U-Net and residual learning

    Parameters
    ----------
    n_dim : int
        number of image dimensions (2 or 3)
    n_depth : int
        number of resolution levels of U-Net architecture
    kern_size : int
        size of convolution filter in all image dimensions
    n_first : int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out : int
        number of channels of the predicted output image
    residual : bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal   
    last_activation : str
        name of activation function for the final output layer

    Returns
    -------
    function
        Function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = unet(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
    """
    def _build_this(input_shape):
        return custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,)*n_dim, pool_size=(2,)*n_dim, n_channel_out=n_channel_out, residual=residual)
    return _build_this

def custom_unet(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                pool_size=(2,2,2),
                n_channel_out=1,
                residual=False,
                eps_scale=1e-3):

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    
    n_dim = len(kernel_size)
    conv = Conv3D

    input = Input(input_shape, name = "input")
    unet_body = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet_body)
    if residual:
        if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
            raise ValueError("number of input and output channels must be the same for a residual net.")
        final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    return Model(inputs=input, outputs=final)

def unet_block(n_depth=2, n_filter_base=16, kernel_size=(3,3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2,2),
               prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block =  conv_block3
    pooling    =  MaxPooling3D
    upsampling =  UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func



def conv_block3(n_filter, n1, n2, n3,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func

# Define different losses 
    
def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)


def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[...,:n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:,:n,...] - y_true))
        return mae

def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[...,:n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:,:n,...] - y_true))
        return mse

    
def prepare_model(model, optimizer, loss, metrics=('mse','mae'),
                  loss_bg_thresh=0, loss_bg_decay=0.06, Y=None):

    from keras.optimizers import Optimizer
    isinstance(optimizer,Optimizer) or _raise(ValueError())

    loss_standard   = eval('loss_%s()'%loss)
    _metrics        = [eval('loss_%s()'%m) for m in metrics]
    callbacks       = [TerminateOnNaN()]
   
    # loss
    if loss_bg_thresh == 0:
        _loss = loss_standard
    else:
        freq = np.mean(Y > loss_bg_thresh)
        # print("class frequency:", freq)
        alpha = K.variable(1.0)
        loss_per_pixel = eval('loss_{loss}(mean=False)'.format(loss=loss))
        _loss = loss_thresh_weighted_decay(loss_per_pixel, loss_bg_thresh,
                                           0.5 / (0.1 + (1 - freq)),
                                           0.5 / (0.1 +      freq),
                                           alpha)
        callbacks.append(ParameterDecayCallback(alpha, loss_bg_decay, name='alpha'))
        if not loss in metrics:
            _metrics.append(loss_standard)


    # compile model
    model.compile(optimizer=optimizer, loss=_loss, metrics=_metrics)

    return callbacks

def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss


class ParameterDecayCallback(Callback):
    def __init__(self, parameter, decay, name=None, verbose=0):
        self.parameter = parameter
        self.decay = decay
        self.name = name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        old_val = K.get_value(self.parameter)
        if self.name:
            logs = logs or {}
            logs[self.name] = old_val
        new_val = old_val * (1. / (1. + self.decay * (epoch + 1)))
        K.set_value(self.parameter, new_val)
        if self.verbose:
            print("\n[ParameterDecayCallback] new %s: %s\n" % (self.name if self.name else 'parameter', new_val))

class DataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        return self.X[idx], self.Y[idx]

def tile_overlap(n_depth, kern_size):
    rf = {(1, 3):   9, (1, 5):  17, (1, 7): 25,
          (2, 3):  22, (2, 5):  43, (2, 7): 62,
          (3, 3):  46, (3, 5):  92, (3, 7): 138,
          (4, 3):  94, (4, 5): 188, (4, 7): 282,
          (5, 3): 190, (5, 5): 380, (5, 7): 570}
    try:
        return rf[n_depth, kern_size]
    except KeyError:
        raise ValueError('tile_overlap value for n_depth=%d and kern_size=%d not available.' % (n_depth, kern_size))


def to_tensor(x,channel=None,single_sample=True):
    if single_sample:
        x = x[np.newaxis]
        if channel is not None and channel >= 0:
            channel += 1
    if channel is None:
        x, channel = np.expand_dims(x,-1), -1
    return move_channel_for_backend(x,channel)


def from_tensor(x,channel=0,single_sample=True):
    return np.moveaxis((x[0] if single_sample else x), (-1 if backend_channels_last() else 1), channel)

def predict_direct(keras_model,x,channel_in=None,channel_out=0,single_sample=True,**kwargs):
    return from_tensor(keras_model.predict(to_tensor(x,channel=channel_in,single_sample=single_sample),**kwargs),
                       channel=channel_out,single_sample=single_sample)

def predict_tiled(keras_model,x,n_tiles,block_size,tile_overlap,channel_in=None,channel_out=0,**kwargs):
    channel_in  = (channel_in  + x.ndim) % x.ndim
    channel_out = (channel_out + x.ndim) % x.ndim

    def _remove_and_insert(x,a):
        # remove element at channel_in and insert a at channel_out
        lst = list(x)
        if channel_in is not None:
            del lst[channel_in]
        lst.insert(channel_out,a)
        return tuple(lst)

    # largest axis (that is not channel_in)
    axis = [i for i in np.argsort(x.shape) if i != channel_in][-1]

    if isinstance(n_tiles,(list,tuple)):
        0 < len(n_tiles) <= x.ndim-(0 if channel_in is None else 1) or _raise(ValueError())
        n_tiles, n_tiles_remaining = n_tiles[0], n_tiles[1:]
    else:
        n_tiles_remaining = []

    n_block_overlap = int(np.ceil(tile_overlap / block_size))
    # n_block_overlap += -1
    # n_block_overlap = 10
    # print(tile_overlap,n_block_overlap)

    dst = None
    for tile, s_src, s_dst in tile_iterator(x,axis=axis,n_tiles=n_tiles,block_size=block_size,n_block_overlap=n_block_overlap):

        if len(n_tiles_remaining) == 0:
            pred = predict_direct(keras_model,tile,channel_in=channel_in,channel_out=channel_out,**kwargs)
        else:
            pred = predict_tiled(keras_model,tile,n_tiles_remaining,block_size,tile_overlap,channel_in=channel_in,channel_out=channel_out,**kwargs)

        if dst is None:
            dst_shape = _remove_and_insert(x.shape, pred.shape[channel_out])
            dst = np.empty(dst_shape, dtype=x.dtype)

        s_src = _remove_and_insert(s_src, slice(None))
        s_dst = _remove_and_insert(s_dst, slice(None))

        dst[s_dst] = pred[s_src]

    return dst

class NoResizer(Resizer):
    """No resizing.

    Raises
    ------
    ValueError
        In :`before`, if image resizing is necessary.
    """

    def before(self, x, div_n, exclude):
        exclude = self._normalize_exclude(exclude, x.ndim)
        consume ((
            (s%div_n==0) or _raise(ValueError('%d (axis %d) is not divisible by %d.' % (s,i,div_n)))
            for i,s in enumerate(x.shape) if (i not in exclude)
        ))
        return x

    def after(self, x, exclude):
        return x


       
class _TensorBoard(Callback):
    # This function is copied from CARE network. 
    def __init__(self, log_dir='./logs',
                 freq=1,
                 compute_histograms=False,
                 n_images=3,
                 prob_out=False,
                 write_graph=False,
                 prefix_with_timestamp=True,
                 write_images=False):
        super(_TensorBoard, self).__init__()
        is_tf_backend() or _raise(RuntimeError('TensorBoard callback only works with the TensorFlow backend.'))

        self.freq = freq
        self.image_freq = freq
        self.prob_out = prob_out
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.n_images = n_images
        self.compute_histograms = compute_histograms

        if prefix_with_timestamp:
            log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))

        self.log_dir = log_dir

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        tf_sums = []

        if self.compute_histograms and self.freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    tf_sums.append(tf.summary.histogram(weight.name, weight))

                if hasattr(layer, 'output'):
                    tf_sums.append(tf.summary.histogram('{}_out'.format(layer.name),
                                                        layer.output))

        # outputs
        backend_channels_last() or _raise(NotImplementedError())

        n_channels_in = self.model.input_shape[-1]
        n_dim_in = len(self.model.input_shape)

        if n_dim_in > 4:
            # print("tensorboard shape: %s"%str(self.model.input_shape))
            input_layer = Lambda(lambda x: K.max(K.max(x, axis=1), axis=-1, keepdims=True))(self.model.input)
        else:
            if n_channels_in > 3:
                input_layer = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(self.model.input)
            elif n_channels_in == 2:
                input_layer = Lambda(lambda x: K.concatenate([x,x[...,:1]], axis=-1))(self.model.input)
            else:
                input_layer = self.model.input

        n_channels_out = self.model.output_shape[-1]
        n_dim_out = len(self.model.output_shape)

        sep = n_channels_out
        if self.prob_out:
            # first half of output channels is mean, second half scale
            # assert n_channels_in*2 == n_channels_out
            # if n_channels_in*2 != n_channels_out:
            #     raise ValueError('prob_out: must be two output channels for every input channel')
            n_channels_out % 2 == 0 or _raise(ValueError())
            sep = sep // 2

        if n_dim_out > 4:
            output_layer = Lambda(lambda x: K.max(K.max(x[...,:sep], axis=1), axis=-1, keepdims=True))(self.model.output)
        else:
            if sep > 3:
                output_layer = Lambda(lambda x: K.max(x[...,:sep], axis=-1, keepdims=True))(self.model.output)
            elif sep == 2:
                output_layer = Lambda(lambda x: K.concatenate([x[...,:sep],x[...,:1]], axis=-1))(self.model.output)
            else:
                output_layer = Lambda(lambda x: x[...,:sep])(self.model.output)

        if self.prob_out:
            # scale images
            if n_dim_out > 4:
                scale_layer = Lambda(lambda x: K.max(K.max(x[...,sep:], axis=1), axis=-1, keepdims=True))(self.model.output)
            else:
                if sep > 3:
                    scale_layer = Lambda(lambda x: K.max(x[...,sep:], axis=-1, keepdims=True))(self.model.output)
                elif sep == 2:
                    scale_layer = Lambda(lambda x: K.concatenate([x[...,sep:],x[...,-1:]], axis=-1))(self.model.output)
                else:
                    scale_layer = Lambda(lambda x: x[...,sep:])(self.model.output)

        
        tf_sums.append(tf.summary.image('input', input_layer, max_outputs=self.n_images))
        if self.prob_out:
            tf_sums.append(tf.summary.image('mean', output_layer, max_outputs=self.n_images))
            tf_sums.append(tf.summary.image('scale', scale_layer, max_outputs=self.n_images))
        else:
            tf_sums.append(tf.summary.image('output', output_layer, max_outputs=self.n_images))

        with tf.name_scope('merged'):
            self.merged = tf.summary.merge(tf_sums)
            # self.merged = tf.summary.merge([foo])

        with tf.name_scope('summary_writer'):
            if self.write_graph:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.freq:
            if epoch % self.freq == 0:
                if self.model.uses_learning_phase:
                   # cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = list(v[:self.n_images] for v in self.validation_data)
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]

                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Model config and training
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
"""    


class Config(argparse.Namespace):
    """Default configuration for a 3D Unet model for SNR recovery.

    Parameters
    ----------
    axes : str
        Axes of the neural network (channel axis optional).
    n_channel_in : int
        Number of channels of given input image.
    n_channel_out : int
        Number of channels of predicted output image.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> config = Config('YX', unet_n_depth=3)

    Attributes
    ----------
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: `` 'mae' ``
    train_epochs : int
        Number of training epochs. Default: ``100``
    train_steps_per_epoch : int
        Number of parameter update steps per epoch. Default: ``400``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_batch_size : int
        Batch size for training. Default: ``16``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable. Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable. Default: ``{'factor': 0.5, 'patience': 10}``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes, n_channel_in=1, n_channel_out=1, probabilistic=False, **kwargs):
        """See class docstring."""

        # parse and check axes
        axes = axes_check_and_normalize(axes)
        ax = axes_dict(axes)
        ax = {a: (ax[a] is not None) for a in ax}

        (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
        not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))

        axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
        axes = axes.replace('S','') # remove sample axis if it exists

        n_dim = 3 if (ax['Z'] or ax['T']) else 2
        

        if backend_channels_last():
            if ax['C']:
                axes[-1] == 'C' or _raise(ValueError('channel axis must be last for backend (%s).' % K.backend()))
            else:
                axes += 'C'
        else:
            if ax['C']:
                axes[0] == 'C' or _raise(ValueError('channel axis must be first for backend (%s).' % K.backend()))
            else:
                axes = 'C'+axes

        # directly set by parameters
        self.n_dim                 = n_dim
        self.axes                  = axes
        self.n_channel_in          = int(n_channel_in)
        self.n_channel_out         = int(n_channel_out)
        self.probabilistic         = bool(probabilistic)

        # default config (can be overwritten by kwargs below)
        self.unet_residual         = self.n_channel_in == self.n_channel_out
        self.unet_n_depth          = 2
        self.unet_kern_size        = 5 if self.n_dim==2 else 3
        self.unet_n_first          = 32
        self.unet_last_activation  = 'linear'
        if backend_channels_last():
            self.unet_input_shape  = self.n_dim*(None,) + (self.n_channel_in,)
        else:
            self.unet_input_shape  = (self.n_channel_in,) + self.n_dim*(None,)

        self.train_loss            = 'mae'
        self.train_epochs          = 100
        self.train_steps_per_epoch = 400
        self.train_learning_rate   = 0.0004
        self.train_batch_size      = 16
        self.train_tensorboard     = True
        self.train_checkpoint      = 'weights_best.h5'
        self.train_reduce_lr       = {'factor': 0.5, 'patience': 10}

        
        for k in kwargs:
            setattr(self, k, kwargs[k])


    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.

        Returns
        -------
        bool
            Flag that indicates whether the current configuration values are valid.
        """
        def _is_int(v,low=None,high=None):
            return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
            )

        ok = {}
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except:
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)
        ok['probabilistic'] = isinstance(self.probabilistic,bool)

        ok['unet_residual'] = (
            isinstance(self.unet_residual,bool) and
            (not self.unet_residual or (self.n_channel_in==self.n_channel_out))
        )
        ok['unet_n_depth']         = _is_int(self.unet_n_depth,1)
        ok['unet_kern_size']       = _is_int(self.unet_kern_size,1)
        ok['unet_n_first']         = _is_int(self.unet_n_first,1)
        ok['unet_last_activation'] = self.unet_last_activation in ('linear','relu')
        ok['unet_input_shape'] = (
            isinstance(self.unet_input_shape,(list,tuple)) and
            len(self.unet_input_shape) == self.n_dim+1 and
            self.unet_input_shape[-1] == self.n_channel_in and
            all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1]))
        )
        ok['train_loss'] = (
            (    self.probabilistic and self.train_loss == 'laplace'   ) or
            (not self.probabilistic and self.train_loss in ('mse','mae'))
        )
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())



class Unet_Denoising_hyperspectral(object):
    """Standard Unet_Denoising_hyperspectral network for spectroscopic image SNR recovery

    Parameters
    ----------
    config : :class:`Config` or None
        
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
        
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Example
    -------
    >>> model = Unet_Denoising_hyperspectral(config, 'my_model')

    Attributes
    ----------
    config : :class:`Config`
        Configuration of Unet_Denoising_hyperspectral network.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config, name=None, basedir='.'):

        config is None or isinstance(config,Config) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))

        name is None or isinstance(name,string_types) or _raise(ValueError())
        isinstance(basedir,(string_types,Path)) or _raise(ValueError())
        self.config = config
        self.basedir = Path(basedir)
        self.name = name
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()


    def _set_logdir(self):
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.logdir = self.basedir / self.name

        config_file =  self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                self.config = Config(**config_dict)
                if not self.config.is_valid():
                    invalid_attr = self.config.is_valid(True)[1]
                    raise ValueError('Invalid attributes in loaded config: ' + ', '.join(invalid_attr))
            else:
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
        else:
            if self.logdir.exists():
                warnings.warn('output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), str(config_file))


    def _find_and_load_weights(self,prefer='best'):
        from itertools import chain
        # get all weight files and sort by modification time descending (newest first)
        weights_ext   = ('*.h5','*.hdf5')
        weights_files = chain(*(self.logdir.glob(ext) for ext in weights_ext))
        weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
        weights_files = list(weights_files)
        if len(weights_files) == 0:
            warnings.warn("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
            return
        weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
        weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
        print("Loading network weights from '%s'." % weights_chosen.name)
        self.load_weights(weights_chosen.name)


    def _build(self):
        return unet(
            n_dim           = self.config.n_dim,
            n_channel_out   = self.config.n_channel_out,
            prob_out        = self.config.probabilistic,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
        )(self.config.unet_input_shape)


    def load_weights(self, name='weights_best.h5'):
        """Load neural network weights from model folder.

        Parameters
        ----------
        name : str
            Name of HDF5 weight file (as saved during or after training).
        """
        self.keras_model.load_weights(str(self.logdir/name))


    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.

        Calls `prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`prepare_model`.

        """
        if optimizer is None:
            from keras.optimizers import Adam
            optimizer = Adam(lr=self.config.train_learning_rate)
        self.callbacks = prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.config.train_checkpoint is not None:
            from keras.callbacks import ModelCheckpoint
            self.callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True, save_weights_only=True))

        if self.config.train_tensorboard:
            #from ..utils.tf import _TensorBoard
            self.callbacks.append(_TensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=self.config.probabilistic))

        if self.config.train_reduce_lr is not None:
            from keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X,Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """

        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))
        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)
        div_by = 2**self.config.unet_n_depth
        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
        for a in axes_relevant:
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axes %s"
                    " (axis %s has incompatible size %d)" % (div_by,axes_relevant,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        training_data = DataWrapper(X, Y, self.config.train_batch_size)

        history = self.keras_model.fit_generator(generator=training_data, validation_data=validation_data,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)

        self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

        if self.config.train_checkpoint is not None:
            self.load_weights(self.config.train_checkpoint)

        return history


    def export_TF(self):
        """Export neural network via :func:`export_SavedModel`."""
        fout = self.logdir / 'TF_SavedModel.zip'
        meta = {
            'type':          self.__class__.__name__,
            #'version':       package_version,
            'probabilistic': self.config.probabilistic,
            'axes':          self.config.axes,
            'axes_div_by':   [(2**self.config.unet_n_depth if a in 'XYZT' else 1) for a in self.config.axes],
            'tile_overlap':  tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size),
        }
        export_SavedModel(self.keras_model, str(fout), meta=meta)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fout.resolve()))


    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image
        axes : str
            Axes of the input ``img``.
        normalizer : :class:`Normalizer` or None
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :'Resizer` or None
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        n_tiles : int
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes the number of tiles. Note that if the number of tiles is too low,
            it is adaptively increased until OOM errors are avoided, albeit at the expense of runtime.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes semantics are the same as in the input image. Only if the output is multi-channel and
            the input image didn't have a channel axis, then output channels are appended at the end.

        """
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]


   

    def _predict_mean_and_scale(self, img, axes, normalizer, resizer, n_tiles=1):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
           returns the restored image via a tuple `(restored,None)`

        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        axes = axes_check_and_normalize(axes,img.ndim)
        _permute_axes = self._make_permute_axes(axes, self.config.axes)

        x = _permute_axes(img)
        channel = axes_dict(self.config.axes)['C']

        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())

        # normalize
        x = normalizer.before(x,self.config.axes)
        # resize: make divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x = resizer.before(x,div_n,exclude=channel)

        done = False
        while not done:
            try:
                if n_tiles == 1:
                    x = predict_direct(self.keras_model,x,channel_in=channel,channel_out=channel)
                else:
                    overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
                    x = predict_tiled(self.keras_model,x,channel_in=channel,channel_out=channel,
                                      n_tiles=n_tiles,block_size=div_n,tile_overlap=overlap)
                done = True
            except tf.errors.ResourceExhaustedError:
                n_tiles = max(4, 2*n_tiles)
                print('Out of memory, retrying with n_tiles = %d' % n_tiles)

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        x.shape[channel] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x,exclude=channel)

        mean, scale = mean, scale = x, None

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            mean, scale = normalizer.after(mean, scale)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)

        return mean, scale


    def _make_permute_axes(self,axes_in,axes_out=None):
        if axes_out is None:
            axes_out = self.config.axes
        channel_in  = axes_dict(axes_in) ['C']
        channel_out = axes_dict(axes_out)['C']
        assert channel_out is not None

        def _permute_axes(data,undo=False):
            if data is None:
                return None
            if undo:
                if channel_in is not None:
                    return move_image_axes(data, axes_out, axes_in, True)
                else:
                    # input is single-channel and has no channel axis
                    data = move_image_axes(data, axes_out, axes_in+'C', True)
                    # output is single-channel -> remove channel axis
                    if data.shape[-1] == 1:
                        data = data[...,0]
                    return data
            else:
                return move_image_axes(data, axes_in, axes_out, True)
        return _permute_axes

    def _check_normalizer_resizer(self, normalizer, resizer):
        if normalizer is None:
            normalizer = NoNormalizer()
        if resizer is None:
            resizer = NoResizer()
        isinstance(resizer,Resizer) or _raise(ValueError())
        isinstance(normalizer,Normalizer) or _raise(ValueError())
        if normalizer.do_after:
            if self.config.n_channel_in != self.config.n_channel_out:
                warnings.warn('skipping normalization step after prediction because ' +
                              'number of input and output channels differ.')

        return normalizer, resizer