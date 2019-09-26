# Sept. 2019
# Tran Le Anh, MSc Student
# Satellite Image Processing Lab, Myongji Univ., Yongin, South Korea
# tranleanh.nt@gmail.com
# https://sites.google.com/view/leanhtran

# Expanded Residual Unet Model (Architecture)

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def res_block(in_put, n_kernels):

    conv = BatchNormalization(axis=3)(in_put)
    conv = Activation('relu')(conv)
    conv = Conv2D(n_kernels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(n_kernels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv)

    conv_shortcut = Conv2D(n_kernels, 3, padding = 'same', kernel_initializer = 'he_normal')(in_put)
    conv_shortcut = BatchNormalization(axis=3)(conv_shortcut)

    conv = Add()([conv, conv_shortcut])

    return conv

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)

    # Level 1
    conv1 = BatchNormalization(axis=3)(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    
    conv1_shortcut = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_shortcut = BatchNormalization(axis=3)(conv1_shortcut)
    conv1 = Add()([conv1, conv1_shortcut])

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Level 2
    conv2 = res_block(pool1, n_kernels = 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Level 3
    conv3 = res_block(pool2, n_kernels = 256) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Level 4 - Bridge
    conv4 = res_block(pool3, n_kernels = 512) 

    # Level 5
    up5 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv3,up5], axis = 3)
    conv5 = res_block(merge5, n_kernels = 256)

    # Level 6
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2, up6], axis = 3)
    conv6 = res_block(merge6, n_kernels = 128)

    # Level 7
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1, up7], axis = 3)
    conv7 = res_block(merge7, n_kernels = 64)

    out_put = Conv2D(1, 1, activation = 'sigmoid')(conv7)

    model = Model(input = inputs, output = out_put)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
