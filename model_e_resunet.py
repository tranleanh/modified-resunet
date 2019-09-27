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
    conv = Conv2D(n_kernels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(n_kernels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)

    conv_shortcut = Conv2D(n_kernels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(in_put)
    conv_shortcut = BatchNormalization(axis=3)(conv_shortcut)

    conv = Add()([conv, conv_shortcut])

    return conv

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)

    # Level 1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Add()([conv1, inputs])

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Level 2
    conv2 = res_block(pool1, n_kernels = 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Level 3
    conv3 = res_block(pool2, n_kernels = 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Level 4
    conv4 = res_block(pool3, n_kernels = 512) 
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Level 5
    conv5 = res_block(pool4, n_kernels = 1024) 
    drop5 = Dropout(0.5)(conv5)

    # Level 6
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = res_block(merge6, n_kernels = 512)

    # Level 7
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = res_block(merge7, n_kernels = 256)

    # Level 8
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))    
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = res_block(merge8, n_kernels = 128)

    # Level 9
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = res_block(merge9, n_kernels = 64)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
