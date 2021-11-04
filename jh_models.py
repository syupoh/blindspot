from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow import keras

models = keras.models
layers = keras.layers
initializers = keras.initializers
regularizers = keras.regularizers

def conv_block(x, filters, block_num, conv_num, strides=(1,1)):
    name = 'block{}_conv{}_'.format(block_num, conv_num)
    # conv-BN-relu
    x = layers.Conv2D(filters, (3,3), strides=(2,2), use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=name+'bn')(x)
    x = layers.Activation('relu', name=name+'act')(x)
    return x

def separable_conv_block(x, filters, block_num, conv_num, pre_activation=None):
    name = 'block{}_sepconv{}_'.format(block_num, conv_num)
    if pre_activation is True:
        x = layers.Activation('relu', name=name+'act')(x)
    # (relu)-sepconv-BN-(relu)
    x = layers.SeparableConv2D(filters, (3,3), padding='same', use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=name+'bn')(x)
    if pre_activation is False:
        x = layers.Activation('relu', name=name+'act')(x)
    return x

def middle_flow_block(x, filters, block_num):
    # middle flow
    residual = x
    x = separable_conv_block(x, filters, block_num=block_num, conv_num='1', pre_activation=True)
    x = separable_conv_block(x, filters, block_num=block_num, conv_num='2', pre_activation=True)
    x = separable_conv_block(x, filters, block_num=block_num, conv_num='3', pre_activation=True)
    return layers.add([x, residual])

def xception_block(x, filters, block_num, pre_activation=True):
    block = 'block{}_'.format(block_num)
    filter_conv1, filter_conv2 = filters
    # residual conv branch
    residual = layers.Conv2D(filter_conv2, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    # separable conv block
    x = separable_conv_block(x, filter_conv1, block_num=block_num, conv_num='1', pre_activation=pre_activation)
    x = separable_conv_block(x, filter_conv2, block_num=block_num, conv_num='2', pre_activation=True)
    # downsampling and merging
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=block+'pool')(x)
    return layers.add([x, residual])


def Xception(input_shape=(299,299,3), classes=26):
    """Instantiates the Xception architecture.
    """
    img_input = layers.Input(shape=input_shape)
    #===========ENTRY FLOW==============
    #Block 1
    x = conv_block(img_input, 32, block_num='1', conv_num='1', strides=(2,2))
    x = conv_block(x, 64, block_num='1', conv_num='2')
    #Block 2
    x = xception_block(x, (128, 128), '2', pre_activation=False)
    #Block 3
    x = xception_block(x, (256, 256), '3')
    #Block 4
    x = xception_block(x, (728, 728), '4')
    #===========MIDDLE FLOW===============
    for i in range(8):
        block_num = str(5+i)
        x = middle_flow_block(x, 728, block_num)
    #========EXIT FLOW============
    #Block 13
    x = xception_block(x, (728, 1024), '13') # second conv is different
    # Block 14
    x = separable_conv_block(x, 1536, block_num='14', conv_num='1', pre_activation=False)
    x = separable_conv_block(x, 2048, block_num='14', conv_num='2', pre_activation=False)
    # logistic regression
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    # Create model.
    model = models.Model(inputs=img_input, outputs=x, name='xception')
    return model


if __name__ =="__main__":
    xception = Xception()
    xception.summary()