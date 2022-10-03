# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
import tools.config as config


def _CBA(x,filters,kernel_size,strides=1,name='block1_conv1'):
    x=Conv2D(
        filters, kernel_size, strides=strides, padding='SAME', name=name+'_conv')(x)
    x=BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name + '_bn')(x)
    x=Activation('relu',name=name)(x)
    return x

def Model(   
        input_shape=(config.IMG_HEIGHT,config.IMG_WIDTH,config.INPUT_CHANNEL)):
  img_input = Input(shape=input_shape)

  # Block 1
  x = _CBA(img_input,
      64, kernel_size=7, strides=2, name='block1_conv1')
  
  x = _CBA(x,
      64, kernel_size=3, name='block1_conv2')
  #x = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='block1_pool')(x)

  # Block 2
  x = _CBA(x,
      64, kernel_size=3,strides=2, name='block2_conv1')
  x = _CBA(x,
      64,kernel_size=3, name='block2_conv2')
  x = _CBA(x,
      128,kernel_size=3, name='block2_conv3')
  #x = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='block2_pool')(x)

  # Block 3
  x = _CBA(x,
      64, kernel_size=3,strides=2, name='block3_conv1')
  x = _CBA(x,
      64, kernel_size=3, name='block3_conv2')
  x = _CBA(x,
      256, kernel_size=3, name='block3_conv3')
  #x = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='block3_pool')(x)

  # Block 4
  x = _CBA(x,
      128, kernel_size=3,strides=2, name='block4_conv1')
  x = _CBA(x,
      128,kernel_size=3, name='block4_conv2')
  x = _CBA(x,
      512, kernel_size=3, name='block4_conv3')
  #x = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='block4_pool')(x)

  # Block 5
  x = _CBA(x,
      128,kernel_size=3,strides=2, name='block5_conv1')
  x = _CBA(x,
      128, kernel_size=3, name='block5_conv2')
  x = _CBA(x,
      512, kernel_size=3, name='block5_conv3')
  #x = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='block5_pool')(x)

  # Create model.
  model = tf.keras.models.Model(img_input, x, name='mynet')

 
  return model