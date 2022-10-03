# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
import tools.config as config
def VGG16(     
                input_shape=(config.IMG_HEIGHT,config.IMG_WIDTH,config.INPUT_CHANNEL)):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
                    img_input)
    x = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    # Block 2
    x = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

    # Block 5
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name='vgg16')

 
    return model