# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

#code_15 config file
# -*- coding: utf-8 -*-

#NUM_CLASS=1

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_num_step', '-ts', type=int, default=50000)
parser.add_argument('--base_model', '-bm', type=str, default='ournet') # can be changed to ournet, resnet50, vgg16 and mobilnetV2
parser.add_argument('--sequence', '-i', type=int, default=1)


args = parser.parse_args()
TRAIN_NUM_STEP=args.train_num_step
BASE_MODEL=args.base_model #ournet 、 resnet50 、 vgg16 、 myResNet


INPUT_CHANNEL=3
#BASE_MODEL='vgg16' #ournet 、 resnet50 、 vgg16 、 myResNet
#TRAIN_NUM_STEP=25000#25000#train_config.num_steps
BOX_PREDICTOR_DEPTH=512#box_predictor_depth
MINIBATCH_SIZE=2000

#USE_STATIC_SHAPES=True
NMS_IOU_THRESHOLD=0.67 #should between [0,1]
LOCALIZATION_LOSS_WEIGHT=2.0
OBJECTNESS_LOSS_WEIGHT=1.0
IMG_HEIGHT=400 #input size
IMG_WIDTH=400 #input size

#LABEL_MAP_PATH=None# 'data/label.pbtxt'
BATCH_SIZE1=7 #batch of data that has angle data
BATCH_SIZE2=5 #batch of data that has no angle data

TFRECORD_PATH1= ['D:/USER/Document/Python1/Identification/dataset/tfrecord/Sheep_angle.tfrecord']#The url of dataset1 which has angle data
TFRECORD_PATH2= ["D:/USER/Document/Python1/Identification/dataset/tfrecord/Sheep_noangle_amp.tfrecord"] #The url of dataset1 which has no angle data

ROTATE_IGNORE_THERSHOLD=0.025 #should between [0,1]

LEARNING_RATE_BASE=0.04
WARMUP_LEARNING_RATE=0.01333 #WARMUP_LEARNING_RATE should less then LEARNING_RATE_BASE
WARMUP_STEPS=2000 #WARMUP_STEPS should less then TRAIN_NUM_STEP
HOLD_BASE_RATE_STEPS=0 #used in config learning rate


RANDOM_HORIZONTAL_FLIP_PROBABILITY1=0.5 # PROBABILITY of HORIZONTAL FLIP in dataset 1
RANDOM_VERTICAL_FLIP_PROBABILITY1=0.5 # PROBABILITY of VERTICAL FLIP in dataset 1
RANDOM_ROTATION90_PROBABILITY1=0.5 # PROBABILITY of ROTATION90 in dataset 1
MERGERED_1X1PROBABILITY1=0.2 
MERGERED_2X2PROBABILITY1=0.8 # PROBABILITY of 2X2 merger HORIZONTAL FLIP in dataset 1
MERGERED_3X3PROBABILITY1=0 # PROBABILITY of 3X3 merger HORIZONTAL FLIP in dataset 1
#Note MERGERED_1X1PROBABILITY1 + MERGERED_2X2PROBABILITY1 + MERGERED_3X3PROBABILITY1 should equal to 1

RANDOM_HORIZONTAL_FLIP_PROBABILITY2=0.5 # PROBABILITY of HORIZONTAL FLIP in dataset 2
RANDOM_VERTICAL_FLIP_PROBABILITY2=0.5 # PROBABILITY of VERTICAL FLIP in dataset 2
RANDOM_ROTATION90_PROBABILITY2=0 # PROBABILITY of ROTATION90 in dataset 2
MERGERED_1X1PROBABILITY2=1
MERGERED_2X2PROBABILITY2=0 # PROBABILITY of 2X2 merger HORIZONTAL FLIP in dataset 2
MERGERED_3X3PROBABILITY2=0 # PROBABILITY of 3X3 merger HORIZONTAL FLIP in dataset 2
#Note MERGERED_1X1PROBABILITY2 + MERGERED_2X2PROBABILITY2 + MERGERED_3X3PROBABILITY2 should equal to 1


