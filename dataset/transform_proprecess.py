# Copyright (c) 2023 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

# -*- coding: utf-8 -*-

import tensorflow as tf
import inspect
import tools.standard_fields as fields
import tools.config as config
ADD_RATIO=1e-8
def _flip_boxes_left_right(boxes):
    """Left-right flip the boxes.
        boxes: Float32 tensor containing the bounding boxes -> [..., 4].
                     Boxes are in normalized form meaning their coordinates vary
                     between [0, 1].
                     Each last dimension is in the form of [ymin, xmin, ymax, xmax].
    """
    ymin, xmin, ymax, xmax, ratio = tf.split(value=boxes, num_or_size_splits=5, axis=-1)
    
    ratio=-ratio
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax, ratio], axis=-1)
    return flipped_boxes


def _flip_boxes_up_down(boxes):
    """Up-down flip the boxes.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
        Boxes are in normalized form meaning their coordinates vary
        between [0, 1].
        Each row is in the form of [ymin, xmin, ymax, xmax].
    """
    ymin, xmin, ymax, xmax, ratio = tf.split(value=boxes, num_or_size_splits=5, axis=1)
 
    Symbol=tf.less(ratio,0.)
    ratio=tf.subtract(1.,tf.abs(ratio))
    ratio=tf.where(Symbol,-ratio,ratio)
    
    flipped_ymin = tf.subtract(1.0, ymax)
    flipped_ymax = tf.subtract(1.0, ymin)
    flipped_boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax, ratio], 1)
    return flipped_boxes



def _rot90_boxes(boxes):
    """Rotate boxes counter-clockwise by 90 degrees.

   
     boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
                Boxes are in normalized form meaning their coordinates vary
                between [0, 1].
                Each row is in the form of [ymin, xmin, ymax, xmax].

    """
    ymin, xmin, ymax, xmax,ratio    = tf.split(value=boxes, num_or_size_splits=5, axis=1)
    rotated_ymin = tf.subtract(1.0, xmax)
    rotated_ymax = tf.subtract(1.0, xmin)
    real_ratio=tf.add(tf.multiply(ratio, 180.),90)
    cond=tf.less(180.,real_ratio)
    ratio=tf.where(cond,tf.subtract(real_ratio, 360.)/180. , real_ratio/180.)
    rotated_xmin = ymin
    rotated_xmax = ymax
    rotated_boxes = tf.concat(
            [rotated_ymin, rotated_xmin, rotated_ymax, rotated_xmax, ratio], 1)
    return rotated_boxes

def random_horizontal_flip(image,
                        boxes=None,
                        probability=0.5):
    """Randomly flips the image and detections horizontally.
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: rank 2 float32 tensor with shape [N, 4] 
    probability: the probability of performing this augmentation.   
    """

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped


    with tf.name_scope('RandomHorizontalFlip__test'):
        result = []
        do_a_flip_random =tf.random.uniform([])
        do_a_flip_random = tf.less(do_a_flip_random, probability)

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        result.append(image)
        #tf.print(image.shape)
        # flip boxes
        if boxes is not None:
            boxes = tf.where(do_a_flip_random,    _flip_boxes_left_right(boxes),
                                            boxes)
            result.append(boxes)
        return tuple(result)


def random_vertical_flip(image,
                        boxes=None,
                        probability=0.5):
    """Randomly flips the image and detections vertically.

        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: rank 2 float32 tensor with shape [N, 4]
        probability: the probability of performing this augmentation.
    """

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped


    with tf.name_scope('RandomVerticalFlip_0001'):
        result = []
        do_a_flip_random =tf.random.uniform([])
        do_a_flip_random = tf.less(do_a_flip_random, probability)

        # flip image
        image = tf.where(do_a_flip_random, _flip_image(image), image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_up_down(boxes),
                                            lambda: boxes)
            result.append(boxes)


        return tuple(result)


def random_rotation90(image,
                    boxes=None,
                    probability=0.5):
    """
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: rank 2 float32 tensor with shape [N, 4]
        probability: the probability of performing this augmentation.  
    """

    def _rot90_image(image):
        # flip image
        image_rotated = tf.image.rot90(image)
        return image_rotated

    with tf.name_scope('RandomRotation90'):
        result = []
        do_a_rot90_random =tf.random.uniform([])
        do_a_rot90_random = tf.less(do_a_rot90_random, probability)

        # flip image
        image = tf.cond(do_a_rot90_random, lambda:_rot90_image(image),
                                        lambda:image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.where(do_a_rot90_random, _rot90_boxes(boxes),
                                            boxes)
            result.append(boxes)
        return tuple(result)

def process(tensor_dict,record_seq):
    probability={}
    if record_seq==1:
        probability[random_horizontal_flip]=config.RANDOM_HORIZONTAL_FLIP_PROBABILITY1
        probability[random_vertical_flip]=config.RANDOM_VERTICAL_FLIP_PROBABILITY1
        probability[random_rotation90]=config.RANDOM_ROTATION90_PROBABILITY1
       
    elif record_seq==2:
        probability[random_horizontal_flip]=config.RANDOM_HORIZONTAL_FLIP_PROBABILITY2
        probability[random_vertical_flip]=config.RANDOM_VERTICAL_FLIP_PROBABILITY2
        probability[random_rotation90]=config.RANDOM_ROTATION90_PROBABILITY2
        
    else :
        raise Exception("record_seq should be 1 or 2")
    
    
    data_argu_funcs=[random_horizontal_flip,
        random_vertical_flip,
        random_rotation90]
      
    argarr={
        'image':fields.InputDataFields.image,
        'boxes':fields.InputDataFields.groundtruth_boxes}
    for func in data_argu_funcs:

        arg_spec = inspect.getfullargspec(func).args
        args=[tensor_dict[argarr[a]]for a in arg_spec if a in argarr] #getvalue
        results = func(*args,probability=probability[func])
        
        if not isinstance(results, (list, tuple)):
                results = (results,)
   
        arg_names=[argarr[a] for a in arg_spec if a in argarr] #getvalue
        for arg_name,res    in zip(arg_names,results):
                tensor_dict[arg_name] = res

    return tensor_dict