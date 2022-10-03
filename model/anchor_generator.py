# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;
# code_08:build model subprog Anchor Generator
"""Generates grid anchors
"""

import tensorflow.compat.v1 as tf

import boxlist.box_list as box_list
import boxlist.box_list_ops as box_list_ops
class AnchorGenerator(object):
    def __init__(self,
               min_level=3, max_level=6):
        self._scales = [1,2**0.5]
        self._aspect_ratios = [1.0, 2.0, 0.5]
        self._base_anchor_size = [2**b*4 for b in range(min_level,max_level+1)]
        self._anchor_stride = [[2**b,2**b] for b in range(min_level,max_level+1)]
        self._anchor_generator_k=[]
        for level in range(min_level,max_level+1):
            scales = [1,2**0.5]
            aspect_ratios = [1.0, 2.0, 0.5]
            base_anchor_size = [2**level*4,2**level*4]
            anchor_stride = [2**level,2**level]
            self._anchor_generator_k.append([scales,aspect_ratios,base_anchor_size,anchor_stride])
    def num_anchors_per_location(self):
        return len(self._base_anchor_size)*[len(self._scales) * len(self._aspect_ratios)]

    def generate(self, feature_map_shape_list):
    
        with tf.name_scope('AnchorGenerator'):
            if not (isinstance(feature_map_shape_list, list)):
                raise ValueError('feature_map_shape_list must be a list')
            if not all([isinstance(list_item, tuple) and len(list_item) == 2
                        for list_item in feature_map_shape_list]):
                raise ValueError('feature_map_shape_list must be a list of pairs.')
        
            with tf.init_scope():
                self._base_anchor_size = tf.cast(tf.convert_to_tensor(
                    self._base_anchor_size), dtype=tf.float32)
                self._anchor_stride = tf.cast(tf.convert_to_tensor(
                    self._anchor_stride), dtype=tf.float32)
            anchors=[]
            for feature_map_shape,anchor_generator_k in zip(feature_map_shape_list,self._anchor_generator_k):
                ag = AnchorGeneratorOne(*anchor_generator_k)
                anchor_one = ag.generate(feature_map_shape_list=[feature_map_shape])
                anchors.append(anchor_one)
        
            return box_list_ops.concatenate(anchors)

class AnchorGeneratorOne(object):
    def __init__(self,
               scales,#=(0.5, 1.0, 2.0),
               aspect_ratios,#=(0.5, 1.0, 2.0),
               base_anchor_size,#=[256,256],
               anchor_stride):#,=[16,16]):

        self._scales = scales
        self._aspect_ratios = aspect_ratios
        self._base_anchor_size = base_anchor_size
        self._anchor_stride = anchor_stride
    
    def num_anchors_per_location(self):
        return [len(self._scales) * len(self._aspect_ratios)]
    
    def _marge(self,x,y):
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        x_exp_shape = tf.concat([tf.ones(tf.rank(y),dtype=tf.int32),tf.shape(x)], 0)
        y_exp_shape = tf.concat([tf.shape(y),tf.ones(tf.rank(x),dtype=tf.int32)], 0)
        xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
        ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)   
        return xgrid, ygrid
    
    def generate(self, feature_map_shape_list):
    
        with tf.name_scope('AnchorGenerator_one'):
            if not (isinstance(feature_map_shape_list, list)
                    and len(feature_map_shape_list) == 1):
              raise ValueError('feature_map_shape_list must be a list of length 1.')
            if not all([isinstance(list_item, tuple) and len(list_item) == 2
                        for list_item in feature_map_shape_list]):
              raise ValueError('feature_map_shape_list must be a list of pairs.')
        
          
            with tf.init_scope():
              self._base_anchor_size = tf.cast(tf.convert_to_tensor(
                  self._base_anchor_size), dtype=tf.float32)
              self._anchor_stride = tf.cast(tf.convert_to_tensor(
                  self._anchor_stride), dtype=tf.float32)
        
        
            grid_height, grid_width = feature_map_shape_list[0]
    
            scales_grid,aspect_ratios_grid=self._marge(self._scales,self._aspect_ratios)
            scales_grid=tf.reshape(scales_grid,[-1])
            aspect_ratios_grid=tf.reshape(aspect_ratios_grid,[-1])
            
            ratio_sqrts = tf.sqrt(aspect_ratios_grid)
            
            heights = scales_grid / ratio_sqrts * self._base_anchor_size[0]
            widths = scales_grid * ratio_sqrts * self._base_anchor_size[1]
          
            # Get a grid of box centers
            y_centers = tf.cast(tf.range(grid_height), dtype=tf.float32)
            y_centers = y_centers * self._anchor_stride[0] 
            x_centers = tf.cast(tf.range(grid_width), dtype=tf.float32)
            x_centers = x_centers * self._anchor_stride[1] 
            x_centers_g,y_centers_g = self._marge(x_centers, y_centers)
            
            
            
            widths_grid, x_centers_grid = self._marge(widths, x_centers_g)
            heights_grid, y_centers_grid = self._marge(heights, y_centers_g)
            
    
            
            
            bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
            bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
            bbox_centers = tf.reshape(bbox_centers, [-1, 2])
            bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
            bbox_corners = tf.concat([bbox_centers - .5 * bbox_sizes, bbox_centers + .5 * bbox_sizes], 1)
            bbox_corners=tf.pad(bbox_corners,[[0,0],[0,1]])
            #angle=tf.zeros([bbox_corners.shape[0],1])
            #anchors=box_list.BoxList(tf.concat([bbox_corners,angle],axis=-1))
            anchors=box_list.BoxList(bbox_corners)
            
        
            return anchors
