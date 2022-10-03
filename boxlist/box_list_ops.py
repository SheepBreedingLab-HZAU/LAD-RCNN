# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;
#code_13 BoxList opera

"""Bounding Box List operations."""

import tensorflow as tf

import boxlist.box_list as box_list

def area(boxlist):
    """
    boxlist: BoxList holding N boxes
    """
    with tf.name_scope('Area'):
        y_min, x_min, y_max, x_max, _ = tf.split(
                value=boxlist.get(), num_or_size_splits=5, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])



def scale(boxlist, y_scale, x_scale):
    """
  
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    """
    with tf.name_scope('Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        y_min, x_min, y_max, x_max, angle = tf.split(
                value=boxlist.get(), num_or_size_splits=5, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxlist = box_list.BoxList(
                tf.concat([y_min, x_min, y_max, x_max, angle], 1))
        #return _copy_extra_fields(scaled_boxlist, boxlist)
        return scaled_boxlist




def clip_to_window(boxlist, window, sorted_scores=None, filter_nonoverlapping=True):
    """
    boxlist: BoxList holding M_in boxes
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
            window to which the op should clip boxes.
    filter_nonoverlapping: whether to filter out boxes that do not overlap at
            all with the window.
    """
    with tf.name_scope('ClipToWindow'):
        y_min, x_min, y_max, x_max, angle = tf.split(
                value=boxlist.get(), num_or_size_splits=5, axis=1)
        win_y_min = window[0]
        win_x_min = window[1]
        win_y_max = window[2]
        win_x_max = window[3]
        y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
        y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
        x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
        x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
        clipped = box_list.BoxList(
                tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped, angle],
                                    1))
        #clipped = _copy_extra_fields(clipped, boxlist)
        if filter_nonoverlapping:
            areas = area(clipped)
            nonzero_area_indices = tf.cast(
                    tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
            clipped = gather(clipped, nonzero_area_indices)
            if sorted_scores !=None:
                    sorted_scores=gather(sorted_scores, nonzero_area_indices)
                    return clipped,sorted_scores
        return clipped


def intersection(boxlist1, boxlist2):
    """
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    """
    with tf.name_scope('Intersection'):
        y_min1, x_min1, y_max1, x_max1, _ = tf.split(
                value=boxlist1.get(), num_or_size_splits=5, axis=1)
        y_min2, x_min2, y_max2, x_max2, _ = tf.split(
                value=boxlist2.get(), num_or_size_splits=5, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths




def iou(boxlist1, boxlist2):
    """
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    """
    with tf.name_scope('IOU'):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = (
                tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
                tf.equal(intersections, 0.0),
                tf.zeros_like(intersections), tf.truediv(intersections, unions))


def change_coordinate_frame(boxlist, window):
    """
    boxlist: A BoxList object holding N boxes.
    window: A rank 1 tensor [4].
    """
    with tf.name_scope('ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale(box_list.BoxList(
                boxlist.get() - [window[0], window[1], window[0], window[1], 0.]),
                                                1.0 / win_height, 1.0 / win_width)
        #boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
        return boxlist_new



def gather(boxlist, indices):
    """
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64
    """
    with tf.name_scope('Gather'):
        if len(indices.shape.as_list()) != 1:
            raise ValueError('indices should have rank 1')
        if indices.dtype != tf.int32 and indices.dtype != tf.int64:
            raise ValueError('indices should be an int32 / int64 tensor')
        if isinstance(boxlist,box_list.BoxList):
                subboxlist = box_list.BoxList(tf.gather(boxlist.get(), indices))
        else:
                subboxlist =tf.gather(boxlist, indices)
        return subboxlist


def concatenate(boxlists):
    """
    boxlists: list of BoxList objects
    """
    with tf.name_scope('Concatenate'):
        concatenated = box_list.BoxList(
                tf.concat([boxlist.get() for boxlist in boxlists], 0))
        return concatenated


def sort_by_score(boxlist, score, descendorder=True):
    """
    boxlist: BoxList holding N boxes.
    score: A float32 tensor with shape[N].
    descendorder: descend or ascend. Default is descend.
    """
    with tf.name_scope('SortByScore'):
        # if score.ndim !=1:
        #     raise ValueError('score should have rank 1')
        if boxlist.num_boxes_static() != score.get_shape()[0]:
            raise ValueError('boxlist and score should has same element')
        _, sorted_indices = tf.nn.top_k(score, boxlist.num_boxes(), sorted=True)
        if not descendorder:
            sorted_indices = tf.reverse_v2(sorted_indices, [0])
        return box_list.BoxList(gather(boxlist, sorted_indices)),tf.gather(score,sorted_indices)





def to_normalized_coordinates(boxlist, height, width):
    """
    boxlist: BoxList with coordinates in terms of pixel-locations.
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    """
    with tf.name_scope('ToNormalizedCoordinates'):
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        return scale(boxlist, 1 / height, 1 / width)


def to_absolute_coordinates(boxlist, height, width):
    """
    boxlist: BoxList with coordinates in range [0, 1].
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    """
    with tf.name_scope('ToAbsoluteCoordinates'):
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        return scale(boxlist, height, width)








