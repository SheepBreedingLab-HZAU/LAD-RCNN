# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;
# -*- coding: utf-8 -*-
# code_09:build model subprog assigner target and anchor

import tensorflow as tf


import boxlist.box_list_ops as box_list_ops
import boxlist.box_coder as box_coder
import boxlist.box_list as box_list

class Match(object):
    
    def __init__(self, match_results):

        if match_results.shape.ndims != 1:
            raise ValueError('match_results should have rank 1')
        if match_results.dtype != tf.int32:
            raise ValueError('match_results should be an int32 or int64 scalar '
                                             'tensor')
        self._match_results = match_results
        #self._gather_op = ops.matmul_gather_on_zeroth_axis#TODO(sunling) should test if its ok
        self._gather_op=tf.gather
    @property
    def match_results(self):
        return self._match_results


    def gather_based_on_match(self, input_tensor, unmatched_value,
                                                        ignored_value):
        """
        input_tensor: Tensor to gather values from.
        unmatched_value: Constant tensor value for unmatched columns.
        ignored_value: Constant tensor value for ignored columns.
        """
        if input_tensor==None:
                input_tensor=tf.ones([self._match_results.shape[0]])

        input_tensor = tf.concat(
                [tf.stack([ignored_value, unmatched_value]),
                 input_tensor],
                axis=0)
        gather_indices = tf.maximum(self.match_results + 2, 0)
        gathered_tensor = self._gather_op(input_tensor, gather_indices)
        return gathered_tensor


class Matcher(object):
    def __init__(self,
                 matched_threshold=0.5,#TODO(sunling) 0.7->0.8
                 unmatched_threshold=0.3):
        self._matched_threshold = matched_threshold
        self._unmatched_threshold = unmatched_threshold


    def match(self, similarity_matrix):
        with tf.name_scope('Match'):
            return Match(self._match(similarity_matrix))
 
    def _match(self, similarity_matrix):
        matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)#indicate for which groundtrouth box 
     
        matched_vals = tf.reduce_max(similarity_matrix, 0)
        below_unmatched_threshold = tf.greater(self._unmatched_threshold,
                                               matched_vals)
        between_thresholds = tf.logical_and(
                tf.greater_equal(matched_vals, self._unmatched_threshold),
                tf.greater(self._matched_threshold, matched_vals))

        matches = self._set_values_using_indicator(matches,
                                                   below_unmatched_threshold,
                                                   -1)
        matches = self._set_values_using_indicator(matches,
                                                   between_thresholds,
                                                   -2) #ingnore
        
      
        return matches #final_matches

    def _set_values_using_indicator(self, x, indicator, val):

        indicator = tf.cast(indicator, x.dtype)
        return tf.add(tf.multiply(x, 1 - indicator), val * indicator)

class TargetAssigner(object):
 
    def __init__(self):

        self._similarity_calc = box_list_ops.iou
        self._box_coder = box_coder.FasterRcnnBoxCoder()
        self._matcher = Matcher()

    @property
    def box_coder(self):
        return self._box_coder

    def assign(self,
            anchors,
            groundtruth_boxes,
            groundtruth_labels=None,
            unmatched_class_label=None):
        """
        anchors: a BoxList representing N anchors
        groundtruth_boxes: a BoxList representing M groundtruth boxes
        groundtruth_labels:    a tensor of shape [M, d_1, ... d_k]
        unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
        """
        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')

        if unmatched_class_label is None:
            unmatched_class_label = tf.constant([0], tf.float32)

        if groundtruth_labels is None:
            groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),0))
            groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)
 
        match_quality_matrix = self._similarity_calc(groundtruth_boxes,
                                                     anchors)#TODO (sunling) cal IOU
        match = self._matcher.match(match_quality_matrix)
        reg_targets = self._create_regression_targets(anchors,
                                                      groundtruth_boxes,
                                                      match)
        cls_targets = self._create_classification_targets(groundtruth_labels,
                                                          unmatched_class_label,
                                                          match)
        reg_weights = self._create_regression_weights(match)

        cls_weights = self._create_classification_weights(match)

        class_label_shape = tf.shape(cls_targets)[1:]
        weights_shape = tf.shape(cls_weights)
        weights_multiple = tf.concat(
                [tf.ones_like(weights_shape), class_label_shape],
                axis=0)
        for _ in range(len(cls_targets.get_shape()[1:])):
            cls_weights = tf.expand_dims(cls_weights, -1)
        cls_weights = tf.tile(cls_weights, weights_multiple)

        num_anchors = anchors.num_boxes_static()
        if num_anchors is not None:
            reg_targets = self._reset_target_shape(reg_targets, num_anchors)
            cls_targets = self._reset_target_shape(cls_targets, num_anchors)
            reg_weights = self._reset_target_shape(reg_weights, num_anchors)
            cls_weights = self._reset_target_shape(cls_weights, num_anchors)

        return (cls_targets, cls_weights, reg_targets, reg_weights,
                        match.match_results)

    def _reset_target_shape(self, target, num_anchors):
        """
        target: the target tensor. Its first dimension will be overwritten.
        num_anchors: the number of anchors, which is used to override the target's
                first dimension.
        """
        target_shape = target.get_shape().as_list()
        target_shape[0] = num_anchors
        target.set_shape(target_shape)
        return target

    def _create_regression_targets(self, anchors, groundtruth_boxes, match):
        """
        anchors: a BoxList representing N anchors
        groundtruth_boxes: a BoxList representing M groundtruth_boxes
        match: a matcher.Match object
        """
        matched_gt_boxes = match.gather_based_on_match(
                groundtruth_boxes.get(),
                unmatched_value=tf.zeros(5),
                ignored_value=tf.zeros(5))
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)

        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
        match_results_shape = match.match_results.shape.as_list()

        unmatched_ignored_reg_targets = tf.tile(
                tf.constant([[0,0,0,0,0]], tf.float32), [match_results_shape[0], 1])
        matched_anchors_mask =    tf.greater_equal(match.match_results, 0)
        reg_targets = tf.compat.v1.where(matched_anchors_mask,
                                         matched_reg_targets,
                                         unmatched_ignored_reg_targets)
        return reg_targets


    def _create_classification_targets(self, groundtruth_labels,
                                       unmatched_class_label, match):
        """
        groundtruth_labels:    a tensor of shape [num_gt_boxes, d_1, ... d_k]
             
        unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
              
        match: a matcher.Match object that provides a matching between anchors
                and groundtruth boxes.
        """
        return match.gather_based_on_match(
                groundtruth_labels,
                unmatched_value=unmatched_class_label,
                ignored_value=unmatched_class_label)

    def _create_regression_weights(self, match):
        """Set regression weight for each anchor.

        Args:
            match: a matcher.Match object that provides a matching between anchors
                and groundtruth boxes.
        Returns:
            a float32 tensor with shape [num_anchors] representing regression weights.
        """
        return match.gather_based_on_match(None,
                ignored_value=0., unmatched_value=0.)

    def _create_classification_weights(self,
                                       match):
        """
        match: a matcher.Match object that provides a matching between anchors
                and groundtruth boxes.
        """
        return match.gather_based_on_match(None,
                ignored_value=0.,
                unmatched_value=1.0)





def batch_assign(target_assigner,
                anchors_batch,
                gt_box_batch,
                gt_class_targets_batch,
                unmatched_class_label=None):
    """
        target_assigner: a target assigner.
        anchors_batch: BoxList representing N box anchors or list of BoxList objects
            with length batch_size representing anchor sets.
        gt_box_batch: a list of BoxList objects with length batch_size
            representing groundtruth boxes for each image in the batch
        gt_class_targets_batch: a list of tensors with length batch_size, where
            each tensor has shape [num_gt_boxes_i, classification_target_size] and
            num_gt_boxes_i is the number of boxes in the ith boxlist of
            gt_box_batch.
        unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
            which is consistent with the classification target for each
            anchor (and can be empty for scalar targets).    This shape must thus be
            compatible with the groundtruth labels that are passed to the "assign"
            function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
    """
    if not isinstance(anchors_batch, list):
        anchors_batch = len(gt_box_batch) * [anchors_batch]
    if not all(
            isinstance(anchors, box_list.BoxList) for anchors in anchors_batch):
        raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
    if not (len(anchors_batch)
                    == len(gt_box_batch)
                    == len(gt_class_targets_batch)):
        raise ValueError('batch size incompatible with lengths of anchors_batch, '
                                         'gt_box_batch and gt_class_targets_batch.')
    cls_targets_list = []
    cls_weights_list = []
    reg_targets_list = []
    reg_weights_list = []
    match_list = []
    # if gt_weights_batch is None:
    #     gt_weights_batch = [None] * len(gt_class_targets_batch)
    for anchors, gt_boxes, gt_class_targets in zip(
            anchors_batch, gt_box_batch, gt_class_targets_batch):
        (cls_targets, cls_weights,
         reg_targets, reg_weights, match) = target_assigner.assign(
                 anchors, gt_boxes, gt_class_targets, unmatched_class_label)
        cls_targets_list.append(cls_targets)
        cls_weights_list.append(cls_weights)
        reg_targets_list.append(reg_targets)
        reg_weights_list.append(reg_weights)
        match_list.append(match)
    batch_cls_targets = tf.stack(cls_targets_list)
    batch_cls_weights = tf.stack(cls_weights_list)
    batch_reg_targets = tf.stack(reg_targets_list)
    batch_reg_weights = tf.stack(reg_weights_list)
    batch_match = tf.stack(match_list)
    return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
                    batch_reg_weights, batch_match)

batch_assign_targets = batch_assign