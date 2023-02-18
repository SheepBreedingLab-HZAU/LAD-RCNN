# Copyright (c) 2023 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;
#code_11 boxcoder and decoder
import tensorflow as tf
import boxlist.box_list as box_list
EPSILON = 1e-8


class FasterRcnnBoxCoder(object):
    """Faster RCNN box coder.
    The coding schema described below reference http://arxiv.org/abs/1506.01497
        ty = (y - ya) / ha
        tx = (x - xa) / wa
        th = log(h / ha)
        tw = log(w / wa)
        where x, y, w, h denote the box's position;
        xa, ya, wa, ha denote the anchor's position;
        tx, ty, tw and th denote the encoded position.
    """

    def __init__(self, scale_factors=[10.0, 10.0, 5.0, 5.0]):
        """
            scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        """
        self._scale_factors = scale_factors
        
    def encode(self, boxes, anchors):
        """
        Encode a box collection with respect to anchor collection.
        boxes: BoxList holding N boxes to be encoded.
        anchors: BoxList of anchors.
        """
        with tf.name_scope('Encode'): 
            ycenter_a, xcenter_a, ha, wa, _ = anchors.get_center_size_type_box()
            ycenter, xcenter, h, w, angle = boxes.get_center_size_type_box()
            # Avoid NaN in division and log below.
            ha += EPSILON
            wa += EPSILON
            h += EPSILON
            w += EPSILON
    
            tx = (xcenter - xcenter_a) / wa
            ty = (ycenter - ycenter_a) / ha
            tw = tf.math.log(w / wa)
            th = tf.math.log(h / ha)

            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
            
            return tf.transpose(tf.stack([ty, tx, th, tw, angle]))
    def decode(self, rel_codes, anchors):
        """

        Args:
            rel_codes: a tensor representing N anchor-encoded boxes.
            anchors: BoxList of anchors.

        Returns:
            boxes: BoxList holding N bounding boxes.
        """
        with tf.name_scope('Decode'):
            ycenter_a, xcenter_a, ha, wa,_ = anchors.get_center_size_type_box()
            ty, tx, th, tw, angle = tf.unstack(tf.transpose(rel_codes))
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
            
            w = tf.exp(tw) * wa
            h = tf.exp(th) * ha
            ycenter = ty * ha + ycenter_a
            xcenter = tx * wa + xcenter_a
            
            ymin = ycenter - h / 2.
            xmin = xcenter - w / 2.
            ymax = ycenter + h / 2.
            xmax = xcenter + w / 2.
            
            return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax ,angle])))
