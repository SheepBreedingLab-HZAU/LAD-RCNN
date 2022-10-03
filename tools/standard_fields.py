# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;
#code_16 config file


class InputDataFields(object):
    image = 'image'
    original_image = 'original_image'
    original_image_spatial_shape = 'original_image_spatial_shape'
    filename = 'filename'
    groundtruth_boxes = 'groundtruth_boxes'
    proposal_boxes = 'proposal_boxes'
    proposal_objectness = 'proposal_objectness'
    num_groundtruth_boxes = 'num_groundtruth_boxes'
    true_image_shape = 'true_image_shape'
    image_height = 'image_height'
    image_width = 'image_width'


class DetectionResultFields(object):
    detection_boxes = 'detection_boxes'
    detection_scores = 'detection_scores'
    raw_detection_boxes = 'raw_detection_boxes'
    raw_detection_scores = 'raw_detection_scores'


class BoxListFields(object):
    boxes = 'boxes'
    scores = 'scores'
    objectness = 'objectness'


class PredictionFields(object):
    feature_maps = 'feature_maps'
    anchors = 'anchors'
    raw_detection_boxes = 'raw_detection_boxes'
    raw_detection_feature_map_indices = 'raw_detection_feature_map_indices'


class TfExampleFields(object):
    image_encoded = 'image/encoded'
    image_format = 'image/format'  # format is reserved keyword
    filename = 'image/filename'
    height = 'image/height'
    width = 'image/width'
    source_id = 'image/source_id'
    image_class_text = 'image/class/text'
    image_class_label = 'image/class/label'
    object_class_text = 'image/object/class/text'
    object_class_label = 'image/object/class/label'
    object_bbox_ymin = 'image/object/bbox/ymin'
    object_bbox_xmin = 'image/object/bbox/xmin'
    object_bbox_ymax = 'image/object/bbox/ymax'
    object_bbox_xmax = 'image/object/bbox/xmax'
    detection_class_label = 'image/detection/label'
    detection_bbox_ymin = 'image/detection/bbox/ymin'
    detection_bbox_xmin = 'image/detection/bbox/xmin'
    detection_bbox_ymax = 'image/detection/bbox/ymax'
    detection_bbox_xmax = 'image/detection/bbox/xmax'
    detection_score = 'image/detection/score'