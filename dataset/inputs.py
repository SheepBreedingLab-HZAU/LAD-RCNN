# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

#code_03:build dataset
"""Model input function for tf-learn object detection model."""



import functools

import tensorflow.compat.v1 as tf

import dataset.dataset_builder as dataset_builder
import dataset.transform_proprecess as trans_proprecess
import dataset.imgmerge_proprecess as merge_proprecess
import tools.standard_fields as fields
import boxlist.box_list as box_list
import boxlist.box_list_ops as box_list_ops


import tools.config as config

_LABEL_OFFSET = 1


def transform_input_data(tensor_dict,
                         model_preprocess_fn,record_seq):
    """
        tensor_dict: dictionary containing input tensors keyed by
            fields.InputDataFields.
        model_preprocess_fn: model's preprocess function to apply on image tensor.
            This function must take in a 4-D float tensor and return a 4-D preprocess
            float tensor and a tensor containing the true image shape.
        record_seq: a int of 1 or 2 which response the seq of tfrecord defined in config

    """
    out_tensor_dict = tensor_dict.copy()

    input_fields = fields.InputDataFields
    
    out_tensor_dict=trans_proprecess.process(out_tensor_dict,record_seq) #dataArgu

        
    image = out_tensor_dict[input_fields.image]
    preprocessed_resized_image, true_image_shape = model_preprocess_fn(
            tf.expand_dims(tf.cast(image, dtype=tf.float32), axis=0))#apply model._image_resizer_fn

    preprocessed_shape = tf.shape(preprocessed_resized_image)
    new_height, new_width = preprocessed_shape[1], preprocessed_shape[2]

    im_box = tf.stack([
            0.0, 0.0,
            tf.cast(new_height,tf.float32) / tf.cast(true_image_shape[0, 0],tf.float32),
            tf.cast(new_width,tf.float32) / tf.cast(true_image_shape[0, 1],tf.float32)
    ])

    
    boxlist = box_list.BoxList(out_tensor_dict[input_fields.groundtruth_boxes])
    realigned_bboxes = box_list_ops.change_coordinate_frame(boxlist, im_box)#TODO(sunling)coordinate frame to resized img
        
    out_tensor_dict[
                input_fields.groundtruth_boxes] = realigned_bboxes.get()
 
    out_tensor_dict[input_fields.image] = tf.squeeze(
            preprocessed_resized_image, axis=0)
    out_tensor_dict[input_fields.true_image_shape] = tf.squeeze(
            true_image_shape, axis=0)

    out_tensor_dict[input_fields.num_groundtruth_boxes] = tf.shape(
                out_tensor_dict[input_fields.groundtruth_boxes])[0]
    return out_tensor_dict

def image_radom_merge_fn(features,labels,record_seq):
    """
    features: Dictionary of feature tensors.
    labels: Dictionary of groundtruth tensors.
    record_seq: a int of 1 or 2 which response the seq of tfrecord defined in config
    """
    out_tensor_dict={}
    out_tensor_dict.update(features)
    out_tensor_dict.update(labels)
    out_tensor_dict.keys()
    out_tensor_dict = merge_proprecess.process(out_tensor_dict,record_seq)
   
    return out_tensor_dict

def pad_input_data_to_static_shapes(tensor_dict,
                                    max_num_boxes,
                                    spatial_image_shape=None):
    
    """
    tensor_dict: Tensor dictionary of input data
    max_num_boxes: Max number of groundtruth boxes needed to compute shapes for
            padding.
    spatial_image_shape: A list of two integers of the form [height, width]
            containing expected spatial shape of the image.
    """
    height, width = spatial_image_shape
    input_fields = fields.InputDataFields
    
    num_channels = tensor_dict[input_fields.image].shape[2]# type int

    padding_shapes = {
            input_fields.image: [height, width, num_channels],
            input_fields.original_image_spatial_shape: [2],
            input_fields.groundtruth_boxes: [max_num_boxes, 5],
            input_fields.num_groundtruth_boxes: [],
            input_fields.filename: [],
            input_fields.true_image_shape: [3],
    }

    
    padded_tensor_dict = {}
    
    def pad_or_clip_nd(tensor, output_shape):#TODO(sunling) 优化这个函数

        tensor_shape = tf.shape(tensor)
        clip_size = [
            tf.where(tensor_shape[i] - shape > 0, shape, -1)
            for i, shape in enumerate(output_shape)
        ]
        clipped_tensor = tf.slice(
            tensor,
            begin=tf.zeros(len(clip_size), dtype=tf.int32),
            size=clip_size)

        clipped_tensor_shape = tf.shape(clipped_tensor)
        trailing_paddings = [
            shape - clipped_tensor_shape[i] for i, shape in enumerate(output_shape)
        ]
        paddings = tf.stack(
            [
                tf.zeros(len(trailing_paddings), dtype=tf.int32),
                trailing_paddings
            ],
            axis=1)
        padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
        output_static_shape = [
            dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
        ]
        padded_tensor.set_shape(output_static_shape)
        return padded_tensor
    
    
    for tensor_name in tensor_dict:
        padded_tensor_dict[tensor_name] = pad_or_clip_nd(
            tensor_dict[tensor_name], padding_shapes[tensor_name])


    padded_tensor_dict[input_fields.num_groundtruth_boxes] = (
        tf.minimum(
            padded_tensor_dict[input_fields.num_groundtruth_boxes],
            max_num_boxes))
    
    return padded_tensor_dict




def _get_dicts(input_dict):
    """Extracts labels dict from input dict."""
    labels={
        fields.InputDataFields.num_groundtruth_boxes:
                input_dict[fields.InputDataFields.num_groundtruth_boxes],
        fields.InputDataFields.groundtruth_boxes:
                input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.filename:
                input_dict[fields.InputDataFields.filename]
    }
    features = {
        fields.InputDataFields.image:
            input_dict[fields.InputDataFields.image],
        
        fields.InputDataFields.true_image_shape:
            input_dict[fields.InputDataFields.true_image_shape],
        fields.InputDataFields.original_image_spatial_shape:
            input_dict[fields.InputDataFields.original_image_spatial_shape]
    }
    return features,labels




def train_input(model, record_seq):
    """
    model: A pre-constructed Detection Model.
    record_seq: a int of 1 or 2 which response the seq of tfrecord defined in config
    """
    model_preprocess_fn = model.preprocess

    def transform_and_pad_input_data_fn(tensor_dict):
        """Combines transform and pad operation."""
        transform_data_fn = functools.partial(
                transform_input_data, model_preprocess_fn=model_preprocess_fn,record_seq=record_seq)

        tensor_dict = pad_input_data_to_static_shapes(
                tensor_dict=transform_data_fn(tensor_dict),
                max_num_boxes=200,
                spatial_image_shape=[config.IMG_HEIGHT,config.IMG_WIDTH])
        return _get_dicts(tensor_dict)

    def image_merge_fn(features,labels):
        """Combines transform and pad operation."""
               
        tensor_dict = pad_input_data_to_static_shapes(
                tensor_dict=image_radom_merge_fn(features,labels,record_seq),
                max_num_boxes=200,
                spatial_image_shape=[config.IMG_HEIGHT,config.IMG_WIDTH])
        return _get_dicts(tensor_dict)
    dataset = dataset_builder.build(
            record_seq=record_seq,
            transform_input_data_fn=transform_and_pad_input_data_fn,
            image_merge_fn=image_merge_fn
            )
    return dataset


