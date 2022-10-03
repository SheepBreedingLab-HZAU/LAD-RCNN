# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;

# code_04:build dataset subprog analysis tfrecord file
# 

import tensorflow as tf
import tools.config as config

import tools.standard_fields as fields
import functools


def build(record_seq, transform_input_data_fn=None,image_merge_fn=None):
    """
    record_seq should be 1 or 2
    transform_input_data_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.
    image_merge_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.

    """
  
    decoder = TfExampleDecoder()
    if record_seq==1:
        batch_size=config.BATCH_SIZE1
        input_files=config.TFRECORD_PATH1    
        mp2X2=config.MERGERED_2X2PROBABILITY1
        mp3X3=config.MERGERED_3X3PROBABILITY1
    elif record_seq==2:
        batch_size=config.BATCH_SIZE2
        input_files=config.TFRECORD_PATH2
        mp2X2=config.MERGERED_2X2PROBABILITY2
        mp3X3=config.MERGERED_3X3PROBABILITY2
    else :
        raise Exception("record_seq should be 1 or 2")
    filenames = tf.io.gfile.glob(input_files) #get all files that match the given pattern(s).

    if not filenames:
        raise RuntimeError('Did not find any input files matching the glob pattern '
                            '{}'.format(input_files))
    dataset = tf.data.TFRecordDataset(filenames,buffer_size=8 * 1000 * 1000)#read tfrecord
    decode=functools.partial(decoder.decode,
                             add_angle=(record_seq==1))
    dataset = dataset.map(decode, tf.data.AUTOTUNE)
    dataset=dataset.shuffle(buffer_size=2048)#size should 
    #transfrom_imput_data
    if transform_input_data_fn:
        dataset=dataset.map(transform_input_data_fn, tf.data.AUTOTUNE)
    if image_merge_fn:
        if mp3X3>0:
            bt1=9
        elif mp2X2>0:
            bt1=4
        else:
            bt1=1
        dataset = dataset.batch(bt1,drop_remainder=True)
        dataset=dataset.map(image_merge_fn, tf.data.AUTOTUNE)
    if batch_size:
        dataset = dataset.batch(batch_size,drop_remainder=True)#TODO(sunling) replace 
    dataset = dataset.prefetch(2)
    return dataset




class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(self):
        """Constructor sets keys_to_features and items_to_handlers."""
        self.keys_to_features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/object/bbox/xmin':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/angle':
                tf.io.VarLenFeature(tf.float32),
        }
        self.items_to_handlers = {
            fields.InputDataFields.image:'image/encoded',
            #fields.InputDataFields.source_id: 'image/source_id',
            fields.InputDataFields.filename: 'image/filename',
            #fields.InputDataFields.groundtruth_classes:'image/object/class/label',
        }
        self.groundtruth_boxes={}
        for k in ['ymin', 'xmin', 'ymax', 'xmax', 'angle']:
            self.groundtruth_boxes.update({k:'image/object/bbox/'+k})
        self.groundtruth_boxes2={}
        for k in ['ymin', 'xmin', 'ymax', 'xmax']:
            self.groundtruth_boxes2.update({k:'image/object/bbox/'+k})
    def decode(self, tf_example_string_tensor,add_angle=False):
        """
          tf_example_string_tensor: a string tensor holding a serialized tensorflow
            example proto.
        """
        serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    
        tensor_dict={}
        tensor_dict1 = tf.io.parse_single_example(
                serialized_example,
                features=self.keys_to_features
            )
        #channel=1 if tensor_dict1['image/format']=='bmp' else 3
        for k in self.items_to_handlers:
            tensor_dict[k]=tensor_dict1[self.items_to_handlers[k]] #no SparseTensor
          
        groundtruth_boxes=[]
        if add_angle:
            for k in self.groundtruth_boxes:
                groundtruth_boxes.append(tf.expand_dims(tensor_dict1[self.groundtruth_boxes[k]].values,1))
            #groundtruth_boxes[-1]=tf.where(tf.less(0,groundtruth_boxes[-1].get_shape()[0]),groundtruth_boxes[-1],tf.zeros(groundtruth_boxes[0].get_shape(),dtype=tf.float32))   
            groundtruth_boxes=tf.concat(groundtruth_boxes,1)
        else:
            for k in self.groundtruth_boxes2:
                groundtruth_boxes.append(tf.expand_dims(tensor_dict1[self.groundtruth_boxes2[k]].values,1))
            #groundtruth_boxes[-1]=tf.where(tf.less(0,groundtruth_boxes[-1].get_shape()[0]),groundtruth_boxes[-1],tf.zeros(groundtruth_boxes[0].get_shape(),dtype=tf.float32))   
            #print(groundtruth_boxes)
            groundtruth_boxes=tf.concat(groundtruth_boxes,1)
            groundtruth_boxes=tf.pad(groundtruth_boxes,[[0,0],[0,1]],mode='CONSTANT',constant_values=0.)
        tensor_dict[fields.InputDataFields.groundtruth_boxes]=groundtruth_boxes  
        tensor_dict[fields.InputDataFields.image]=tf.image.decode_image(tensor_dict[fields.InputDataFields.image])
        tensor_dict[fields.InputDataFields.image].set_shape([None, None, config.INPUT_CHANNEL])
        tensor_dict[fields.InputDataFields.original_image_spatial_shape] = tf.shape(
            tensor_dict[fields.InputDataFields.image])[:2]
    
        return tensor_dict