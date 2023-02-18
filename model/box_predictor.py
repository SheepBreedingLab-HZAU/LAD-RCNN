# Copyright (c) 2023 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;
# -*- coding: utf-8 -*-
# code_07:build model subprog box predictor header


import tensorflow as tf

class BoxPredictorHead(tf.keras.layers.Layer):

    def __init__(self,
        is_training,
        num_predictions_per_location,
        name=None):
        """
            is_training: Indicates whether the BoxPredictor is in training mode.
            num_predictions_per_location: int
            name: A string name scope to assign to the model. 
        """
 
        
        super(BoxPredictorHead, self).__init__(name=name)
        
        self._is_training = is_training

        self._inplace_batchnorm_update = False
        #self.call=self._predict

        self._Conv2D=(
            tf.keras.layers.Conv2D(
                    num_predictions_per_location * 9,
                    [3,3],
                    padding='SAME',
                    name='BOXPredictorWithAngle',
                    bias_initializer=tf.constant_initializer(0.0),
                    kernel_regularizer=tf.keras.regularizers.l2(float(0.0 * 0.5)),
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                            mean=0.0,stddev=0.01),
                    activation=None,
                    use_bias=True))

    def call(self, image_features,**kwargs):
            return self._predict(image_features)
    
    def _predict(self, image_features):
        """
            image_features: A list of float tensors of shape [batch_size, height_i,
                width_i, channels_i] containing features for a batch of images.
        """
        
        batch_size = tf.shape(image_features[0])[0]
        predictions = {}
        
        convpreds=[]
        for f in image_features:
            convpred_=self._Conv2D(f)
            convpred_=tf.reshape(convpred_,[batch_size, -1, 9])
            convpreds.append(convpred_)
        convpred=tf.concat(convpreds,axis=1)
        
        sp=tf.split(value=convpred, num_or_size_splits=9, axis=-1)
        objectness=tf.concat(sp[:2],axis=-1)
        box_encodings=tf.concat(sp[2:6],axis=-1)
        angle_direction=tf.concat(sp[6:8],axis=-1)
        angle_value=sp[8]

        angle_value=tf.math.sigmoid(angle_value)
        angle_encodings=tf.concat([angle_direction,angle_value],axis=-1)# angle between [-1,1]
        predictions['box_encodings']=box_encodings
        predictions['objectness'] = objectness
        predictions['angle_encodings']=angle_encodings
        return predictions


