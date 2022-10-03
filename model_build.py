# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;
#code_02:build model

import tensorflow as tf
import functools
import math


import boxlist.box_list as box_list
import boxlist.box_list_ops as box_list_ops
import model.box_predictor as box_predictor
import tools.standard_fields as fields
import tools.config as config
import model.feature_extractor as ResnetFE
import model.TargetAssigner as TargetAssigner
import model.anchor_generator as anchor_generator
import model.balanced_positive_negative_sampler as sampler


IMAGE_HEIGHT=config.IMG_HEIGHT
IMAGE_WIDTH=config.IMG_WIDTH 

class Model(tf.keras.layers.Layer):
 

    def __init__(self,
                    is_training,
                    minibatch_size=3000,
                    iou_thershold=0.5,
                    localization_loss_weight=1.0,
                    objectness_loss_weight=5.0,
                    angle_value_loss_weight=1.0,
                    angle_direction_loss_weight=10.0):                        
        """
            is_training: A boolean indicating whether the training version of the
                computation graph should be constructed.
            minibatch_size: The "batch size" to use for computing the
                objectness and location loss. This
                "batch size" refers to the number of anchors selected as contributing
                to the loss function
            localization_loss_weight: float
        """
      
        super(Model, self).__init__()

        self._groundtruth_lists = {}
        self._training_step = None
        
        self._is_training = is_training

     
        self._feature_extractor = ResnetFE.KerasFeatureExtractor(
                    is_training=is_training,
                    basemodel=config.BASE_MODEL)
     

        self._proposal_target_assigner = TargetAssigner.TargetAssigner()
    
        self._box_coder = self._proposal_target_assigner.box_coder

        self._anchor_generator = anchor_generator.AnchorGenerator()
     
 

        self._minibatch_size = minibatch_size
        self._sampler = sampler.BalancedPositiveNegativeSampler(
                positive_fraction=0.2 #TODO(sunling) 0.5->0.2
                )
        num_anchors_per_location = (
                self._anchor_generator.num_anchors_per_location())


        self._box_predictor = (
                box_predictor.BoxPredictorHead(
                        is_training=self._is_training,
                     
                        num_predictions_per_location=num_anchors_per_location[0],
                        name='BoxPredictor'))

        self._nms_fn=functools.partial( #TODO(sunling)testNMS
                tf.image.non_max_suppression,
                max_output_size=300,
                iou_threshold=0.4,
                score_threshold=1e-8
        )

        self._loc_loss_weight = localization_loss_weight
        self._obj_loss_weight = objectness_loss_weight
        self._angle_value_loss_weight=angle_value_loss_weight
        self._angle_direction_loss_weight=angle_direction_loss_weight
        anchor_shape=(config.IMG_HEIGHT, config.IMG_WIDTH)
        anchor_shapes=[]
        for i in range(1,7):
            anchor_shape=(math.ceil(anchor_shape[0]/2),math.ceil(anchor_shape[1]/2))#conv 4th
            if i in range(3,7):
                anchor_shapes.append(anchor_shape)
        anchors = (
                self._anchor_generator.generate(anchor_shapes))
        
        clip_window = tf.cast(tf.stack([0, 0, config.IMG_HEIGHT, config.IMG_WIDTH]),
                                                    dtype=tf.float32)
        anchors_boxlist = box_list_ops.clip_to_window(
                         anchors, clip_window, filter_nonoverlapping=False) # limit
        self._anchors = anchors_boxlist
   

    @property
    def anchors(self,scope=None):
        return self._anchors
    def _image_resizer(self,image,
                            img_dimension=config.IMG_HEIGHT,
                            method=tf.image.ResizeMethod.BILINEAR,
                            per_channel_pad_value=(0, 0, 0)):
        """
            image: A 3D tensor of shape [height, width, channels]
            img_dimension: maximum allowed size of the larger image dimension.
            method: (optional) interpolation method used in resizing. Defaults to BILINEAR.
            per_channel_pad_value: A tuple of per-channel scalar value to use for
                padding. By default pads zeros.

        """
        if len(image.get_shape()) != 3:
            raise ValueError('Image should be 3D tensor')

        with tf.name_scope('ImageResize'):
            new_image=tf.image.resize(
                    image, tf.stack([img_dimension, img_dimension]), method=method,
                    antialias=True, preserve_aspect_ratio=True)
            
            if new_image.get_shape().is_fully_defined(): #image.get_shape().is_fully_defined() is false on @tf.function 
                new_size = tf.constant(new_image.get_shape().as_list())
            else:
                new_size = tf.shape(new_image) 
            channels = tf.unstack(new_image, axis=2)
            new_image = tf.stack(
                    [
                            tf.pad(
                                    channels[i], [[0, img_dimension - new_size[0]],
                                                                [0, img_dimension - new_size[1]]],
                                    constant_values=per_channel_pad_value[i])
                            for i in range(len(channels))
                    ],
                    axis=2)
            new_image.set_shape([img_dimension, img_dimension, len(channels)])
            result = [new_image]
            result.append(new_size)
            return result
    def call(self, images):
        """
        images: a [batch_size, height, width, channels] float tensor.

        Returns:
             detetcions: The dict of tensors returned by the postprocess function.
        """

        preprocessed_images, shapes = self.preprocess(images)
        prediction_dict = self.predict(preprocessed_images, shapes)
        return self.postprocess(prediction_dict, shapes)

    def preprocess(self, inputs):
          with tf.name_scope('Preprocessor'):
                outputs = [self._image_resizer(arg) for arg in tf.unstack(inputs)]
                outputs = [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
                (resized_inputs,true_image_shapes)=(outputs[0],outputs[1])

                return (resized_inputs,true_image_shapes)
    def regularization_losses(self):
        """
        Returns:
            A list of regularization loss tensors.
        """
        losses = []
        slim_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
      
        if slim_losses:
            losses.extend(slim_losses)

        losses.extend(self._box_predictor.losses)

        losses.extend(self._feature_extractor.losses)
       
        return losses

    def _focal_loss(self, logits, labels, gamma=2):
        '''
        :param logits: [batch_size, n_class]
        :param labels: [batch_size]
        :return: -(1-y)^r * log(y)
        '''
        labels=tf.convert_to_tensor(labels)
        logits=tf.convert_to_tensor(logits)
        
        softmax = tf.math.softmax(logits)  # [batch_size * n_class]
        softmax_loss=-tf.math.multiply(tf.math.pow(tf.math.subtract(1.,softmax),gamma),tf.math.log(softmax))
        loss=tf.reduce_sum(tf.math.multiply(labels,softmax_loss),axis=-1)
        return loss
    def _localization_and_angle_loss(self,predict_box,predict_angle, target_tensor, weights):
            """
                prediction_tensor: A float tensor of shape [batch_size, num_anchors,
                    4] representing the (encoded) predicted locations of objects.
                predict_angle: A float tensor of shape [batch_size, num_anchors,
                    3] representing the regression targets
                weights: a float tensor of shape [batch_size, num_anchors]
            """          
            cy, cx, w, h , target_angle = tf.split(target_tensor,num_or_size_splits=5,axis=-1)
            target_box      = tf.concat([cy, cx, w, h],axis=-1)
            target_angle    = tf.squeeze(target_angle,-1)
            target_angle_direction = tf.cast(tf.math.greater(target_angle,0),tf.int32)
            target_angle_direction_one_hot = tf.one_hot(target_angle_direction,2,dtype=tf.float32)
            angle_direct_weight=tf.where(
                tf.logical_and(
                    tf.greater(target_angle, -config.ROTATE_IGNORE_THERSHOLD),
                    tf.less(target_angle, config.ROTATE_IGNORE_THERSHOLD)),
                0.,
                1.)
            
            target_angle_value  = tf.math.abs(target_angle)
            
            predict_angle_direction=tf.slice(predict_angle,[0,0,0],[predict_angle.shape[0],predict_angle.shape[1],2])
            predict_angle_value=tf.slice(predict_angle,[0,0,2],[predict_angle.shape[0],predict_angle.shape[1],1])
            predict_angle_value=tf.squeeze(predict_angle_value,-1)
            
            
          
            with tf.name_scope('Loss'):
                #----cal boundbox loss----#
                localization_loss=tf.reduce_sum(
                       #tf.math.multiply(
                           tf.compat.v1.losses.huber_loss(
                               target_box,
                               predict_box,
                               delta=1.0,
                               weights=tf.expand_dims(weights, axis=2),
                               loss_collection=None,
                               reduction=tf.losses.Reduction.NONE),
                       #batchweights),# mulity angle by 10 
                   axis=2)
               
                #----cal angledirection loss ------#
  
                
                
                batch_angle_direct_weight=tf.math.multiply(weights,angle_direct_weight)
                
                angle_direction_loss=tf.math.multiply(
                    ##tf.nn.softmax_cross_entropy_with_logits(
                    self._focal_loss(
                        labels=target_angle_direction_one_hot,
                        logits=predict_angle_direction),
                    batch_angle_direct_weight)
                #----cal angel value loss----#
           
                batch_angle_value_weight1=tf.tile(tf.constant([[30.]]),[config.BATCH_SIZE1,1])
                batch_angle_value_weight2=tf.tile(tf.constant([[0.0]]),[config.BATCH_SIZE2,1])
                batch_angle_value_weight=tf.concat([batch_angle_value_weight1,batch_angle_value_weight2],axis=0)
                batch_angle_value_weight=tf.math.multiply(weights,batch_angle_value_weight)
                
                anglesubstract = tf.math.abs(tf.math.subtract(target_angle_value,predict_angle_value))
                #anglelossline=tf.math.minimum(anglesubstract,tf.math.subtract(2.,anglesubstract))
                  
                angle_value_loss=tf.math.multiply(
                                tf.math.multiply(
                                    tf.convert_to_tensor(0.5, dtype=tf.float32),
                                    tf.multiply(anglesubstract, anglesubstract)),
                                batch_angle_value_weight)
                
                
                
                normalizer = tf.maximum(
                         tf.reduce_sum(weights, axis=1), 1.0)
                normalizer2 = tf.maximum(
                         tf.reduce_sum(batch_angle_direct_weight, axis=1), 1.0)  
                
                
                localization_loss = tf.reduce_mean(
                        tf.reduce_sum(localization_loss, axis=1) / normalizer)    #each loss
      
                angle_value_loss = tf.reduce_mean(
                        tf.reduce_sum(angle_value_loss, axis=1) / normalizer)    #each loss
                      
                angle_direction_loss = tf.reduce_mean(
                    tf.reduce_sum(angle_direction_loss, axis=1) / normalizer2)
                
                
                
                
                return localization_loss,angle_value_loss,angle_direction_loss
              
                
    def _objectness_loss(self,
                            prediction_tensor,
                            target_tensor,
                            weights):
            #loss=tf.nn.softmax_cross_entropy_with_logits(
            loss=self._focal_loss(
                labels=target_tensor,
                logits=prediction_tensor)
            
            normalizer = tf.maximum(
                    tf.reduce_sum(weights, axis=1), 1.0)
            objectness_losses = tf.math.multiply(loss,weights)
            objectness_loss = tf.reduce_mean(
                    tf.reduce_sum(objectness_losses, axis=1) / normalizer)
            return objectness_loss
                
    def predict(self, preprocessed_inputs):
        """
        preprocessed_inputs: a [batch, height, width, channels] float tensor
                representing a batch of images.

        """
     
        features_maps= self._feature_extractor(preprocessed_inputs,training=self._is_training) #getFeature

        box_predictions = self._box_predictor(
                    features_maps,training=self._is_training) #get location and class from feature
        
        box_encodings = box_predictions['box_encodings']
        objectness =(
                box_predictions['objectness'])
        angle_encodings=box_predictions['angle_encodings']
        prediction_dict = {
                'box_encodings':
                        box_encodings,
                'objectness':
                        objectness,
                'angle_encodings':
                        angle_encodings,
        }
        return prediction_dict




    def loss(self, prediction_dict, true_image_shapes):
            """
            prediction_dict: a dictionary holding prediction tensors (
                        contain `box_encodings`,
                    `objectness_predictions_with_background`)
            """
            with tf.name_scope('Loss'):
                (groundtruth_boxlists, groundtruth_classes_with_background_list
                ) = self._format_groundtruth_data() #get groundtruth_data
                            
                (batch_cls_targets, batch_cls_weights, batch_reg_targets,
                 batch_reg_weights, _) = TargetAssigner.batch_assign_targets(
                         target_assigner=self._proposal_target_assigner,
                         anchors_batch=self.anchors,
                         gt_box_batch=groundtruth_boxlists,
                         gt_class_targets_batch=(len(groundtruth_boxlists) * [None])) #get target and weight
                         
                         
                batch_cls_weights = tf.reduce_mean(batch_cls_weights, axis=2)    #batch_cls_weights [batch，6912,1]->[batch，6912]
                batch_cls_targets = tf.squeeze(batch_cls_targets, axis=2)            #batch_cls_targets [batch，6912,1]->[batch，6912]
                #as negative object is setting to 1 in cls_weights, batch_cls_weights is very big
                def _minibatch_subsample_fn(inputs):
                    cls_targets, cls_weights = inputs
                    return self._sampler.subsample(
                            tf.cast(cls_weights, tf.bool),
                            self._minibatch_size, tf.cast(cls_targets, tf.bool))
            
                arg_tuples = zip(*[tf.unstack(elem) for elem in [batch_cls_targets, batch_cls_weights]])
                batch_sampled_indices = tf.cast(tf.stack([_minibatch_subsample_fn(arg_tuple) for arg_tuple in arg_tuples]),dtype=tf.float32) #pick 256 objects
                #shape=[(12, 6912)],tf.reduce_sum(batch_sampled_indices,axis=1)=[255,255,255,255,255,255,255,255,255,255,255,255]
                #blance
        
                # Normalize by number of examples in sampled minibatch
                
                batch_one_hot_targets = tf.one_hot(
                        tf.cast(batch_cls_targets, dtype=tf.int32), depth=2) #TensorShape([12, 6912, 2])
                sampled_reg_indices = tf.multiply(batch_sampled_indices, batch_reg_weights) #pick XXX object to cal loss
                
                
                #shape=[(12, 6912)],tf.reduce_sum(sampled_reg_indices,axis=1)=[32.,35.,......]
             
                localization_losses,angle_value_losses,angle_direction_losses = self._localization_and_angle_loss(
                        prediction_dict['box_encodings'], prediction_dict['angle_encodings'], batch_reg_targets, weights=sampled_reg_indices)#, limit to positive object weights
                        #batch_reg_targets range (-1.7401459, 1.6572816) get from ground_truth和anchor，and encoded by "box_coder"
                objectness_losses = self._objectness_loss(
                        prediction_dict['objectness'],
                        batch_one_hot_targets,
                        weights=batch_sampled_indices)
                
                    
                
        
                localization_loss = tf.multiply(
                    self._loc_loss_weight,
                    localization_losses,
                    name='localization_loss')
                angle_value_loss = tf.multiply(
                    self._angle_value_loss_weight,
                    angle_value_losses,
                    name='angle_value_loss')
                angle_direction_loss = tf.multiply(
                    self._angle_direction_loss_weight,
                    angle_direction_losses,
                    name='angle_direction_loss')
                objectness_loss = tf.multiply(
                    self._obj_loss_weight, 
                    objectness_losses, 
                    name='objectness_loss')
                
                loss_dict = {'Loss/localization_loss': localization_loss,
                             'Loss/angle_value_loss': angle_value_loss,
                             'Loss/angle_direction_loss': angle_direction_loss,
                             'Loss/objectness_loss': objectness_loss}
            return loss_dict

    def _format_groundtruth_data(self):
        """

        Returns:
            groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
                of the groundtruth boxes.
            groundtruth_classes_with_background_list: A list of 2-D one-hot
                (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
                class targets with the 0th index assumed to map to the background class.
        """
       
        groundtruth_boxlists = [
                box_list_ops.to_absolute_coordinates(
                        box_list.BoxList(boxes), IMAGE_HEIGHT, IMAGE_WIDTH)# box_list.BoxList(boxes), image_shapes[i, 0], image_shapes[i, 1])
                for i, boxes in enumerate(
                        self.groundtruth_lists(fields.BoxListFields.boxes))
        ]
        groundtruth_objectness_list = []
        for one_hot_encoding in self.groundtruth_lists(
                fields.BoxListFields.objectness):
            groundtruth_objectness_list.append(one_hot_encoding)
        return (groundtruth_boxlists, groundtruth_objectness_list)



    def groundtruth_lists(self, field):

        if field not in self._groundtruth_lists:
            raise RuntimeError('Groundtruth tensor {} has not been provided'.format(
                    field))
        return self._groundtruth_lists[field]




    def provide_groundtruth(
                self,
                groundtruth_boxes_list,
                training_step=None):
            """
            Args:
                groundtruth_boxes_list: a list of 2-D tf.float32 tensors of shape
                    [num_boxes, 4] containing coordinates of the groundtruth boxes.
                        Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
                        format and assumed to be normalized and clipped
                        relative to the image window with y_min <= y_max and x_min <= x_max.
                training_step: An integer denoting the current training step. This is
                    useful when models want to anneal loss terms.
            """
            self._groundtruth_lists[fields.BoxListFields.boxes] = groundtruth_boxes_list
            self._groundtruth_lists[
                     fields.BoxListFields.objectness] = [tf.pad(tf.expand_dims(tf.cast(tf.reduce_sum(single,axis=-1)>0.,tf.float32),axis=-1),[[0,0],[1,0]],mode='CONSTANT') for single in groundtruth_boxes_list]
            if training_step is not None:
                self._training_step = training_step

   
    def postprocess(self, prediction_dict, true_image_shapes):
        """
          prediction_dict:  contain `box_encodings`,
            `objectness_predictions_with_background` fields.
          true_image_shapes: int32 tensor of shape [batch, 3] 
            
        """
        with tf.name_scope('Postprocessor'):
            
            box_encodings_batch=prediction_dict['box_encodings']
            
            predict_angle=prediction_dict['angle_encodings']
            predict_angle_direction=tf.slice(predict_angle,[0,0,0],[predict_angle.shape[0],predict_angle.shape[1],2])
            predict_angle_value=tf.slice(predict_angle,[0,0,2],[predict_angle.shape[0],predict_angle.shape[1],1])
            
            positiveprob=tf.nn.softmax(
                    predict_angle_direction)[:,:,1:2]
            angle_symbol=tf.where(tf.less(positiveprob,0.5),-1.,1.)
            angle_batch=tf.math.multiply(predict_angle_value,angle_symbol)
            box_encodings_with_angle_batch=tf.concat([box_encodings_batch,angle_batch], axis=-1)
            raw_proposal_boxes = self._batch_decode_boxes(box_encodings_with_angle_batch) #decodebox TensorShape([3, 6912,5])
            #raw_proposal_boxes = tf.squeeze(proposal_boxes, axis=2) # TensorShape([3, 6912, 4])
            objectness_softmax = tf.nn.softmax(
                    prediction_dict['objectness'])     #shape=(3, 6912, 2)
            objectness= objectness_softmax[:, :, 1]
            #predictions=[self._postprocess_one(*keys) for keys in zip(raw_proposal_boxes,objectness,true_image_shapes)]
            predictions=self._postprocess_one(raw_proposal_boxes[0],objectness[0],true_image_shapes[0])
            return predictions
            

    def _postprocess_one(self,proposal_boxes,objectness,true_image_shape):
        sorted_boxes,sorted_scores = box_list_ops.sort_by_score(box_list.BoxList(proposal_boxes),
                                                                objectness)
        clip_window = tf.cast(tf.pad(true_image_shape,[[2,0]])[:-1],tf.float32)
        sorted_boxes,sorted_scores = box_list_ops.clip_to_window(
                sorted_boxes,
                clip_window,
                sorted_scores,
                filter_nonoverlapping=True)#clip window

        slide= self._nms_fn(
                    sorted_boxes.get()[:,:4],
                    sorted_scores)
        sorted_boxes=box_list_ops.gather(sorted_boxes,slide)
        sorted_scores=box_list_ops.gather(sorted_scores,slide)
        normalized_sorted_boxes= box_list_ops.to_normalized_coordinates(
                    sorted_boxes,true_image_shape[0],
                    true_image_shape[1]).get()
        return {
                fields.DetectionResultFields.detection_boxes:
                        normalized_sorted_boxes,
                fields.DetectionResultFields.detection_scores:
                        sorted_scores
        }
    def _batch_decode_boxes(self, box_encodings_with_angle_batch):
        """
        box_encodings: a 4-D tensor with shape
            [batch_size, num_anchors, 5]
            representing box encodings.
        anchor_boxes: [batch_size, self._box_coder.code_size]
            representing decoded bounding boxes. If using a shared box across
            classes the shape will instead be
            [total_num_proposals, 1, self._box_coder.code_size].
        """
        batch_size = box_encodings_with_angle_batch.shape.as_list()[0]#Note:batch_size in this is not same as config file
        anchor_boxes = tf.tile(
                tf.expand_dims(self.anchors.get(), 0), [batch_size, 1, 1])#TensorShape([batch, 6912, 5])
        
        anchors_boxlist = box_list.BoxList(
                tf.reshape(anchor_boxes, [-1, 5]))
        decoded_boxes = self._box_coder.decode(
                tf.reshape(box_encodings_with_angle_batch, [-1, 5]),
                anchors_boxlist)
        return tf.reshape(decoded_boxes.get(),
                tf.stack([batch_size, -1, 5]))

