# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

# code_06:build model subprog extractor features

import tensorflow as tf
import model.vgg16 as vgg16
import model.ournet as ournet
import tools.config as config
class KerasFeatureExtractor(
    tf.keras.Model): 
    def __init__(self,
               is_training,
               basemodel='ournet',
               name='FeatureExtractor'):
        """
          is_training: whether the network is in training mode.
          basemodel should be one of ['ournet','mobilnetV2','vgg16','resnet50']
          name: a string name scope to assign to the model. If 'None', Keras
            will auto-generate one from the class name.
        """
        super(KerasFeatureExtractor, self).__init__(name=name)
        self._is_training = is_training
        self._basemodel=basemodel
        if (basemodel not in['ournet','mobilnetV2','vgg16','resnet50']):
          raise ValueError('basemodel should be ournet, mobilnetV2, vgg16 or resnet50')
    
   
    def call(self, inputs, **kwargs):
        return self._extract_features(inputs)

  
    def build(self, input_shape):
        self._variable_dict = {}
      
        self.built = True
        if self._basemodel=='ournet':
            model =  ournet.Model()
            outputs = [model.get_layer(output_layer_name).output
                      for output_layer_name in  ['block3_conv3','block4_conv3', 'block5_conv3']]
        elif self._basemodel=='vgg16':
            model =  vgg16.VGG16()
            outputs = [model.get_layer(output_layer_name).output
                      for output_layer_name in  ['block3_pool','block4_pool', 'block5_pool']]
        elif self._basemodel=='resnet50' :
            model = tf.keras.applications.resnet.ResNet50( 
              input_shape=(config.IMG_HEIGHT,config.IMG_WIDTH,config.INPUT_CHANNEL),
              classes=None,
               weights=None,
               include_top=False)
            outputs = [model.get_layer(output_layer_name).output
                      for output_layer_name in  ['conv3_block4_out','conv4_block6_out', 'conv5_block3_out']]
     
        elif self._basemodel=='mobilnetV2' :
            model =  tf.keras.applications.mobilenet_v2.MobileNetV2( 
                    input_shape=(config.IMG_HEIGHT,config.IMG_WIDTH,config.INPUT_CHANNEL),
                    classes=None,
                    weights=None,
                    include_top=False)
            outputs = [model.get_layer(output_layer_name).output
                        for output_layer_name in  ['block_6_expand_relu','block_13_expand_relu', 'out_relu']]
        self.classification_backbone = tf.keras.Model(
           inputs=model.inputs,
           outputs=outputs)
        depth=128
        self._conv=[]
        for i in range(3):
           self._conv.append(tf.keras.Sequential([
                   tf.keras.layers.Conv2D(depth, 1,
                                 strides=1,
                                 padding='SAME',
                                 use_bias=False,
                                 name="boxPred_conv_{}".format(i)),
                   tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.997,name="boxPred_norm_{}".format(i)),
                   tf.keras.layers.Activation('relu',name='boxPredActivation_{}'.format(i))
               ],name='boxPred_{}'.format(i)))
      
        self._aftconvList=[]
        for i in range(5,6):
            layers = []
            layer_name = 'bottom_block{}'.format(i) 
            layers.append(
              tf.keras.layers.Conv2D(
                depth,
                [3, 3],
                padding='SAME',
                strides=2,
                name=layer_name + '_conv',
                use_bias=False)) 
            layers.append( 
              tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.997,name=layer_name+"_norm"))  
            layers.append(
              tf.keras.layers.Activation('relu',name=layer_name)) 
            self._aftconvList.append(layers)
        self._box_predictor_conv =[]
        for i in range(4):
            self._box_predictor_conv.append(
               tf.keras.Sequential([
                     tf.keras.layers.Conv2D(
                             128,
                             kernel_size=[3,3],
                             padding='SAME',
                             name='BoxPredictorConv_{}'.format(i),
                             kernel_regularizer=tf.keras.regularizers.l2(float(0.0 * 0.5)),
                             kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                     mean=0.0,stddev=0.01),
                             activation=None,
                             use_bias=True), 
                     #tf.keras.layers.Lambda(tf.identity),
                     tf.keras.layers.Lambda(
                             tf.nn.relu6,
                             name='BoxPredictorActivation_{}'.format(i))
             ], name='BoxPredictorFeatures_{}'.format(i)))
    
    def _extract_features(self, preprocessed_inputs):
        """
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
        """
    
        image_features = self.classification_backbone(preprocessed_inputs)
        featureL3=self._conv[0](image_features[0])
        featureL4=self._conv[1](image_features[1])
        featureL5=self._conv[2](image_features[2])
        upsamplingL4=tf.image.resize(
                  featureL5, [tf.shape(featureL4)[1], tf.shape(featureL4)[2]],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#upsampling
        featureL4+=upsamplingL4
        upsamplingL3=tf.image.resize(
                  featureL4, [tf.shape(featureL3)[1], tf.shape(featureL3)[2]],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#upsampling
        featureL3+=upsamplingL3
       
        features_=[featureL3,featureL4,featureL5]
       
        lastfeature=features_[-1]
        for aftconv in self._aftconvList:
            for layer in aftconv:
                lastfeature=layer(lastfeature)
            features_.append(lastfeature)
        features=[]
        for i,feature in enumerate(features_):
            features.append(self._box_predictor_conv[i](feature))
        return features




