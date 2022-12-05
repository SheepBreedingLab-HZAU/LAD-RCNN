# Copyright (c) 2022 Jiang Xunping and Sun Ling

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Licensed under the MIT license;

#code_01 train
import os


import tensorflow as tf
import time

import model_build
import tools.standard_fields as fields
import dataset.inputs as inputs
import tools.config as config
import tools.optimizer_builder as optimizer_builder

args=config.args
sequence=args.sequence

NUM_STEPS_PER_ITERATION = 1


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf.compat.v1.logging.info("start")

import logging
logger = logging.getLogger()
while logger.handlers:
         logger.handlers.pop()



print(config.BASE_MODEL,config.TRAIN_NUM_STEP,sequence)




def _compute_losses_and_predictions_dicts(
    model, features, labels, training_step=None):
    """
        model: a DetectionModel.
        features: Dictionary of feature tensors from the input dataset.
        labels: A dictionary of groundtruth tensors post-unstacking. 
        training_step: int, the current training step.
    """

    model.provide_groundtruth(
            groundtruth_boxes_list=labels[fields.InputDataFields.groundtruth_boxes],
            training_step=training_step)    
    preprocessed_images = features[fields.InputDataFields.image]
    prediction_dict=model.predict(preprocessed_images)
    losses_dict = model.loss(
            prediction_dict, features[fields.InputDataFields.true_image_shape])
    
    
    regularization_losses = model.regularization_losses()
    regularization_loss = tf.add_n(
                    regularization_losses, name='regularization_loss')

    losses_dict['Loss/regularization_loss'] = regularization_loss
    losses = [loss_tensor for loss_tensor in losses_dict.values()]
    total_loss = tf.add_n(losses, name='total_loss')
    losses_dict['Loss/total_loss'] = total_loss
    
    return losses_dict, prediction_dict





def eager_train_step(detection_model,
                    features,
                    labels, 
                    optimizer,
                    training_step):
    """
        detection_model: A DetectionModel (based on Keras) to train.
        features: Dictionary of feature tensors from the input dataset.
        labels: A dictionary of groundtruth tensors. 
        optimizer: The training optimizer that will update the variables.
        training_step: int, the training step number.
        num_replicas: The number of replicas in the current distribution strategy.
    """
    is_training = True

    detection_model._is_training = is_training 
    labels={
            key: tf.unstack(tensor) for key, tensor in labels.items()
    }

    with tf.GradientTape() as tape:
        losses_dict, _ = _compute_losses_and_predictions_dicts(
                detection_model, features, labels,
                training_step=training_step)



    trainable_variables = detection_model.trainable_variables

    total_loss = losses_dict['Loss/total_loss']
    gradients = tape.gradient(total_loss, trainable_variables)
 
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return losses_dict


def train_loop(train_steps=None,#None
        save_final_config=False,
        model_dir='savemodel/',
        record_summaries=True,
        num_steps_per_iteration=NUM_STEPS_PER_ITERATION,
        checkpoint_every_n=1000,
        performance_summary_exporter=None,
        checkpoint_max_to_keep=10,
        loggstep=100):
    steps_per_sec_list = []

    if train_steps is None and config.TRAIN_NUM_STEP != 0:
        train_steps = config.TRAIN_NUM_STEP


 
    detection_model = _build_faster_rcnn_model(is_training=True)

    def train_dataset_fn():
        """Callable to create train input."""

        train_input1=inputs.train_input(model=detection_model,record_seq=1)
        train_input2=inputs.train_input(model=detection_model,record_seq=2)
        train_input1=train_input1.repeat()
        train_input2=train_input2.repeat()
        
        return train_input1,train_input2

    
    train_input=train_dataset_fn()#


    global_step = tf.Variable(
        0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
        aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)
    
    
    optimizer, (learning_rate,) = optimizer_builder.build(global_step=global_step)


    summary_writer_filepath=os.path.join(model_dir, 'train')
    
    summary_writer = tf.compat.v2.summary.create_file_writer(
            summary_writer_filepath)

    with summary_writer.as_default():
        with tf.compat.v2.summary.record_if(
                lambda: global_step % num_steps_per_iteration == 0):
    
            ckpt = tf.compat.v2.train.Checkpoint(
                    step=global_step, model=detection_model, optimizer=optimizer)
            
            
            manager_dir = model_dir

            manager = tf.compat.v2.train.CheckpointManager(
                    ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)

            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            ckpt.restore(latest_checkpoint)
            tf.print(global_step.value())
            @tf.function
            def train_step_fn(features, labels):
                """Single train step."""
                if record_summaries:
                    tf.compat.v2.summary.image(
                        name='train_input_images',
                        step=global_step,
                        data=features[fields.InputDataFields.image],
                        max_outputs=3)
                losses_dict = eager_train_step(
                    detection_model,
                    features,
                    labels,
                    optimizer,
                    training_step=global_step)#,
            
                global_step.assign_add(1)
                return losses_dict

            
            def _dist_train_step(data_iterator):
                """A distributed train step."""
                features1, labels1 = data_iterator[0].next()
                features2, labels2 = data_iterator[1].next()
                features={}
                labels={}
                for f in features1:
                        features[f]=tf.concat([features1[f],features2[f]],axis=0)
                for f in labels1:
                        labels[f]=tf.concat([labels1[f],labels2[f]],axis=0)
                return train_step_fn(features, labels)

            
            train_input_iter = iter(train_input[0]),iter(train_input[1])

            if int(global_step.value()) == 0:
                manager.save()

            checkpointed_step = int(global_step.value())
            logged_step = global_step.value()

            last_step_time = time.time()
            for _ in range(global_step.value(), train_steps,num_steps_per_iteration):
             
                losses_dict = _dist_train_step(train_input_iter)
                
                time_taken = time.time() - last_step_time
                last_step_time = time.time()
                steps_per_sec = num_steps_per_iteration * 1.0 / time_taken

                tf.compat.v2.summary.scalar(
                        'steps_per_sec', steps_per_sec, step=global_step)

                steps_per_sec_list.append(steps_per_sec)

                logged_dict = losses_dict.copy()
                logged_dict['learning_rate'] = learning_rate()

                for key, val in logged_dict.items():
                    tf.compat.v2.summary.scalar(key, val, step=global_step)

                if global_step.value() - logged_step >= loggstep:#TODO(sunling)
                    logged_dict_np = {name: value.numpy() for name, value in
                                                        logged_dict.items()}
                    tf.compat.v1.logging.info(
                            'Step {} per-step time {:.3f}s'.format(
                                    global_step.value(), time_taken / num_steps_per_iteration))
                    tf.compat.v1.logging.info(
                        'totalloss {:.3f} learning_rate {:.3f} objectness_loss {:.3f} localization_loss {:.3f} angle_value_loss {:.3f} angle_direction_loss {:.3f} regularization_loss {:.3f}'.format(
                            float(logged_dict_np['Loss/total_loss']),
                            float(logged_dict_np['learning_rate']),
                            float(logged_dict_np['Loss/objectness_loss']),
                            float(logged_dict_np['Loss/localization_loss']),
                            float(logged_dict_np['Loss/angle_value_loss']),
                            float(logged_dict_np['Loss/angle_direction_loss']),
                            float(logged_dict_np['Loss/regularization_loss'])))
                    
         
                    logged_step = global_step.value()

                if ((int(global_step.value()) - checkpointed_step) >=
                        checkpoint_every_n):
                    manager.save()
                    checkpointed_step = int(global_step.value())


def _build_faster_rcnn_model(is_training):
    return model_build.Model(
        is_training=is_training,     
        minibatch_size=config.MINIBATCH_SIZE,
        localization_loss_weight=config.LOCALIZATION_LOSS_WEIGHT,
        objectness_loss_weight=config.OBJECTNESS_LOSS_WEIGHT,
        iou_thershold=config.NMS_IOU_THRESHOLD)


        
if __name__=="__main__":
    train_loop()        
 