# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;

#code_15 config file
"""Functions to build DetectionModel training optimizers."""

import tensorflow as tf
import tools.config as config
import numpy as np

def build(global_step=None):
    """
        global_step: A variable representing the current step.
    """
    
    optimizer = None
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()
    summary_vars = []
    
    learning_rate = learn_rate_fn(
        global_step,
        learning_rate_base=config.LEARNING_RATE_BASE,
        total_steps=config.TRAIN_NUM_STEP,
        warmup_learning_rate=config.WARMUP_LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        hold_base_rate_steps=config.HOLD_BASE_RATE_STEPS)
    summary_vars.append(learning_rate)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate,
        momentum=0.90)

    return optimizer, summary_vars
def learn_rate_fn(global_step,
        learning_rate_base,
        total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=0,
        hold_base_rate_steps=0):
    """
        global_step: int64 (scalar) tensor representing global step.
        learning_rate_base: base learning rate.
        total_steps: total number of training steps.
        warmup_learning_rate: initial learning rate for warm up.
        warmup_steps: number of warmup steps.
        hold_base_rate_steps: Optional number of steps to hold base learning rate
            before decaying.
    """
    def eager_decay_rate():
        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
                np.pi *
                (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
                ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
            learning_rate = tf.where(
                    global_step > warmup_steps + hold_base_rate_steps,
                    learning_rate, learning_rate_base)
        if warmup_steps > 0:
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * tf.cast(global_step,
                                                                        tf.float32) + warmup_learning_rate
            learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                                             learning_rate)
        return tf.where(global_step > total_steps, 0.0, learning_rate,
                                        name='learning_rate')

    return eager_decay_rate