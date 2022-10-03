# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT;
import tensorflow as tf
#from tools import ops



class BalancedPositiveNegativeSampler(object):
    def __init__(self, positive_fraction=0.5):
        """
        positive_fraction: desired fraction of positive examples (scalar in [0,1])
            in the batch.
        """
        if positive_fraction < 0 or positive_fraction > 1:
            raise ValueError('positive_fraction should be in range [0,1]. '
                                             'Received: %s.' % positive_fraction)
        self._positive_fraction = positive_fraction
     

    def _get_num_pos_neg_samples(self, sorted_indices_tensor, sample_size):
        """
        sorted_indices_tensor: A sorted int32 tensor of shape [N] which contains
            the signed indices of the examples where the sign is based on the label
            value. The examples that cannot be sampled are set to 0. It samples
            atmost sample_size*positive_fraction positive examples and remaining
            from negative examples.
        sample_size: Size of subsamples.
        """
        input_length = tf.shape(sorted_indices_tensor)[0]
        valid_positive_index = tf.greater(sorted_indices_tensor,
                                          tf.zeros(input_length, tf.int32))
        num_sampled_pos = tf.reduce_sum(tf.cast(valid_positive_index, tf.int32))
        max_num_positive_samples = tf.constant(
                int(sample_size * self._positive_fraction), tf.int32)
        num_positive_samples = tf.minimum(max_num_positive_samples, num_sampled_pos)
        num_negative_samples = tf.constant(sample_size,
                                           tf.int32) - num_positive_samples

        return num_positive_samples, num_negative_samples

 
    def _static_subsample(self, indicator, batch_size, labels):
        """
        indicator: boolean tensor of shape [N] whose True entries can be sampled.
            N should be a complie time constant.
        batch_size: desired batch size. This scalar cannot be None.
        labels: boolean tensor of shape [N] denoting positive(=True) and negative
            (=False) examples.

        """
        # Check if indicator and labels have a static size.
        if not indicator.shape.is_fully_defined():
            raise ValueError('indicator must be static in shape when is_static is'
                                             'True')
        if not labels.shape.is_fully_defined():
            raise ValueError('labels must be static in shape when is_static is'
                                             'True')
        if not isinstance(batch_size, int):
            raise ValueError('batch_size has to be an integer when is_static is'
                                             'True.')

        input_length = tf.shape(indicator)[0]
        num_true_sampled = tf.reduce_sum(tf.cast(indicator, tf.float32))
        additional_false_sample = tf.less_equal(
                tf.cumsum(tf.cast(tf.logical_not(indicator), tf.float32)),
                batch_size - num_true_sampled)
        indicator = tf.logical_or(indicator, additional_false_sample)


        permutation = tf.random.shuffle(tf.range(input_length))
        indicator = tf.gather(indicator, permutation, axis=0)
        labels = tf.gather(labels, permutation, axis=0)

 
        indicator_idx = tf.where(
                indicator, tf.range(1, input_length + 1),
                tf.zeros(input_length, tf.int32))

        signed_label = tf.where(
                labels, tf.ones(input_length, tf.int32),
                tf.scalar_mul(-1, tf.ones(input_length, tf.int32)))

        signed_indicator_idx = tf.multiply(indicator_idx, signed_label)
        sorted_signed_indicator_idx = tf.nn.top_k(
                signed_indicator_idx, input_length, sorted=True).values

        [num_positive_samples,
         num_negative_samples] = self._get_num_pos_neg_samples(
                 sorted_signed_indicator_idx, batch_size)

        sampled_idx=tf.concat([sorted_signed_indicator_idx[:num_positive_samples],sorted_signed_indicator_idx[-num_negative_samples:]],axis=0)

        sampled_idx = tf.abs(sampled_idx) - tf.ones(batch_size, tf.int32) #shape=[256]

        sampled_idx_indicator = tf.cast(tf.reduce_sum(
                tf.one_hot(sampled_idx, depth=input_length),
                axis=0), tf.bool)

        idx_indicator = tf.scatter_nd(
                tf.expand_dims(permutation, -1), sampled_idx_indicator,
                shape=(input_length,))
        return idx_indicator

    def subsample(self, indicator, batch_size, labels):
        """
            indicator: boolean tensor of shape [N] whose True entries can be sampled.
            batch_size: desired batch size. If None, keeps all positive samples and
                randomly selects negative samples so that the positive sample fraction
                matches self._positive_fraction. It cannot be None is is_static is True.
            labels: boolean tensor of shape [N] denoting positive(=True) and negative
                    (=False) examples.
            scope: name scope.
        """
        if len(indicator.get_shape().as_list()) != 1:
            raise ValueError('indicator must be 1 dimensional, got a tensor of '
                                             'shape %s' % indicator.get_shape())
        if len(labels.get_shape().as_list()) != 1:
            raise ValueError('labels must be 1 dimensional, got a tensor of '
                                             'shape %s' % labels.get_shape())
        if labels.dtype != tf.bool:
            raise ValueError('labels should be of type bool. Received: %s' %
                                             labels.dtype)
        if indicator.dtype != tf.bool:
            raise ValueError('indicator should be of type bool. Received: %s' %
                                             indicator.dtype)
        with tf.name_scope('BalancedPositiveNegativeSampler'):
                return self._static_subsample(indicator, batch_size, labels)



