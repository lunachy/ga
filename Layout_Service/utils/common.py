from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import keras
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.nets as nets


def _conv2d_fixed_padding(inputs: tf.Tensor, filters: int, kernel_size: int, strides: int=1, layer_name: str=None, reuse: bool=False):
    """

    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param layer_name:
    :return:
    """
    assert layer_name is not None
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs=inputs, num_outputs=filters, kernel_size=kernel_size,
                         stride=strides, scope=layer_name, padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def se_block(residual, name, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = residual.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(residual, axis=[1, 2], keepdims=True)
        assert squeeze.get_shape()[1:] == (1, 1, channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel // ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel)
        # top = tf.multiply(bottom, se, name='scale')
        scale = residual * excitation
    return scale


def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        print("CBAM Hello")
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat


def conv2d(inputs, filter, kernel, strides, padding="same"):
    inputs = keras.layers.Conv2D(filter, kernel, strides=strides, padding=padding, activation=None,
                                 kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(0.01),
                                 activity_regularizer=keras.regularizers.l1(0.01))(inputs)
    inputs = keras.layers.LeakyReLU(alpha=0.3)(inputs)
    outputs = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros',
                                              moving_variance_initializer='ones', beta_regularizer=None,
                                              gamma_regularizer=None,
                                              beta_constraint=None, gamma_constraint=None)(inputs)
    return outputs
