# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    u_net_layers
    Description :    
    Author      :    zhaowen
    date        :    2019/6/4
____________________________________________________________________
    Change Activity:
                        2019/6/4:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'

# copy from https://github.com/jakeret/tf_unet/blob/master/tf_unet/layers.py

# from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf


def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)


def deconv2d(x, W, stride):
    with tf.name_scope("deconv2d"):
        # x_shape = tf.shape(x)#
        x_shape = x.shape.as_list()  # tf.shape(x)
        output_shape = [x_shape[0], x_shape[1] * 2, x_shape[2] * 2,
                        x_shape[3] // 2]  # tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        #output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        # if not output_shape[0]:
        #     output_shape[0] = -1
        # print("deconv2d:output_shape{0},x_shape{1}".format(output_shape, x_shape))
        return tf.nn.conv2d_transpose(x, W, tf.convert_to_tensor(output_shape), strides=[1, stride, stride, 1],
                                      padding='VALID',
                                      name="conv2d_transpose")


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")
