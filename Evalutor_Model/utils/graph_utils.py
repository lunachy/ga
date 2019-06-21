import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
from tensorflow.contrib.slim import nets


def BatchNormal(x: tf.Tensor,
                name: str,
                epsilon: float=1e-5,
                trainable: bool=True,
                dtype = tf.float32):
    """
    调用官方api实现 batchnorm
    :param x:
    :param name:
    :param epsilon:
    :param trainable:  是否参与训练
    :return:
    """
    shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        gamma = tf.get_variable("gamma",
                                [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02),
                                trainable=trainable,
                                dtype=dtype)
        beta = tf.get_variable("beta",
                               [shape[-1]],
                               initializer=tf.constant_initializer(0.),
                               trainable=trainable,
                               dtype=dtype)
    mean, variance = tf.nn.moments(x, [0, 1, 2])
    return tf.nn.batch_norm_with_global_normalization(x,
                                                      mean,
                                                      variance,
                                                      beta,
                                                      gamma,
                                                      epsilon,
                                                      scale_after_normalization=True)


def Normalize(x: tf.Tensor,
              name: str,
              epsilon: float=1e-8,
              trainable: bool=True,
              dtype = tf.float32):
    """
    attention is all you need 模型当中的normal
    :param x:
    :param name:
    :param epsilon:
    :param trainable:
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        inputs_shape = x.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
        beta = tf.get_variable("beta",
                               params_shape,
                               initializer=tf.zeros_initializer(),
                               trainable=trainable,
                               dtype=dtype)
        gamma = tf.get_variable("gamma",
                                params_shape,
                                initializer=tf.ones_initializer(),
                                trainable=trainable,
                                dtype=dtype)
        normalized = (x - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def Basic2dConv(x: tf.Tensor,
                d_out: int,
                name: str,
                ksize: tuple = (3, 3),
                stride: tuple = (1, 1),
                active=None,
                trainable: bool=True,
                use_bias: bool=True,
                dtype = tf.float32,
                padding: str='SAME'):
    """
    卷积层
    :param x tensor
    :param d_out int 卷积核数目
    :param ksize list 卷积核尺寸
    :param active 激活函数
    :param trainable
    :param padding SAME或者VALID
    """
    d_in = x.get_shape()[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        kernel = tf.get_variable(name="w",
                                 shape=[ksize[0], ksize[1], d_in, d_out],
                                 dtype=dtype,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 trainable=trainable)
        bias = tf.get_variable(initializer=tf.constant(0.0, dtype=dtype, shape=[d_out]),
                               trainable=trainable,
                               name='b')
        conv = tf.nn.conv2d(x, kernel, [1, stride[0], stride[1], 1], padding=padding)
        l2_loss = tf.nn.l2_loss(kernel) + tf.nn.l2_loss(bias)
        if use_bias:
            conv_plus_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_plus_bias = conv
        if active is not None:
            return active(conv_plus_bias, name="active"), l2_loss
        else:
            return conv_plus_bias, l2_loss


def MultiFilter2dConv(x: tf.Tensor,
                      d_outs: list,
                      name: str,
                      ksizes: list,
                      strides: list,
                      active=None,
                      trainable: bool=True,
                      dtype=tf.float32):
    """
    多个 尺寸的卷积核卷积
    :param x:
    :param d_outs:
    :param name:
    :param ksizes:
    :param strides:
    :param active:
    :param trainable
    :return:
    """
    assert len(d_outs) == len(ksizes) == len(strides)
    conv_results = []
    l2_loss = tf.constant(0.0)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        for i in range(len(d_outs)):
            ksize = ksizes[i]
            stride = strides[i]
            d_out = d_outs[i]
            conv_result, conv_result_l2_loss = Basic2dConv(x=x,
                                                           d_out=d_out,
                                                           name="{0}_{1}".format(name, i),
                                                           ksize=ksize,
                                                           stride=stride,
                                                           active=active,
                                                           trainable=trainable,
                                                           dtype=dtype)
            conv_results.append(conv_result)
            l2_loss += conv_result_l2_loss
    return conv_results, l2_loss


def Basic2dDilatedConv(x: tf.Tensor,
                       d_out: int,
                       name: str,
                       ksize: list = [3, 3],
                       rate: int=1,
                       active=None,
                       trainable: bool=True,
                       use_bias: bool=True,
                       dtype = tf.float32):
    """
    空洞卷积
    :param x:
    :param d_out:
    :param name:
    :param ksize:
    :param rate:
    :param active:
    :param trainable:
    :param use_bias
    :return:
    """
    d_in = x.get_shape()[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        kernel = tf.get_variable(name="w",
                                 shape=[ksize[0], ksize[1], d_in, d_out],
                                 dtype=dtype,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 trainable=trainable)
        bias = tf.get_variable(tf.constant(0.0, dtype=dtype, shape=[d_out]), trainable=True, name='b')
        l2_loss = tf.nn.l2_loss(kernel) + tf.nn.l2_loss(bias)
        conv = tf.nn.atrous_conv2d(x, filters=kernel, rate=rate, padding="SAME")
        if use_bias:
            conv_plus_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_plus_bias = conv
        if active is not None:
            return active(conv_plus_bias, name="active"), l2_loss
        else:
            return conv_plus_bias, l2_loss


def MultiRate2dDilatedConv(x: tf.Tensor,
                           d_outs: int,
                           name: str,
                           ksizes: list,
                           rates: list,
                           active=None,
                           dtype=tf.float32):
    """
    多尺度的空洞卷积
    :return:
    """
    assert len(d_outs) == len(ksizes) == len(rates)
    dilated_conv_results = []
    l2_loss = tf.constant(0.0, dtype=dtype)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        for i in range(len(d_outs)):
            d_out = d_outs[i]
            ksize = ksizes[i]
            rate = rates[i]
            atrous_conv_result, atrous_conv_l2_loss = Basic2dDilatedConv(x=x,
                                                                         d_out=d_out,
                                                                         name="{0}_{1}".format(name, i),
                                                                         ksize=ksize,
                                                                         rate=rate,
                                                                         active=active,
                                                                         dtype=dtype)
            dilated_conv_results.append(atrous_conv_result)
            l2_loss += atrous_conv_l2_loss
    return dilated_conv_results, l2_loss


def MaxPool(x: tf.Tensor,
            name: str,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1]):
    """
    池化层
    :param x tensor
    :param name str
    :param ksize  核尺寸
    :param strides  步长
    """
    activation = tf.nn.max_pool(x, ksize, strides, padding='SAME', name=name)
    return activation


def FullConnect(x: tf.Tensor,
                out: int,
                name: str,
                active=None,
                keep_prob: float=None,
                trainable: bool=True,
                dtype=tf.float32):
    """
    全连接层
    :param x tensor
    :param out 输出维度
    :param name
    :param active 激活函数
    :param keep_prob
    :param name
    """
    d_in = x.get_shape()[-1].value
    initializer = tf.glorot_uniform_initializer()  #
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable(shape=(d_in, out),
                            initializer=initializer,
                            name="w",
                            dtype=dtype,
                            trainable=trainable)
        b = tf.get_variable(initializer=tf.constant(0.0, dtype=dtype, shape=[out]), trainable=trainable, name='b')
        l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        x_w_b = tf.nn.xw_plus_b(x, w, b)
        if keep_prob is not None:
            x_w_b = tf.nn.dropout(x=x_w_b, keep_prob=keep_prob)
        if active is not None:
            return active(x_w_b, name="active"), l2_loss
        else:
            return x_w_b, l2_loss


def SpatialAttention(x: tf.Tensor, name: str, k: int=1024):
    """
    空间注意力转移  https://www.e-learn.cn/content/qita/678740  与原始论文有区别
    :param x:  [batch_size, height, width, channel]
    :param name:
    :param k:
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        _, H, W, C = x.get_shape()
        w = tf.get_variable(name="attention_w", shape=[C, 1], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(initializer=tf.constant(0.0, dtype=tf.float32, shape=[1]), trainable=True, name='attention_b')
        spatial_attention = tf.matmul(tf.reshape(x, [-1, C]), w) + b  # 每一个空间点位置的attention  多个通道的同一个位置 生成一个概率
        spatial_attention = tf.nn.sigmoid(tf.reshape(spatial_attention, [-1, W * H]))  # batch_size, w*h
        spatial_attention = tf.tile(input=spatial_attention, multiples=[1, C])  # batch_size, w*h*c
        attention = tf.reshape(spatial_attention, [-1, H, W, C])  # batch_size, height, w, channel
        attention_x = tf.multiply(x=x, y=attention)
        return attention_x


def ChannelWiseAttention(x: tf.Tensor, name: str):
    """
    通道注意力转移
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        _, H, W, C = x.get_shape()
        w = tf.get_variable("attention_w", [C, C], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(initializer=tf.constant(0.0, dtype=tf.float32, shape=[C]), trainable=True, name='attention_b')
        transpose_feature_map = tf.transpose(tf.reduce_mean(x, [1, 2], keepdims=True), perm=[0, 3, 1, 2])  # 每一个通道
        channel_wise_attention = tf.matmul(tf.reshape(transpose_feature_map,  [-1, C]), w) + b  # b, c
        channel_wise_attention = tf.nn.sigmoid(channel_wise_attention)
        channel_wise_attention = tf.tile(input=channel_wise_attention, multiples=[1, H*W])
        attention = tf.reshape(channel_wise_attention, [-1, H, W, C])
        attention_x = tf.multiply(x=x, y=attention)
        return attention_x


def Vgg16Layer(input_tensor: tf.Tensor,
               name: str,
               trainable: bool=True,
               filter_list: list = [64, 128, 256, 512, 512],
               normal_model: str = "batch_norm",
               dtype = tf.float32
               ):
    """
    vgg16卷积特征层
    :param input_tensor
    :param name 名称
    :param trainable
    :param filter_list vgg每一层的卷积核数目
    """
    conv1_1, conv1_1_l2_loss = Basic2dConv(input_tensor, filter_list[0], '{0}_conv1_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv1_2, conv1_2_l2_loss = Basic2dConv(conv1_1, filter_list[0], '{0}_conv1_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    if normal_model == "batch_norm":
        conv1_2 = BatchNormal(x=conv1_2, name="{0}_bn_1".format(name),  trainable=trainable, dtype=dtype)
    else:
        conv1_2 = Normalize(x=conv1_2,  name="{0}_bn_1".format(name), trainable=trainable, dtype=dtype)
    pool1 = MaxPool(conv1_2, '{0}_pool1'.format(name))

    conv2_1, conv2_1_l2_loss = Basic2dConv(pool1, filter_list[1], '{0}_conv2_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv2_2, conv2_2_l2_loss = Basic2dConv(conv2_1, filter_list[1], '{0}_conv2_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    if normal_model == "batch_norm":
        conv2_2 = BatchNormal(x=conv2_2, name="{0}_bn_2".format(name), trainable=trainable, dtype=dtype)
    else:
        conv2_2 = Normalize(x=conv2_2, name="{0}_bn_2".format(name), trainable=trainable, dtype=dtype)
    pool2 = MaxPool(conv2_2, '{0}_pool2'.format(name))

    conv3_1, conv3_1_l2_loss = Basic2dConv(pool2, filter_list[2], '{0}_conv3_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv3_2, conv3_2_l2_loss = Basic2dConv(conv3_1, filter_list[2], '{0}_conv3_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv3_3, conv3_3_l2_loss = Basic2dConv(conv3_2, filter_list[2], '{0}_conv3_3'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    if normal_model == "batch_norm":
        conv3_3 = BatchNormal(x=conv3_3, name="{0}_bn_3".format(name), trainable=trainable, dtype=dtype)
    else:
        conv3_3 = Normalize(x=conv3_3, name="{0}_bn_3".format(name), trainable=trainable, dtype=dtype)
    pool3 = MaxPool(conv3_3, '{0}_pool3'.format(name))
    conv4_1, conv4_1_l2_loss = Basic2dConv(pool3, filter_list[3], '{0}_conv4_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv4_2, conv4_2_l2_loss = Basic2dConv(conv4_1, filter_list[3], '{0}_conv4_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv4_3, conv4_3_loss = Basic2dConv(conv4_2, filter_list[3], '{0}_conv4_3'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    if normal_model == "batch_norm":
        conv4_3 = BatchNormal(x=conv4_3, name="{0}_bn_4".format(name), trainable=trainable, dtype=dtype)
    else:
        conv4_3 = Normalize(x=conv4_3, name="{0}_bn_4".format(name), trainable=trainable, dtype=dtype)
    pool4 = MaxPool(conv4_3, '{0}_pool4'.format(name))

    conv5_1, conv5_1_l2_loss = Basic2dConv(pool4, filter_list[4], '{0}_conv5_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv5_2, conv5_2_l2_loss = Basic2dConv(conv5_1, filter_list[4], '{0}_conv5_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv5_3, conv5_3_l2_loss = Basic2dConv(conv5_2, filter_list[4], '{0}_conv5_3'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    if normal_model == "batch_norm":
        conv5_3 = BatchNormal(x=conv5_3, name="{0}_bn_5".format(name), trainable=trainable, dtype=dtype)
    else:
        conv5_3 = Normalize(x=conv5_3, name="{0}_bn_5".format(name), trainable=trainable, dtype=dtype)
    pool5 = MaxPool(conv5_3, '{0}_pool5'.format(name))
    l2_loss = conv1_1_l2_loss + conv1_2_l2_loss + conv2_1_l2_loss + conv2_2_l2_loss + conv3_1_l2_loss + conv3_2_l2_loss\
        + conv3_3_l2_loss + conv4_1_l2_loss + conv4_2_l2_loss + conv4_3_loss + conv5_1_l2_loss + conv5_2_l2_loss \
        + conv5_3_l2_loss
    return pool5, l2_loss


def InceptionLayer(x: tf.Tensor, num_3x3red: int, num_3x3: int, num_5x5red: int, num_5x5: int, ksize: list=[3, 3], stride: list=[1, 1], name: str="inception"):
    """
    inception layer
    3*3 5*5 多个卷积核支持
    :param num_3x3red int 第一个3*3 卷积核数目
    :param num_3x3 int 第二个3*3 卷积核数目
    :param num_5x5red int 第一个5*5 卷积核数目
    :param num_5x5 int 第二个5*5 卷积核数目
    :return:
    """
    # 3*3 reduce + 3*3
    c3xc3r, c3xc3r_l2_loss = Basic2dConv(x=x, d_out=num_3x3red, ksize=ksize, stride=stride, name="{0}_c3xc3_reduce".format(name), active=tf.nn.relu)
    c3xc3, c3xc3_l2_loss = Basic2dConv(x=c3xc3r, d_out=num_3x3, ksize=ksize, stride=stride, name="{0}_c3xc3".format(name), active=tf.nn.relu)
    # 5*5 reduce + 5*5
    c5xc5r, c5xc5r_l2_loss = Basic2dConv(x=x, d_out=num_5x5red, ksize=ksize, stride=stride, name="{0}_c5xc5_reduce".format(name), active=tf.nn.relu)
    c5xc5, c5xc5_l2_loss = Basic2dConv(x=c5xc5r, d_out=num_5x5, ksize=ksize, stride=stride, name="{0}_c5xc5".format(name), active=tf.nn.relu)

    c3xc3_pooling = MaxPool(x=c3xc3, name="{0}_c3xc3_pool".format(name))
    c5xc5_pooling = MaxPool(x=c5xc5, name="{0}_c5xc5_pool".format(name))
    pooling = MaxPool(x=x, name="{0}_proj_pool".format(name))
    filter_concat = tf.concat(values=[c3xc3_pooling, c5xc5_pooling, pooling], axis=3)
    l2_loss = c3xc3r_l2_loss + c3xc3_l2_loss + c5xc5r_l2_loss + c5xc5_l2_loss
    return filter_concat, l2_loss


def GoogleNetLayer(input_tensor: tf.Tensor, name: str):
    """
    并不是标准的googlenet
    卷积+4个inception
    :return:
    """
    conv1_1, conv1_1_l2_loss = Basic2dConv(input_tensor, 64, '{0}_conv1_1'.format(name), ksize=[7, 7], active=tf.nn.relu)
    pool_1 = MaxPool(x=conv1_1, name="{0}_pool_1".format(name))

    inception1, inception1_l2_loss = InceptionLayer(x=pool_1, num_3x3red=32, num_3x3=64, num_5x5red=32, num_5x5=64, name="inception1")
    inception2, inception2_l2_loss = InceptionLayer(x=inception1, num_3x3red=64, num_3x3=128, num_5x5red=64, num_5x5=128, name="inception2")
    inception3, inception3_l2_loss = InceptionLayer(x=inception2, num_3x3red=64, num_3x3=128, num_5x5red=64, num_5x5=128, name="inception3")
    inception4, inception4_l2_loss = InceptionLayer(x=inception3, num_3x3red=64, num_3x3=128, num_5x5red=64, num_5x5=128, name="inception4")
    l2_loss = conv1_1_l2_loss + inception1_l2_loss + inception2_l2_loss + inception3_l2_loss + inception4_l2_loss
    return inception4, l2_loss


def GetTokenEmbedding(vocab_size: int,
                      num_units: int,
                      zero_pad: bool=True,
                      scope: str="embedding",
                      trainable: bool=True,
                      dtype=tf.float32):
    """
    获取embedding的weight
    :param vocab_size:
    :param num_units:
    :param zero_pad:  是否对0进行 masking操作
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):  # todo 禁止使用cpu 占用
        w = tf.get_variable("w", dtype=dtype, shape=(vocab_size, num_units), initializer=tf.glorot_uniform_initializer(), trainable=trainable)
        if zero_pad:
            w = tf.concat((tf.zeros(shape=[1, num_units], dtype=dtype), w[1:, :]), axis=0, name="concat")
    return w


def DeformableSquareCnn(x: tf.Tensor, name: str, trainable: bool):
    """
    可变卷积 仅仅支持方形的数据  这种实现方式似乎与原始论文有区别
    :param x [b, h, w, c]
    :param name
    :param trainable
    :return:
    """
    x_shape = x.get_shape().as_list()
    assert x_shape[1] == x_shape[2]
    # [b, h, w, 2c]  每一个像素点对应两个数值
    offsets, offsets_l2_loss = Basic2dConv(x=x, d_out=x_shape[-1]*2, name="{0}_offset_cnn".format(name), trainable=trainable, use_bias=False)
    # [b, h, w, 2c] -> (b*c, h, w, 2)
    offsets = _to_bc_h_w_2(x=offsets, x_shape=x_shape)
    # [b, h, w, c] -> (b*c, h, w)
    x = _to_bc_h_w(x=x, x_shape=x_shape)
    # offsets的经过双线性映射 映射到 "x"空间 生成新的特征值
    x_offset = tf_batch_map_offsets(x, offsets)
    x_offset = _to_b_h_w_c(x=x_offset, x_shape=x_shape)
    return x_offset, offsets_l2_loss


def _to_bc_h_w_2(x: tf.Tensor, x_shape: list):
    """
    (b, h, w, 2c) -> (b*c, h, w, 2)
    :param x:
    :param x_shape:
    :return:
    """
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
    return x


def _to_bc_h_w(x: tf.Tensor, x_shape: list):
    """
    (b, h, w, c) -> (b*c, h, w)
    :param x:
    :param x_shape:
    :return:
    """
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
    return x


def _to_b_h_w_c(x: tf.Tensor, x_shape: list):
    """

    :param x:
    :param x_shape:
    :return:
    """
    x = tf.reshape(x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
    x = tf.transpose(x, [0, 2, 3, 1])
    return x


def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    # https://github.com/tensorflow/tensorflow/issues/8521
    assert len(a.get_shape()) == 1

    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a


def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """
    仅在 h = w 的情况下生效
    :param inputs:
    :param coords:
    :return:
    """
    # 将coords里面的所有数值全部都限制在(0, h^2-1)之间
    coords = coords.clip(0, inputs.shape[1] - 1)
    # 使用双线性差值算法 找到合适的位置
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals


def sp_batch_map_offsets(input, offsets):

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)
    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def tf_batch_map_offsets(input: tf.Tensor, offsets: tf.Tensor, order=1):
    """
    双线性差值定理
    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s)
    offsets: tf.Tensor. shape = (b, s, s, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(
        tf.range(input_size), tf.range(input_size), indexing='ij'
    )
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = offsets + grid
    # 双线性差值 计算新的特征图
    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals


def tf_mse_loss(logits: tf.Tensor, targets: tf.Tensor):
    """
    均方误差函数
    :param logits:  [batch_size, ]
    :param targets:  [batch_size, ]
    :return:
    """
    mse_loss = tf.reduce_mean(tf.pow(tf.subtract(x=logits, y=targets), y=2))
    return mse_loss


def tf_mae_loss(logits: tf.Tensor, targets: tf.Tensor):
    """
    绝对误差函数
    :param logits:
    :param targets:
    :return:
    """
    mae_loss = tf.reduce_mean(tf.abs(tf.subtract(x=logits, y=targets)))
    return mae_loss


def tf_huber_loss(logits: tf.Tensor, targets: tf.Tensor, delta: float=1.0):
    """
    huber损失函数  https://blog.csdn.net/hyk_1996/article/details/79570915 相关实现
    :param logits:
    :param targets:
    :param delta:
    :return:
    """
    # [b, ] - [b, ]
    residual = tf.abs(tf.subtract(x=logits, y=targets))
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    where = tf.reduce_mean(tf.where(condition=condition, x=small_res, y=large_res))
    return where


def tf_alpha_beta(values: list, name: str, trainable: bool):
    """
    a_1 * values[0] + a_2 * values[1]  + a_3 * values[2] + b
    各个维度尺寸必须保证一致
    :param values:
    :return:
    """
    assert len(values) > 0
    addtion = tf.zeros(shape=(1), name="{0}_add".format(name))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(len(values)):
            value = values[i]
            alpha = tf.get_variable(name="{0}_alpha_{1}".format(name, i), shape=(1), initializer=tf.glorot_uniform_initializer(), dtype=tf.float32, trainable=trainable)
            addtion = tf.add(addtion, alpha * value)
        beta = tf.get_variable(name="{0}_beta".format(name), shape=(1), initializer=tf.glorot_uniform_initializer(), dtype=tf.float32, trainable=trainable)
        addtion = tf.add(addtion, beta)
    return addtion


def tf_alpha_beta_weight(values: list, name: str, trainable: bool):
    """
    矩阵形式的
    weight_1 * values[0] + weight_2 * values[1]  + weight_3 * values[2] + weight4
    :param values:
    :param name:
    :param trainable:
    :return:
    """
    assert len(values) > 0
    shape = values[0].get_shape()[-1]
    addtion = tf.zeros(shape=shape, name="{0}_add".format(name))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(len(values)):
            value = values[i]
            alpha = tf.get_variable(name="{0}_alpha_{1}".format(name, i), shape=(shape), initializer=tf.glorot_uniform_initializer(), dtype=tf.float32, trainable=trainable)
            addtion = tf.add(addtion, tf.multiply(x=value, y=alpha))
        beta = tf.get_variable(name="{0}_beta".format(name), shape=(shape), initializer=tf.glorot_uniform_initializer(), dtype=tf.float32, trainable=trainable)
        addtion = tf.add(addtion, beta)
    return addtion


def residual_unit(x: tf.Tensor, num_filter, stride, dim_match, name: str, bottle_neck=True, train_able: bool=True):
    """
    残差结构  (这里的卷积)
    :param x:   输入的tensor [b, h, w, c]
    :param num_filter:   卷积核数目
    :param stride:   卷积的步长
    :param dim_match:
    :param name:   名称
    :param bottle_neck:
    :param bn_mom:  batch_norm 参数
    :param train_able:  参数是否训练
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if bottle_neck:
            bn1 = BatchNormal(x=x, name="{0}_bn1".format(name), trainable=train_able)
            act1 = tf.nn.relu(bn1, name="{0}_relu1".format(name))
            conv1, conv1_l2_loss = Basic2dConv(x=act1, d_out=int(num_filter/4), name="{0}_cnn1".format(name), ksize=(1, 1),
                                               stride=(1, 1), active=None, trainable=train_able, use_bias=False, padding="VALID")

            bn2 = BatchNormal(x=conv1, name="{0}_bn2".format(name), trainable=train_able)
            act2 = tf.nn.relu(bn2, name="{0}_relu2".format(name))
            conv2, conv2_l2_loss = Basic2dConv(x=act2, d_out=int(num_filter/4), name="{0}_cnn2".format(name), ksize=(3, 3),
                                               stride=stride, active=None, trainable=train_able, use_bias=False, padding="SAME")

            bn3 = BatchNormal(x=conv2, name="{0}_bn3".format(name), trainable=train_able)
            act3 = tf.nn.relu(bn3, name="{0}_relu3".format(name))
            conv3, conv3_l2_loss = Basic2dConv(x=act3, d_out=num_filter, name="{0}_cnn3".format(name), ksize=(1, 1),
                                               stride=(1, 1), active=None, trainable=train_able, use_bias=False, padding="VALID")
            if dim_match:
                short_cut = x
            else:
                short_cut, short_cut_l2_loss = Basic2dConv(x=act1, d_out=num_filter, name="{0}_cnn4".format(name), ksize=(1, 1),
                                                        stride=stride, active=None, trainable=train_able, use_bias=False, padding="VALID")
            return conv3 + short_cut


# x = tf.placeholder(name="a", shape=(None, 128, 128, 3), dtype=tf.float32)
# t = residual_unit(x=x, name="t", num_filter=128, stride=(3, 3), dim_match=False, bottle_neck=True)
# print(t)
