import tensorflow as tf


def BatchNormal(x: tf.Tensor,
                name: str,
                training: tf.Tensor,
                trainable: bool=True,
                decay: float=0.99,
                dtype=tf.float32):
    """
    调用官方api实现batchnorm  batch_norm的实现仅仅再cpu上
    :param x:  [b, h, w, c]
    :param name:
    :param epsilon:
    :param trainable:  是否参与训练  训练过程使用 batch生成的 mean variance 预测过程使用训练数据当中的平滑结果
    :param decay 平滑因子
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        y = tf.layers.batch_normalization(x, trainable=trainable, training=training, momentum=0.9,
                                          name="{0}_bn".format(name), epsilon=1e-5)  # 直接使用封装好的)
    return y


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

        conv = tf.nn.conv2d(x, kernel, [1, stride[0], stride[1], 1], padding=padding)
        if use_bias:
            bias = tf.get_variable(initializer=tf.constant(0.0, dtype=dtype, shape=[d_out]),
                                   trainable=trainable,
                                   name='b')
            conv_plus_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_plus_bias = conv
        if active is not None:
            return active(conv_plus_bias, name="active")
        else:
            return conv_plus_bias


def MaxPool(x: tf.Tensor,
            name: str,
            ksize: tuple=(1, 2, 2, 1),
            strides: tuple=(1, 2, 2, 1)):
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
        x_w_b = tf.nn.xw_plus_b(x, w, b)
        if keep_prob is not None:
            x_w_b = tf.nn.dropout(x=x_w_b, keep_prob=keep_prob)
        if active is not None:
            return active(x_w_b, name="active")
        else:
            return x_w_b


def Vgg16Layer(input_tensor: tf.Tensor,
               name: str,
               training: tf.Tensor,
               trainable: bool=True,
               filter_list: list = [64, 128, 256, 512, 512],
               dtype = tf.float32
               ):
    """
    vgg16卷积特征层
    :param input_tensor
    :param name 名称
    :param trainable
    :param filter_list vgg每一层的卷积核数目
    """
    conv1_1 = Basic2dConv(input_tensor, filter_list[0], '{0}_conv1_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv1_2 = Basic2dConv(conv1_1, filter_list[0], '{0}_conv1_2'.format(name), active=None, trainable=trainable, dtype=dtype)
    conv1_2 = BatchNormal(x=conv1_2, name="{0}_bn_1".format(name),  trainable=trainable, dtype=dtype, training=training)
    conv1_2 = tf.nn.relu(conv1_2)
    pool1 = MaxPool(conv1_2, '{0}_pool1'.format(name))
    conv2_1 = Basic2dConv(pool1, filter_list[1], '{0}_conv2_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv2_2 = Basic2dConv(conv2_1, filter_list[1], '{0}_conv2_2'.format(name), active=None, trainable=trainable, dtype=dtype)
    conv2_2 = BatchNormal(x=conv2_2, name="{0}_bn_2".format(name), trainable=trainable, dtype=dtype, training=training)
    conv2_2 = tf.nn.relu(conv2_2)
    pool2 = MaxPool(conv2_2, '{0}_pool2'.format(name))

    conv3_1 = Basic2dConv(pool2, filter_list[2], '{0}_conv3_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv3_2 = Basic2dConv(conv3_1, filter_list[2], '{0}_conv3_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv3_3 = Basic2dConv(conv3_2, filter_list[2], '{0}_conv3_3'.format(name), active=None, trainable=trainable, dtype=dtype)
    conv3_3 = BatchNormal(x=conv3_3, name="{0}_bn_3".format(name), trainable=trainable, dtype=dtype, training=training)
    conv3_3 = tf.nn.relu(conv3_3)
    pool3 = MaxPool(conv3_3, '{0}_pool3'.format(name))
    
    conv4_1 = Basic2dConv(pool3, filter_list[3], '{0}_conv4_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv4_2 = Basic2dConv(conv4_1, filter_list[3], '{0}_conv4_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv4_3 = Basic2dConv(conv4_2, filter_list[3], '{0}_conv4_3'.format(name), active=None, trainable=trainable, dtype=dtype)
    conv4_3 = BatchNormal(x=conv4_3, name="{0}_bn_4".format(name), trainable=trainable, dtype=dtype, training=training)
    conv4_3 = tf.nn.relu(conv4_3)
    pool4 = MaxPool(conv4_3, '{0}_pool4'.format(name))

    conv5_1 = Basic2dConv(pool4, filter_list[4], '{0}_conv5_1'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv5_2 = Basic2dConv(conv5_1, filter_list[4], '{0}_conv5_2'.format(name), active=tf.nn.relu, trainable=trainable, dtype=dtype)
    conv5_3 = Basic2dConv(conv5_2, filter_list[4], '{0}_conv5_3'.format(name), active=None, trainable=trainable, dtype=dtype)
    conv5_3 = BatchNormal(x=conv5_3, name="{0}_bn_5".format(name), trainable=trainable, dtype=dtype, training=training)
    conv5_3 = tf.nn.relu(conv5_3)
    pool5 = MaxPool(conv5_3, '{0}_pool5'.format(name))
    return pool5


def GetTokenEmbedding(vocab_size: int,
                      num_units: int,
                      zero_pad: bool=True,
                      scope: str="embedding",
                      trainable: bool=True,
                      dtype=tf.float64):
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


def EmbeddingLayers(input_tensor_list: list, vocab_size_list: list, num_unit_list: list, scopes: list, trainable: bool, dtype=tf.float64):
    """
    输入tensor 并且返回每一个tensor的embedding结果
    :param input_tensor_list:  输入的tensor列表
    :param vocab_size_list:
    :param num_unit_list:
    :param scopes:
    :param trainable:
    :param dtype:
    :return:
    """
    valid_index = []
    for i in range(len(input_tensor_list)):  # 记录有效的index
        if input_tensor_list[i] is not None:
            valid_index.append(i)

    if len(valid_index) == 1:
        look_up_table_0 = GetTokenEmbedding(vocab_size=vocab_size_list[valid_index[0]], num_units=num_unit_list[0],
                                            zero_pad=False,
                                            trainable=trainable,
                                            scope=scopes[valid_index[0]],
                                            dtype=dtype)
        embedding_0 = tf.nn.embedding_lookup(params=look_up_table_0, ids=input_tensor_list[valid_index[0]])
        return embedding_0
    else:
        embedding_list = []
        for id in range(len(valid_index)):
            look_up_table = GetTokenEmbedding(vocab_size=vocab_size_list[valid_index[id]], num_units=num_unit_list[valid_index[id]],
                                              zero_pad=False,
                                              trainable=trainable,
                                              scope=scopes[valid_index[id]],
                                              dtype=dtype)
            embedding = tf.nn.embedding_lookup(params=look_up_table, ids=input_tensor_list[valid_index[id]])
            embedding_list.append(embedding)
        return embedding_list


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


def fixed_padding(input: tf.Tensor, kernel_size: int, mode: str='CONSTANT'):
    """
    数据填充
    :param input:  输入的tensor [b, h, w, c]
    :param kernel_size:  卷积核
    :param mode
    :return:
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(input, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def conv_fixed_padding(input: tf.Tensor, filters: int, kernel_size: int, stride_size: int, name: str, trainable: bool,
                       dtype=tf.float32, padding=None):
    """
    先padding在进行卷积
    :param input: 输入的tensor [b, h, w, c]
    :param filters:  卷积核数目
    :param kernel_size:  卷积核尺寸
    :param stride_size:   卷积步长
    :param name:  名称
    :param trainable:  是否训练
    :param dtype:  数据类型
    :return:
    """
    if stride_size > 1:
        input = fixed_padding(input=input, kernel_size=kernel_size)
    if padding is None:
        padding = "SAME" if stride_size == 1 else "VALID"
    return Basic2dConv(x=input,
                       d_out=filters,
                       name=name,
                       ksize=(kernel_size, kernel_size),
                       stride=(stride_size, stride_size),
                       padding=padding,
                       use_bias=False,
                       trainable=trainable,
                       active=None,
                       dtype=dtype)


def deep_conv(input: tf.Tensor, name: str, ksize_list: list, stride_list: list, filter_list: list, trainable: bool, dtype=tf.float32, active=None,
              batch_norm: bool=False, training: tf.Tensor=False, last_active: bool=True):
    """
    深度卷积网络 连续多层卷积
    :param input:  输入的tensor
    :param name:
    :param ksize_list:   卷积核尺寸
    :param stride_list:  卷积步长
    :param last_active: 最后一层是否激活
    :return:
    """
    assert len(ksize_list) == len(stride_list)
    for i, ksize in enumerate(ksize_list):
        input = conv_fixed_padding(input=input,
                                   name="{0}_{1}_cnn".format(name, i),
                                   kernel_size=ksize_list[i],
                                   stride_size=stride_list[i],
                                   filters=filter_list[i],
                                   trainable=trainable,
                                   dtype=dtype)
        if batch_norm:
            input = BatchNormal(x=input, name="{0}_{1}_bn".format(name, i), trainable=trainable, dtype=dtype,
                                  training=training)
        if i == len(ksize_list) - 1:
            if last_active and active is not None:
                input = active(input, name="{0}_{1}_active".format(name, i))
        elif active is not None:
            input = active(input, name="{0}_{1}_active".format(name, i))
    return input


def wide_conv(input: tf.Tensor, name: str, ksize_lists: list, stride_lists: list, filter_lists:list, trainable: bool,
              dtype=tf.float32, active=None, batch_norm: bool=False, training: tf.Tensor=False, padding=None, debug=False):
    """
    宽度卷积网络 最后 concat 一起
    :param input:  输入的tensor
    :param name:
    :param ksize_lists:   卷积核尺寸  例如[[3, 4], [3]]
    :param stride_lists:  卷积步长  要求与ksize_list的维度一样  例如 [[1, 1], [1]]
    :param filter_lists:  卷积核 要求与ksize_list的维度一样 例如 [[32, 32], [32]]
    return:  返回最终 concat的值
    """
    concat_list = []
    for i, (ksize_list, stride_list, filter_list) in enumerate(zip(ksize_lists, stride_lists, filter_lists)):
        input_i = input
        for j, (ksize, stride, filter) in enumerate(zip(ksize_list, stride_list, filter_list)):
            input_i = conv_fixed_padding(input=input_i, name="{0}_{1}_{2}".format(name, i, j), kernel_size=ksize,
                                       stride_size=stride, trainable=trainable, filters=filter, padding=padding)
            if batch_norm:
                input_i = BatchNormal(x=input_i, name="{0}_{1}_{2}_bn".format(name, i, j), trainable=trainable, dtype=dtype,
                                    training=training)
            if active:
                input_i = active(input_i, name="{0}_{1}_{2}_active".format(name, i, j))
        concat_list.append(input_i)
    return tf.concat(values=concat_list, axis=-1)


def resnet_block(input: tf.Tensor, name: str, ksize_list: list, filter_list: list, trainable: bool,
                 dtype=tf.float32, active=None, batch_norm: bool=False, training: tf.Tensor=False, last_active:bool=True):
    """
    残差结构网络
    :param input:
    :param name:
    :param ksize_list:
    :param filter_lists:
    :param trainable:
    :param dtype:
    :param active:
    :return:
    """
    output = deep_conv(input=input, name="{0}_deep_cnn".format(name), ksize_list=ksize_list, stride_list=[1]*len(ksize_list),
                       filter_list=filter_list, trainable=trainable, dtype=dtype, active=active, batch_norm=batch_norm,
                       training=training, last_active=last_active)

    return input + output


def attention_concat(input: tf.Tensor, attention_input: tf.Tensor):
    """
    经过resnet卷积 生成与input一样尺寸的tensor  用于attention
    :param input:
    :param attention_input:
    :return:
    """
    _, height, width, c = input.get_shape()
    # todo


def average_gradients(tower_grads: list):
    """
    计算平均梯度
    :param tower_grads:
    :return:
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:  # 防止图中意外创建 没有用到的 variable
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        if len(grads) == 0:  # 防止图中意外创建 没有用到的 variable
            continue
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_trainable_l2_loss():
    """
    获取所有变量的l2正则化损失
    :return:
    """
    l2_loss = tf.constant(value=0.0, dtype=tf.float32)
    for v in tf.trainable_variables():
        if "embedding/w:0" in v.name:
            continue
        l2_loss += tf.nn.l2_loss(v)
    return l2_loss


def get_variable_l2_loss(variables: list):
    """
    获取输入变量的l2损失
    :param variables:
    :return:
    """
    if not isinstance(variables, list):
        variables = [variables]
    l2_loss = tf.constant(value=0.0, dtype=tf.float32)
    for v in variables:
        l2_loss += tf.nn.l2_loss(v)
    return l2_loss


def get_trainable_l1_loss():
    """
    获取所有变量的l1正则化损失
    :return:
    """
    l1_loss = tf.constant(value=0.0, dtype=tf.float32)
    for v in tf.trainable_variables():
        if "embedding/w:0" in v.name:
            continue
        l1_loss += tf.reduce_sum(tf.abs(v))
    return l1_loss


def get_variable_l1_loss(variables: list):
    """
    获取输入变量的l1损失
    :param variables:
    :return:
    """
    if not isinstance(variables, list):
        variables = [variables]
    l1_loss = tf.constant(value=0.0, dtype=tf.float32)
    for v in variables:
        l1_loss += tf.reduce_sum(tf.abs(v))
    return l1_loss


def get_train_param_nums():
    train_param_sum = 0
    for v in tf.trainable_variables():
        if "embedding/w:0" in v.name:
            continue
        train_param_sum += sum(v.get_shape().as_list())
    return train_param_sum


