# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    transformer
    Description :    basic units for transformer
    Author      :    zhaowen
    date        :    2019/3/28
____________________________________________________________________
    Change Activity:
                        2019/3/28:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'

#  refer to https://github.com/princewen/tensorflow_practice/blob/master/basic/Basic-Transformer-Demo/modules.py

from skimage import transform

import tensorflow as tf
import numpy as np
import copy, os


def get_mid_beamsearch(logits, topk, i, preds=[], scores=[], max_length=5, debug=False):
    logits = logits[:, i, :]
    arg_topk = np.argsort(-logits, axis=-1, )[:, 0:topk]
    if debug:
        print("预测中间k", arg_topk, i)

    _pred = []
    _scores = []
    if i == 0:
        for j in range(topk):
            _pred.append([arg_topk[0][j]])
            _scores.append(logits[0][arg_topk[0][j]])
        if debug:
            print("第一次：_yid,_scores", _pred, _scores)
    else:
        for j in range(topk):
            for k in range(len(arg_topk[j])):
                _pred.append(preds[j] + [arg_topk[j][k]])
                if debug:
                    print("logits", logits.shape, len(arg_topk[j]))
                _scores.append(scores[j] + logits[j][arg_topk[j][k]])
        if debug:
            print("第{0}次,_pred:{1},scores:{2}".format(i, _pred, _scores))
        _scores = np.array(_scores)
        _arg_topk = np.argsort(-_scores, axis=-1, )[0:topk]
        _pred = [_pred[k] for k in _arg_topk]
        _scores = [_scores[k] for k in _arg_topk]
    preds = []
    scores = []
    for k in range(topk):
        preds.append(_pred[k])
        scores.append(_scores[k])
    if debug:
        print("当前的有：", preds, scores)

    if i == max_length - 1:
        pred = preds[0]
        pred_k = preds
    return preds, scores


def get_accu(pred, batch_targets, batch_size, maxlength=5):
    '''
    根据预测结果
    获取：
        准确率
        全路径准确率
    excemple:
        input:     get_accu([1,2,3,0],[1,2,3,0],batch_size=4,maxlength=4)
        return:    (1.0, 1.0)

        input:     get_accu([1,2,3,0,1,3,5,2],[1,2,3,0,1,3,0,0],batch_size=8,maxlength=4)
        return:   (1.0, 1.0)

        input:     get_accu([1,2,3,0,1,2,5,2],[1,2,3,0,1,3,0,0],batch_size=8,maxlength=4)
        return:    (0.8, 0.5)

    '''
    right = 0
    wrong = 0
    whole_right = 0
    whole_wrong = 0
    all_right = 0
    all_wrong = 0
    real = batch_targets
    for i in range(batch_size):
        if (i + 1) % maxlength == 0:
            if all_right > 0 and all_wrong == 0:
                whole_right += 1
                all_right, all_wrong = 0, 0
            elif i == 0:
                pass
            else:
                whole_wrong += 1
                all_right = 0
                all_wrong = 0
        if real[i] != 0:
            if real[i] == pred[i]:
                right += 1
                all_right += 1
            else:
                wrong += 1
                all_wrong += 1

    acc_ = right / (right + wrong)
    # print(whole_right,whole_wrong)
    whole_acc = whole_right / (whole_right + whole_wrong)
    return acc_, whole_acc


def get_accu_topk(pred, batch_targets, topk=3, maxlength=5):
    '''
    获取TOPk完全正确的准确率

    pred = np.array([[[ 972,  652, 672,672,672],
        [ 972,  672,  652,672,672],
        [ 672,  652,  972, 672,672]
        ]])
    # [b, topk, l]
    pred.shape

    batch_targets = np.array([[972, 652, 0, 0, 0]])
    # [b,l]
    batch_targets.shape

    '''
    assert topk == np.shape(pred)[1]
    assert np.shape(pred)[0] == np.shape(batch_targets)[0]
    assert np.shape(pred)[2] == np.shape(batch_targets)[1]
    acc_whole_num = 0
    num = 0
    acc_whole_num = 0
    for batch in range(np.shape(pred)[0]):
        pred_b = pred[batch]
        batch_targets_b = batch_targets[batch]
        num += 1

        for top in range(topk):
            pred_1 = pred_b[top]
            acc, acc_whole = get_accu(pred=pred_1, batch_targets=batch_targets_b, batch_size=maxlength,
                                      maxlength=maxlength)
            if acc_whole > 0:
                acc_whole_num += 1
                break
    acc_whole_topk = acc_whole_num / num

    return acc_whole_topk


def featureMapGen(inputs, scope="cnnFeatureMap", reuse=None, trainable=True,
                  filter_list: list = [64, 128, 256, 512, 512]):
    NotImplemented


def featureGen_vgg(inputs, scope="cnnfeatureMap_vgg", reuse=None, trainable=True,
                   filter_list: list = [64, 128, 256, 512, 512]):
    '''
    生成卷积特征图
    :param inputs:
    :param scope:
    :return:
    '''
    from utils.graph_utils import GetTokenEmbedding, Basic2dConv, MultiFilter2dConv, MaxPool, FullConnect, BatchNormal, \
        Vgg16Layer
    with tf.variable_scope(scope, reuse=reuse):
        infrastructure_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=inputs, name="fill_furn_vgg",
                                                             trainable=trainable, filter_list=filter_list)

        _, height, width, c = infrastructure_pool5.get_shape()
        # 卷积层特征 [b*l, h//32, w//32, 72] -> [b, l, h//32*w//32*72]
        infrastructure_cnn_feature = tf.reshape(tensor=infrastructure_pool5, shape=[-1, height * width * c],
                                                name="fill_frastructure_cnn_feature")

    return infrastructure_cnn_feature


from Layout_Service.utils import common


def _darknet53_block(inputs, filters, name=None):
    """
    implement residuals block in darknet53
    """
    shortcut = inputs
    inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
    inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = common.cbam_block(inputs, name)

    inputs = inputs + shortcut
    return inputs


def _block(inputs: tf.Tensor, name):
    _, w, h, _ = inputs.get_shape()
    print("--debug:w,h", w, h)

    if w == 128:
        strides = [1, 2, 2]
    elif w == 64:
        strides = [1, 2, 2]  # [1, 1, 2]
    elif w == 32:
        strides = [1, 1, 1]
    else:  # 暂未实现
        pass
    inputs = common._conv2d_fixed_padding(inputs, 32, 3, strides=strides[0])
    inputs = common._conv2d_fixed_padding(inputs, 64, 3, strides=strides[1])
    inputs = _darknet53_block(inputs, 32, name)
    inputs = common._conv2d_fixed_padding(inputs, 128, 3, strides=strides[2])
    return inputs


def featureGen_DarkNet(inputs_room, inputs_furniture, scope="yolo_v3", reuse=None, ):
    '''
    生成yolo_v3 的卷积特征图
    :param inputs:
    :param scope:
    :param reuse:
    :param trainable:
    :return:
    '''
    with tf.variable_scope(scope, reuse=reuse):
        print("--debug:inputs", inputs_room)
        inputs_room = _block(inputs_room, '_block_inputs_room')
        inputs_furniture = _block(inputs_furniture, "_block_furniture")
        # temp,todo modify
        inputs = tf.concat([inputs_room, inputs_furniture], axis=-1)
        print("--debug:inputs", inputs)
        for i in range(2):
            name = 'block1' + str(i)
            print("--debug:inputs", name, inputs)
            inputs = _darknet53_block(inputs, 128, name)

        print("--debug:_conv2d_fixed_padding", inputs)
        inputs = common._conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            name = 'block2' + str(i)
            print("--debug:block2", name, inputs)
            inputs = _darknet53_block(inputs, 128, name)

        route_1 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 512, 3, strides=2)
        print("--debug:_conv2d_fixed_padding", inputs)
        for i in range(8):
            name = 'block3' + str(i)
            print("--debug:block3", name, inputs)
            inputs = _darknet53_block(inputs, 256, name)

        route_2 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            name = 'block4' + str(i)
            inputs = _darknet53_block(inputs, 512, name)

    return route_1, route_2, inputs


def featureGen(inputs, scope="cnnfeatureMap", reuse=None, trainable=True):
    '''
    生成卷积特征图
    :param inputs:
    :param scope:
    :return:
    '''
    from utils.graph_utils import GetTokenEmbedding, Basic2dConv, MultiFilter2dConv, MaxPool, FullConnect, BatchNormal
    with tf.variable_scope(scope, reuse=reuse):
        infrastructure_cnn1, in_cnn1_l2_loss = Basic2dConv(x=inputs, d_out=72, ksize=(3, 3),
                                                           name="infrastructure_cnn1", active=tf.nn.relu,
                                                           trainable=trainable)
        infrastructure_cnn1 = BatchNormal(x=infrastructure_cnn1, name="infrastructure_cnn1_bn", trainable=trainable)
        infrastructure_pool1 = MaxPool(x=infrastructure_cnn1, name="infrastructure_pool1")

        # 普通 cnn 卷积 (64, 64) -> (32, 32)
        infrastructure_cnn2, in_cnn2_l2_loss = Basic2dConv(x=infrastructure_pool1, d_out=72, ksize=(3, 3),
                                                           name="infrastructure_cnn2", active=tf.nn.relu,
                                                           trainable=trainable)
        infrastructure_cnn2 = BatchNormal(x=infrastructure_cnn2, name="infrastructure_cnn2_bn", trainable=trainable)
        infrastructure_pool2 = MaxPool(x=infrastructure_cnn2, name="infrastructure_pool2")

        # 普通 cnn 卷积  (32, 32) -> (16, 16)
        infrastructure_cnn3, in_cnn3_l2_loss = Basic2dConv(x=infrastructure_pool2, d_out=72, ksize=(3, 3),
                                                           name="infrastructure_cnn3", active=tf.nn.relu,
                                                           trainable=trainable)
        infrastructure_cnn3 = BatchNormal(x=infrastructure_cnn3, name="infrastructure_cnn3_bn", trainable=trainable)
        infrastructure_pool3 = MaxPool(x=infrastructure_cnn3, name="infrastructure_pool3")

        # 多种核尺寸 cnn 卷积  (32, 32) -> (16, 16)
        d_outs = [24, 24, 24]
        ksizes = [[3, 3], [5, 5], [7, 7]]
        strides = [[1, 1], [1, 1], [1, 1]]

        infrastructure_cnn4, in_cnn4_l2_loss = MultiFilter2dConv(x=infrastructure_pool3, d_outs=d_outs, ksizes=ksizes,
                                                                 strides=strides, active=tf.nn.relu,
                                                                 name="infrastructure_cnn4", trainable=trainable)
        infrastructure_cnn4 = tf.concat(values=infrastructure_cnn4, axis=-1)
        infrastructure_pool4 = MaxPool(x=infrastructure_cnn4, name="infrastructure_pool4")

        # 多种核尺寸 cnn 卷积  (16, 16) -> (8, 8)
        d_outs = [24, 24, 24]
        ksizes = [[3, 3], [5, 5], [7, 7]]
        strides = [[1, 1], [1, 1], [1, 1]]

        infrastructure_cnn5, in_cnn5_l2_loss = MultiFilter2dConv(x=infrastructure_pool4, d_outs=d_outs, ksizes=ksizes,
                                                                 strides=strides, active=tf.nn.relu,
                                                                 name="infrastructure_cnn5", trainable=trainable)
        infrastructure_cnn5 = tf.concat(values=infrastructure_cnn5, axis=-1)
        # [b*l, h//32, w//32, 72]
        infrastructure_pool5 = MaxPool(x=infrastructure_cnn5, name="infrastructure_pool5")
        image_height = 128
        image_width = 128

        height, weight = int(image_height / 32), int(image_width / 32)
        max_length = 5
        # 卷积层特征 [b*l, h//32, w//32, 72] -> [b, l, h//32*w//32*72]
        infrastructure_cnn_feature = tf.reshape(tensor=infrastructure_pool5,
                                                shape=[-1, height * weight * 72],
                                                name="infrastructure_cnn_feature")

    return infrastructure_cnn_feature


def gen_furniture_image(dx, dy, w, h, cenpoint, fill_value=1):
    '''
    在一个宽度为w, 高度为h的图像 中，以 cenx,ceny为中心点
    dx,dy为宽度高度生成一张图(w,h,1)的图.

    '''
    cenx, ceny = cenpoint[0], cenpoint[1]

    img = np.zeros(shape=(w, h))
    stx = np.floor(cenx - int(dx / 2))
    sty = np.floor(ceny - int(dy / 2))
    edx = np.ceil(int(dx / 2) + cenx)
    edy = np.ceil(int(dy / 2) + ceny)
    # print(stx, sty, edx, edy)
    stx, sty, edx, edy = int(stx), int(sty), int(edx), int(edy)
    img[stx:edx, sty:edy] = fill_value
    return img


def GenPng_numpy(room_ori, furniture_dx, furniture_dy, furniture_cid, furniture_value, length,
                 debug=False, m=20, n=20, r=90, w=128, h=128, add_title=''):
    '''
    根据预测结果生成新的户型信息图(numpy 版)(预测的结果是m*n 实际的图片是128*128的因此要做一个转换)
    :param room_ori: 户型图
    :param furniture_dx: 家具的宽度
    :param furniture_dy: 家具的高度
    :param furniture_cid: 家具的cid
    furniture_value ：可以分解为：
        :param furniture_x: 家具位置的 x
        :param furniture_y: 家具位置的 y
        :param furniture_r: 家具的旋转角度
    :return: 家具在户型图中存放的位置
    '''
    from utils.display import show_img
    if isinstance(furniture_value, int):
        k = furniture_value // (m * n)
        ij = furniture_value % (m * n)
        cen_x = ij // m
        cen_y = ij % n
    else:
        cen_x, cen_y, k = furniture_value[0], furniture_value[1], furniture_value[2]

    grid_size = length / w
    cen_x = int((cen_x / m) * w)
    cen_y = int((cen_y / n) * h)

    furniture_grid_dx = int(furniture_dx / grid_size)
    furniture_grid_dy = int(furniture_dy / grid_size)
    if debug:
        furniture = gen_furniture_image(furniture_grid_dx, furniture_grid_dy, w, h, cenpoint=(cen_x, cen_y),
                                        fill_value=1)
    else:
        furniture = gen_furniture_image(furniture_grid_dx, furniture_grid_dy, w, h, cenpoint=(cen_x, cen_y),
                                        fill_value=furniture_cid)
    if int(k * r) in [90, 270]:
        # furniture = transform.rotate(angle= k * r, image=furniture, center=(cen_x, cen_y))
        if debug:
            print("旋转之前")
            show_img(furniture, "before")
        furniture = np.transpose(furniture, axes=[1, 0])

        if debug:
            print(np.shape(furniture))
            print("旋转了角度")
            show_img(furniture, "after")
        # furniture_copy = np.zeros_like(furniture)
        # for i in range(furniture.shape[0]):
        #     for j in range(furniture.shape[1]):
        #         furniture_copy[j][i]  = furniture[i][j]
        # furniture = furniture_copy
    # furniture = transform.rotate(angle= k * r, image=furniture, center=(cen_x, cen_y))
    if debug:
        import matplotlib.pyplot as plt
        print("--debug:", cen_x, cen_y, k * r, k, r)

        # show_img(room_ori + furniture, add_title)
        fig, ax = plt.subplots()
        if add_title:
            plt.title(add_title)
        plt.text(x=cen_x, y=cen_y, s=str(furniture_cid))
        ax.imshow(furniture + room_ori)
        plt.show()
    return room_ori + furniture


def denseWithL2loss(inputs, units, activation=None, reuse=None, use_bias=True, scope="Dense_with_L2"):
    with tf.variable_scope(scope, reuse=reuse):
        layer_dense = tf.layers.Dense(units, activation=activation, use_bias=use_bias)
        k = layer_dense(inputs)
        if len(layer_dense.weights) == 2:
            l2_loss = tf.nn.l2_loss(layer_dense.weights[0]) + tf.nn.l2_loss(layer_dense.weights[1])
        else:
            l2_loss = tf.nn.l2_loss(layer_dense.weights)
        return k, l2_loss


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    For example,
    ```
    import tensorflow as tf
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    ```
    import tensorflow as tf
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def add_positional_embedding_nd(x, max_length, name=None):
    """Adds n-dimensional positional embedding.

    The embeddings add to all positional dimensions of the tensor.

    Args:
      x: Tensor with shape [batch, p1 ... pn, depth]. It has n positional
        dimensions, i.e., 1 for text, 2 for images, 3 for video, etc.
      max_length: int representing static maximum size of any dimension.
      name: str representing name of the embedding tf.Variable.

    Returns:
      Tensor of same shape as x.
    """
    with tf.name_scope("add_positional_embedding_nd"):
        x_shape = x.shape.as_list()
        num_dims = len(x_shape) - 2
        depth = x_shape[-1]
        base_shape = [1] * (num_dims + 1) + [depth]
        base_start = [0] * (num_dims + 2)
        base_size = [-1] + [1] * num_dims + [depth]
        for i in range(num_dims):
            shape = base_shape[:]
            start = base_start[:]
            size = base_size[:]
            shape[i + 1] = max_length
            size[i + 1] = x_shape[i + 1]
            var = tf.get_variable(
                name + "_%d" % i,
                shape,
                initializer=tf.random_normal_initializer(0, depth ** -0.5))
            var = var * depth ** 0.5
            x += tf.slice(var, start, size)
        return x


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=True):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


def multihead_attention(queries, keys, num_units=None,
                        num_heads=0,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="mulithead_attention",
                        reuse=None):
    '''Applies multihead attention.
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projection
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  #
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #

        # Split and Concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  #
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # t = tf.cast((float(K_.get_shape().as_list()[-1]) ** 0.5), "float32")
        # print("outputs,t", outputs, t)
        # outputs = tf.cast(outputs, "float32")
        # outputs = outputs / t
        # outputs = tf.cast(outputs, "int32")

        # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # 这里其实就是进行一个mask操作，不给模型看到未来的信息。
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            #
            # tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        # Query Mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)

        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)

    return outputs


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    For example,
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
    outputs = label_smoothing(inputs)
    with tf.Session() as sess:
        print(sess.run([outputs]))
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def scaled_dotproduct_attention(queries, keys, num_units=None,
                                num_heads=0,
                                dropout_rate=0,
                                is_training=True,
                                causality=False,
                                scope="mulithead_attention",
                                reuse=None):
    '''Applies multihead attention.
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projection
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  #
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # 这里其实就是进行一个mask操作，不给模型看到未来的信息。
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            # tf: 1.10
            # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()

            # tf:1.12 +
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        # Query Mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)

    return outputs


def Besic2dConvForTrans(x: tf.Tensor,
                        d_out: int,
                        name: str,
                        ksize: tuple = (3, 3),
                        stride: tuple = (1, 1),
                        active=None,
                        trainable: bool = True,
                        use_bias: bool = True,
                        dtype=tf.float32,
                        padding: str = 'SAME'):
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


def convForTrans(trainable):
    '''
    根据论文http://www.cs.columbia.edu/~vondrick/transformer.pdf
    构建的conv层
    :return:
    '''

    NotImplemented


class glowForTrans():
    '''
    glow 流模型
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/video/next_frame_glow.py
    :return:
    '''

    def __init__(self, ):
        NotImplemented


class image_transformer():

    def __init__(self):
        NotImplemented

    def body(self, inputs, targets):
        # 输入、输出

        inputs_shape = inputs.shape.as_list()
        targets_shape = inputs.shape.as_list()
        hp = ''

        # 预测时
        if not (tf.get_variable_scope().reuse or hp.mode == "predict"):
            tf.summary.image("inputs", inputs, max_outputs=1)
            tf.summary.image("targets", targets, max_outputs=1)

        encoder_input = self.prepare_encoder(inputs, hp)
        encoder_output = self.transformer_encoder_layers(
            encoder_input,
            hp.num_encoder_layers,
            hp,
            attention_type=hp.enc_attention_type,
            name="encoder")

        decoder_input, rows, cols = self.prepare_decoder(targets, hparams=hp)

        # 配置生成的条件信息 todo:适配
        decoder_input += tf.reshape(inputs, [targets_shape[0], 1, 1, hp.hidden_size])
        decoder_output = self.transfomer_decoder_layers(decoder_input, None, hp.num_decoder_layers, hp,
                                                        attention_type=hp.dec_attention_type, name="decoder")

        output = self.create_output(decoder_output, rows, cols, targets, hp)

    def prepare_decoder(self, targets, hparams):
        NotImplemented

    def transfomer_decoder_layers(self):
        NotImplemented

    def create_output(self):
        NotImplemented


import Layout_Service.utils.nn as nn
from tensorflow.contrib.framework.python.ops import arg_scope


#
# def feature_cnn_pix(batch_image, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160,
#                     nr_logistic_mix=10, resnet_nonlinearity='concat_elu', energy_distance=False,
#                     scope="feature_cnn_pix",
#                     reuse=None):
#     '''
#     获取 形状为(b,h,w,d1) 的image Tensor 返回一个 (B,H,W,D2)的特征图
#     :param batch_image:
#     :param h:
#     :param init:
#     :param ema:
#     :param dropout_p:
#     :param nr_resnet:
#     :param nr_filters:
#     :param nr_logistic_mix:
#     :param resnet_nonlinearity:
#     :param energy_distance:
#     :return:
#     '''
#
#     with tf.variable_scope(scope, reuse=reuse):
#         counters = {}
#         x = batch_image
#         with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema,
#                        dropout_p=dropout_p):
#
#             # parse resnet nonlinearity argument
#             if resnet_nonlinearity == 'concat_elu':
#                 resnet_nonlinearity = nn.concat_elu
#             elif resnet_nonlinearity == 'elu':
#                 resnet_nonlinearity = tf.nn.elu
#             elif resnet_nonlinearity == 'relu':
#                 resnet_nonlinearity = tf.nn.relu
#             else:
#                 raise ('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')
#
#             with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):
#
#                 # ////////// up pass through pixelCNN ////////
#                 xs = x.shape.as_list()  # nn.int_shape(x)
#                 x_pad = tf.concat([x, tf.ones_like(x[:, :, :, 0:1])],
#                                   -1)  # add channel of ones to distinguish image from padding later on
#                 u_list = [nn.down_shift(
#                     nn.down_shifted_conv2d(x_pad, num_filters=nr_filters,
#                                            filter_size=[2, 3]))]  # stream for pixels above
#                 ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) + \
#                            nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,
#                                                                                                                    1]))]  # stream for up and to the left
#
#                 with tf.variable_scope("u_list_1", reuse=reuse):
#                     for rep in range(nr_resnet):
#                         u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
#                         ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
#
#                     u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
#                     ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
#
#                 with tf.variable_scope("u_list_2", reuse=reuse):
#                     for rep in range(nr_resnet):
#                         u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
#                         ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
#                     u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
#                     ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
#                 with tf.variable_scope("u_list_3", reuse=reuse):
#                     for rep in range(nr_resnet):
#                         u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
#                         ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
#
#                 # remember nodes
#                 for t in u_list + ul_list:
#                     tf.add_to_collection('checkpoints', t)
#
#                 # /////// down pass ////////
#                 print("ul_list",ul_list)
#                 print("u_list", u_list)
#                 u = u_list.pop()
#                 ul = ul_list.pop()
#
#                 with tf.variable_scope("r_u_list_1", reuse=reuse):
#                     for rep in range(nr_resnet):
#                         u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
#                         ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
#                         tf.add_to_collection('checkpoints', u)
#                         tf.add_to_collection('checkpoints', ul)
#                     u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
#                     ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
#                 with tf.variable_scope("r_u_list_2", reuse=reuse):
#                     for rep in range(nr_resnet + 1):
#                         u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
#                         ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
#                         tf.add_to_collection('checkpoints', u)
#                         tf.add_to_collection('checkpoints', ul)
#
#                     u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
#                     ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
#
#                 with tf.variable_scope("r_u_list_3", reuse=reuse):
#                     for rep in range(nr_resnet + 1):
#                         u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
#                         ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
#                         tf.add_to_collection('checkpoints', u)
#                         tf.add_to_collection('checkpoints', ul)
#                     print(ul, "output before")
#                 x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)
#
#                 assert len(u_list) == 0
#                 assert len(ul_list) == 0
#
#                 return x_out


def feature_cnn_pix(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160,
                    nr_logistic_mix=10, resnet_nonlinearity='concat_elu', energy_distance=False,
                    scope="feature_cnn_pix",
                    reuse=None):
    '''
    获取 形状为(b,h,w,d1) 的image Tensor 返回一个 (B,H,W,D2)的特征图
    :param batch_image:
    :param h:
    :param init:
    :param ema:
    :param dropout_p:
    :param nr_resnet:
    :param nr_filters:
    :param nr_logistic_mix:
    :param resnet_nonlinearity:
    :param energy_distance:
    :return:
    '''

    with tf.variable_scope(scope, reuse=reuse):
        counters = {}

        with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema,
                       dropout_p=dropout_p):

            # parse resnet nonlinearity argument
            if resnet_nonlinearity == 'concat_elu':
                resnet_nonlinearity = nn.concat_elu
            elif resnet_nonlinearity == 'elu':
                resnet_nonlinearity = tf.nn.elu
            elif resnet_nonlinearity == 'relu':
                resnet_nonlinearity = tf.nn.relu
            else:
                raise ('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

            with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

                # ////////// up pass through pixelCNN ////////
                xs = nn.int_shape(x)
                print("xs", xs)
                x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])],
                                  3)  # add channel of ones to distinguish image from padding later on
                u_list = [nn.down_shift(
                    nn.down_shifted_conv2d(x_pad, num_filters=nr_filters,
                                           filter_size=[2, 3]))]  # stream for pixels above
                print("u_list", u_list)
                ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) + \
                           nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,
                                                                                                                   1]))]  # stream for up and to the left
                print("ul_list", ul_list)
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
                print("u_list", u_list)
                print("ul_list", ul_list)

                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
                print("u_list", u_list)

                print("ul_list", ul_list)

                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
                print("u_list", u_list)
                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
                print("u_list", u_list)
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
                print("u_list", u_list)
                # remember nodes
                for t in u_list + ul_list:
                    tf.add_to_collection('checkpoints', t)
                print("u_list", u_list)
                # /////// down pass ////////
                u = u_list.pop()
                ul = ul_list.pop()
                for rep in range(nr_resnet):
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                if energy_distance:
                    f = nn.nin(tf.nn.elu(ul), 64)

                    # generate 10 samples
                    fs = []
                    for rep in range(10):
                        fs.append(f)
                    f = tf.concat(fs, 0)
                    fs = nn.int_shape(f)
                    f += nn.nin(tf.random_uniform(shape=fs[:-1] + [4], minval=-1., maxval=1.), 64)
                    f = nn.nin(nn.concat_elu(f), 64)
                    x_sample = tf.tanh(nn.nin(nn.concat_elu(f), 3, init_scale=0.1))

                    x_sample = tf.split(x_sample, 10, 0)

                    assert len(u_list) == 0
                    assert len(ul_list) == 0

                    return x_sample

                else:
                    x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

                    assert len(u_list) == 0
                    assert len(ul_list) == 0

                    return x_out


from collections import OrderedDict
from Layout_Service.utils.u_net_layers import (weight_variable, weight_variable_devonc, bias_variable,
                                               conv2d, deconv2d, max_pool, crop_and_concat)


def create_conv_Unet(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2,
                     scope="feature_cnn_pix",
                     reuse=None):
    """
    U-net 的卷积特征提取层
    Creates a new convolutional unet for the given parametrization.
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    # Placeholder for the input image
    with tf.variable_scope(scope, reuse=reuse):
        with tf.name_scope("preprocessing"):
            x_shape = x.shape.as_list()
            nx = x_shape[1]
            ny = x_shape[2]
            x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
            print("x_image", x_image)
            in_node = x_image
            batch_size = tf.shape(x_image)[0]
            print(nx, ny, channels, batch_size, "nx, ny, channels, batch_size")

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size
        # down layers
        for layer in range(0, layers):
            with tf.name_scope("down_conv_{}".format(str(layer))):
                features = 2 ** layer * features_root
                # print("features", features)
                stddev = np.sqrt(2 / (filter_size ** 2 * features))
                # print("stddev", stddev)
                if layer == 0:
                    w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
                else:
                    w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")
                # print("w1", w1)

                w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
                b1 = bias_variable([features], name="b1")
                b2 = bias_variable([features], name="b2")

                conv1 = conv2d(in_node, w1, b1, keep_prob)
                tmp_h_conv = tf.nn.relu(conv1)
                conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob)
                dw_h_convs[layer] = tf.nn.relu(conv2)

                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))

                size -= 2 * 2 * (filter_size // 2)  # valid conv
                if layer < layers - 1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]
                    size /= pool_size

        in_node = dw_h_convs[layers - 1]
        # print(in_node, "in_node")

        # up layers
        for layer in range(layers - 2, -1, -1):
            with tf.name_scope("up_conv_{}".format(str(layer))):
                features = 2 ** (layer + 1) * features_root
                stddev = np.sqrt(2 / (filter_size ** 2 * features))
                wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")

                bd = bias_variable([features // 2], name="bd")

                h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                deconv[layer] = h_deconv_concat
                print("up layers:wd:{0},bd:{1},h_deconv:{2},h_deconv_concat:{3},".format(wd, bd, h_deconv,
                                                                                         h_deconv_concat))

                w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
                w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
                b1 = bias_variable([features // 2], name="b1")
                b2 = bias_variable([features // 2], name="b2")

                conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
                h_conv = tf.nn.relu(conv1)
                conv2 = conv2d(h_conv, w2, b2, keep_prob)
                print(conv2, "conv2_up")
                in_node = tf.nn.relu(conv2)
                print(in_node, "in_node_up")
                up_h_convs[layer] = in_node

                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))

                size *= pool_size
                size -= 2 * 2 * (filter_size // 2)  # valid conv

        # Output Map
        with tf.name_scope("output_map"):
            weight = weight_variable([1, 1, features_root, n_class], stddev)
            bias = bias_variable([n_class], name="bias")
            conv = conv2d(in_node, weight, bias, tf.constant(1.0))
            output_map = tf.nn.relu(conv)
            up_h_convs["out"] = output_map

        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return output_map, variables, int(in_size - size)


from checkCrash.CrashChecker import CrashChecker


def get_XYR(label):
    configpaths = ["./config.props", "./checkCrash/config.props", "../checkCrash/config.props"]
    configpaths = [x for x in configpaths if os.path.exists(x)]
    assert configpaths, 'no config file exists!'

    ck = CrashChecker(configPath=configpaths[0])
    grids = ck.label2InputGrids(label, 16)
    x_a, y_a = grids["centerX"], grids["centerY"]
    real_pos = ck.convert2real(label=label)
    r_a = real_pos["r"]
    return x_a, y_a, r_a


def calcLabelSimlarity(label_a, label_b, debug=False):
    # 给定两个label 计算 两个label 之间相似性的得分
    # 越相似 得分越接近 1 ，否则 得分越接近 0
    # 默认 label是 16 * 16 * 4 的
    x_a, y_a, r_a = get_XYR(label_a)
    x_b, y_b, r_b = get_XYR(label_b)

    one_grid_dis = 1

    dis_x = int(abs(x_b - x_a))
    dis_y = int(abs(y_b - y_a))
    dis_r = int(abs(r_a - r_b))

    sim_score = 1

    # 角度
    # 90度相差 或者270度 相差的可以认为 很不相似，

    if dis_r in [90, 270]:
        sim_score -= 0.3
    elif dis_r in [180]:
        sim_score -= 0.15

    # 距离
    dis = np.sqrt((dis_x) ** 2 + (dis_y) ** 2)
    x = sim_score - 0.7 * (dis + 1e-4 / one_grid_dis) ** (-2)
    if x < 0:
        x = 0
    sim_score -= x
    if debug:
        max_grid_dis = 15 * np.sqrt(2)
        print("x:{}".format(x))
        print("x_a:{},y_a:{},r_a:{}".format(x_a, y_a, r_a))
        print("x_b:{},y_b:{},r_b:{}".format(x_b, y_b, r_b))
        print("dis:{},max_grid_dis:{}".format(dis, max_grid_dis))
    return sim_score
