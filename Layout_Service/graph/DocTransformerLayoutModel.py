# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    TransformerLayoutModel_Reshapechange
    Description :    
    Author      :    zhaowen
    date        :    2019/4/25
____________________________________________________________________
    Change Activity:
                        2019/4/25:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'

"""
Transformer 
"""
import tensorflow as tf

from graph.CNNGlobalLayoutModel import GlobalVgg16Cnn
from utils.transformer import embedding, multihead_attention, feedforward, normalize, \
    denseWithL2loss, featureGen_vgg
from utils.Hyper_parameter import Hyperparams as hp
from utils.graph_utils import Basic2dConv


class LayoutDocTransformer_vgg(GlobalVgg16Cnn):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 ):
        """
        :param image_width:
        :param image_height:
        :param furniture_nums:
        :param label_size:
        :param furniture_width:
        :param furniture_height:
        :param furniture_embedding_size:
        :param max_length:
        """
        super(LayoutDocTransformer_vgg, self).__init__(image_width=image_width,
                                                       image_height=image_height,
                                                       furniture_nums=furniture_nums,
                                                       label_size=label_size,
                                                       furniture_width=furniture_width,
                                                       furniture_height=furniture_height,
                                                       furniture_embedding_size=furniture_embedding_size,
                                                       max_length=max_length)

        # 自定义的一些配置

        self.num_blocks = hp.num_blocks
        self.hidden_units = hp.hidden_units
        self.ebd_size = hp.ebd_size
        self.num_heads = hp.num_heads
        self.dropout_rate = hp.dropout_rate
        self.sinusoid = hp.sinusoid
        self.src_voc_size = furniture_nums
        self.lr = hp.lr
        self.top_k = hp.top_k

        self.use_furniture_cnn = True
        self.use_fill_furniture = True
        self.use_cnn_predict_encoder = hp.use_cnn_predict_encoder
        self.use_cnn_predict_predict = hp.use_cnn_predict_predict
        self.use_single = False
        self.use_decoder = True
        self.use_decoder_target = True
        self.use_cnn_reshape = hp.use_cnn_reshape

        self.use_cnn = hp.use_cnn
        self.sequence_data = False

        self.use_cnn_loss = hp.use_cnn_loss
        print("--info:是否使用cnn 进行reshape:{}".format(self.use_cnn_reshape))
        print("--info:是否使用cnn_out：{}".format(self.use_cnn_predict_predict))

    def build_attention_graph_tensor(self,
                                     target, room='', furniture='',
                                     trainable: bool = True, ):

        with tf.variable_scope("get_data", reuse=tf.AUTO_REUSE):
            # 使用 VGG 模型
            if self.use_cnn:
                self.cnn_out = self.cnn_output_distribute
                if self.use_cnn_reshape:
                    self.room_context = self.middle_state_cnn_not_flatten
                    self.furniture_fea = self.furniture_cnn_not_flatten
                else:
                    self.room_context = self.middle_state_cnn_feature
                    self.furniture_fea = self.furniture_cnn_feature
            else:
                # 不使用VGG 模型
                if self.sequence_data:
                    self.room_context = tf.reshape(room, shape=[-1, self.image_width, self.image_height])
                    target = tf.reshape(target, shape=[-1, ])
                else:
                    self.room_context = room
                self.room_context = embedding(self.room_context,
                                              vocab_size=5000,
                                              num_units=self.ebd_size,
                                              zero_pad=True,  # 让padding一直是0
                                              scale=True,
                                              scope="room_context_embed")
                self.room_context = tf.layers.dropout(self.room_context, rate=self.dropout_rate,
                                                      training=tf.convert_to_tensor(trainable))
                self.room_context = tf.layers.flatten(self.room_context)

                if self.sequence_data:
                    self.furniture_fea = tf.reshape(furniture, shape=[-1, self.image_width, self.image_height])
                else:
                    self.furniture_fea = furniture
                self.furniture_fea = embedding(self.furniture_fea,
                                               vocab_size=5000,
                                               num_units=self.ebd_size,
                                               zero_pad=True,  # 让padding一直是0
                                               scale=True,
                                               scope="furniture_fea_embed")
                self.furniture_fea = tf.layers.dropout(self.furniture_fea, rate=self.dropout_rate,
                                                       training=tf.convert_to_tensor(trainable))
                self.furniture_fea = tf.layers.flatten(self.furniture_fea)

            self.y = target
            self.target = target

        with tf.variable_scope("room_context"):

            if self.use_cnn_reshape and self.use_cnn:
                self.room_context, l2_loss_room_context = Basic2dConv(x=self.room_context, d_out=32, name="cnn_reshape",
                                                                      active=tf.nn.relu, trainable=trainable)

                self.room_context = tf.reshape(self.room_context, [-1, self.hidden_units])
                print("cnn:", self.room_context)

            else:
                self.room_context, l2_loss_room_context = denseWithL2loss(self.room_context, 1 * self.hidden_units,
                                                                          scope="room_context_reshape",
                                                                          activation=tf.nn.relu, use_bias=False)
            self.room_context = tf.reshape(self.room_context,
                                           [-1, self.max_length, self.room_context.shape.as_list()[-1]])

            self.room_context += embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.room_context)[1]), 0),
                        [tf.shape(self.room_context)[0], 1]),
                vocab_size=hp.maxlen,
                num_units=hp.hidden_units,
                zero_pad=False,
                scale=False,
                scope="room_c_pe")
            self.room_context = tf.layers.dropout(self.room_context, rate=self.dropout_rate,
                                                  training=tf.convert_to_tensor(trainable))

            with tf.variable_scope("room_context_attention", reuse=tf.AUTO_REUSE):
                for i in range(self.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.room_context = multihead_attention(queries=self.room_context,
                                                                keys=self.room_context,
                                                                num_units=self.hidden_units,
                                                                num_heads=self.num_heads,
                                                                dropout_rate=self.dropout_rate,
                                                                is_training=trainable,
                                                                causality=True,
                                                                scope="self_att"
                                                                )
                        self.room_context = feedforward(self.room_context,
                                                        num_units=[4 * self.hidden_units, self.hidden_units])

        with tf.variable_scope("encoder"):

            with tf.variable_scope("furniture_fea", reuse=tf.AUTO_REUSE):

                with tf.variable_scope("furniture_concat_process", reuse=tf.AUTO_REUSE):

                    if self.use_cnn_reshape and self.use_cnn:
                        self.furniture_fea, l2_loss_enc_3 = Basic2dConv(x=self.furniture_fea, d_out=32,
                                                                        name="cnn_reshape", active=tf.nn.relu,
                                                                        trainable=trainable)

                        print("cnn_before:", self.furniture_fea)
                        self.furniture_fea = tf.reshape(self.furniture_fea, [-1, self.hidden_units])
                        print("cnn:", self.furniture_fea)
                        self.enc = self.furniture_fea
                    else:

                        if self.use_cnn_predict_encoder:
                            self.enc = tf.concat(
                                [self.furniture_fea, self.cnn_out], axis=-1)
                        else:
                            self.enc = tf.concat(
                                [self.furniture_fea], axis=-1)
                    if self.use_single:
                        enc_shape = self.enc.shape.as_list()
                        enc_shape_ = [-1, 1, enc_shape[-1]]
                        self.enc = tf.reshape(self.enc, shape=enc_shape_)
                    else:
                        enc_shape = self.enc.shape.as_list()
                        enc_shape[0] = -1
                        enc_shape = self.enc.shape.as_list()
                        enc_shape_ = [-1, self.max_length, enc_shape[-1]]
                        self.enc = tf.reshape(self.enc, shape=enc_shape_)

                    # self.enc, l2_loss_enc_1 = denseWithL2loss(self.enc, 4 * self.hidden_units, scope="enc_shape_1",
                    #                                           activation=tf.nn.relu,use_bias=False)
                    # self.enc, l2_loss_enc_2 = denseWithL2loss(self.enc, 2 * self.hidden_units, scope="enc_shape_2",
                    #                                           activation=tf.nn.relu,use_bias=False)
                    if not self.use_cnn_reshape:
                        self.enc, l2_loss_enc_3 = denseWithL2loss(self.enc, 1 * self.hidden_units, scope="enc_shape_3",
                                                                  activation=tf.nn.relu, use_bias=False)

                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.enc)[1]), 0), [tf.shape(self.enc)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                    self.enc = tf.layers.dropout(self.enc, rate=self.dropout_rate,
                                                 training=tf.convert_to_tensor(trainable))

            with tf.variable_scope("encoder_attention", reuse=tf.AUTO_REUSE):
                for i in range(self.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=self.hidden_units,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=trainable,
                                                       causality=True,
                                                       scope="self_att"
                                                       )
                        self.enc = multihead_attention(queries=self.room_context,
                                                       keys=self.enc,
                                                       num_units=self.hidden_units,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=trainable,
                                                       causality=True,
                                                       scope="furniture_context_att"
                                                       )

                        self.enc = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])
        if self.use_fill_furniture:
            with tf.variable_scope("decoder"):

                if not self.use_single:
                    self.y = target
                    self.y = tf.reshape(self.y, [-1, self.max_length])
                    self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 0, self.y[:, :-1]),
                                                    -1)  # 0代表<S>，是decoder的初始输入

                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=self.label_size,
                                     num_units=self.hidden_units,
                                     scale=True,
                                     scope="fill_targets_embed")

                self.dec = tf.layers.dropout(self.dec, rate=self.dropout_rate,
                                             training=tf.convert_to_tensor(trainable))

                self.dec += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                            [tf.shape(self.decoder_inputs)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe")

                with tf.variable_scope("dec_attention", reuse=tf.AUTO_REUSE):
                    for i in range(self.num_blocks):
                        with tf.variable_scope("num_blocks_{}".format(i)):
                            self.dec = multihead_attention(queries=self.dec,
                                                           keys=self.dec,
                                                           num_units=self.hidden_units,
                                                           num_heads=self.num_heads,
                                                           dropout_rate=self.dropout_rate,
                                                           is_training=trainable,
                                                           causality=True,
                                                           scope="self_att"
                                                           )
                            self.dec = multihead_attention(queries=self.dec,
                                                           keys=self.enc,
                                                           num_units=self.hidden_units,
                                                           num_heads=self.num_heads,
                                                           dropout_rate=self.dropout_rate,
                                                           is_training=trainable,
                                                           causality=True,
                                                           scope="self_att_dec"
                                                           )
                            self.dec = feedforward(self.dec,
                                                   num_units=[4 * self.hidden_units, self.hidden_units])

                self.enc = self.dec

        with tf.variable_scope("label"):
            # 归一化
            self.enc = normalize(self.enc)

            if self.use_cnn_predict_predict:
                self.cnn_out_ = tf.reshape(self.cnn_out, [-1, self.max_length, self.label_size])
                self.enc = tf.concat(
                    [self.enc, self.cnn_out_], axis=-1)

            # 预测值：单个时:(?,1,1601),整体时:(?,5,1601)

            self.logits, self.predict_l2_loss = denseWithL2loss(self.enc, self.label_size)

            # 将多个与单个的结果全都转成单个的来进行计算
            self.logits = tf.reshape(self.logits, shape=[-1, self.label_size])
            labels = self.target
            target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)

            batch_target = labels
            istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充

            # label smoothing
            label_smoothing = 0.1

            new_labels = ((1. - label_smoothing) * tf.cast(target_one_hot, tf.float32) + (
                    label_smoothing / float(self.label_size)))

            attention_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=new_labels, logits=self.logits)

            attention_score = tf.nn.softmax(self.logits, name="attention_score")
            self.attention_score = attention_score
            self.attention_output_distribute = attention_score

            # [b, label_size] -> [b, ]

            attention_predict = tf.argmax(attention_score, 1, name="attention_prediction", output_type=tf.int32)
            self.pred = attention_predict

            # (目标 且预测正确) / (目标数目)
            correct = tf.cast(tf.equal(batch_target, attention_predict), "float") * istarget / (tf.reduce_sum(istarget))
            accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
            self.attention_acc = accuracy

            self.l2_loss_attention = self.predict_l2_loss + l2_loss_enc_3 + l2_loss_room_context  # l2_loss_enc_1 + l2_loss_enc_2

            self.attention_loss = tf.reduce_sum(attention_loss_ * istarget) / (tf.reduce_sum(istarget))

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # 暂时填充用0 todo：删除在代码中
            self.attention_acc_topk = self.attention_acc
            self.preds_topk = self.pred

            if trainable:

                if self.use_cnn_predict_predict and self.use_cnn_loss:
                    self.attention_loss = self.attention_loss + self.l2_loss_attention * 1e-3 + 0.2 * self.cnn_loss
                else:
                    self.attention_loss = self.attention_loss + self.l2_loss_attention * 1e-3

                train_vars = tf.trainable_variables()

                decay_rate = 0.98
                global_steps = self.global_step
                decay_steps = 2000

                c = tf.maximum(0.0001,
                               tf.train.exponential_decay(hp.lr, global_steps, decay_steps, decay_rate, staircase=True))
                self.lr_decay = c

                # self.optimizer = tf.train.AdamOptimizer(learning_rate=c, beta1=0.9, beta2=0.98, epsilon=1e-8, )
                tvars = tf.trainable_variables()
                max_grad_norm = 10  # 梯度计算
                grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.attention_loss, tvars), max_grad_norm)
                train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
                self.attention_train_op = train_op

                # self.attention_train_op = self.optimizer.minimize(
                #     self.attention_loss,
                #     global_step=self.global_step, var_list=train_vars)
                # self.attention_train_op_all = self.optimizer.minimize(
                #     self.attention_loss,
                #     global_step=self.global_step)
                print("done!")

    def build_attention_graph(self, trainable: bool = True):

        if self.use_cnn:
            self.build_attention_graph_tensor(
                self.target, trainable)
        else:
            self.build_attention_graph_tensor(
                self.target, room=self.room, furniture=self.furniture, trainable=trainable)
