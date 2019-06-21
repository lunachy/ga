# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    Seq2SeqLayoutModel
    Description :    
    Author      :    zhaowen
    date        :    2019/4/24
____________________________________________________________________
    Change Activity:
                        2019/4/24:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'

import tensorflow as tf

from graph.CNNGlobalLayoutModel import GlobalVgg16Cnn
from utils.transformer import embedding, multihead_attention, feedforward, normalize, \
    denseWithL2loss, featureGen_vgg
from utils.Hyper_parameter import Hyperparams as hp
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq


class LayoutSeq2Seq_vgg(GlobalVgg16Cnn):

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
        super(LayoutSeq2Seq_vgg, self).__init__(image_width=image_width,
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
        self.use_cnn_predict_encoder = True
        self.use_cnn_predict_predict = True
        self.use_single = False
        self.use_decoder = True
        self.use_decoder_target = True

    def build_attention_graph_tensor(self,
                                     target, batch_size,
                                     trainable: bool = True, ):

        # =================================1, 定义模型的输入数据()
        with tf.variable_scope("get_data", reuse=tf.AUTO_REUSE):
            self.room_context = self.middle_state_cnn_feature
            self.furniture_fea = self.furniture_cnn_feature

            self.y = target
            self.target = target
            self.decoder_targets = tf.reshape(self.target, [-1, self.max_length])

            self.cnn_out = self.cnn_output_distribute

            # 序列的长度, 布局中input 与 output 的长度相同
            self.seq_length = tf.reshape(self.target, [-1, self.max_length])
            self.mask = self.seq_length
            self.seq_length = tf.reduce_sum(self.seq_length, axis=-1)
            self.encoder_inputs_length = self.seq_length

            # batch_size
            self.batch_size = self.seq_length.shape[0]
            print("---debug: self.batch_size:", self.batch_size)
            self.batch_size = tf.Print(self.batch_size, [self.batch_size], message="--debug: self.batch_size")

        # Encoder
        with tf.variable_scope("encoder"):

            embedding = tf.get_variable('embedding', [self.label_size, self.ebd_size])

            # 家具特征
            with tf.variable_scope("furniture_fea", reuse=tf.AUTO_REUSE):

                if self.use_furniture_cnn:
                    self.furniture_fea = self.furniture_fea

                with tf.variable_scope("furniture_concat_process", reuse=tf.AUTO_REUSE):
                    if self.use_cnn_predict_encoder:
                        self.enc = tf.concat(
                            [self.furniture_fea, self.cnn_out, self.room_context], axis=-1)
                    else:
                        self.enc = tf.concat(
                            [self.furniture_fea, self.room_context], axis=-1)
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

            with tf.variable_scope("furniture_encoder", reuse=tf.AUTO_REUSE):

                def single_rnn_cell():
                    # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
                    # 的列表中最终模型会发生错误
                    single_cell = tf.contrib.rnn.LSTMCell(hp.hidden_units)
                    # 添加dropout
                    cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=1 - self.dropout_rate)
                    return cell

                encoder_cell = rnn.MultiRNNCell(
                    [single_rnn_cell() \
                     for i in range(self.num_blocks * self.num_blocks)])

                encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs=self.enc, dtype=self.enc.dtype,
                                                                  sequence_length=self.encoder_inputs_length)

        # =================================2, 定义模型的encoder部分
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            encoder_inputs_length = self.encoder_inputs_length
            # encoder_output = tf.concat(encoder_output, -1)
            # BahdanauAttention 与 LuongAttention 主要不同点再对齐函数上：在计算第 i个位置的score，
            # 前者是需要使用 s_{i-1}和h_{j} 来进行计算，后者使用s_{i}和h_{j}计算，这么来看还是后者直观上更合理些，
            # 逻辑上也更顺滑。两种机制在不同任务上的性能貌似差距也不是很大，具体的细节还待进一步做实验比较。
            #
            # attention_mechanim = seq2seq.BahdanauAttention(self.hidden_units, encoder_output,
            #                                                self.max_length, normalize=True)
            attention_mechanim = seq2seq.LuongAttention(self.hidden_units, encoder_output,
                                                        self.max_length, scale=True,
                                                        memory_sequence_length=encoder_inputs_length)

            batch_size = self.batch_size
            decoder_cell = rnn.MultiRNNCell(
                [single_rnn_cell() \
                 for i in range(self.num_blocks * self.num_blocks)])
            decoder_cell = seq2seq.AttentionWrapper(decoder_cell, attention_mechanim,
                                                    attention_layer_size=self.hidden_units,
                                                    name="Attention_Wrapper")

            #  定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=encoder_state)

            output_layer = tf.layers.Dense(self.label_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if trainable:
                self.y = tf.reshape(self.y, [-1, self.max_length])
                self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 0, self.y[:, :-1]),
                                                -1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.decoder_input)

                training_helper = seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                         sequence_length=self.encoder_inputs_length,
                                                         time_major=False)

                training_decoder = seq2seq.BasicDecoder(decoder_cell, training_helper,
                                                        decoder_initial_state, output_layer=output_layer)

                decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder=training_decoder,
                                                               impute_finished=True,
                                                               maximum_iterations=tf.convert_to_tensor(self.max_length,
                                                                                                       dtype=tf.int32))

                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.logits = self.decoder_logits_train
                self.logits = tf.reshape(self.logits, shape=[-1, self.label_size])
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                self.pred = tf.reshape(self.decoder_predict_train, sahpe=[-1])
                # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets, weights=self.mask)

                correct = tf.cast(tf.equal(self.decoder_targets, self.decoder_predict_train), "float") * self.mask / (
                    tf.reduce_sum(self.mask))
                accuracy = tf.reduce_sum(correct, name="cnn_accuracy")

                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
                self.attention_train_op = train_op
                self.attention_acc = accuracy

            # else:
            #     start_tokens = tf.ones([self.batch_size, ], tf.int32) * 0
            #     if self.beam_search:
            #         inference_decoder = seq2seq.BeamSearchDecoder(cell=decoder_cell,
            #                                                       embedding=embedding,
            #                                                       start_tokens=start_tokens,)

    def build_attention_graph(self, batch_size, trainable: bool = True, ):

        self.build_attention_graph_tensor(
            self.target, batch_size=batch_size, trainable=trainable)
