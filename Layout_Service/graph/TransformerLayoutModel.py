# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    TransformerLayoutModel
    Description :    使用 attention 2 attention 进行布局测试全路径与碰撞检测
    Author      :    zhaowen
    date        :    2019/3/28
____________________________________________________________________
    Change Activity:
                        2019/3/28:
____________________________________________________________________

'''
__author__ = 'zhaowen'

"""
Transformer 布局模块
"""
import copy
import tensorflow as tf
import numpy as np
from graph.CNNGlobalLayoutModel import GlobalVgg16Cnn
from utils.transformer import embedding, multihead_attention, feedforward, normalize, \
    denseWithL2loss, featureGen_vgg, featureGen_DarkNet
from utils.Hyper_parameter import Hyperparams as hp
from utils.show_img import show_img, show_state
from checkCrash.CrashChecker import CrashChecker
from datetime import datetime


class LayoutTransformer_vgg(GlobalVgg16Cnn):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 configPath: str = "../checkCrash/config.props",
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
        super(LayoutTransformer_vgg, self).__init__(image_width=image_width,
                                                    image_height=image_height,
                                                    furniture_nums=furniture_nums,
                                                    label_size=label_size,
                                                    furniture_width=furniture_width,
                                                    furniture_height=furniture_height,
                                                    furniture_embedding_size=furniture_embedding_size,
                                                    max_length=max_length,
                                                    )

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
        self.use_cnn_reshape = False
        self.use_cnn = hp.use_cnn
        # 不使用cnn(vgg)时,指定use_DarkNet才会成功
        # 启用需要：self.use_cnn = False,self.use_DarkNet = True
        self.use_DarkNet = hp.use_DarkNet

        # 统计信息
        self.count_info = {}
        self.count_info["num_info"] = 0
        self.count_info["not_label_and_crash_to_label"] = 0
        self.count_info["label_but_crash"] = 0

        self.loadCrashChecker(configPath=configPath)

    def build_attention_graph_tensor(self,
                                     target, room="", furniture="",
                                     trainable: bool = True, ):

        with tf.variable_scope("get_data", reuse=tf.AUTO_REUSE):
            if self.use_cnn:

                self.room_context = self.middle_state_cnn_feature
                self.furniture_fea = self.furniture_cnn_feature
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

                self.furniture_fea = furniture
                self.furniture_fea = embedding(self.furniture_fea,
                                               vocab_size=5000,
                                               num_units=self.ebd_size,
                                               zero_pad=True,  # 让padding一直是0
                                               scale=True,
                                               scope="furniture_fea_embed")
                self.furniture_fea = tf.layers.dropout(self.furniture_fea, rate=self.dropout_rate,
                                                       training=tf.convert_to_tensor(trainable))
                if self.use_DarkNet:
                    print("use_DarkNet", self.room_context)
                    _, self.furniture_fea, self.room_context = featureGen_DarkNet(
                        self.room_context, self.furniture_fea, "room_context_Dark_net")

                self.room_context = tf.layers.flatten(self.room_context)
                self.furniture_fea = tf.layers.flatten(self.furniture_fea)

            self.y = target
            self.target = target
            self.cnn_out = self.cnn_output_distribute

            # self.room_context = tf.Print(self.room_context, [self.room_context], "room_context")
            # self.furniture_fea = tf.Print(self.furniture_fea, [self.furniture_fea], "furniture_fea")

        with tf.variable_scope("encoder"):

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

                    # self.enc, l2_loss_enc_1 = denseWithL2loss(self.enc, 4 * self.hidden_units, scope="enc_shape_1",
                    #                                           activation=tf.nn.relu,use_bias=False)
                    # self.enc, l2_loss_enc_2 = denseWithL2loss(self.enc, 2 * self.hidden_units, scope="enc_shape_2",
                    #                                           activation=tf.nn.relu,use_bias=False)
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

            self.cnn_out_ = tf.reshape(self.cnn_out, [-1, self.max_length, self.label_size])
            if self.use_cnn_predict_predict:
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

            self.l2_loss_attention = self.predict_l2_loss + l2_loss_enc_3  # l2_loss_enc_1 + l2_loss_enc_2

            self.attention_loss = tf.reduce_sum(attention_loss_ * istarget) / (tf.reduce_sum(istarget))

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # 暂时填充用0 todo：删除在代码中
            self.attention_acc_topk = self.attention_acc
            self.preds_topk = self.pred

            if trainable:

                if self.use_cnn_predict_predict:
                    self.attention_loss = self.attention_loss + self.l2_loss_attention * 1e-3
                else:
                    self.attention_loss = self.attention_loss + self.l2_loss_attention * 1e-3

                train_vars = tf.trainable_variables()

                decay_rate = 0.98
                global_steps = self.global_step
                decay_steps = 2000

                c = tf.maximum(0.0001,
                               tf.train.exponential_decay(hp.lr, global_steps, decay_steps, decay_rate, staircase=True))
                self.lr_decay = c

                # self.attention_optimizer = tf.train.AdamOptimizer(learning_rate=c, beta1=0.9, beta2=0.98, epsilon=1e-8, )
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

    def predict_route(self, room_json: str, top_k: int, room_type_id: int):
        pred, pred_topk = self.hidden_predict_func(roomJson=room_json, top_k=top_k, roomTypeId=room_type_id)
        pred_topk_dst = []
        for i in range(top_k):
            pred_topk_ds = []
            for pre in pred_topk[i]:
                pred_topk_ds.append(pre - 1)
            pred_topk_dst.append(pred_topk_ds)
        return pred_topk_dst

    def hidden_predict_func(self, roomJson, top_k, roomTypeId=1):

        sess = self.sess
        max_length = self.max_length
        ck = self.clash_checker
        ck.init_room(roomTypeId=roomTypeId, room_str=roomJson)
        st_state = ck.roomMats[0].tolist()  # 0 代表 zid
        furniture_mats = []
        assert len(ck.furniture_mats) > 0  # 要布局的家具数需要大于0
        shp = np.shape(ck.furniture_mats[0][0])  ## 0 代表 zid
        for i in range(max_length):

            if i < len(ck.furniture_mats) - 1:
                furniture_mats.append(ck.furniture_mats[i][0].tolist())
            else:
                furniture_mats.append(np.zeros(shape=shp))

        data_infos = ck.data_infos
        temp_all_ready_furnitures = np.zeros_like([st_state] * max_length * top_k)
        temp_all_ready_furnitures = temp_all_ready_furnitures.tolist()
        temp_targets = [0] * max_length * top_k
        # print(np.shape(temp_targets), "temp_targets")
        # print(np.shape(temp_all_ready_furnitures), "temp_all_ready_furnitures")
        length_no_zero = len([x for x in data_infos if x["zid"] > 0.1])
        pred_ = []

        for furniture_id in range(length_no_zero):

            if furniture_id == 0:
                temp_all_ready_furnitures[furniture_id] = st_state
                for topn in range(top_k):
                    ind = furniture_id + max_length * topn
                    # print("ind::", ind, np.shape(st_state), np.shape(temp_all_ready_furnitures))
                    temp_all_ready_furnitures[ind] = st_state
                    temp_targets[ind] = 0
            else:
                for topn in range(top_k):

                    ind = furniture_id + max_length * topn
                    for __ix in range(furniture_id):
                        temp_targets[__ix + max_length * topn] = preds[topn][__ix]

                    # print(furniture_id, temp_targets,"furniture_id, temp_targets")

                    for __ix in range(furniture_id + 1):
                        # print("__ix", __ix, "topn", topn)
                        # print(__ix + max_length * topn, "__ix + max_length * topn")
                        # print("shape:中间状态{},shape：中间状态填充:{}".format(np.shape(mid_states),
                        #                                             np.shape(temp_all_ready_furnitures)))
                        temp_all_ready_furnitures[__ix + max_length * topn] = mid_states[topn][__ix]
                # if furniture_id == length_no_zero - 1:
                #     for ix, x in enumerate(temp_all_ready_furnitures):
                #         print("验证中间状态:",ix)
                #         print(preds)
                #
                #         show_img(temp_all_ready_furnitures[ix],title="No.{} mid state".format(ix))

            feed_dict = {
                self.furniture: furniture_mats * top_k,
                self.room: temp_all_ready_furnitures,
                self.target: temp_targets,
            }
            pred, logits = sess.run([self.pred, self.logits], feed_dict=feed_dict)
            if furniture_id == 0:
                preds = []
                scores = []
                out_ids = []
                mid_states = [st_state] * top_k
                # print(top_k, "top_k", np.shape(mid_states))

            ori_shape = np.shape(logits)
            now_shape = list(ori_shape)
            now_shape.insert(0, 1)
            logits = np.reshape(logits, newshape=now_shape)

            ori_shape = np.shape(pred)
            now_shape = list(ori_shape)
            now_shape.insert(0, 1)
            pred = pred.reshape(now_shape)

            temp_log = np.reshape(logits, newshape=[top_k, max_length, -1])
            pred = np.reshape(pred, newshape=[top_k, -1])

            if furniture_id < length_no_zero:
                if isinstance(preds, list):
                    pass
                else:
                    preds = preds.tolist()
            preds, scores, out_ids, mid_states = self.get_mid_beamsearch_add_check_Crash(temp_log, top_k, furniture_id,
                                                                                         preds, scores,
                                                                                         max_length=max_length,
                                                                                         debug=False, start_ind=0,
                                                                                         ck=ck, out_ids=out_ids,
                                                                                         add_check_Crash=True,
                                                                                         states=mid_states)
            pred_.append(pred[:, furniture_id][0])

        return pred_, preds

    def get_mid_beamsearch_add_check_Crash(self, logits, topk, i, preds=[], scores=[], \
                                           out_ids=[], states=[], \
                                           max_length=5, debug=False, start_ind=0, \
                                           add_check_Crash=True, ck='', target=[]):
        '''
        含有碰撞检测的beamsearch
        '''
        # 碰撞检测统计信息
        count_info = self.count_info
        count_info["num_info"] += 1

        logits_shape = list(np.shape(logits))
        logits_shape_1 = logits_shape[0]
        logits_shape_2 = logits_shape[1]
        if logits_shape_2 != max_length:
            logits_shape[1] = int(max_length)
            logits_shape[0] = int(logits_shape_1 * (logits_shape_2 / max_length))

        logits = np.reshape(logits, newshape=logits_shape)
        logits = logits[:, i, :]

        # 预测结果 topk 个 + extends 个结果
        topk_extend = 40
        arg_topk = np.argsort(-logits, axis=-1, )[:, 0:(topk_extend + topk)]

        topk_new = topk + topk_extend

        # 保留原始的数据,以便碰撞检测都不通过时保留原来通过的结果
        ori_preds = copy.deepcopy(preds)
        ori_scores = copy.deepcopy(scores)
        ori_out_ids = copy.deepcopy(out_ids)
        ori_out_states = copy.deepcopy(states)
        image_grid_num = 64
        _pred = []
        _scores = []
        _mid_states = []
        # 记录之前的ind
        if debug:
            print(np.shape(states), "states shape", type(states), np.shape(preds))

        _in_indexs = []
        _out_indexs = []
        if i == start_ind:
            _in_inndexs = []
            for j in range(topk_new):
                _in_inndexs.append([-1])
                _out_indexs.append([-1])
                _pred.append([arg_topk[0][j]])
                if j < topk_new and add_check_Crash:
                    _mid_states.append([states[0]])
                _scores.append(logits[0][arg_topk[0][j]])
            if debug:
                print(("第一次：_pid,_mid_states", _pred, np.shape(_mid_states)))
        else:
            for j in range(len(preds)):
                for k in range(len(arg_topk[j])):
                    _pred.append(preds[j] + [arg_topk[j][k]])
                    _in_indexs.append(out_ids[j])
                    _scores.append(scores[j] + logits[j][arg_topk[j][k]])
                    if add_check_Crash:
                        _mid_states.append(states[j])

            _scores = np.array(_scores)
            _arg_topk = np.argsort(-_scores, axis=-1, )[0:topk_new]
            _pred = [_pred[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
            _in_indexs = [_in_indexs[k] for k in _arg_topk]
            if add_check_Crash:
                _mid_states = [_mid_states[k] for k in _arg_topk]
            if debug:
                print(("第{0}次,_pred:{1},scores:{2},_in_indexs,_mid_states{3}".format(i, _pred, _scores, _in_indexs, )))

        _mid_states = np.array(_mid_states, dtype=np.int32).tolist()

        preds = []
        scores = []
        out_ids = []
        states = []
        ck_num = 0
        all_num = 0

        ind = -1
        ind_list = []
        for k in range(topk_new):
            # 进行碰撞检测 筛选不碰撞的结果
            if len(preds) >= topk and all_num >= topk:
                if debug:
                    print("已经找到了topk个,停止碰撞检测", topk, np.shape(states), np.shape(preds), k)
                    print(preds)
                break
            else:
                all_num += 1
                pred_now = {}
                label = int(_pred[k][-1]) - 1
                pred_now["label"] = label

                if add_check_Crash:

                    if i == start_ind:

                        in_ind_ = 0
                        ret, hitMask = ck.detect_crash_label(pred_now, index=in_ind_, tag=i)
                    else:
                        in_ind_ = _in_indexs[k]
                        ret, hitMask = ck.detect_crash_label(pred_now, index=in_ind_, tag=i)
                    if ret == 0:
                        if target:
                            if k > topk and target[0] == label + 1:
                                count_info["not_label_and_crash_to_label"] += 1

                        ind_list.append(k)
                        ind += 1
                        out_ids.append(ind)
                        preds.append(_pred[k])
                        scores.append(_scores[k])
                        mask_data = ck.buildEmbeddingMat(hitMask, numSegs=image_grid_num)
                        states.append(_mid_states[k] + [(np.array(mask_data[0], dtype=np.int32)).tolist()])  # 0 表示zid

                        if debug:
                            print("input:{}".format(pred_now))
                            print("第{0}个家具检测：当前无碰撞,out_ind:{1},in_ind:{2},input{3}".format(i, ind, in_ind_,
                                                                                           pred_now))
                            show_img(mask_data[0],
                                     "NO.{0} funtinue check out_ind:{1},in_ind:{2}".format(i, ind,
                                                                                           in_ind_))
                        show_state(mask_data[0],
                                   "{3}NO{0} funtinue check out_ind{1}in_ind{2}".format(i, ind,
                                                                                        in_ind_,
                                                                                        str(datetime.now())[:12]),
                                   save=True)

                    else:
                        if debug:
                            print("碰撞返回值:", ret)
                        if target:
                            if int(_pred[k][-1]) == target[0]:
                                count_info["label_but_crash"] += 1
                        ck_num += 1
                else:

                    out_ids.append(ind)
                    preds.append(_pred[k])
                    scores.append(_scores[k])
        if len(preds) == 0:
            n_preds = []
            n_scores = []
            n_out_ids = []
            n_out_states = []
            for k, p in enumerate(ori_preds):
                p.append(0)
                n_preds.append(p)
                n_scores.append(ori_scores[k])
                n_out_ids.append(ori_out_ids[k])
                try:
                    n_out_states.append(ori_out_states[k])
                except Exception as e:
                    print(e)
                    pass
            if i == max_length - 1:
                if add_check_Crash:
                    ck.step_finash()
            preds = n_preds
            scores = n_scores
            out_ids = n_out_ids
            states = n_out_states

        else:
            if add_check_Crash:
                ck.step_finash()
        if debug:
            print("选择的列表：", ind_list)
            print(("当前的有：", preds, scores))
            print("beam_search_return", preds, scores, out_ids, i, start_ind)
            print("碰撞次数:", ck_num)
            print("总次数:", )

        return preds, scores, out_ids, states

    def loadCrashChecker(self, configPath):
        self.clash_checker = CrashChecker(configPath=configPath)
