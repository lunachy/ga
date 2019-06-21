# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    predict_helper
    Description :    
    Author      :    zhaowen
    date        :    2019/4/25
____________________________________________________________________
    Change Activity:
                        2019/4/25:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'
import time
import numpy as np
from utils.transformer import GenPng_numpy
from utils.Hyper_parameter import Hyperparams as hp
from utils.metric import get_mid_beamsearch


def hidden_predict_func(batch_rooms, batch_furnitures, batch_furniture_dxs, batch_furniture_dys, \
                        batch_furniture_cids, batch_targets, batch_room_context, tensor_layout_model, sess, topk,
                        check_crash_time, type_predict="att", Hprarms=hp):
    '''

    :param batch_rooms:
    :param batch_furnitures:
    :param batch_furniture_dxs:
    :param batch_furniture_dys:
    :param batch_furniture_cids:
    :param batch_targets:
    :param batch_room_context:
    :param tensor_layout_model:
    :param sess:
    :param topk:
    :param check_crash_time:
    :param type_predict:
    :param max_length:
    :return:
    '''
    print(type_predict)
    print("--debug: type_predict", type_predict)
    hp = Hprarms
    topk_max = int(max(topk))

    temp_all_ready_furnitures = np.zeros_like(batch_room_context * topk_max)
    temp_all_ready_furnitures = temp_all_ready_furnitures.tolist()
    temp_targets = np.zeros_like(batch_targets * topk_max)
    temp_targets = temp_targets.tolist()

    length_no_zero = len([x for x in batch_furniture_cids if x > 0.1])

    pred_ = []
    pred_cnn_ = []
    max_length = hp.max_length
    for furniture_id in range(hp.max_length):

        if furniture_id >= length_no_zero:
            pred_.append(0)
            pred_cnn_.append(0)
            continue

        if furniture_id == 0 or int(batch_furniture_cids[furniture_id]) == 0:

            temp_all_ready_furnitures[furniture_id] = batch_room_context[furniture_id]
            for topn in range(topk_max):
                temp_all_ready_furnitures[
                    furniture_id + max_length * topn] = batch_room_context[furniture_id]
        else:
            for topn in range(topk_max):
                ind = furniture_id + max_length * topn

                preds = np.array(preds)

                temp_all_ready_furnitures[ind] = GenPng_numpy(room_ori=temp_all_ready_furnitures[ind - 1],
                                                              furniture_dx=(batch_furniture_dxs * topk_max)[
                                                                  ind - 1],
                                                              furniture_dy=(batch_furniture_dys * topk_max)[
                                                                  ind - 1],
                                                              furniture_value=int(preds[:, furniture_id - 1][topn]), \
                                                              length=hp.length, m=hp.label_grid, n=hp.label_grid,
                                                              r=hp.angle,
                                                              w=hp.furniture_width, h=hp.furniture_height,
                                                              furniture_cid=(batch_furniture_cids * topk_max)[
                                                                  ind - 1])

        feed_dict = {
            tensor_layout_model.furniture:
                batch_furnitures * topk_max,
            tensor_layout_model.room:
                temp_all_ready_furnitures,
            tensor_layout_model.target:
                temp_targets,
            tensor_layout_model.furniture_cids:
                batch_furniture_cids * topk_max,
        }
        if type_predict == "att":

            pred, logits = sess.run([
                tensor_layout_model.pred,
                tensor_layout_model.logits,
            ],
                feed_dict=feed_dict)
        elif type_predict == "cnn":
            pred, logits = sess.run([
                tensor_layout_model.cnn_output_distribute,
                tensor_layout_model.cnn_logits,
            ],
                feed_dict=feed_dict)
            pred = np.argmax(a=pred, axis=-1)

        if furniture_id == 0:
            preds = []
            scores = []

        ori_shape = np.shape(logits)
        now_shape = list(ori_shape)
        now_shape.insert(0, 1)
        logits = np.reshape(logits, newshape=now_shape)

        ori_shape = np.shape(pred)
        now_shape = list(ori_shape)
        now_shape.insert(0, 1)
        pred = pred.reshape(now_shape)



        now_shape = list(ori_shape)
        now_shape.insert(0, 1)


        topk_max = max(topk)

        temp_log = np.reshape(logits, newshape=[topk_max, max_length, -1])
        pred = np.reshape(pred, newshape=[topk_max, -1])


        if furniture_id < length_no_zero:
            if isinstance(preds, list):
                pass
            else:
                preds = preds.tolist()
            preds, scores = get_mid_beamsearch(
                temp_log,
                topk_max,
                furniture_id,
                preds,
                scores,
                max_length=max_length,
                debug=False,
            )

            e_st = time.clock()

            # todo: use_checkCrash
            # 使用CheckCrash进行检测

            d_st = time.clock()
            check_crash_time += (d_st - e_st)

        pred_.append(pred[:, furniture_id][0])
    if len(preds[0]) < max_length:
        preds_pad = []
        for p in preds:
            pad_length = max_length - len(p)
            p = p + [0] * pad_length
            preds_pad.append(p)
        preds = preds_pad

    return pred_, preds


def predict_for_use(datasets,
                    tensor_layout_model,
                    sess,
                    max_length=5,
                    topk=[8],
                    use_checkCrash=False,
                    type_predict="att", Hprarms=hp):
    # 碰撞检测使用的时间
    check_crash_time = 0

    # 记录用时
    st_a = time.clock()

    batch_rooms, batch_furnitures, batch_furniture_dxs, batch_furniture_dys, \
    batch_furniture_cids, batch_targets, batch_room_context = datasets["rooms"], datasets["furnitures"], datasets[
        "dxs"], datasets["dys"], datasets["cids"], datasets["targets"], datasets["room_context"]

    pred_, preds_pad, = hidden_predict_func(batch_rooms, batch_furnitures, batch_furniture_dxs,
                                            batch_furniture_dys, \
                                            batch_furniture_cids, batch_targets, batch_room_context,
                                            tensor_layout_model, sess, topk, check_crash_time,
                                            type_predict=type_predict, Hprarms=Hprarms, )
    st_b = time.clock()

    preds = preds_pad

    pred = pred_
    return pred, preds
