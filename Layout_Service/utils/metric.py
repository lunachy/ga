# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    metric
    Description :    
    Author      :    zhaowen
    date        :    2019/4/9
____________________________________________________________________
    Change Activity:
                        2019/4/9:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'
import numpy as np
import tensorflow as tf
import copy
import json
import time
import os
from checkCrash.Crash_check import returnEmbeddingMask, get_cidlist_fromjson, crash_Check
from utils.show_img import show_img
from tqdm import tqdm
from utils.Hyper_parameter import Hyperparams as hp
from data.StateFurnitureDataSet import SingleGlobalStateFurnitureDataSets

hp.max_length = 8
hp.maxlen = 8
hp.image_width = 64
hp.image_height = 64
hp.use_cnn_loss = False
hp.use_cnn = False
hp.use_cnn_predict_predict = False
hp.furniture_width = 64
hp.furniture_height = 64

# 统计是label却被检测碰撞的个数
count_info = {}
# 是label 但是却被检测碰撞了
count_info["label_but_crash"] = 0
count_info["label_but_crash_tag"] = []
# 不是label 但是由于碰撞检测的原因获取到了label的个数
count_info["not_label_and_crash_to_label"] = 0
count_info["not_label_and_crash_to_label_tag"] = []
# 检测的label个数
count_info["num_info"] = 0




def get_mid_beamsearch(logits, topk, i, preds=[], scores=[], max_length=5, debug=False, start_ind=0):
    logits_shape = list(np.shape(logits))
    logits_shape_1 = logits_shape[0]
    logits_shape_2 = logits_shape[1]
    if logits_shape_2 != max_length:
        logits_shape[1] = int(max_length)
        logits_shape[0] = int(logits_shape_1 * (logits_shape_2 / max_length))

    logits = np.reshape(logits, newshape=logits_shape)
    logits = logits[:, i, :]
    arg_topk = np.argsort(-logits, axis=-1, )[:, 0:topk]
    if debug:
        print("预测中间k", arg_topk, i)

    _pred = []
    _scores = []
    if i == start_ind:
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


def get_accu(pred, batch_targets, batch_size, maxlength=5, supervision=0, get_num=False):
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
            if i % maxlength < supervision:
                continue

            if real[i] == pred[i]:
                right += 1
                all_right += 1
            else:
                wrong += 1
                all_wrong += 1

    acc_ = right / (right + wrong + 1e-4)
    # print(whole_right,whole_wrong)
    whole_acc = whole_right / (whole_right + whole_wrong + 1e-4)
    if get_num:
        return {"right_num": right, "wrong_num": wrong, "whole_right": whole_right, "whole_wrong": whole_wrong}
    else:
        return acc_, whole_acc


def get_accu_topk(pred, batch_targets, topk=3, maxlength=5, supervision=0, get_num=False):
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
    # 注意每次只能一个户型的数据
    assert topk == np.shape(pred)[1]
    assert np.shape(pred)[0] == np.shape(batch_targets)[0]
    assert np.shape(pred)[2] == np.shape(batch_targets)[1]

    num = 0
    whole_right_num = 0
    whole_wrong_num = 0
    acc_whole_num = 0
    for batch in range(np.shape(pred)[0]):
        pred_b = pred[batch]
        batch_targets_b = batch_targets[batch]
        num += 1

        for top in range(topk):
            pred_1 = pred_b[top]
            if get_num:
                info = get_accu(pred=pred_1, batch_targets=batch_targets_b, batch_size=maxlength,
                                maxlength=maxlength, supervision=supervision, get_num=True)
                whole_right_num += info["whole_right"]
                whole_wrong_num += info["whole_wrong"]
            else:
                acc, acc_whole = get_accu(pred=pred_1, batch_targets=batch_targets_b, batch_size=maxlength,
                                          maxlength=maxlength, supervision=supervision, get_num=False)
                if acc_whole > 0:
                    acc_whole_num += 1
                    break
    acc_whole_topk = acc_whole_num / num
    if get_num:
        info = {}
        if whole_right_num > 0:

            info["whole_right"] = 1
            info["whole_wrong"] = 0
        elif whole_right_num == whole_wrong_num and whole_right_num == 0:
            info["whole_right"] = 0
            info["whole_wrong"] = 0
        else:
            info["whole_right"] = 0
            info["whole_wrong"] = 1
        return info

    return acc_whole_topk


def get_accu_(pred, batch_targets, batch_size, maxlength=5):
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


def get_accu_topk_(pred, batch_targets, topk=3, maxlength=5):
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


def get_mid_beamsearch_add_check_Crash(logits, topk, i, preds=[], scores=[], \
                                       out_ids=[], states=[], \
                                       max_length=5, debug=False, start_ind=0, \
                                       add_check_Crash=False, ck='', info={},
                                       result_all=[], target=[]):
    '''
    含有碰撞检测的beamsearch
    '''
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

    image_grid_num = 64

    results = []
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
            if j < topk and add_check_Crash:
                _mid_states.append([states[j]])
            _scores.append(logits[0][arg_topk[0][j]])
        if debug:
            print(("第一次：_yid,_scores,_mid_states", _pred, _scores, np.shape(_mid_states)))
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
    if add_check_Crash:
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
        result = {}
        # 进行碰撞检测 筛选不碰撞的结果
        if len(preds) >= topk:
            if debug:
                print("已经找到了topk个,停止碰撞检测", topk)
            break
        else:
            all_num += 1
            pred_now = {}
            pred_now["zid"] = int(info["zid"])
            label = int(_pred[k][-1]) - 1
            pred_now["label"] = label

            if add_check_Crash:

                if i == start_ind:

                    in_ind_ = -1
                    ret, hitMask = ck.detect_crash_label(pred_now, index=in_ind_, tag=info["tag"])
                else:
                    in_ind_ = _in_indexs[k]
                    ret, hitMask = ck.detect_crash_label(pred_now, index=in_ind_, tag=info["tag"])

                if ret == 0:
                    if k > topk and target[0] == label + 1:
                        count_info["not_label_and_crash_to_label"] += 1
                        count_info["not_label_and_crash_to_label_tag"].append(info)

                    ind_list.append(k)

                    ind += 1

                    out_ids.append(ind)
                    preds.append(_pred[k])
                    scores.append(_scores[k])
                    mask_data = returnEmbeddingMask(hitMask, image_grid_size=image_grid_num)
                    t_emp = [(np.array(mask_data["zid"], dtype=np.int32) + 1).tolist()]
                    states.append(_mid_states[k] + [(np.array(mask_data["zid"], dtype=np.int32) + 1).tolist()])

                    if debug:
                        print("input:{}".format(pred_now))
                        print("第{0}个家具检测：当前无碰撞,out_ind:{1},in_ind:{2},input{3}".format(i, ind, in_ind_,
                                                                                       pred_now))
                #                         show_img(mask_data["zid"],
                #                                  "NO.{0} funtinue check out_ind:{1},in_ind:{2}".format(i, ind,
                #                                                                                        in_ind_))
                else:
                    if target:
                        if int(_pred[k][-1]) == target[0]:
                            count_info["label_but_crash"] += 1
                            count_info["label_but_crash_tag"].append(info)

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

        if i == max_length - 1:
            if add_check_Crash:
                ck.step_finash()
        preds = n_preds
        scores = n_scores
        out_ids = n_out_ids

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


def hidden_predict_func(batch_rooms, batch_furnitures, batch_furniture_dxs, batch_furniture_dys, \
                        batch_furniture_cids, batch_targets, batch_room_context, batch_room_json, batch_zid_tags, \
                        tensor_layout_model, sess, topk, check_crash_time, \
                        type_predict="correct", max_length=5, room_id=0, just_cnn=False, use_Check_Crash=False):
    topk_max = int(max(topk))

    json_data = batch_room_json[0]
    furnist = get_cidlist_fromjson(json_data)
    zid_tags = batch_zid_tags  # get_zoneTag_fromjson(json_data)
    room_str = json.dumps(json_data, ensure_ascii=False)
    label_grid_num = 16
    image_grid_num = 64
    get_hitMask_grid_num = 32
    objGranularity = 100
    max_room_size = 8000


    topk = topk_max
    if use_Check_Crash:
        ck = crash_Check(model_dir='', room_str=room_str, furniturelist=furnist, label_grid_num=label_grid_num,
                         image_grid_num=image_grid_num,
                         max_room_size=max_room_size, objGranularity=objGranularity,
                         get_hitMask_grid_num=get_hitMask_grid_num, topk=topk)
    else:
        ck = ''

    try:

        ind = -1

        if type_predict == "correct":
            temp_all_ready_furnitures = np.zeros_like(batch_room_context)
            temp_all_ready_furnitures = temp_all_ready_furnitures.tolist()
            temp_targets = np.zeros_like(batch_targets)
            temp_targets = temp_targets.tolist()
        elif type_predict == "predict":

            temp_all_ready_furnitures = np.zeros_like(batch_room_context * topk_max)
            temp_all_ready_furnitures = temp_all_ready_furnitures.tolist()
            temp_targets = np.zeros_like(batch_targets * topk_max)
            temp_targets = temp_targets.tolist()

        length_no_zero = len([x for x in batch_furniture_cids if x > 0.1])
        length_zero = max_length - length_no_zero

        pred_ = []
        pred_cnn_ = []
        for furniture_id in range(max_length):

            if furniture_id >= length_no_zero:
                pred_.append(0)
                pred_cnn_.append(0)
                continue
            # 用真实值填充进去
            if type_predict == "correct":
                temp_all_ready_furnitures[furniture_id] = batch_room_context[
                    furniture_id]
                temp_targets[furniture_id] = batch_targets[furniture_id]
                feed_dict = {
                    tensor_layout_model.furniture: batch_furnitures,
                    tensor_layout_model.room: temp_all_ready_furnitures,
                    tensor_layout_model.target: temp_targets,
                    tensor_layout_model.furniture_cids: batch_furniture_cids,
                }
                if not just_cnn:
                    pred, accu, accu_k, logits, pred_cnn = sess.run([tensor_layout_model.pred, \
                                                                     tensor_layout_model.attention_acc, \
                                                                     tensor_layout_model.attention_acc_topk,
                                                                     tensor_layout_model.logits,
                                                                     tensor_layout_model.cnn_output_distribute],
                                                                    feed_dict=feed_dict)
                    print(np.shape(logits), "logits")
                else:
                    pred, accu, accu_k, logits, pred_cnn = sess.run([tensor_layout_model.pred, \
                                                                     tensor_layout_model.attention_acc, \
                                                                     tensor_layout_model.attention_acc_topk,
                                                                     tensor_layout_model.cnn_logits,
                                                                     tensor_layout_model.cnn_output_distribute],
                                                                    feed_dict=feed_dict)
                    print(np.shape(logits), "logits")

            elif type_predict == "predict":
                if furniture_id == 0 or int(batch_furniture_cids[furniture_id]) == 0:
                    print(np.shape(batch_room_context), furniture_id)

                    temp_all_ready_furnitures[furniture_id] = batch_room_context[furniture_id]
                    for topn in range(topk_max):
                        ind = furniture_id + max_length * topn
                        temp_all_ready_furnitures[ind
                        ] = batch_room_context[furniture_id]



                else:

                    for topn in range(topk_max):
                        ind = furniture_id + max_length * topn
                        # preds = np.array(preds)

                        #                     print(preds,"preds")
                        #                     print(np.shape(preds))
                        #                     print(type(mid_states))

                        #                     print(np.shape(mid_states))
                        #                     print(np.shape(temp_all_ready_furnitures))
                        # print("预测的长度与中间状态图的信息：",np.shape(preds),np.shape(mid_states),np.shape(temp_all_ready_furnitures))

                        for __ix in range(furniture_id):
                            print("ind-__ix", ind - __ix, __ix, "__ix", "topn", topn)
                            try:
                                temp_targets[__ix + max_length * topn] = preds[topn][__ix]
                            except Exception as e:
                                print("furniture_id:{}".format(furniture_id))
                                print("error:{},preds:{},batch_furniture_dxs:{},\
                                batch_furniture_dys:{},batch_targets:{},shape_batch_furnitures:{},\
                                shape_batch_room_context:{},batch_furniture_cids:{}" \
                                      .format(e, preds, batch_furniture_dxs,
                                              batch_furniture_dys, batch_targets,
                                              np.shape(batch_furnitures),
                                              np.shape(batch_room_context),
                                              batch_furniture_cids))
                                raise (e)

                        for __ix in range(furniture_id + 1):
                            temp_all_ready_furnitures[__ix + max_length * topn] = mid_states[topn][__ix]

                    if furniture_id == length_no_zero - 1:
                        for ix, x in enumerate(temp_all_ready_furnitures):
                            #                         print("验证中间状态:",ix)
                            #                         print(preds)

                            # show_img(temp_all_ready_furnitures[ix],title="No.{} mid state".format(ix))
                            pass

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
                if just_cnn:
                    pred, accu, accu_k, logits, pred_cnn = sess.run([
                        tensor_layout_model.pred, tensor_layout_model.attention_acc,
                        tensor_layout_model.attention_acc_topk,
                        tensor_layout_model.cnn_logits,  # tensor_layout_model.logits,
                        tensor_layout_model.cnn_output_distribute,
                    ],
                        feed_dict=feed_dict)
                else:
                    pred, accu, accu_k, logits, pred_cnn = sess.run([
                        tensor_layout_model.pred, tensor_layout_model.attention_acc,
                        tensor_layout_model.attention_acc_topk,
                        tensor_layout_model.logits,  # tensor_layout_model.logits,
                        tensor_layout_model.cnn_output_distribute,
                    ],
                        feed_dict=feed_dict)

            if furniture_id == 0:
                preds = []
                scores = []
                out_ids = []
                mid_states = [batch_room_context[0]] * topk_max
                print("最开始中间状态的形状", np.shape(mid_states))

            print(type_predict, "add", use_Check_Crash)
            ori_shape = np.shape(logits)
            now_shape = list(ori_shape)
            now_shape.insert(0, 1)
            logits = np.reshape(logits, newshape=now_shape)

            ori_shape = np.shape(pred)
            now_shape = list(ori_shape)
            now_shape.insert(0, 1)
            pred = pred.reshape(now_shape)

            pred_cnn = np.argmax(a=pred_cnn, axis=-1)
            ori_shape = np.shape(pred_cnn)
            now_shape = list(ori_shape)
            now_shape.insert(0, 1)
            pred_cnn = pred_cnn.reshape(now_shape)

            if type_predict == "correct":
                temp_log = np.repeat(a=logits, axis=0, repeats=topk_max)
            elif type_predict == "predict":
                temp_log = np.reshape(logits, newshape=[topk_max, max_length, -1])
                pred = np.reshape(pred, newshape=[topk_max, -1])
                pred_cnn = np.reshape(pred_cnn, newshape=[topk_max, -1])

            if furniture_id < length_no_zero:
                if isinstance(preds, list):
                    pass
                else:
                    preds = preds.tolist()

                preds, scores, out_ids, mid_states = get_mid_beamsearch_add_check_Crash(
                    temp_log,
                    topk_max,
                    furniture_id,
                    preds,
                    scores,
                    max_length=max_length,
                    debug=True, start_ind=0, ck=ck, out_ids=out_ids, add_check_Crash=use_Check_Crash, states=mid_states,
                    target=[batch_targets[furniture_id]],
                    info={"zid": batch_furniture_cids[furniture_id], "tag": zid_tags[furniture_id]}

                )

                e_st = time.clock()

                # todo: use_checkCrash
                # 使用CheckCrash进行检测

                d_st = time.clock()
                check_crash_time += (d_st - e_st)

            pred_.append(pred[:, furniture_id][0])
            pred_cnn_.append(pred_cnn[:, furniture_id][0])

        if len(preds[0]) < max_length:
            preds_pad = []
            for p in preds:
                pad_length = max_length - len(p)
                p = p + [0] * pad_length
                preds_pad.append(p)
            preds = preds_pad
        if use_Check_Crash:
            ck.room_finish()
    except Exception as e:
        if use_Check_Crash:
            ck.room_finish()
        raise (e)
    return pred_, preds, pred_cnn_


def predict_By_correct(eval_datasets,
                       tensor_layout_model,
                       sess,
                       max_length=5,
                       topk=[8],
                       use_single=False,
                       use_Check_Crash=False,
                       type_predict="correct",
                       just_cnn=False):
    '''
    用布局正确的上下文信息预测
    :param eval_datasets:
    :param tensor_layout_model:
    :param sess:
    :param max_length:
    :param topk:
    :param use_single:
    :return:
    '''

    # 整条路径对的个数
    whole_right = 0
    whole_right_cnn = 0
    whole_wrong = 0
    whole_wrong_cnn = 0

    # 正确布局结果的个数
    right_num = 0
    right_num_cnn = 0
    error_num = 0
    error_num_cnn = 0

    # 存储topk 准确率的列表
    top_k_ac_all_right = [0] * len(topk)
    top_k_ac_all_wrong = [0] * len(topk)

    # 一个户型一个户型的预测
    batch_size = 1

    num = 0
    # 碰撞检测使用的时间
    check_crash_time = 0

    # 记录用时
    st_a = time.clock()
    error_num = 0
    n_temp = 0
    error_ids = []
    for room_id in tqdm(range(eval_datasets.train_nums // int(batch_size))):
        try:

            batch_furniture_cid_images, batch_furniture_zid_images, batch_state_cid_images, batch_state_zid_images, \
            batch_zids, batch_dxs, batch_dys, batch_labels, batch_room_jsons, batch_zid_tags = \
                eval_datasets.next_batch(batch_size=1, max_length=max_length, reverse=False, height=hp.image_width,
                                         width=hp.image_width
                                         )
            n_temp += 1

            print("-" * 30)
            print("room_id{}".format(room_id))
        except Exception as e:

            print("ERROR IN NEXT BANCH:", eval_model)
            error_ids.append(room_id)
            raise (e)
            continue

        json_data = batch_room_jsons[0]
        furnist = get_cidlist_fromjson(json_data)
        zid_tags = [x for x in batch_zid_tags if x != 0]
        if len(zid_tags) != len([x for x in batch_zids if x > 0.1]):
            print("ERROR tags 长度与 zid 长度不一致:", zid_tags, batch_zids)
            continue

        batch_rooms, batch_furnitures, batch_furniture_dxs, batch_furniture_dys, \
        batch_furniture_cids, batch_targets, batch_room_context = np.array(batch_state_zid_images) + 1, np.array(
            batch_furniture_zid_images) + 1, batch_dxs, batch_dys, batch_zids, batch_labels, np.array(
            batch_state_zid_images) + 1

        batch_rooms = batch_rooms.tolist()
        batch_room_context = batch_room_context.tolist()
        batch_furnitures = batch_furnitures.tolist()

        try:

            pred_, preds_pad, pred_cnn_ = hidden_predict_func(batch_rooms, batch_furnitures, batch_furniture_dxs,
                                                              batch_furniture_dys, \
                                                              batch_furniture_cids, batch_targets, batch_room_context,
                                                              batch_room_jsons, batch_zid_tags, tensor_layout_model, \
                                                              sess, topk, check_crash_time, type_predict=type_predict,
                                                              room_id=room_id, \
                                                              just_cnn=just_cnn, max_length=max_length,
                                                              use_Check_Crash=use_Check_Crash)
        except Exception as e:
            print("Error\n" * 20, e)
            error_num += 1
            if error_num > 100000:
                raise (e)
            error_ids.append(room_id)
            continue

        preds = preds_pad

        pred = pred_
        # pred = preds[0]

        info = get_accu(pred=pred,
                        batch_targets=batch_targets,
                        batch_size=5,
                        maxlength=5,
                        get_num=True)
        info_cnn = get_accu(pred=pred_cnn_,
                            batch_targets=batch_targets,
                            batch_size=5,
                            maxlength=5,
                            get_num=True)
        # print(pred,batch_targets,info)

        for i, top in enumerate(topk):
            preds_i = np.array(preds)
            preds_i = preds_i[:top, :]
            top_k_info = get_accu_topk(batch_targets=[batch_targets],
                                       maxlength=max_length,
                                       pred=[preds_i],
                                       topk=top,
                                       get_num=True)
            top_k_ac_all_right[i] += top_k_info["whole_right"]
            top_k_ac_all_wrong[i] += top_k_info["whole_wrong"]
        # print(info_cnn)
        # print("pred:{},real:{}".format(pred_cnn_,batch_targets))
        # print("预测：{0}，实际:{1},准确率:{2},整体准确率:{3},top_k_ac:{4}".format(pred,batch_targets,accu,whole_acc,top_k_ac))
        # break

        right_num += info["right_num"]
        error_num += info["wrong_num"]

        whole_right += info["whole_right"]
        whole_wrong += info["whole_wrong"]

        right_num_cnn += info_cnn["right_num"]
        error_num_cnn += info_cnn["wrong_num"]

        whole_right_cnn += info_cnn["whole_right"]
        whole_wrong_cnn += info_cnn["whole_wrong"]

    print("TOP1:准确率：", right_num / (right_num + error_num + 1e-8))
    print("TOP1:单条数据完全正确的准确率:",
          whole_right / (whole_right + whole_wrong + 1e-8))
    print("TOP1:准确率 CNN：",
          right_num_cnn / (right_num_cnn + error_num_cnn + 1e-8))
    print("TOP1:单条数据完全正确的准确率: cnn",
          whole_right_cnn / (whole_right_cnn + whole_wrong_cnn + 1e-8))
    ed_a = time.clock()
    print(check_crash_time, "碰撞检测使用时间")
    print(ed_a - st_a, "总共使用时间")

    print("topk完全正确的准确率:", [
        "  TOP:" + str(topk[i]) + " accu:" + str(x /
                                                 (x + top_k_ac_all_wrong[i]))
        for i, x in enumerate(top_k_ac_all_right)
    ])
    accu = right_num / (right_num + error_num + 1e-8)
    accu_whole = whole_right / (whole_right + whole_wrong + 1e-8)
    accu_topk_whole = [
        x / (x + top_k_ac_all_wrong[i] + 1e-8)
        for i, x in enumerate(top_k_ac_all_right)
    ]
    from pprint import pprint
    print("+" * 50)
    pprint(error_ids)

    return accu, accu_whole, accu_topk_whole


def eval_model(cnn_model_path="weights/vgg16_128_16_model/model-190",
               attention_model_path="weights/test28_emb/model-32",
               topk=[
                   3,
               ],
               model_class='',
               limit_num=1.0,
               type_predict="correct",
               use_single=True,
               just_cnn=False,
               use_seq=False, use_Check_Crash=False, test_files=["d:/workspace/64_test_ck.txt"]):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.reset_default_graph()

    image_width = hp.image_width
    image_height = hp.image_height
    furniture_nums = hp.furniture_nums
    label_size = hp.label_size
    furniture_width = hp.furniture_width
    furniture_height = hp.furniture_height
    furniture_embedding_size = hp.furniture_embedding_size
    max_length = hp.max_length
    keep_prob = 1.0

    datasets = SingleGlobalStateFurnitureDataSets(path_list=test_files,
                                                  top_path_list=[])

    eval_datasets = datasets

    eval_datasets.train_nums = int(eval_datasets.train_nums * limit_num)
    print(eval_datasets.train_nums, "eval_nums")

    transformer_layout_model = model_class(
        image_width=image_width,
        image_height=image_height,
        furniture_nums=furniture_nums,
        label_size=label_size,
        furniture_width=furniture_width,
        furniture_height=furniture_height,
        furniture_embedding_size=furniture_embedding_size,
        max_length=max_length)

    sess = transformer_layout_model.create_sess()

    transformer_layout_model.build_cnn_graph(
        trainable=False,
        filter_list=[64, 128, 256, 512, 512],
        normal_model="norm",
        keep_prob=keep_prob, mask=True)
    var_to_restore_cnn = tf.global_variables()
    transformer_layout_model.initializer(sess=sess)
    if cnn_model_path:
        transformer_layout_model.restore(model_path=cnn_model_path, sess=sess)

    if use_seq:
        transformer_layout_model.build_attention_graph(
            trainable=False, batch_size=tf.convert_to_tensor(max(topk)))
    else:
        transformer_layout_model.build_attention_graph(trainable=False, )
    vars = tf.global_variables()
    var_to_restore_rnn = [val for val in vars if val not in var_to_restore_cnn]
    var_to_init = var_to_restore_rnn
    sess.run(tf.variables_initializer(var_to_init))
    vars = tf.global_variables()
    var_to_restore_rnn = [val for val in vars if val not in var_to_restore_cnn]
    var_to_init = var_to_restore_rnn
    sess.run(tf.initialize_variables(var_to_init))
    print("attention 模型构建完成")
    print("--" * 20)
    if just_cnn:
        print("直接使用CNN 不使用ATT")
    else:
        transformer_layout_model.restore(sess=sess,
                                         model_path=attention_model_path)
        print("RESTORE OK!")
    print(transformer_layout_model.label_size, "tensor_layout_model")
    topk = topk
    if type_predict == "correct":
        print("有监督的布局结果测试：")
    elif type_predict == "predict":
        print("无监督的布局结果测试：")
    if use_single:
        print("单个布局")

        predict_By_correct(eval_datasets,
                           transformer_layout_model,
                           sess,
                           max_length=max_length,
                           topk=topk,
                           use_single=use_single,
                           type_predict=type_predict,
                           just_cnn=just_cnn, use_Check_Crash=use_Check_Crash)
