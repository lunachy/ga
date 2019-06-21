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


def get_mid_beamsearch(logits, topk, i, preds=[], scores=[], max_length=5, debug=False,start_ind = 0):
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
            if i  % maxlength < supervision:
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
                acc,acc_whole = get_accu(pred=pred_1, batch_targets=batch_targets_b, batch_size=maxlength,
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
        elif whole_right_num == whole_wrong_num and whole_right_num == 0 :
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
