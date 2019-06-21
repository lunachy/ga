"""
评估指标
"""
import tensorflow as tf
import numpy as np
import json


def top_k_acc(targets: np.ndarray, predicts: np.ndarray, top_k: int=4, ignore_label: int=None):
    """
    计算 top_k 准确率  (单步准确率计算)
    :param targets:  真实目标   (batch_size, )
    :param predicts:  预测结果  (batch_size, label_size)
    :param top_k:
    :param ignore_label:  忽略计算的target数据
    :return:
    """
    top_k_acc = 0  # top_k 准确次数
    top_k_error = 0  # top_k 错误次数
    assert len(targets) == len(predicts)
    top_k_predicts = np.argsort(-predicts, axis=1)[:, : top_k]
    for index in range(len(targets)):
        if ignore_label is not None and targets[index] == 0:
            continue
        else:
            if targets[index] in top_k_predicts[index]:
                top_k_acc += 1
            else:
                top_k_error += 1
    return top_k_acc/(top_k_acc + top_k_error + 1e-10)


def top_k_sequence_acc(squence_predicts: list, squence_targets: list, top_k: int=1, ignore_label: int=0):
    """
    计算序列的完全正确的acc数值
    :param squence_predicts:   [batch_size, all_top, step_length]
    :param squence_targets:   [batch_size, step_length]
    :param ignore_label:  当前序列全为此值 不计正确与错误
    :return:
    """
    assert len(squence_predicts) == len(squence_targets)
    top_k_acc = 0  # top_k 准确次数
    top_k_error = 0  # top_k 错误次数
    for i in range(len(squence_predicts)):
        squence_target = squence_targets[i]
        if squence_target in squence_predicts[i][: top_k]:
            top_k_acc += 1
        else:
            top_k_error += 1
    return top_k_acc / (top_k_acc + top_k_error + 1e-10)

