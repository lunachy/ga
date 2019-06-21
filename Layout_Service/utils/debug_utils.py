"""
debug 工具类
"""
import numpy as np
import json
from utils.display import get_image_locate, display_state


def debug_info(target: int, scores: np.ndarray, label_split_size: int, angle_split_size: int = 4, h: int = 128, w: int = 128, top_k: int=4):
    """
    生成debug数据字符串
    :param target:  真实值
    :param scores:   预测分值  [1, label_size]
    :param label_split_size:  输出label栅格化数目
    :param angle_split_size:  旋转角度栅格化数据
    :param h:  输入图片的像素点数目 h
    :param w:  输入图片的像素点数目 w
    :param top_k:  仅记录分值最高的前top_k
    return 返回实际结果info 以及top_k结果预测的info
    """
    top_k_predict = np.argsort(-scores)[: top_k]
    target_x, target_y, target_angle = get_image_locate(value=target-1, size=label_split_size, angle_size=angle_split_size, h=h, w=w)
    target_info = {"label": target, "x": target_x, "y": target_y, "angle": target_angle, "score": 1}
    predict_infos = []
    for predict_index in top_k_predict:
        predict_score = scores[predict_index]
        predict_x, predict_y, predict_angle = get_image_locate(value=predict_index-1, size=label_split_size, angle_size=angle_split_size, h=h, w=w)
        predict_info = {"label": predict_index, "score": predict_score, "x": predict_x, "y": predict_y, "angle": predict_angle}
        predict_infos.append(predict_info)
    return target_info, predict_infos


def room_debug_info(targets: np.ndarray, scoress: np.ndarray, zids: list, label_split_size: int, angle_split_size: int, h: int, w: int, top_k: int, mask_label: int=0):
    """
    返回单个户型的信息
    :param targets:  布局真实label  (单个户型的所有真实的label)
    :param scoress:  布局预测分值  (单个户型的所有预测的label)
    :param zids:  zid数据
    :param label_split_size:  输出label栅格化数目
    :param angle_split_size:  旋转角度栅格化数据
    :param h:  输入图片的像素点数目 h
    :param w:  输入图片的像素点数目 w
    :param top_k:  仅记录分值最高的前top_k
    :param mask_label
    :return:  整个户型的数据信息
    """
    steps_info = []
    for step in range(len(targets)):
        target = targets[step]
        if mask_label is not None and mask_label == target:  # 填充忽略不计的target
            continue
        scores = scoress[step]
        step_target_info, step_predict_infos = debug_info(target=target, scores=scores, label_split_size=label_split_size,
                                                          angle_split_size=angle_split_size, h=h, w=w, top_k=top_k)
        step_info = {"target_info": step_target_info, "predict_infos": step_predict_infos, "zid": zids[step]}
        steps_info.append(step_info)
    return steps_info


