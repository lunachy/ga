"""
常用工具类
"""

import numpy as np
import math


def cal_distance(point1: np.ndarray, point2: np.ndarray):
    """
    计算point1与point2的距离
    :param point1:
    :param point2:
    :return:
    """
    if isinstance(point1, list):
        point1 = np.array(point1)
    if isinstance(point2, list):
        point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def cal_angle(a1: np.ndarray, a2: np.ndarray,
              b1: np.ndarray, b2: np.ndarray):
    """
    计算两个向量之间的夹角
    :param a1:
    :param a2:
    :param b1:
    :param b2:
    :return:  取值范围 [0, 180)
    """
    if isinstance(a1, list):
        a1 = np.array(a1)
    if isinstance(a2, list):
        a2 = np.array(a2)
    if isinstance(b1, list):
        b1 = np.array(b1)
    if isinstance(b2, list):
        b2 = np.array(b2)
    x = a2-a1
    y = b2-b1
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    if Lx == 0 or Ly == 0:
        return 0
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    return angle*360/2/np.pi


def to_polar(x: float, y: float):
    """
    转换为极坐标系
    :param x:
    :param y:
    :return:  半径以及角度
    """
    r = math.hypot(x, y)
    if y > 0:
        if x == 0:
            b = math.pi / 2
        elif x > 0:
            b = math.atan(y / x)
        else:
            b = math.atan(y / x) + math.pi
    elif y < 0:
        if x == 0:
            b = 3 * (math.pi / 2)
        elif x > 0:
            b = math.atan(y / x) + 2*math.pi
        else:
            b = math.atan(y / x) + math.pi
    else:  # 原点
        if x == 0:
            b = 0
        elif x > 0:
            b = 0
        else:
            b = math.pi
    angle = math.degrees(b)
    return r, angle


# print(to_polar(5, 0))
# print(to_polar(5, 5))
# print(to_polar(0, 5))
# print(to_polar(-5, 5))
# print(to_polar(-5, 0))
# print(to_polar(-5, -5))
# print(to_polar(0, -5))
# print(to_polar(5, -5))


def angle_nomalize(angle: int):
    """
    角度归一化到 0~360
    :param angle:
    :return:
    """
    if angle >= 360:
        angle = angle - 360
    if angle < 0:
        angle = angle + 360
    return angle
