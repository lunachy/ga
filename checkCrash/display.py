import numpy as np
import matplotlib.pyplot as plt


def get_image_locate(value: int, size: int = 20, angle_size: int = 4, h: int = 128, w: int = 128):
    """

    :param label
    :param size
    :param angle_size
    :return x 中心点坐标 y 中心点坐标  旋转的角度
    """
    k = value // (size * size)  # 角度
    ij = value % (size * size)
    cen_x = ij // size  # x 中心点
    cen_y = ij % size  # y 中心点
    return cen_x * (h / size), cen_y * (w / size), k * (360 / angle_size)


def display_state(state: list):
    """
    显示状态 ......
    :param state:
    :return:
    """
    h, w = np.shape(state)
    dis = np.zeros_like(state)
    cid_dict = {}
    for i in range(h):
        for j in range(w):
            if state[i][j] not in cid_dict:
                cid_dict[state[i][j]] = len(cid_dict)
            dis[i][j] = cid_dict[state[i][j]]
    return dis


def show_img(img, title):
    """
    可视化矩阵或者可视化列表
    :param img:
    :param title:
    :return:
    """
    img_ = display_state(state=img)
    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    ax.imshow(img_)
    plt.show()