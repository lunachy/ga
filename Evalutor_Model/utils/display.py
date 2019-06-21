import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_image_locate(value: int, size: int=20, angle_size: int=4, h:int=128, w:int=128):
    """
    解析出位置
    :param label
    :param size
    :param angle_size
    :return x 中心点坐标 y 中心点坐标
    """
    k = value // (size*size)
    ij = value % (size*size)
    cen_x = ij // size
    cen_y = ij % size
    return cen_x*(h/20), cen_y*(w/20), k*(360/angle_size)


def display_label(h: int, w: int, furnitures: list, state: list, labels: list, furniture_dxs: list, furniture_dys: list):
    """

    :param h:
    :param w:
    :param furnitures:  家具的cid
    :param state:   状态
    :param labels:   标签
    :param furniture_dxs:    家具dx数值
    :param furniture_dys:    家具dy数值
    :return:
    """
    img = np.zeros((h, w))
    font = cv2.FONT_HERSHEY_SIMPLEX
    furniture_ids = {}
    index = 10
    for i in range(h):
        for j in range(w):
            if state[i][j] == 0:
                continue
            if state[i][j] in (1, 2, 3, 4, 5):
                img[i][j] = state[i][j]
            else:
                if state[i][j] not in furniture_ids:
                    furniture_ids[state[i][j]] = index
                    index += 1
                    img[i][j] = furniture_ids[state[i][j]]
                else:
                    img[i][j] = furniture_ids[state[i][j]]

    if len(furniture_ids) > 0:
        index = max(furniture_ids.values()) + 1
    else:
        index = 10

    for i in range(len(labels)):
        label = labels[i] - 1
        if label < 0:
            continue
        # 获取label中心点
        y, x, angle = get_image_locate(label)
        y, x = int(y), int(x)
        img[y][x] = index
        if angle in [0, 180]:
            furniture_height = furniture_dxs[i]
            furniture_width = furniture_dys[i]
        else:
            furniture_height = furniture_dys[i]
            furniture_width = furniture_dxs[i]
        # 根据中心点还原出 图像真实
        grid_x = (furniture_width / 8000) * w
        grid_y = (furniture_height / 8000) * h
        start_x, end_x = max(0, x-int(grid_x/2)), min(w, x+int(grid_x/2))
        start_y, end_y = max(0, y-int(grid_y/2)), min(h, y+int(grid_y/2))
        # print("当前数据的label:{0}".format(label))
        # print("start_x:{0},  end_x:{1}".format(start_x, end_x))
        # print("start_y:{0}, end_y:{1}".format(start_y, end_y))
        for height_y in range(start_y, end_y):
            for width_x in range(start_x, end_x):
                img[height_y][width_x] = index
    temp_img = cv2.resize(img, dsize=(h * 10, w * 10), interpolation=cv2.INTER_LINEAR)
    for i in range(len(labels)):
        label = labels[i] - 1
        if label < 0:
            continue
        y, x, angle = get_image_locate(label)
        y, x = int(y), int(x)
        img[y][x] = index
        index += 1
        cid = furnitures[i]
        cv2.putText(temp_img, "{0}".format(cid), (x * 10, y * 10), font, 2, (15, 15), 5)
    return temp_img


def display_label2(h: int, w: int, furnitures: list, state: list, labels: list, furniture_dxs: list, furniture_dys: list):
    """

    :param h:
    :param w:
    :param furnitures:  家具的cid
    :param state:   状态
    :param labels:   标签
    :param furniture_dxs:    家具dx数值
    :param furniture_dys:    家具dy数值
    :return:
    """
    img = np.zeros((h, w))
    font = cv2.FONT_HERSHEY_SIMPLEX
    furniture_ids = {}
    index = 10
    for i in range(h):
        for j in range(w):
            if state[i][j] == 0:
                continue
            if state[i][j] in (1, 2, 3, 4, 5):
                img[i][j] = state[i][j]
            else:
                if state[i][j] not in furniture_ids:
                    furniture_ids[state[i][j]] = index
                    index += 1
                    img[i][j] = furniture_ids[state[i][j]]
                else:
                    img[i][j] = furniture_ids[state[i][j]]

    if len(furniture_ids) > 0:
        index = max(furniture_ids.values()) + 1
    else:
        index = 10

    for i in range(len(labels)):
        label = labels[i] - 1
        if label < 0:
            continue
        # 获取label中心点
        y, x, angle = get_image_locate(label)
        y, x = int(y), int(x)
        img[y][x] = index
        if angle in [0, 180]:
            furniture_height = furniture_dxs[i]
            furniture_width = furniture_dys[i]
        else:
            furniture_height = furniture_dys[i]
            furniture_width = furniture_dxs[i]
        # 根据中心点还原出 图像真实
        grid_x = (furniture_width / 8000) * w
        grid_y = (furniture_height / 8000) * h
        start_x, end_x = max(0, x-int(grid_x/2)), min(w, x+int(grid_x/2))
        start_y, end_y = max(0, y-int(grid_y/2)), min(h, y+int(grid_y/2))
        # print("当前数据的label:{0}".format(label))
        # print("start_x:{0},  end_x:{1}".format(start_x, end_x))
        # print("start_y:{0}, end_y:{1}".format(start_y, end_y))
        for height_y in range(start_y, end_y):
            for width_x in range(start_x, end_x):
                img[height_y][width_x] = index
    temp_img = cv2.resize(img, dsize=(h * 10, w * 10), interpolation=cv2.INTER_LINEAR)
    for i in range(len(labels)):
        label = labels[i] - 1
        if label < 0:
            continue
        y, x, angle = get_image_locate(label)
        y, x = int(y), int(x)
        img[y][x] = index
        index += 1
        cid = furnitures[i]
        cv2.putText(temp_img, "{0}".format(cid), (x * 10, y * 10), font, 2, (15, 15), 5)
    return temp_img


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