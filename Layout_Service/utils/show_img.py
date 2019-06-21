# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    show_img
    Description :    
    Author      :    zhaowen
    date        :    2019/4/12
____________________________________________________________________
    Change Activity:
                        2019/4/12:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'
import os


def show_img(img, title, save=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # plt.text(x=cen_x, y=cen_y, s=str(furniture_cid))
    ax.imshow(img)

    if save:
        p = "d:/workspace/log/pngs/{}.png".format(title)
        path = p.replace(" ", "").replace(",", "").replace("", "").replace("-", "_")
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()


def show_state(state, title, save=False, save_path=""):
    """
    显示状态 ......
    :param state:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    h, w = np.shape(state)
    dis = np.zeros_like(state)
    cid_dict = {}
    for i in range(h):
        for j in range(w):
            if state[i][j] not in cid_dict:
                cid_dict[state[i][j]] = len(cid_dict)
            dis[i][j] = cid_dict[state[i][j]]
    if title:
        plt.title(title)
        # plt.text(x=cen_x, y=cen_y, s=str(furniture_cid))
    ax.imshow(dis)

    if save:
        if save_path:
            p = os.path.join(save_path, "{}.png".format(title))
        else:
            p = "d:/workspace/log/pngs/{}.png".format(title)
        path = p.replace(" ", "").replace(",", "").replace("", "").replace("-", "_")
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()
