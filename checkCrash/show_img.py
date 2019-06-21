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

def show_img(img,title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # plt.text(x=cen_x, y=cen_y, s=str(furniture_cid))
    ax.imshow(img)
    plt.show()