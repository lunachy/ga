#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:   roomstr2label.py
@author: chengyong
@time:   2019/6/21 18:12
@desc:   
"""

from checkCrash.CrashChecker import CrashChecker
import pysnooper

roomstr_path = '/ai/data/data-0619/layout-data-BedRoom.json'
out_label_path = '/ai/data/bedroom_label.txt'
ROOMTYPEID = 2
conf_path = './checkCrash/config.props'
ck = CrashChecker(conf_path)
total_row = 0
no_crash_row = 0
out_label_str = ''
# with pysnooper.snoop():
if True:
    with open(roomstr_path) as f1, open(out_label_path, 'a') as f2:
        for line in f1:
            if no_crash_row > 0 and no_crash_row % 10000 == 0:
                break

            total_row += 1
            if total_row % 1000 == 0:
                f2.write(out_label_str)
                out_label_str = ''
                print('processing {} rows'.format(total_row))

            ret = ck.init_room(roomTypeId=ROOMTYPEID, room_str=line)
            if ret != 0:
                ck.room_finish()
                continue

            true_labels = [x['label'] for x in ck.data_infos]
            ret = ck.checkFullPath(true_labels)
            ck.room_finish()
            if ret == 0:
                no_crash_row += 1
                out_label_str = out_label_str + str(true_labels) + '\n'
        print('total_row: ', total_row, 'no_crash_row: ', no_crash_row, 'ratio: ', no_crash_row / total_row)
