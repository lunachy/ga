# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    TEST_PY
    Description :    
    Author      :    zhaowen
    date        :    2019/5/28
____________________________________________________________________
    Change Activity:
                        2019/5/28:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'

import json
import os
import time
import pickle
import numpy as np
from utils.display import display_state
from checkCrash.CrashChecker import buildEmbeddingMat
from utils.show_img import show_img
from utils.MultiThread import Mymultiprocessing, allFiles
from checkCrash.CrashChecker import CrashChecker


def get_cidlist_fromjson(roomjson):
    '''
    从roomj json 中获取家具列表
    :return:
    '''
    cidlist = []
    if isinstance(roomjson, str):
        roomjson = json.loads(roomjson)
    data = roomjson
    for zone in data["functionZones"]:
        if zone["functionZoneType"] == "Undefined":
            continue
        for furniture in zone["subPositions"]:
            cid_i = {}

            cid_i["cid"] = int(furniture["cid"])
            cid_i["dx"] = int(furniture["size"]["dx"])
            cid_i["dy"] = int(furniture["size"]["dy"])
            cid_i["modelId"] = 0
            cid_i["isDrModel"] = 0
            cidlist.append(cid_i)
    return cidlist


def get_zoneTag_fromjson(roomjson):
    '''
    从roomj json 中获取功能区tag
    :return:
    '''
    taglist = []
    if isinstance(roomjson, str):
        roomjson = json.loads(roomjson)
    data = roomjson
    for zone in data["functionZones"]:
        if "functionZoneType" in zone.keys():
            if zone["functionZoneType"] == "Undefined":
                continue
        taglist.append(zone["tag"])
        print(",tag:{}".format(zone["tag"]))

    return taglist


def GenData(arglist, crash_Files=[]):
    inJsonpath = arglist[0]
    outJsonDir = arglist[1]
    ck = arglist[2]
    outPath = os.path.join(outJsonDir, os.path.basename(inJsonpath))
    jsonpath = inJsonpath
    with open(jsonpath, "r", encoding="utf-8") as f:
        room_str = f.read()
    try:
        ck.init_room(room_str=room_str, roomTypeId=1)
        return_data = ck.gen_data(enhance=False)
    except Exception as e:
        # raise e
        crash_Files.append({"path": inJsonpath, "error": str(e)})
    pickle.dump(obj=return_data, file=open(outPath.replace(".json", ".pkl"), "wb"))
    return crash_Files


def MultiProcessGendatas(process_num, inJsonDir, outJsonDir):
    jsonFiles = list(allFiles(inJsonDir, "*.json"))
    wait_time = 0
    st = time.clock()

    DATALISTS = []
    filelist = jsonFiles
    tempmod = len(filelist) % (process_num)
    CD = int((len(filelist) + 1 + tempmod) / (process_num))
    file_num = len(jsonFiles)
    num_len = 0
    temp_files = []
    for i in range(process_num):
        if i == process_num - 1:
            DATALISTS.append([filelist[i * CD:], outJsonDir])
            temp_files.extend(filelist[i * CD:])
            num_len += len(filelist[i * CD:])
        else:
            DATALISTS.append([filelist[(i * CD):((i + 1) * CD)], outJsonDir])
            num_len += len(filelist[(i * CD):((i + 1) * CD)])
            temp_files.extend(filelist[(i * CD):((i + 1) * CD)])

    print(file_num, num_len)
    print("遗漏：", [x for x in temp_files if x not in jsonFiles])

    assert file_num == num_len

    worker = Mymultiprocessing(DATALISTS, 0, GenDatas, process_num)

    worker.multiprocessingOnly()
    ed = time.clock()
    print('多进程使用时间:', ed - st)


def GenDatas(arg):
    inJsonFiles, outJsonDir = arg[0], arg[1]
    print(outJsonDir, "OUT")
    if os.path.exists(outJsonDir):
        pass
    else:
        os.makedirs(outJsonDir)
    ck = CrashChecker(configPath="config.props")  # "../checkCrash/config.props"
    argsList = [[file, outJsonDir, ck] for file in inJsonFiles]
    print("需要处理的文件个数:", len(argsList))
    time.sleep(10)
    from tqdm import tqdm
    crash_Files = []
    for info in tqdm(argsList):
        crash_Files = GenData(arglist=info, crash_Files=crash_Files)
    print("#" * 30)
    print("碰撞信息:", crash_Files)


if __name__ == "__main__":
    MultiProcessGendatas(inJsonDir=r"D:\workspace\JSON_ORI\layout-cnn-room-test-BedRoom-0.0-5.30",
                         outJsonDir=r"D:\workspace\JSON_Gen\BedRoom_61", process_num=7)
