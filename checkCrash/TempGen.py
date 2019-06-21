# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    CrashChecker
    Description :    
    Author      :    zhaowen
    date        :    2019/5/28
____________________________________________________________________
    Change Activity:
                        2019/5/28:
____________________________________________________________________

'''
__author__ = 'zhaowen'

import ctypes
from checkCrash import structures
from utils.MultiThread import Mymultiprocessing, MyTMultithread
import pickle
import os
from checkCrash.CrashChecker import buildEmbeddingMat,CrashChecker


def worker(jsonfile, outJsonDir=r"D:\workspace\JSON_Gen\BedRoom8"):
    print(jsonfile)
    data = json.load(open(jsonfile, "r", encoding="utf-8"))
    room_str = json.dumps(data, ensure_ascii=False)
    room_str_copy = room_str
    room_str = room_str.encode("gb18030")

    ret, DetectorIdx, hitMask, furniture_mats, data_infos, embedding_room, embedding_furniture = api_init_room(
        1, room_str, numSegs)
    threadid = DetectorIdx

    assert ret == 0

    furnitures = []
    for i, furniture in enumerate(furniture_mats):
        mask_furnitures = buildEmbeddingMat(furniture, image_grid_size=64)
        furnitures.append(mask_furnitures)

    data_infos = data_infos
    room_context = hitMask
    data_return = {}
    data_return["room_str"] = room_str_copy
    data_return["furniture_mats"] = furnitures

    enhance = True

    if enhance:
        num_range = 4
    else:
        num_range = 1

    for i in range(num_range):
        data_enhance_ind = i
        data_return[str(data_enhance_ind * 90)] = {}
        room_context_enhance = rotateEmbedding(embedding_room, i)
        data_return[str(data_enhance_ind * 90)]["room_mats"] = room_context_enhance
        data_return[str(data_enhance_ind * 90)]["mid_states"] = []
        data_return[str(data_enhance_ind * 90)]["mid_states"].append(room_context_enhance)
        data_return[str(data_enhance_ind * 90)]["labels"] = []
        data_return[str(data_enhance_ind * 90)]["zids"] = []
        data_return[str(data_enhance_ind * 90)]["tags"] = []

    try:

        for ind, data in enumerate(data_infos):
            try:
                label = data["label"]
                tag = data["tag"]
                zid = data["zid"]
                if zid < 0:
                    continue
                pred = {}
                pred["label"] = label
                ret, hitMask, embedding_hitMask = detect_crash_label(pred=pred, index=0, tag=ind,
                                                                     threadid=threadid)  # tag
                print(ret, "ret")

                assert ret == 0
                for i in range(num_range):
                    data_enhance_ind = i
                    data_return[str(data_enhance_ind * 90)]["zids"].append(zid)
                    data_return[str(data_enhance_ind * 90)]["tags"].append(tag)
                    if i == 0:
                        data_return[str(data_enhance_ind * 90)]["labels"].append(label)
                    else:
                        data_return[str(data_enhance_ind * 90)]["labels"].append(
                            api_roate_label(label, numAddRoate=i))

                    data_return[str(data_enhance_ind * 90)]["mid_states"].append(
                        rotateEmbedding(embedding_hitMask, i))
            except Exception as e:
                print(e)

            finally:
                lib.API_finish(threadid)
    finally:
        lib.API_finish(threadid)
    outPath = os.path.join(outJsonDir, os.path.basename(jsonfile))
    if data_return:
        # print(data_return)
        # json.dump(obj=data_return, fp=open(outPath, "w", encoding="utf-8"))

        pickle.dump(obj=data_return, fp=open(outPath, "w", encoding="utf-8"))


if __name__ == "__main__":

    from utils.MultiThread import allFiles
    from multiprocessing import Pool

    import multiprocessing
    import json, time

    inJsonDir = r"D:\workspace\JSON_ORI\layout-cnn-room-BedRoom-0.0"
    outJsonDir = r"D:\workspace\JSON_Gen\BedRoom8"

    jsonFiles = list(allFiles(inJsonDir, "*.json"))
    print(len(jsonFiles))
    wait_time = 3
    process_num = 16

    print("Will process {} Files ,after {} sec ,In {} worker".format(len(jsonFiles), wait_time, process_num))

    time.sleep(wait_time)
    st = time.clock()

    num_process = 7
    filelist = jsonFiles

    p = Pool(num_process)
    DATALISTS = []
    tempmod = len(filelist) % (num_process)
    CD = int((len(filelist) + 1 + tempmod) / (num_process))
    for i in range(num_process):
        if i == num_process:
            DATALISTS.append(filelist[i * CD:-1])
        DATALISTS.append(filelist[(i * CD):((i + 1) * CD)])

    try:
        processes = []
        for i in range(num_process):
            # print('wait add process:',i+1,time.clock())
            # print(eval(self.funname),DATALISTS[i])
            MultThread = MyTMultithread(DATALISTS[i], i, worker, 1)
            configPath = "config.props"
            print(os.path.exists(configPath))
            ret = api_system_init(configPath=configPath)
            print("sys init:", ret)
            numSegs = get_numsegs_from_config(configPath)
            p = multiprocessing.Process(target=MultThread.startrun())
            processes.append(p)
        for p in processes:
            print('wait join ')
            p.start()

        print('waite over')
    except Exception as e:
        print('error :', e)
    print('end process')

    # worker = MyTMultithread(jsonFiles, 0, worker, process_num)

    # worker.startrun()
    # worker.multiprocessingWithReturn()
    # json.dump(obj=result, fp=open("Gendatas.log", "w", encoding="utf-8"), ensure_ascii=False)
    ed = time.clock()
    print('多进程使用时间:', ed - st)
