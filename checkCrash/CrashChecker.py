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
import os
import numpy as np

from show_img import show_img
import pysnooper

ERROR_CODE = """
 #/*错误码*/
#define OK 0
//#define NO_COLLISION 1
#define HAS_COLLISION 1
#define INVALID_PARAM -1
#define NO_MODEL -2
#define EXCEED_SEARCH_WIDTH -3
#define NO_ROOM_INFO -4
#define JSON_ERROR -5
#define PROBUF_PARSE_ERROR -6
#define SEARCH_WIDTH_PARAM_ERROR -7
#define PATH_INDEX_PARAM_ERROR -8
#define GRIDNUM_PARAM_ERROR -9
#define GRIDNUM_MISMATCH -10
#define JNI_MEM_ERROR -11
#define GRANULARITY_ERROR -12
#define THREADS_POOL_FULL -13
#define DETECTOR_IDX_ERROR -14
#define ZONE_TAG_NOT_FOUND -15
#define DETECTOR_BUSILY -16
#define UNKONWN_ERROR -20
#define INIT_FAIL_ERROR -21
"""

'''
加载3D模型，碰撞检测基于3D obj文件生成的模型数据进行检测
'''
# paths = ["temporary.dll", "checkCrash/temporary.dll", "../checkCrash/temporary.dll"]
import sys

platform = sys.platform
if platform == "linux":
    paths = ["./libcollision_check.so", "./checkCrash/libcollision_check.so", "../checkCrash/libcollision_check.so"]
else:
    paths = ["ConsoleApplication1.dll", "checkCrash/ConsoleApplication1.dll",
             "../checkCrash/ConsoleApplication1.dll"]

path = [x for x in paths if os.path.exists(x)][0]
# path = os.path.join(*(os.path.split(__file__)[:-1] + (path,)))
print(path)
print(os.path.exists(path))
try:
    dll_name = path
    lib = ctypes.cdll.LoadLibrary(dll_name)

    if platform == "linux":
        print(os.path.exists(dll_name), "so load sucess")
    else:
        print(os.path.exists(dll_name), "dll load sucess")
except Exception as e:

    print(os.path.exists(dll_name), "dll  Not exists", e)
    raise (e)


class CrashChecker():

    def __init__(self, configPath="checkCrash/config.props", debug=False):

        '''
        碰撞检测的API
        :param room_str: json  字符串 可以使用 json.dumps(data, ensure_ascii=False) 生成
        :param room_str:
        :param configPath:
        :param roomTypeId:
        '''

        self.lib = lib
        print(os.path.exists(configPath))
        ret = api_system_init(configPath=configPath)
        print("sys init:", ret)
        self.debug = debug
        self.numSegs = get_numsegs_from_config(configPath)
        self.configPath = configPath
        self.ignore_zids = [89]
        self.room_length_dict = get_roomLen_from_config(self.configPath)
        print(self.room_length_dict)

    def init_room(self, roomTypeId, room_str):

        self.room_str = room_str
        room_str = room_str.encode("gb18030")
        self.room_length = self.room_length_dict[str(roomTypeId)]

        ret, DetectorIdx, data_infos, embedding_room, embedding_furniture = api_init_room(
            roomTypeId, room_str, self.numSegs)
        if ret != 0:
            return ret
        # assert ret == 0, "房间初始化失败,ret:{}".format(ret)

        furnitures = []
        furnitures_ori = []

        for i in range(len(data_infos)):
            furnitures_ori.append(buildEmbeddingMat(
                embedding_furniture[i * self.numSegs * self.numSegs:(i + 1) * self.numSegs * self.numSegs],
                self.numSegs, stIndexs=0))
            if int(data_infos[i]["zid"]) in self.ignore_zids:
                continue
            if "designed" in data_infos[i].keys():  # 设计师放置的
                if data_infos[i]["designed"]:
                    continue
            mask_furnitures = buildEmbeddingMat(
                embedding_furniture[i * self.numSegs * self.numSegs:(i + 1) * self.numSegs * self.numSegs],
                self.numSegs, stIndexs=0)
            furnitures.append(mask_furnitures)
            if self.debug:
                show_img(mask_furnitures[0], "{} furniture".format(i))
        if self.debug:
            show_img(self.roomMats[0], "room")

        # print("GET :DetectorIdx", DetectorIdx)

        self.threadid = DetectorIdx
        self.ori_data_infos = data_infos
        self.data_infos = [x for x in data_infos if x["zid"] not in self.ignore_zids and not x["designed"]]

        self.roomMats = buildEmbeddingMat(embedding_room, numSegs=self.numSegs)

        self.furniture_mats = furnitures
        self.ori_furniture_mats = furnitures_ori
        self.embedding_room = embedding_room
        self.embedding_furniture = embedding_furniture
        self.image_grid_num = self.numSegs

        if self.debug:
            print(self.threadid, "--  debug:进程id ret.threadId")
            print("初始化完成,threadId{}".format(self.threadid))
        return ret

    def gen_data(self, enhance=False):
        '''
        生成训练数据
        :return:
        '''
        data_infos = self.data_infos

        room_context = self.roomMats
        data_return = {}
        data_return["room_str"] = self.room_str
        data_return["furniture_mats"] = self.furniture_mats

        if enhance:
            num_range = 4
        else:
            num_range = 1

        for i in range(num_range):
            data_enhance_ind = i
            data_return[str(data_enhance_ind * 90)] = {}
            room_context_enhance = self.rotateEmbedding(self.embedding_room, i)
            data_return[str(data_enhance_ind * 90)]["room_mats"] = room_context_enhance
            data_return[str(data_enhance_ind * 90)]["mid_states"] = []
            data_return[str(data_enhance_ind * 90)]["mid_states"].append(room_context_enhance)
            data_return[str(data_enhance_ind * 90)]["labels"] = []
            data_return[str(data_enhance_ind * 90)]["zids"] = []
            data_return[str(data_enhance_ind * 90)]["tags"] = []

        try:

            for ind, data in enumerate(data_infos):
                print("get_data:", data_infos)
                try:
                    label = data["label"]
                    tag = data["tag"]
                    zid = data["zid"]
                    if int(zid) in self.ignore_zids:
                        continue
                    pred = {}
                    pred["label"] = label
                    ret, embedding_hitMask = self.detect_crash_label(pred=pred, index=0, tag=tag)  # tag

                    assert ret == 0, "碰撞:第{}个功能区,ret:{}".format(ind, ret)
                    for i in range(num_range):
                        data_enhance_ind = i
                        data_return[str(data_enhance_ind * 90)]["zids"].append(zid)
                        data_return[str(data_enhance_ind * 90)]["tags"].append(tag)
                        if i == 0:
                            data_return[str(data_enhance_ind * 90)]["labels"].append(label)
                        else:
                            data_return[str(data_enhance_ind * 90)]["labels"].append(self.rotateLabel(label, i))

                        data_return[str(data_enhance_ind * 90)]["mid_states"].append(
                            self.rotateEmbedding(embedding_hitMask, i))
                except Exception as e:
                    print(e)
                    raise (e)
                finally:
                    self.finish_step()
        except Exception as e:
            print(e)
            raise e
        finally:

            self.room_finish()
        return data_return

    def rotateEmbedding(self, embedding_room, numAddRoate):
        numSegs = self.numSegs
        embde_rotate = api_rotate_mat(srcHitmask=embedding_room, numAddRoate=numAddRoate,
                                      numSegs=numSegs)
        rotate_mats = buildEmbeddingMat(embde_rotate, self.image_grid_num
                                        )
        return rotate_mats

    def rotateLabel(self, label, numAddRoate):
        return api_roate_label(label, numAddRoate=numAddRoate)

    def checkFullPath(self, labelList):
        '''
        numLabels: 输入要检测的碰撞功能区个数
        labelList：每个功能区对应的label
        return : OK：0,否则为错误码
        '''
        numLabels = len(labelList)
        labels_int = (ctypes.c_int * numLabels)()
        for i in range(numLabels):
            labels_int[i] = ctypes.c_int(labelList[i])
        return lib.API_checkFullPath(numLabels, labels_int, self.threadid)

    def checkFullPathEx(self, labelList):
        '''
        numLabels: 输入要检测的碰撞功能区个数
        labelList：每个功能区对应的label
        return : OK：0,否则为错误码
        '''
        numLabels = len(labelList)
        labels_int = (ctypes.c_int * numLabels)()
        for i in range(numLabels):
            labels_int[i] = ctypes.c_int(labelList[i])

        grid_num = self.numSegs * self.numSegs
        EmbeddingArrayType = structures.EmbeddingCode * grid_num
        embedding_code_list = EmbeddingArrayType()
        for i in range(grid_num):
            structures.init_embedding_code(embedding_code_list[i])

        ret = lib.API_checkFullPathEx(numLabels, labels_int, self.threadid, embedding_code_list)
        if ret == 0:
            return ret, embedding_code_list
        else:
            return ret, None

    def detect_crash_label(self, pred, index, tag):
        '''
        修改后的碰撞检测 获取位置 由 c++ 实现
        :param pred:
        :param index:
        :return:
        '''
        pred_label = pred["label"]
        numSegs = self.numSegs
        zoneLabel = pred_label

        ret, embedding_mask = api_check_crash(zoneIndex=tag, label=zoneLabel,
                                              thredidx=self.threadid, route_index=index,
                                              numSegs=numSegs)

        return ret, embedding_mask

    def check_data(self, layoutLables):
        '''
        layoutLabels : [ 221,21,32 ]
        检测指定类型房间的布局结果是否有碰撞，若有返回False ,{}
        若无：返回True 与户型的相关信息
        :return:
        '''
        data_infos = self.data_infos
        data_return = {}
        data_return["room_str"] = self.room_str
        data_return["furniture_mats"] = self.furniture_mats
        data_return['0'] = {}
        room_context_enhance = self.embedding_room
        data_return['0']["room_mats"] = room_context_enhance
        data_return['0']["mid_states"] = []
        data_return['0']["mid_states"].append(room_context_enhance)
        data_return['0']["labels"] = []
        data_return['0']["zids"] = []
        data_return['0']["tags"] = []
        ret_list = []
        hitMask_list = []
        for ind, data in enumerate(data_infos):
            if ind < len(layoutLables):
                tag = data["tag"]
                zid = data["zid"]
                if int(zid) in self.ignore_zids or zid["designed"]:
                    continue
                data["label"] = layoutLables[ind]
                label = data["label"]
                pred = {}
                pred["label"] = label
                ret, embedding_hitMask = self.detect_crash_label(pred=pred, index=0, tag=tag)
                ret_list.append(ret)
                hitMask_list.append(embedding_hitMask)
                data_return['0']["zids"].append(zid)
                data_return['0']["tags"].append(tag)
                data_return['0']["labels"].append(label)
                if ret != 0:
                    break
                if ret == 0:
                    ret_array = buildEmbeddingMat(embedding_hitMask, 64, stIndexs=0)
                else:
                    ret_array = embedding_hitMask
                data_return['0']["mid_states"].append(ret_array)
                self.finish_step()
        if set(ret_list) == {0}:
            self.room_finish()
            return [True, data_return]

        else:
            self.room_finish()
            return [False, {}]

    def finish_step(self):
        lib.API_finishStep(self.threadid)

    def room_finish(self):
        lib.API_finish(self.threadid)

    def convert2real(self, label, z=0, cid=0):
        '''
        将预测的结果转为碰撞检测的坐标
        :param label:
        :return:
        '''
        furniture_pos = label
        self.label_grid_num = 16
        m = self.label_grid_num
        n = self.label_grid_num
        self.split_angle = 90
        split_angle = self.split_angle
        self.max_room_size = self.room_length
        length = self.max_room_size
        if isinstance(furniture_pos, int):
            k = furniture_pos // (m * n)
            ij = furniture_pos % (m * n)
            cen_x = ij // m
            cen_y = ij % n
        else:
            cen_x, cen_y, k = furniture_pos[0], furniture_pos[1], furniture_pos[2]

        roate = int(k * split_angle)

        grid_size = length / m

        crash_cen_x = -(length / 2) + cen_x * grid_size + grid_size / 2
        crash_cen_y = -(length / 2) + cen_y * grid_size + grid_size / 2

        pred = {}
        pred["z"] = int(z)
        pred["x"] = int(crash_cen_x)
        pred["y"] = int(crash_cen_y)
        pred["r"] = int(roate)
        pred["cid"] = cid
        return pred

    def label2InputGrids(self, label, inputSegs):
        '''
        由label 获取 其在输入图像中的位置
        :param label:
        :param inputSegs:
        :param labelSegs:
        :return:
        '''
        preds = self.convert2real(label)
        realx, realy, realz = preds["x"], preds["y"], preds["z"]
        grid_x = int(realx / (self.max_room_size / inputSegs) + 0.5)
        grid_y = int(realy / (self.max_room_size / inputSegs) + 0.5)
        if grid_x < 0:
            grid_x -= 1
        if grid_y < 0:
            grid_y -= 1
        grid_y = int(grid_y)
        grid_x = int(grid_x)

        result = {}
        result["centerX"] = grid_x + inputSegs / 2
        result["centerY"] = grid_y + inputSegs / 2

        return result

    def buildEmbeddingMat(self, EmbeddingArray, numSegs=64, stIndexs=0):
        data_return = np.zeros(shape=[4, numSegs, numSegs])
        st_ind = stIndexs * numSegs * numSegs
        ed_ind = st_ind + numSegs * numSegs
        for k in range(st_ind, ed_ind):
            i = int(k / numSegs)
            j = k % numSegs
            data_return[0][i][j] = EmbeddingArray[k].zid
            data_return[1][i][j] = EmbeddingArray[k].cid
            data_return[2][i][j] = EmbeddingArray[k].mid
            data_return[3][i][j] = EmbeddingArray[k].scid

        return data_return

    def api_distStat(self, zoneIndex, numLabels, labelList, dis_Result):

        labels_int = (ctypes.c_int * numLabels)()
        n_ = self.numSegs * self.numSegs
        dis_Result_int = (ctypes.c_int * n_)()
        for i in range(numLabels):
            labels_int[i] = ctypes.c_int(labelList[i])
        for i in range(self.numSegs * self.numSegs):
            dis_Result_int[i] = ctypes.c_int(dis_Result[i])
        lib.API_distStat(zoneIndex, numLabels, labels_int, self.threadid, dis_Result_int)
        for i in range(self.numSegs * self.numSegs):
            dis_Result[i] = dis_Result_int[i]

    def checkCrashs(self, numLabels, labelList):
        '''
		numLabels: 输入要检测的碰撞功能区个数
		labelList：每个功能区对应的label
		return : OK：0,否则为错误码
        '''
        labels_int = (ctypes.c_int * numLabels)()
        for i in range(numLabels):
            labels_int[i] = ctypes.c_int(labelList[i])
        return lib.API_checkCrashs(numLabels, labels_int, self.threadid)


def api_check_crash(zoneIndex, label, route_index, thredidx, numSegs):
    grid_num = numSegs * numSegs
    EmbeddingArrayType = structures.EmbeddingCode * grid_num
    embedding_code_list = EmbeddingArrayType()
    for i in range(grid_num):
        structures.init_embedding_code(embedding_code_list[i])

    ret = lib.API_checkCrash(zoneIndex, label, route_index, thredidx, embedding_code_list)
    if 0 == ret:
        return ret, embedding_code_list
    else:
        return ret, None


def api_system_init(configPath):
    configPathByte = bytes(configPath, encoding="utf-8")
    configPath_ptr = ctypes.c_char_p(configPathByte)
    structures.InitResult()
    return lib.API_systemInit(configPath_ptr)


def api_init_room(roomTypeId, room_str, numSegs):
    '''
    初始化
    :param roomTypeId:
    :param room_str:
    :param numSegs:
    :return:
    '''
    room_p = ctypes.c_char_p(room_str)
    seq_max_length = 20
    grid_num = numSegs * numSegs
    furnist_grids = grid_num * seq_max_length
    EmbeddingArrayType = structures.EmbeddingCode * grid_num
    EmbeddingArrayType_furniture = structures.EmbeddingCode * furnist_grids
    embedding_code_list_room_context = EmbeddingArrayType()
    embedding_code_list_furniture = EmbeddingArrayType_furniture()

    # TODO: modify (cost 0.25-0.3sec)
    for i in range(grid_num):
        structures.init_embedding_code(embedding_code_list_room_context[i])

    for i in range(seq_max_length * grid_num):
        try:
            structures.init_embedding_code(embedding_code_list_furniture[i])
        except Exception as e:
            print("e:{},i:{}".format(e, i))
            raise (e)
    outDetectorIdx = ctypes.c_int(8)
    outDetectorIdx_p = ctypes.pointer(outDetectorIdx)

    outZoneNum = ctypes.c_int(seq_max_length)
    outZoneNum_p = ctypes.pointer(outZoneNum)  # ctypes.c_int(outZoneNum)

    ZoneArrayType = structures.Zone * seq_max_length
    outZones = ZoneArrayType()
    try:

        ret = lib.API_init(roomTypeId, room_p, outDetectorIdx_p, outZoneNum_p, outZones,
                           embedding_code_list_room_context,
                           embedding_code_list_furniture)
    except Exception as e:
        print(e)
        raise (e)
    finally:
        pass

    if ret >= 0:
        hit_mask = []
        datas = []
        furnist_mat = []

        for i in range(int(outZoneNum.value)):
            try:

                data = {}
                data["zid"] = outZones[i].id
                data["tag"] = outZones[i].tag
                data["label"] = outZones[i].label
                data["designed"] = outZones[i].designed

                datas.append(data)
            except Exception as e:
                print("outZoneNum.value", outZoneNum.value, i, e)
                continue

        datas.sort(key=lambda x: x["tag"])

        return ret, outDetectorIdx.value, datas, embedding_code_list_room_context, embedding_code_list_furniture

    else:
        return ret, outDetectorIdx.value, None, None, None


def api_rotate_mat(srcHitmask, numAddRoate, numSegs):
    '''
    输入旋转前的Embedding 图，返回旋转后的
    :param srcHitmask: 旋转前的 hitmask 图
    :param numAddRoate: 旋转的 index, 指在按照角度分割后的旋转，1表示旋转90度，2表示180，依次类推
    :return:旋转后的图
    '''
    grid_num = numSegs * numSegs
    EmbeddingArrayTypeDest = structures.EmbeddingCode * grid_num
    dest_embedding_code_list = EmbeddingArrayTypeDest()

    for i in range(grid_num):
        structures.init_embedding_code(dest_embedding_code_list[i])

    lib.API_rotateMat(srcHitmask, dest_embedding_code_list, numAddRoate)

    return dest_embedding_code_list


def api_roate_label(label, numAddRoate):
    '''
    给定旋转前的label,返回旋转后的label，
    :param label: 旋转前的label
    :param numAddRoate: 旋转的数目（单位90度）
    :return: 旋转后的label
    '''
    return lib.API_rotateLabel(label, numAddRoate)


def get_numsegs_from_config(configPath):
    try:
        f = open(configPath, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            if (line.find("input_segs") != -1 and not line.startswith("#")):
                splits = line.split("=")
                if (len(splits) != 2):
                    return -1
                segStr = splits[1]
                segNumStr = segStr.strip()
                value = int(segNumStr)

                return value
        return -1
    finally:
        if f:
            f.close()


def get_roomLen_from_config(configPath):
    try:
        room_len_dict = {}
        f = open(configPath, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            if (line.find("room_metadatas") != -1):
                splits = line.split("=")
                if (len(splits) != 2):
                    return -1
                segStr = splits[1]
                segNumStr = segStr.strip()
                configs = segNumStr.split(",")
                for config in configs:
                    k_str, v_str = config.split(":")
                    room_len_dict[str(k_str.strip())] = int(v_str)
                return room_len_dict
        return -1
    finally:
        if f:
            f.close()


def buildEmbeddingMat(EmbeddingArray, numSegs, stIndexs=0):
    data_return = np.zeros(shape=[4, numSegs, numSegs])
    st_ind = stIndexs * numSegs * numSegs
    ed_ind = st_ind + numSegs * numSegs
    for k in range(st_ind, ed_ind):
        i = int(k / numSegs)
        j = k % numSegs
        data_return[0][i][j] = EmbeddingArray[k].zid
        data_return[1][i][j] = EmbeddingArray[k].cid
        data_return[2][i][j] = EmbeddingArray[k].mid
        data_return[3][i][j] = EmbeddingArray[k].scid

    return data_return


if __name__ == "__main__":

    # test
    pass

    import matplotlib.pyplot as plt
    from utils.display import display_state
    import json, pickle

    configpaths = ["./config.props", "./checkCrash/config.props", "../checkCrash/config.props"]
    configpath = [x for x in configpaths if os.path.exists(x)][0]
    print("配置文件{}".format(configpath))

    ck = CrashChecker(configPath=configpath)

    res = [0] * 64 * 64
    # test_json_data = '{"dataId":2998,"houseId":1023468,"functionZones":[{"id":48,"mainPositions":[{"sample":true,"main":true,"tag":"KVyyc3O1","cid":318,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-85,"y":-784,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1039,"dy":1523,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"positions":[{"sample":true,"main":true,"tag":"KVyyc3O1","cid":318,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-85,"y":-784,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1039,"dy":1523,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"F9WlOe8b","cid":100,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1136,"y":-1414,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":504,"dy":287,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-1051.2725664122102,"m31":-629.3622945137386,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-1051,"y":-629,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"FlhADecE","cid":319,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-748,"y":-1368,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":283,"dy":337,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-663.0969639876009,"m31":-583.987609571012,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-663,"y":-583,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"guUOyerR","cid":319,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":588,"y":-1371,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":292,"dy":346,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":673.6400485433519,"m31":-586.7131521866563,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":673,"y":-586,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":-85,"y":-784,"z":0},"bound":{"x":-1388,"y":-1557,"dx":2123,"dy":1534},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":0},{"id":38,"mainPositions":[{"sample":true,"main":true,"tag":"KGdpJQ8H","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"location":{"x":1112,"y":-530,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1991,"dy":539,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"positions":[{"sample":true,"main":true,"tag":"KGdpJQ8H","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"location":{"x":1112,"y":-530,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1991,"dy":539,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":1112,"y":-530,"z":0},"bound":{"x":116,"y":-800,"dx":1991,"dy":539},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"tag":1},{"id":99,"mainPositions":[{"sample":true,"main":true,"tag":"J9y9X54i","cid":96,"metaData":{"name":"UNKNOWN","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":0,"y":0,"z":0},"locationInHouse":{"x":0,"y":0,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1,"dy":1,"dz":1},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"isDrModel":false}],"positions":[],"center":{"x":0,"y":0,"z":0},"bound":{"x":0,"y":0,"dx":0,"dy":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":2}],"roomType":"BedRoom","drRoomType":"SecondBedRoom","roomName":"次卧","walls":[{"scid":1,"wallPoints":[{"x":-1591,"y":1156},{"x":-1388,"y":1156},{"x":-1591,"y":-1656},{"x":-1388,"y":-1656}]},{"scid":14,"wallPoints":[{"x":-1391,"y":-1553},{"x":1391,"y":-1553},{"x":-1391,"y":-1756},{"x":1391,"y":-1756}]},{"scid":16,"wallPoints":[{"x":-1591,"y":1656},{"x":-1388,"y":1656},{"x":-1591,"y":1153},{"x":-1388,"y":1153}]},{"scid":14,"wallPoints":[{"x":-1391,"y":1756},{"x":1391,"y":1756},{"x":-1391,"y":1553},{"x":1391,"y":1553}]},{"scid":12,"wallPoints":[{"x":1388,"y":-143},{"x":1591,"y":-143},{"x":1388,"y":-1656},{"x":1591,"y":-1656}]},{"scid":13,"wallPoints":[{"x":1388,"y":1656},{"x":1591,"y":1656},{"x":1388,"y":-146},{"x":1591,"y":-146}]}],"windows":[],"doors":[{"scid":13,"points":[{"x":1392,"y":539},{"x":1392,"y":1440},{"x":1593,"y":1440},{"x":1593,"y":539}],"type":0,"horizontalFlip":true,"verticalFlip":false,"rotate":{"zAxis":179.99996185302734}}],"models":[{"cid":100,"modelSize":{"dx":504,"dy":287,"dz":0},"point":{"x":-1136,"y":-1414,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":120,"modelSize":{"dx":1991,"dy":539,"dz":0},"point":{"x":1112,"y":-530,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":90.0}},{"cid":318,"modelSize":{"dx":1039,"dy":1523,"dz":0},"point":{"x":-85,"y":-784,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":319,"modelSize":{"dx":283,"dy":337,"dz":0},"point":{"x":-748,"y":-1368,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":319,"modelSize":{"dx":292,"dy":346,"dz":0},"point":{"x":588,"y":-1371,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}}],"areaPoints":[{"x":1388,"y":1553},{"x":1388,"y":-143},{"x":1388,"y":-146},{"x":1388,"y":-1553},{"x":-1388,"y":-1553},{"x":-1388,"y":1153},{"x":-1388,"y":1156},{"x":-1388,"y":1553},{"x":1388,"y":1553}],"wallLines":[{"startPoint":{"x":1388,"y":1553,"z":0},"endPoint":{"x":1388,"y":-143,"z":0}},{"startPoint":{"x":1388,"y":-143,"z":0},"endPoint":{"x":1388,"y":-146,"z":0}},{"startPoint":{"x":1388,"y":-146,"z":0},"endPoint":{"x":1388,"y":-1553,"z":0}},{"startPoint":{"x":1388,"y":-1553,"z":0},"endPoint":{"x":-1388,"y":-1553,"z":0}},{"startPoint":{"x":-1388,"y":-1553,"z":0},"endPoint":{"x":-1388,"y":1153,"z":0}},{"startPoint":{"x":-1388,"y":1153,"z":0},"endPoint":{"x":-1388,"y":1156,"z":0}},{"startPoint":{"x":-1388,"y":1156,"z":0},"endPoint":{"x":-1388,"y":1553,"z":0}},{"startPoint":{"x":-1388,"y":1553,"z":0},"endPoint":{"x":1388,"y":1553,"z":0}}],"roomBound":{"x":-1388,"y":-1553,"dx":2777,"dy":3107},"roomLocation":{"x":1271,"y":-2503,"z":0},"roomId":2998022}'
    test_json_data = '{"dataId":2998,"houseId":1023468,"functionZones":[{"id":48,"mainPositions":[{"sample":true,"main":true,"tag":"lRpr86ud","cid":318,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":747,"y":-961,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1216,"dy":1438,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"positions":[{"sample":true,"main":true,"tag":"lRpr86ud","cid":318,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":747,"y":-961,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1216,"dy":1438,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"PCTLpIFy","cid":319,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-45,"y":-1475,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":329,"dy":398,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-792.9151343311742,"m31":-514.6560965068802,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-792,"y":-514,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"YR3drL4H","cid":319,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":1577,"y":-1475,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":423,"dy":408,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":829.8521747503291,"m31":-514.6560965068802,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":829,"y":-514,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":747,"y":-961,"z":0},"bound":{"x":-210,"y":-1680,"dx":1999,"dy":1438},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":0},{"id":38,"mainPositions":[{"sample":true,"main":true,"tag":"36VpwkUF","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1651,"y":-1492,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1291,"dy":419,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":true,"tag":"Q4vtS7Ej","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"location":{"x":-1264,"y":-656,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1274,"dy":408,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"positions":[{"sample":true,"main":true,"tag":"36VpwkUF","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1651,"y":-1492,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1291,"dy":419,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":-632.0165371111075,"m32":645.6375831695373,"m33":1.0,"properties":18},"relativeLocation":{"x":0,"y":-632,"z":645},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":true,"tag":"Q4vtS7Ej","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"location":{"x":-1264,"y":-656,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1274,"dy":408,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":0.0,"m01":1.0,"m02":0.0,"m03":0.0,"m10":-1.0,"m11":0.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":386.83770805938593,"m31":204.31569087643538,"m32":645.6375831695373,"m33":1.0,"properties":18},"relativeLocation":{"x":386,"y":204,"z":645},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":false}],"center":{"x":-1651,"y":-860,"z":-645},"bound":{"x":-2297,"y":-1702,"dx":1291,"dy":1683},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":1},{"id":54,"mainPositions":[{"sample":true,"main":true,"tag":"SpeGJdME","cid":108,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":659,"y":1533,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1738,"dy":279,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"positions":[{"sample":true,"main":true,"tag":"SpeGJdME","cid":108,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":659,"y":1533,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1738,"dy":279,"dz":0},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":659,"y":1533,"z":0},"bound":{"x":-209,"y":1393,"dx":1738,"dy":279},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"tag":2},{"id":99,"mainPositions":[{"sample":true,"main":true,"tag":"M4NocGs9","cid":96,"metaData":{"name":"UNKNOWN","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":0,"y":0,"z":0},"locationInHouse":{"x":0,"y":0,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1,"dy":1,"dz":1},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"isDrModel":false}],"positions":[],"center":{"x":0,"y":0,"z":0},"bound":{"x":0,"y":0,"dx":0,"dy":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":3}],"roomType":"BedRoom","drRoomType":"MasterBedRoom","roomName":"主卧","walls":[{"scid":1,"wallPoints":[{"x":-2401,"y":-1698},{"x":2401,"y":-1698},{"x":-2401,"y":-1901},{"x":2401,"y":-1901}]},{"scid":16,"wallPoints":[{"x":2298,"y":1701},{"x":2501,"y":1701},{"x":2298,"y":-1701},{"x":2501,"y":-1701}]},{"scid":12,"wallPoints":[{"x":-1161,"y":1901},{"x":2301,"y":1901},{"x":-1161,"y":1698},{"x":2301,"y":1698}]},{"scid":14,"wallPoints":[{"x":-2501,"y":291},{"x":-2298,"y":291},{"x":-2501,"y":-1701},{"x":-2298,"y":-1701}]},{"scid":13,"wallPoints":[{"x":-2301,"y":1901},{"x":-1158,"y":1901},{"x":-2301,"y":1698},{"x":-1158,"y":1698}]},{"scid":12,"wallPoints":[{"x":-2501,"y":1801},{"x":-2298,"y":1801},{"x":-2501,"y":288},{"x":-2298,"y":288}]}],"windows":[],"doors":[{"scid":14,"points":[{"x":-2497,"y":-987},{"x":-2497,"y":-186},{"x":-2296,"y":-186},{"x":-2296,"y":-987}],"type":0,"horizontalFlip":true,"verticalFlip":false,"rotate":{"zAxis":179.99996185302734}},{"scid":13,"points":[{"x":-2199,"y":1695},{"x":-2199,"y":1897},{"x":-1298,"y":1897},{"x":-1298,"y":1695}],"type":0,"horizontalFlip":true,"verticalFlip":false,"rotate":{"zAxis":-89.99995422363281}},{"scid":16,"points":[{"x":2297,"y":-798},{"x":2297,"y":1002},{"x":2499,"y":1002},{"x":2499,"y":-798}],"type":2,"horizontalFlip":false,"verticalFlip":false,"rotate":{"zAxis":7.62939453125E-6}}],"models":[{"cid":318,"modelSize":{"dx":1216,"dy":1438,"dz":0},"point":{"x":747,"y":-961,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":319,"modelSize":{"dx":329,"dy":398,"dz":0},"point":{"x":-45,"y":-1475,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":319,"modelSize":{"dx":423,"dy":408,"dz":0},"point":{"x":1577,"y":-1475,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":120,"modelSize":{"dx":1291,"dy":419,"dz":0},"point":{"x":-1651,"y":-1492,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}},{"cid":120,"modelSize":{"dx":1274,"dy":408,"dz":0},"point":{"x":-1264,"y":-656,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":90.0}},{"cid":108,"modelSize":{"dx":1738,"dy":279,"dz":0},"point":{"x":659,"y":1533,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":180.0}},{"cid":100,"modelSize":{"dx":564,"dy":245,"dz":0},"point":{"x":1992,"y":1552,"z":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":-180.0}}],"areaPoints":[{"x":-2298,"y":291},{"x":-2298,"y":1698},{"x":-1161,"y":1698},{"x":-1158,"y":1698},{"x":2298,"y":1698},{"x":2298,"y":-1698},{"x":-2298,"y":-1698},{"x":-2298,"y":288},{"x":-2298,"y":291}],"wallLines":[{"startPoint":{"x":-2298,"y":291,"z":0},"endPoint":{"x":-2298,"y":1698,"z":0}},{"startPoint":{"x":-2298,"y":1698,"z":0},"endPoint":{"x":-1161,"y":1698,"z":0}},{"startPoint":{"x":-1161,"y":1698,"z":0},"endPoint":{"x":-1158,"y":1698,"z":0}},{"startPoint":{"x":-1158,"y":1698,"z":0},"endPoint":{"x":2298,"y":1698,"z":0}},{"startPoint":{"x":2298,"y":1698,"z":0},"endPoint":{"x":2298,"y":-1698,"z":0}},{"startPoint":{"x":2298,"y":-1698,"z":0},"endPoint":{"x":-2298,"y":-1698,"z":0}},{"startPoint":{"x":-2298,"y":-1698,"z":0},"endPoint":{"x":-2298,"y":288,"z":0}},{"startPoint":{"x":-2298,"y":288,"z":0},"endPoint":{"x":-2298,"y":291,"z":0}}],"roomBound":{"x":-2298,"y":-1698,"dx":4597,"dy":3397},"roomLocation":{"x":5161,"y":-4448,"z":0},"roomId":2998082}'
    ck.init_room(room_str=test_json_data, roomTypeId=2)
    print(ck.data_infos)

    ck.api_distStat(0, 3, [318, 222, 180], res)
    print(res)
    print("res:", np.shape(res))
    res = np.reshape(res, [64, 64])
    print(res)
    for i in range(1):
        # jsonpath = r"D:\\workspace\\JSON_ORI\\layout-cnn-room-test-BedRoom-0.0-5.30\\107\\21448-room-json.json"
        # jsonpath = r"D:\workspace\JSON_ORI\layout-cnn-room-test-BedRoom-0.0-5.30\63\12775-room-json.json"

        # test_json_data = json.load(open(jsonpath, "r", encoding="utf-8"))
        # test_json_data = json.dumps(test_json_data, ensure_ascii=False)
        # test_json_data = '{"functionZones":[{"id":52,"mainPositions":[{"sample":true,"main":true,"tag":"H10k58yX","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 三人沙发  [原先名称]\u003d【豪兴  格美】现代   三人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-774,"y":-2789,"z":0},"locationInHouse":{"x":-3374,"y":-6209,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2030,"dy":940,"dz":940},"modelId":16973,"skuId":154834,"categoryId":303,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":true,"tag":"fsFQefYs","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 双人沙发  [原先名称]\u003d【豪兴  格美】现代   双人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-2291,"y":-1628,"z":0},"locationInHouse":{"x":-4891,"y":-5048,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1550,"dy":940,"dz":940},"modelId":16978,"skuId":154835,"categoryId":302,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"H10k58yX","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 三人沙发  [原先名称]\u003d【豪兴  格美】现代   三人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-774,"y":-2789,"z":0},"locationInHouse":{"x":-3374,"y":-6209,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2030,"dy":940,"dz":940},"modelId":16973,"skuId":154834,"categoryId":303,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":797,"y":287,"z":426},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":true},{"sample":true,"main":true,"tag":"fsFQefYs","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 双人沙发  [原先名称]\u003d【豪兴  格美】现代   双人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-2291,"y":-1628,"z":0},"locationInHouse":{"x":-4891,"y":-5048,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1550,"dy":940,"dz":940},"modelId":16978,"skuId":154835,"categoryId":302,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-363,"y":-1229,"z":426},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":5.81858356388325},"isDrModel":true},{"sample":false,"main":false,"tag":"W3QxT1nx","cid":89,"metaData":{"name":"装饰花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":663,"y":-2629,"z":560},"locationInHouse":{"x":-1936,"y":-6049,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":153,"dy":179,"dz":179},"modelId":16929,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":637,"y":1725,"z":986},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":true},{"sample":false,"main":false,"tag":"GVbjpfTV","cid":61,"metaData":{"name":"挂画","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-973,"y":-3249,"z":940},"locationInHouse":{"x":-3573,"y":-6669,"z":940},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1139,"dy":34,"dz":34},"modelId":16911,"skuId":0,"categoryId":88,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":1257,"y":88,"z":1366},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":true}],"center":{"x":-1062,"y":-1991,"z":-426},"bound":{"x":-2329,"y":-3294,"dx":2534,"dy":2604},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.71238898038469},"tag":2},{"id":54,"mainPositions":[{"sample":true,"main":true,"tag":"gRjJpIOG","cid":108,"metaData":{"name":"【简欧  英伦系列】欧式   500-600mm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.212419684575316},"location":{"x":-993,"y":319,"z":0},"locationInHouse":{"x":-3593,"y":-3100,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2360,"dy":401,"dz":401},"modelId":16962,"skuId":153300,"categoryId":327,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"gRjJpIOG","cid":108,"metaData":{"name":"【简欧  英伦系列】欧式   500-600mm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.212419684575316},"location":{"x":-993,"y":319,"z":0},"locationInHouse":{"x":-3593,"y":-3100,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2360,"dy":401,"dz":401},"modelId":16962,"skuId":153300,"categoryId":327,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":false,"tag":"BUuWEPX1","cid":99,"metaData":{"name":"电视","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.07076562260427},"location":{"x":-983,"y":477,"z":850},"locationInHouse":{"x":-3583,"y":-2942,"z":850},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1371,"dy":84,"dz":84},"modelId":16912,"skuId":0,"categoryId":132,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":120,"y":-102,"z":850},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.858345938028954},"isDrModel":true}],"center":{"x":-993,"y":319,"z":0},"bound":{"x":-2173,"y":-868,"dx":2360,"dy":2376},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.212419684575316},"tag":1},{"id":40,"mainPositions":[{"sample":true,"main":true,"tag":"vAPKFQli","cid":118,"metaData":{"name":"余颢凌-情迷地中海-玄关柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"location":{"x":2724,"y":-4062,"z":0},"locationInHouse":{"x":124,"y":-7482,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2000,"dy":360,"dz":360},"modelId":24590,"skuId":0,"categoryId":537,"isCustom":true,"spuId":8100439,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"vAPKFQli","cid":118,"metaData":{"name":"余颢凌-情迷地中海-玄关柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"location":{"x":2724,"y":-4062,"z":0},"locationInHouse":{"x":124,"y":-7482,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2000,"dy":360,"dz":360},"modelId":24590,"skuId":0,"categoryId":537,"isCustom":true,"spuId":8100439,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"center":{"x":2724,"y":-4062,"z":0},"bound":{"x":1724,"y":-4971,"dx":1999,"dy":1817},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"tag":0},{"id":99,"mainPositions":[{"sample":false,"main":false,"tag":"DCaRXj8I","cid":-1,"metaData":{"name":"UNKNOWN","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":0,"y":0,"z":0},"locationInHouse":{"x":0,"y":0,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1,"dy":1,"dz":1},"modelId":0,"skuId":0,"categoryId":0,"isCustom":false,"spuId":0,"mark":0},"isDrModel":false}],"positions":[{"sample":false,"main":false,"tag":"sFfMDUts","cid":173,"metaData":{"name":"[修正名称]\u003d【喜舍  】现代  塑料 塑料吸顶灯  [原先名称]\u003d【天路行  】现代  塑料 塑料吸顶灯","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-2567,"y":-131,"z":2600},"locationInHouse":{"x":-5167,"y":-3551,"z":2600},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":400,"dy":400,"dz":400},"modelId":23661,"skuId":158075,"categoryId":457,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-2567,"y":-131,"z":2600},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":false,"main":false,"tag":"Cg3TqaSP","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2313,"y":-2659,"z":560},"locationInHouse":{"x":-4913,"y":-6079,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":179,"dy":151,"dz":151},"modelId":16921,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-2313,"y":-2659,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"HQi02x8S","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-553,"y":-1259,"z":570},"locationInHouse":{"x":-3153,"y":-4679,"z":570},"scale":{"x":0.978,"y":0.718,"z":0.737},"size":{"dx":432,"dy":241,"dz":241},"modelId":16914,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-553,"y":-1259,"z":570},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":false,"main":false,"tag":"GVn4hEai","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-853,"y":-1209,"z":490},"locationInHouse":{"x":-3453,"y":-4629,"z":490},"scale":{"x":1.0,"y":1.0,"z":1.462},"size":{"dx":217,"dy":218,"dz":218},"modelId":16924,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-853,"y":-1209,"z":490},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"kOaNMYyq","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":6.051747862571702},"location":{"x":-1133,"y":-1299,"z":490},"locationInHouse":{"x":-3733,"y":-4719,"z":490},"scale":{"x":0.906481,"y":0.999564,"z":0.646},"size":{"dx":306,"dy":216,"dz":216},"modelId":16925,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1133,"y":-1299,"z":490},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":6.051747862571702},"isDrModel":true},{"sample":false,"main":false,"tag":"YsYJyGrZ","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.296630477778912},"location":{"x":-1613,"y":-2049,"z":10},"locationInHouse":{"x":-4213,"y":-5469,"z":10},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":468,"dy":485,"dz":485},"modelId":16926,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1613,"y":-2049,"z":10},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.296630477778912},"isDrModel":true},{"sample":false,"main":false,"tag":"abJ4TM7O","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":573,"y":-2869,"z":560},"locationInHouse":{"x":-2026,"y":-6289,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":560,"dy":291,"dz":291},"modelId":16928,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":573,"y":-2869,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"5vga1vBI","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2403,"y":-2789,"z":130},"locationInHouse":{"x":-5003,"y":-6209,"z":130},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":265,"dy":208,"dz":208},"modelId":16931,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":1},"relativeLocation":{"x":-2403,"y":-2789,"z":130},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"xrdLvRTj","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2583,"y":-2639,"z":550},"locationInHouse":{"x":-5183,"y":-6059,"z":550},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":136,"dy":132,"dz":132},"modelId":16918,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-2583,"y":-2639,"z":550},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"SvZ9Jpxs","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":603,"y":-2859,"z":130},"locationInHouse":{"x":-1996,"y":-6279,"z":130},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":265,"dy":208,"dz":208},"modelId":16931,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":2},"relativeLocation":{"x":603,"y":-2859,"z":130},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"ZeV2wrIy","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1753,"y":309,"z":560},"locationInHouse":{"x":-4353,"y":-3110,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":216,"dy":216,"dz":216},"modelId":16905,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1753,"y":309,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"Zz1ttyib","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.0354133288802956},"location":{"x":-403,"y":289,"z":320},"locationInHouse":{"x":-3003,"y":-3130,"z":320},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":172,"dy":237,"dz":237},"modelId":16909,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-403,"y":289,"z":320},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.0354133288802956},"isDrModel":true},{"sample":false,"main":false,"tag":"cdl1nV81","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":5.884755627372201},"location":{"x":-1393,"y":329,"z":320},"locationInHouse":{"x":-3993,"y":-3090,"z":320},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":212,"dy":336,"dz":336},"modelId":16906,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1393,"y":329,"z":320},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":5.884755627372201},"isDrModel":true},{"sample":false,"main":false,"tag":"fwpVHPpZ","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-213,"y":269,"z":540},"locationInHouse":{"x":-2813,"y":-3150,"z":540},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":289,"dy":320,"dz":320},"modelId":16963,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":1},"relativeLocation":{"x":-213,"y":269,"z":540},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"pDuChmyC","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":2820,"y":-3812,"z":1020},"locationInHouse":{"x":220,"y":-7232,"z":1020},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":289,"dy":320,"dz":320},"modelId":16963,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":2},"relativeLocation":{"x":2820,"y":-3812,"z":1020},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":false,"tag":"hU5IaADb","cid":670,"metaData":{"name":"定制背景墙","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-943,"y":411,"z":80},"locationInHouse":{"x":-3543,"y":-3009,"z":80},"scale":{"x":0.952154,"y":0.091743,"z":0.909795},"size":{"dx":4201,"dy":218,"dz":218},"modelId":24384,"skuId":162233,"categoryId":710,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-943,"y":411,"z":80},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"lblpxEC6","cid":677,"metaData":{"name":"艾佳定制吊顶","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"location":{"x":-883,"y":-1399,"z":2340},"locationInHouse":{"x":-3483,"y":-4819,"z":2340},"scale":{"x":0.988924,"y":0.968223,"z":1.0},"size":{"dx":3792,"dy":4028,"dz":4028},"modelId":24626,"skuId":164590,"categoryId":709,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-883,"y":-1399,"z":2340},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"isDrModel":true},{"sample":true,"main":false,"tag":"nwXBFOob","cid":733,"metaData":{"name":"艾佳定制 石材波打线-10cm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-823,"y":-2919,"z":0},"locationInHouse":{"x":-3423,"y":-6339,"z":0},"scale":{"x":1.380434,"y":1.273076,"z":1.0},"size":{"dx":100,"dy":3000,"dz":3000},"modelId":24447,"skuId":162147,"categoryId":703,"isCustom":false,"spuId":0,"mark":1},"relativeLocation":{"x":-823,"y":-2919,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":true,"main":false,"tag":"5SP2avVz","cid":733,"metaData":{"name":"艾佳定制 石材波打线-10cm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-823,"y":341,"z":0},"locationInHouse":{"x":-3423,"y":-3079,"z":0},"scale":{"x":1.366746,"y":1.273076,"z":1.0},"size":{"dx":100,"dy":3000,"dz":3000},"modelId":24447,"skuId":162147,"categoryId":703,"isCustom":false,"spuId":0,"mark":2},"relativeLocation":{"x":-823,"y":341,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":true,"main":false,"tag":"059dkj6G","cid":733,"metaData":{"name":"艾佳定制 石材波打线-10cm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2663,"y":-1289,"z":0},"locationInHouse":{"x":-5263,"y":-4709,"z":0},"scale":{"x":1.430801,"y":1.135553,"z":1.0},"size":{"dx":100,"dy":3000,"dz":3000},"modelId":24447,"skuId":162147,"categoryId":703,"isCustom":false,"spuId":0,"mark":3},"relativeLocation":{"x":-2663,"y":-1289,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"lPSNFPMf","cid":89,"metaData":{"name":"装饰花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":896,"y":279,"z":730},"locationInHouse":{"x":-1703,"y":-3140,"z":730},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":535,"dy":461,"dz":461},"modelId":16910,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":896,"y":279,"z":730},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"zjx4NSH0","cid":89,"metaData":{"name":"装饰花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-903,"y":-1509,"z":550},"locationInHouse":{"x":-3503,"y":-4929,"z":550},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":321,"dy":357,"dz":357},"modelId":16936,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-903,"y":-1509,"z":550},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"center":{"x":0,"y":0,"z":0},"bound":{"x":-2943,"y":-3836,"dx":3999,"dy":5094},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":3}],"usageId":1,"roomName":"客厅","walls":[{"scid":12,"wallPoints":[{"x":-1930,"y":1945},{"x":-1930,"y":2145},{"x":1730,"y":2145},{"x":1930,"y":1945}]},{"scid":1,"wallPoints":[{"x":1930,"y":1945},{"x":1730,"y":2145},{"x":1730,"y":3235},{"x":1930,"y":3215}]},{"scid":12,"wallPoints":[{"x":-1930,"y":-2145},{"x":-1930,"y":-1945},{"x":2170,"y":-1945},{"x":1970,"y":-2145}]},{"scid":16,"wallPoints":[{"x":-2130,"y":-1945},{"x":-2130,"y":1945},{"x":-1930,"y":1945},{"x":-1930,"y":-1945}]}],"windows":[],"doors":[{"scid":16,"points":[{"x":-2133,"y":-910},{"x":-2133,"y":1289},{"x":-1932,"y":1289},{"x":-1932,"y":-910}],"type":2,"horizontalFlip":false,"verticalFlip":false,"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707956610046239}}],"areaPoints":[{"x":-1930,"y":1945},{"x":1930,"y":1945},{"x":1930,"y":1985},{"x":1932,"y":1984},{"x":1934,"y":1984},{"x":1940,"y":1983},{"x":1942,"y":1982},{"x":1947,"y":1980},{"x":1949,"y":1979},{"x":1954,"y":1976},{"x":1956,"y":1975},{"x":1960,"y":1971},{"x":1961,"y":1969},{"x":1964,"y":1964},{"x":1965,"y":1962},{"x":1967,"y":1957},{"x":1968,"y":1955},{"x":1969,"y":1949},{"x":1969,"y":1947},{"x":1970,"y":1945},{"x":1970,"y":-1945},{"x":-1930,"y":-1945},{"x":-1930,"y":1945}],"roomId":0}'
        # test_json_data = '{"functionZones":[{"id":48,"positions":[{"sample":true,"main":true,"tag":"JfudZDnP","cid":40,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":1080,"y":85,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1238,"dy":1934,"dz":1934},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"JMV66PL1","cid":330,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":-134,"y":798,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1047,"dy":510,"dz":510},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":1215,"y":-712,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":1080,"y":85,"z":0},"bound":{"x":-658,"y":-881,"dx":2358,"dy":1934},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"tag":0},{"id":38,"positions":[{"sample":true,"main":true,"tag":"6ollMzLi","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":-1441,"y":628,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1202,"dy":884,"dz":884},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":-1441,"y":628,"z":0},"bound":{"x":-2042,"y":186,"dx":1202,"dy":884},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"tag":1},{"id":99,"positions":[],"center":{"x":0,"y":0,"z":0},"bound":{"x":0,"y":0,"dx":0,"dy":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":2}],"usageId":4,"roomName":"儿童房","walls":[{"scid":14,"wallPoints":[{"x":1842,"y":-1062},{"x":2089,"y":-1062},{"x":1842,"y":-1608},{"x":2089,"y":-1608}]},{"scid":1,"wallPoints":[{"x":1706,"y":1400},{"x":2130,"y":1400},{"x":1706,"y":-166},{"x":2130,"y":-166}]},{"scid":12,"wallPoints":[{"x":2054,"y":1411},{"x":2207,"y":1411},{"x":2054,"y":-6534},{"x":2207,"y":-6534}]},{"scid":11,"wallPoints":[{"x":-6319,"y":1393},{"x":2182,"y":1393},{"x":-6319,"y":1063},{"x":2182,"y":1063}]},{"scid":11,"wallPoints":[{"x":-2373,"y":1411},{"x":-2052,"y":1411},{"x":-2373,"y":306},{"x":-2052,"y":306}]},{"scid":11,"wallPoints":[{"x":-2143,"y":377},{"x":-2052,"y":377},{"x":-2143,"y":-1191},{"x":-2052,"y":-1191}]},{"scid":11,"wallPoints":[{"x":-2362,"y":-1056},{"x":1995,"y":-1056},{"x":-2362,"y":-1192},{"x":1995,"y":-1192}]},{"scid":11,"wallPoints":[{"x":-970,"y":-1062},{"x":-825,"y":-1062},{"x":-970,"y":-2827},{"x":-825,"y":-2827}]}],"windows":[{"scid":1,"points":[{"x":2033,"y":-1063},{"x":2033,"y":-181},{"x":2199,"y":-181},{"x":2199,"y":-1063}],"type":0,"horizontalFlip":false,"verticalFlip":false}],"doors":[{"scid":11,"points":[{"x":-1942,"y":-1185},{"x":-1942,"y":-1031},{"x":-1066,"y":-1031},{"x":-1066,"y":-1185}],"type":0,"horizontalFlip":true,"verticalFlip":true,"rotate":{"zAxis":90.0}}],"areaPoints":[{"x":1995,"y":-1063},{"x":1995,"y":-1056},{"x":-2053,"y":-1056},{"x":-2053,"y":306},{"x":-2053,"y":376},{"x":-2053,"y":1063},{"x":1706,"y":1063},{"x":1706,"y":-166},{"x":2053,"y":-166},{"x":2053,"y":-1063},{"x":1995,"y":-1063}],"roomId":344042}'
        # test_json_data = '{"functionZones":[{"id":38,"mainPositions":[{"sample":false,"main":true,"tag":"cM9lgrL5","cid":120,"metaData":{"name":"东韵雅舍-主卧衣柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-548,"y":-1394,"z":0},"locationInHouse":{"x":-4503,"y":581,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2080,"dy":600,"dz":600},"modelId":24556,"skuId":0,"categoryId":539,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":30},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"cM9lgrL5","cid":120,"metaData":{"name":"东韵雅舍-主卧衣柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-548,"y":-1394,"z":0},"locationInHouse":{"x":-4503,"y":581,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2080,"dy":600,"dz":600},"modelId":24556,"skuId":0,"categoryId":539,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":0.0,"m31":0.0,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"center":{"x":-548,"y":-1394,"z":0},"bound":{"x":-1588,"y":-1694,"dx":2080,"dy":600},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":0},{"id":99,"mainPositions":[{"sample":false,"main":false,"tag":"wXgG2t8Z","cid":-1,"metaData":{"name":"UNKNOWN","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":0,"y":0,"z":0},"locationInHouse":{"x":0,"y":0,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1,"dy":1,"dz":1},"modelId":0,"skuId":0,"categoryId":0,"isCustom":false,"spuId":0,"mark":0},"isDrModel":false}],"positions":[{"sample":false,"main":false,"tag":"pfmmP6co","metaData":{"name":"[喜舍专供]射灯","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1513,"y":1609,"z":2695},"locationInHouse":{"x":-5468,"y":3584,"z":2695},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":90,"dy":90,"dz":90},"modelId":20405,"skuId":0,"categoryId":173,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-1513.541015625,"m31":1609.563232421875,"m32":2695.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-1513,"y":1609,"z":2695},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"UUfM8poe","metaData":{"name":"【喜舍专供】摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.7123900456491254},"location":{"x":-1440,"y":-1233,"z":546},"locationInHouse":{"x":-5395,"y":741,"z":546},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":274,"dy":182,"dz":182},"modelId":20032,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0653027624914126E-6,"m01":-0.9999999999994326,"m02":0.0,"m03":0.0,"m10":0.9999999999994326,"m11":1.0653027624914126E-6,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-1440.0498046875,"m31":-1233.7828369140625,"m32":546.1849975585938,"m33":1.0,"properties":18},"relativeLocation":{"x":-1440,"y":-1233,"z":546},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.7123900456491254},"isDrModel":true},{"sample":false,"main":false,"tag":"SlmsuAsb","metaData":{"name":"【喜舍专供】方形地毯","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.712389113542744},"location":{"x":-781,"y":-173,"z":0},"locationInHouse":{"x":-4736,"y":1801,"z":0},"scale":{"x":0.961271,"y":0.730131,"z":1.0},"size":{"dx":1854,"dy":2718,"dz":2718},"modelId":20053,"skuId":0,"categoryId":76,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.3328003749250113E-7,"m01":-0.9999999999999911,"m02":0.0,"m03":0.0,"m10":0.9999999999999911,"m11":1.3328003749250113E-7,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-781.4833984375,"m31":-173.4306640625,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-781,"y":-173,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.712389113542744},"isDrModel":true},{"sample":false,"main":false,"tag":"ZQXKV0BC","metaData":{"name":"【喜舍专供】书籍杂志","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1440,"y":-1197,"z":1427},"locationInHouse":{"x":-5395,"y":777,"z":1427},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":141,"dy":160,"dz":160},"modelId":20055,"skuId":0,"categoryId":90,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-1440.564453125,"m31":-1197.4473876953125,"m32":1427.463134765625,"m33":1.0,"properties":18},"relativeLocation":{"x":-1440,"y":-1197,"z":1427},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"qXiGFJ1c","metaData":{"name":"【喜舍专供】陶瓷器皿","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.839724620703634},"location":{"x":-1438,"y":-1181,"z":986},"locationInHouse":{"x":-5393,"y":793,"z":986},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":182,"dy":151,"dz":151},"modelId":20058,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.7660442719342556,"m01":-0.6427878136964921,"m02":0.0,"m03":0.0,"m10":0.6427878136964921,"m11":-0.7660442719342556,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-1438.8740234375,"m31":-1181.6617431640625,"m32":986.1841430664062,"m33":1.0,"properties":18},"relativeLocation":{"x":-1438,"y":-1181,"z":986},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.839724620703634},"isDrModel":true},{"sample":false,"main":false,"tag":"FtlI4NJV","metaData":{"name":"【喜舍专供】装饰托盘","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.14159078937703},"location":{"x":-788,"y":322,"z":615},"locationInHouse":{"x":-4743,"y":2297,"z":615},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":480,"dy":376,"dz":376},"modelId":20063,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999982623,"m01":1.8642127633909198E-6,"m02":0.0,"m03":0.0,"m10":-1.8642127633909198E-6,"m11":-0.9999999999982623,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-788.2412109375,"m31":322.9755859375,"m32":615.5826416015625,"m33":1.0,"properties":18},"relativeLocation":{"x":-788,"y":322,"z":615},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.14159078937703},"isDrModel":true},{"sample":false,"main":false,"tag":"5dAu3LMu","metaData":{"name":"【喜舍专供】台灯","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"location":{"x":-2075,"y":1583,"z":495},"locationInHouse":{"x":-6030,"y":3558,"z":495},"scale":{"x":0.64,"y":0.64,"z":0.64},"size":{"dx":322,"dy":322,"dz":322},"modelId":20065,"skuId":0,"categoryId":170,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999996808,"m01":7.989483271744533E-7,"m02":0.0,"m03":0.0,"m10":-7.989483271744533E-7,"m11":-0.9999999999996808,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-2075.998046875,"m31":1583.304443359375,"m32":495.4797058105469,"m33":1.0,"properties":18},"relativeLocation":{"x":-2075,"y":1583,"z":495},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"isDrModel":true},{"sample":false,"main":false,"tag":"eMX8MUhv","metaData":{"name":"【喜舍专供】相框","rotate":{"xAxis":0.17453597118992972,"yAxis":0.0,"zAxis":3.14159078937703},"location":{"x":-1880,"y":1515,"z":501},"locationInHouse":{"x":-5835,"y":3490,"z":501},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":200,"dy":36,"dz":36},"modelId":20066,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999982623,"m01":1.835890196603814E-6,"m02":3.2372274125185935E-7,"m03":0.0,"m10":-1.8642127633909198E-6,"m11":-0.9848072240752292,"m12":-0.17365117738087985,"m13":0.0,"m20":0.0,"m21":-0.1736511773811816,"m22":0.9848072240769405,"m23":0.0,"m30":-1880.8291015625,"m31":1515.58349609375,"m32":501.66973876953125,"m33":1.0,"properties":18},"relativeLocation":{"x":-1880,"y":1515,"z":501},"relativeRotate":{"xAxis":0.17453597118992972,"yAxis":0.0,"zAxis":3.14159078937703},"isDrModel":true},{"sample":false,"main":false,"tag":"iyQvmq2X","metaData":{"name":"【喜舍专供】陶瓷器皿","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1978,"y":1347,"z":560},"locationInHouse":{"x":-5933,"y":3322,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":141,"dy":129,"dz":129},"modelId":20068,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-1978.8388671875,"m31":1347.606201171875,"m32":560.1215209960938,"m33":1.0,"properties":18},"relativeLocation":{"x":-1978,"y":1347,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"67VXJmtn","metaData":{"name":"【喜舍专供】花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2219,"y":1471,"z":491},"locationInHouse":{"x":-6174,"y":3446,"z":491},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":178,"dy":181,"dz":181},"modelId":20071,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-2219.3056640625,"m31":1471.09423828125,"m32":491.44696044921875,"m33":1.0,"properties":18},"relativeLocation":{"x":-2219,"y":1471,"z":491},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"smKmNgmd","metaData":{"name":"n【兴利  璞极】中式  实木 床","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"location":{"x":-825,"y":624,"z":0},"locationInHouse":{"x":-4780,"y":2599,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":3220,"dy":2139,"dz":2139},"modelId":9177,"skuId":43374,"categoryId":305,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999996808,"m01":7.989483271744533E-7,"m02":0.0,"m03":0.0,"m10":-7.989483271744533E-7,"m11":-0.9999999999996808,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-825.0,"m31":624.5,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-825,"y":624,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"isDrModel":true},{"sample":false,"main":false,"tag":"DFm6S5cX","metaData":{"name":"n【兴利  璞极】中式  实木 床头柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"location":{"x":-2067,"y":1510,"z":-10},"locationInHouse":{"x":-6022,"y":3485,"z":-10},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":580,"dy":468,"dz":468},"modelId":9182,"skuId":43379,"categoryId":333,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999996808,"m01":7.989483271744533E-7,"m02":0.0,"m03":0.0,"m10":-7.989483271744533E-7,"m11":-0.9999999999996808,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-2067.96630859375,"m31":1510.2822265625,"m32":-10.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-2067,"y":1510,"z":-10},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"isDrModel":true},{"sample":false,"main":false,"tag":"YSsJCjNu","metaData":{"name":"【天路行  】欧式  布艺 吊灯（功率:20W，适用面积:8-15㎡）","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-881,"y":489,"z":2126},"locationInHouse":{"x":-4836,"y":2464,"z":2126},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":703,"dy":629,"dz":629},"modelId":14995,"skuId":148515,"categoryId":456,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-881.48193359375,"m31":489.020751953125,"m32":2126.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-881,"y":489,"z":2126},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"Yld14SNf","metaData":{"name":"【欧博莱  经济版窗帘】语希(布)咖啡","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.712389113542744},"location":{"x":-2335,"y":886,"z":0},"locationInHouse":{"x":-6290,"y":2861,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":745,"dy":215,"dz":215},"modelId":21713,"skuId":154484,"categoryId":454,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.3328003749250113E-7,"m01":-0.9999999999999911,"m02":0.0,"m03":0.0,"m10":0.9999999999999911,"m11":1.3328003749250113E-7,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-2335.82421875,"m31":886.7998046875,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-2335,"y":886,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.712389113542744},"isDrModel":true},{"sample":false,"main":false,"tag":"1zCFRGmS","metaData":{"name":"[修正名称]\u003d【欧博莱  舒适尊享版窗帘】沁心(纱)本白  [原先名称]\u003d【欧博莱  经济版窗帘】沁心(纱)本白","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.712389113542744},"location":{"x":-2335,"y":-863,"z":0},"locationInHouse":{"x":-6290,"y":1111,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":745,"dy":215,"dz":215},"modelId":21576,"skuId":154526,"categoryId":454,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.3328003749250113E-7,"m01":-0.9999999999999911,"m02":0.0,"m03":0.0,"m10":0.9999999999999911,"m11":1.3328003749250113E-7,"m12":0.0,"m13":0.0,"m20":0.0,"m21":-0.0,"m22":1.0,"m23":0.0,"m30":-2335.82421875,"m31":-863.1929931640625,"m32":0.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-2335,"y":-863,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.712389113542744},"isDrModel":true},{"sample":false,"main":false,"tag":"rkQ3J97Y","metaData":{"name":"喜舍【罗莱  成人款】中式  纯棉 床品套件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"location":{"x":-861,"y":577,"z":215},"locationInHouse":{"x":-4816,"y":2552,"z":215},"scale":{"x":1.111062,"y":0.904182,"z":1.0},"size":{"dx":1562,"dy":1984,"dz":1984},"modelId":22625,"skuId":156225,"categoryId":452,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999996808,"m01":7.989483271744533E-7,"m02":0.0,"m03":0.0,"m10":-7.989483271744533E-7,"m11":-0.9999999999996808,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-861.75830078125,"m31":577.4580078125,"m32":215.80648803710938,"m33":1.0,"properties":18},"relativeLocation":{"x":-861,"y":577,"z":215},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141591854641466},"isDrModel":true},{"sample":false,"main":false,"tag":"W89Tiofn","metaData":{"name":"【艾佳定制】石膏板平顶","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-545,"y":0,"z":2720},"locationInHouse":{"x":-4500,"y":1975,"z":2720},"scale":{"x":0.9,"y":0.8921053,"z":1.0},"size":{"dx":4200,"dy":3800,"dz":3800},"modelId":22639,"skuId":161465,"categoryId":714,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-545.0,"m31":0.0,"m32":2720.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-545,"y":0,"z":2720},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"gJo7d3Uv","metaData":{"name":"定制吊顶造型","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-545,"y":0,"z":2384},"locationInHouse":{"x":-4500,"y":1975,"z":2384},"scale":{"x":0.99473685,"y":0.80714285,"z":1.0},"size":{"dx":3800,"dy":4200,"dz":4200},"modelId":24337,"skuId":162418,"categoryId":714,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":1.0,"m01":0.0,"m02":0.0,"m03":0.0,"m10":-0.0,"m11":1.0,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-545.0,"m31":0.0,"m32":2384.0,"m33":1.0,"properties":18},"relativeLocation":{"x":-545,"y":0,"z":2384},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":false,"tag":"LeZ0LHfc","cid":670,"metaData":{"name":"艾佳定制背景墙","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.14159078937703},"location":{"x":-843,"y":1659,"z":10},"locationInHouse":{"x":-4798,"y":3634,"z":10},"scale":{"x":1.0746784,"y":1.0,"z":0.958768},"size":{"dx":3000,"dy":450,"dz":450},"modelId":24357,"skuId":162420,"categoryId":710,"isCustom":false,"spuId":0,"mark":0},"transform":{"m00":-0.9999999999982623,"m01":1.8642127633909198E-6,"m02":0.0,"m03":0.0,"m10":-1.8642127633909198E-6,"m11":-0.9999999999982623,"m12":0.0,"m13":0.0,"m20":0.0,"m21":0.0,"m22":1.0,"m23":0.0,"m30":-843.78955078125,"m31":1659.183349609375,"m32":10.321968078613281,"m33":1.0,"properties":18},"relativeLocation":{"x":-843,"y":1659,"z":10},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.14159078937703},"isDrModel":true}],"center":{"x":0,"y":0,"z":0},"bound":{"x":-2455,"y":1434,"dx":3224,"dy":449},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":1}],"usageId":2,"roomName":"主卧","walls":[{"scid":1,"wallPoints":[{"x":2715,"y":-3270},{"x":2515,"y":-3070},{"x":2515,"y":-1270},{"x":2715,"y":-1270}]},{"scid":1,"wallPoints":[{"x":-1435,"y":1610},{"x":-1235,"y":1810},{"x":-1065,"y":1810},{"x":-1165,"y":1610}]},{"scid":1,"wallPoints":[{"x":-1435,"y":1610},{"x":-1435,"y":3070},{"x":-1235,"y":3170},{"x":-1235,"y":1810}]},{"scid":1,"wallPoints":[{"x":-1235,"y":-3270},{"x":-1235,"y":-3070},{"x":2515,"y":-3070},{"x":2715,"y":-3270}]},{"scid":14,"wallPoints":[{"x":-1435,"y":-3170},{"x":-1435,"y":-1970},{"x":-1235,"y":-1770},{"x":-1235,"y":-3070}]},{"scid":14,"wallPoints":[{"x":-2615,"y":-1970},{"x":-2515,"y":-1770},{"x":-1235,"y":-1770},{"x":-1435,"y":-1970}]},{"scid":13,"wallPoints":[{"x":-2515,"y":3070},{"x":-2715,"y":3270},{"x":-1435,"y":3270},{"x":-1435,"y":3070}]},{"scid":11,"wallPoints":[{"x":-2715,"y":-1770},{"x":-2715,"y":3270},{"x":-2515,"y":3070},{"x":-2515,"y":-1770}]},{"scid":14,"wallPoints":[{"x":-1165,"y":-90},{"x":-1165,"y":1610},{"x":-965,"y":1610},{"x":-965,"y":110}]},{"scid":14,"wallPoints":[{"x":-1165,"y":-90},{"x":-965,"y":110},{"x":1605,"y":110},{"x":1705,"y":-90}]},{"scid":14,"wallPoints":[{"x":1705,"y":-90},{"x":1805,"y":110},{"x":2515,"y":110},{"x":2065,"y":-90}]},{"scid":1,"wallPoints":[{"x":2065,"y":-1270},{"x":2065,"y":-90},{"x":2715,"y":10},{"x":2715,"y":-1270}]}],"windows":[{"scid":1,"points":[{"x":2519,"y":-2663},{"x":2519,"y":-1263},{"x":2720,"y":-1263},{"x":2720,"y":-2663}],"type":0,"horizontalFlip":false,"verticalFlip":false,"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707956610046239}}],"doors":[{"scid":11,"points":[{"x":-2716,"y":-1621},{"x":-2716,"y":-721},{"x":-2516,"y":-721},{"x":-2516,"y":-1621}],"type":0,"horizontalFlip":false,"verticalFlip":false,"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":-1.570796193636842}},{"scid":14,"points":[{"x":-194,"y":-83},{"x":-194,"y":117},{"x":705,"y":117},{"x":705,"y":-83}],"type":3,"horizontalFlip":false,"verticalFlip":false,"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0}}],"models":[],"areaPoints":[{"x":2515,"y":-1270},{"x":2515,"y":-3070},{"x":-1235,"y":-3070},{"x":-1235,"y":-1770},{"x":-2515,"y":-1770},{"x":-2515,"y":3070},{"x":-1435,"y":3070},{"x":-1435,"y":1610},{"x":-1165,"y":1610},{"x":-1165,"y":-90},{"x":1705,"y":-90},{"x":2065,"y":-90},{"x":2065,"y":-1270},{"x":2515,"y":-1270}],"wallLines":[{"startPoint":{"x":-1235,"y":-3070,"z":0},"endPoint":{"x":2515,"y":-3070,"z":0}},{"startPoint":{"x":2515,"y":-3070,"z":0},"endPoint":{"x":2515,"y":-1270,"z":0}},{"startPoint":{"x":2515,"y":-1270,"z":0},"endPoint":{"x":2065,"y":-1270,"z":0}},{"startPoint":{"x":2065,"y":-1270,"z":0},"endPoint":{"x":2065,"y":-90,"z":0}},{"startPoint":{"x":-2515,"y":-1770,"z":0},"endPoint":{"x":-1235,"y":-1770,"z":0}},{"startPoint":{"x":-1235,"y":-1770,"z":0},"endPoint":{"x":-1235,"y":-3070,"z":0}},{"startPoint":{"x":-1435,"y":3070,"z":0},"endPoint":{"x":-2515,"y":3070,"z":0}},{"startPoint":{"x":-2515,"y":3070,"z":0},"endPoint":{"x":-2515,"y":-1770,"z":0}},{"startPoint":{"x":-1165,"y":1610,"z":0},"endPoint":{"x":-1435,"y":1610,"z":0}},{"startPoint":{"x":-1435,"y":1610,"z":0},"endPoint":{"x":-1435,"y":3070,"z":0}},{"startPoint":{"x":1705,"y":-90,"z":0},"endPoint":{"x":-1165,"y":-90,"z":0}},{"startPoint":{"x":-1165,"y":-90,"z":0},"endPoint":{"x":-1165,"y":1610,"z":0}},{"startPoint":{"x":2065,"y":-90,"z":0},"endPoint":{"x":1705,"y":-90,"z":0}}],"roomBound":{"x":-2515,"y":-3070,"dx":5030,"dy":6140},"roomLocation":{"x":-1145,"y":1340,"z":0},"roomId":0}'
        # ck.init_room(room_str=test_json_data, roomTypeId=2)
        print(ck.data_infos)
        show_img(ck.roomMats[1], "room")
        result_data = ck.gen_data(enhance=True)
        for i, furniture in enumerate(result_data["furniture_mats"]):
            show_img(furniture[0], "furniture:{}".format(i))
            pass

        for angleIndex in range(4):
            print(result_data[str(angleIndex * 90)]["labels"])
            for i, data in enumerate(result_data[str(angleIndex * 90)]["mid_states"]):
                fig, ax = plt.subplots()
                img = display_state(data[0])
                ax.imshow(img)
                if i > 0:
                    res = ck.label2InputGrids(inputSegs=64, label=result_data[str(angleIndex * 90)]["labels"][i - 1])
                    plt.text(x=res["centerY"], y=res["centerX"], s=str(str(angleIndex * 90) + str(i)))
                    ax.scatter(y=res["centerX"], x=res["centerY"], c="r")
                plt.show()
        pass
