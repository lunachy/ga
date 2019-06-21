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
paths = ["ConsoleApplication1.dll","F:\pycharm\WorkSpace\GA\GA_functionZone\checkCrash\ConsoleApplication1.dll", "checkCrash/ConsoleApplication1.dll", "../checkCrash/ConsoleApplication1.dll"
         ]

path = [x for x in paths if os.path.exists(x)][0]
try:
    dll_name = path
    lib = ctypes.cdll.LoadLibrary(dll_name)
    print(os.path.exists(dll_name), "dll exists")
except:

    print(os.path.exists(dll_name), "dll  Not exists")


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

    def init_room(self, roomTypeId, room_str):

        self.room_str = room_str
        room_str = room_str.encode("gb18030")

        ret, DetectorIdx, data_infos, embedding_room, embedding_furniture = api_init_room(
            roomTypeId, room_str, self.numSegs)
        assert ret == 0

        furnitures = []
        for i in range(len(data_infos)):
            if int(data_infos[i]["zid"]) in [99, 98]:
                continue
            mask_furnitures = buildEmbeddingMat(
                embedding_furniture[i * self.numSegs * self.numSegs:(i + 1) * self.numSegs * self.numSegs],
                self.numSegs, stIndexs=0)
            furnitures.append(mask_furnitures)
            if self.debug:
                show_img(mask_furnitures[0], "{} furniture".format(i))
        if self.debug:
            show_img(self.roomMats[0], "room")

        self.threadid = DetectorIdx
        self.data_infos = data_infos
        self.roomMats = buildEmbeddingMat(embedding_room, numSegs=self.numSegs)

        self.furniture_mats = furnitures
        self.embedding_room = embedding_room
        self.embedding_furniture = embedding_furniture
        self.image_grid_num = self.numSegs

        if self.debug:
            print(self.threadid, "--  debug:进程id ret.threadId")
            print("初始化完成,threadId{}".format(self.threadid))


    def check_data(self, layout):

        '''
        生成训练数据
        layout = [[],[]]
        :return:
        '''
        data_infos = self.data_infos
        room_context = self.roomMats
        data_return = {}
        data_return["room_str"] = self.room_str
        data_return["furniture_mats"] = self.furniture_mats

        data_enhance_ind = 0
        data_return[str(data_enhance_ind)] = {}
        room_context_enhance = self.embedding_room
        data_return[str(data_enhance_ind)]["room_mats"] = room_context_enhance
        data_return[str(data_enhance_ind)]["mid_states"] = []
        data_return[str(data_enhance_ind)]["mid_states"].append(room_context_enhance)
        data_return[str(data_enhance_ind)]["labels"] = []
        data_return[str(data_enhance_ind)]["zids"] = []
        data_return[str(data_enhance_ind)]["tags"] = []
        ret_list = []
        hitMask_list = []
        for ind, data in enumerate(data_infos):
            tag = data["tag"]
            zid = data["zid"]
            if int(zid) in [98, 99]:
                continue
            if type(layout) == int:
                data["label"] = layout
            else:
                data["label"] = layout[ind]

            label = data["label"]
            pred = {}
            pred["label"] = label
            print('ind:{}pred:{}'.format(ind,pred))
            ret, embedding_hitMask = self.detect_crash_label(pred=pred, index=0, tag=ind)  # tag
            ret_list.append(ret)
            ### 需要进行 判断None时才能embeding
            hitMask_list.append(embedding_hitMask)

            data_return['0']["zids"].append(zid)
            data_return['0']["tags"].append(tag)
            data_return['0']["labels"].append(label)
            if ret==0:
                ret_array = buildEmbeddingMat(embedding_hitMask, 64, stIndexs=0)
            # print("ret_array", np.shape(ret_array))
            else:
                ret_array = embedding_hitMask
            data_return['0']["mid_states"].append(ret_array)

            # 打印布局过程图
            # show_img(data_return['0']['mid_states'][-1][0], 'rotate')

            # print('最后一张图:',data_return['0']["mid_states"][-1][0])
            # show_img(ck.roomMats[0], "mid_state:{}".format(i))
            self.step_finash()
        if set(ret_list) == {0}:
            print('检测到未碰撞个体')
            self.room_finish()
            # 不碰撞
            return [True, data_return]

        else:
            # 碰撞
            print('检测有个体碰撞')
            self.room_finish()
            return [False, {}]



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

                try:
                    label = data["label"]
                    tag = data["tag"]
                    zid = data["zid"]
                    if int(zid) in [98, 99]:
                        continue
                    pred = {}
                    pred["label"] = label
                    ret, embedding_hitMask = self.detect_crash_label(pred=pred, index=0, tag=ind)  # tag

                    assert ret == 0
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
                        show_img(data_return['0']['mid_states'][-1][0], 'rotate')
                        # show_img(ck.roomMats[0], "mid_state:{}".format(i))

                except Exception as e:
                    print(e)
                    raise (e)
                finally:
                    self.step_finash()
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

    def detect_crash_label(self, pred, index, tag):
        '''
        修改后的碰撞检测 获取位置 由 c++ 实现
        :param pred: {"label":295}
        :param index: 上一级目录
        :param tag: 功能区布局顺序
        :return:ret=0 不碰撞
        '''
        pred_label = pred["label"]
        numSegs = self.numSegs
        zoneLabel = pred_label
        ret, embedding_mask = api_check_crash(zoneIndex=tag, label=zoneLabel,
                                              thredidx=self.threadid, index=index,
                                              numSegs=numSegs)
        return ret, embedding_mask

    def step_finash(self):
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
        self.max_room_size = 6400
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
        result["centerX"] = grid_x + 32
        result["centerY"] = grid_y + 32

        return result

    def buildEmbeddingMat(self, EmbeddingArray, numSegs, stIndexs=0):
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


def api_check_crash(zoneIndex, label, index, thredidx, numSegs):
    grid_num = numSegs * numSegs
    EmbeddingArrayType = structures.EmbeddingCode * grid_num
    embedding_code_list = EmbeddingArrayType()
    for i in range(grid_num):
        structures.init_embedding_code(embedding_code_list[i])

    ret = lib.API_checkCrash(zoneIndex, label, index, thredidx, embedding_code_list)
    # print(ret)
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

                datas.append(data)
            except Exception as e:
                print("outZoneNum.value", outZoneNum.value, i, e)
                continue

        # datas.sort(key=lambda x: x["tag"])
        # for i in range(numSegs):
        #     tmp = []
        #     for j in range(numSegs):
        #         tmp.append(embedding_code_list_room_context[i * numSegs + j])
        #     hit_mask.append(tmp)

        # for i in range(outZoneNum.value):
        #     tmp = []
        #     for j in range(numSegs):
        #         tmp_ = []
        #         for k in range(numSegs):
        #             tmp_.append(embedding_code_list_furniture[i * numSegs * numSegs + j * numSegs + k])
        #         tmp.append(tmp_)
        #     furnist_mat.append(tmp)

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
            if (line.find("input_segs") != -1):
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
    import matplotlib.pyplot as plt
    from checkCrash.display import display_state
    import json, pickle

    ck = CrashChecker(configPath="config.props")
    jsonpath = r"F:\pycharm\WorkSpace\GA\GA_functionZone\layout-cnn-room-test-BedRoom-0.0-5.30\0\22-room-json.json"

    test_json_data = json.load(open(jsonpath, "r", encoding="utf-8"))
    test_json_data = json.dumps(test_json_data, ensure_ascii=False)
    # test_json_data = '{"functionZones":[{"id":52,"mainPositions":[{"sample":true,"main":true,"tag":"H10k58yX","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 三人沙发  [原先名称]\u003d【豪兴  格美】现代   三人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-774,"y":-2789,"z":0},"locationInHouse":{"x":-3374,"y":-6209,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2030,"dy":940,"dz":940},"modelId":16973,"skuId":154834,"categoryId":303,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":true,"tag":"fsFQefYs","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 双人沙发  [原先名称]\u003d【豪兴  格美】现代   双人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-2291,"y":-1628,"z":0},"locationInHouse":{"x":-4891,"y":-5048,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1550,"dy":940,"dz":940},"modelId":16978,"skuId":154835,"categoryId":302,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"H10k58yX","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 三人沙发  [原先名称]\u003d【豪兴  格美】现代   三人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-774,"y":-2789,"z":0},"locationInHouse":{"x":-3374,"y":-6209,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2030,"dy":940,"dz":940},"modelId":16973,"skuId":154834,"categoryId":303,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":797,"y":287,"z":426},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":true},{"sample":true,"main":true,"tag":"fsFQefYs","cid":301,"metaData":{"name":"[修正名称]\u003d【豪兴 格美】现代 双人沙发  [原先名称]\u003d【豪兴  格美】现代   双人沙发","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-2291,"y":-1628,"z":0},"locationInHouse":{"x":-4891,"y":-5048,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1550,"dy":940,"dz":940},"modelId":16978,"skuId":154835,"categoryId":302,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-363,"y":-1229,"z":426},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":5.81858356388325},"isDrModel":true},{"sample":false,"main":false,"tag":"W3QxT1nx","cid":89,"metaData":{"name":"装饰花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":663,"y":-2629,"z":560},"locationInHouse":{"x":-1936,"y":-6049,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":153,"dy":179,"dz":179},"modelId":16929,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":637,"y":1725,"z":986},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":true},{"sample":false,"main":false,"tag":"GVbjpfTV","cid":61,"metaData":{"name":"挂画","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-973,"y":-3249,"z":940},"locationInHouse":{"x":-3573,"y":-6669,"z":940},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1139,"dy":34,"dz":34},"modelId":16911,"skuId":0,"categoryId":88,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":1257,"y":88,"z":1366},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707963267948966},"isDrModel":true}],"center":{"x":-1062,"y":-1991,"z":-426},"bound":{"x":-2329,"y":-3294,"dx":2534,"dy":2604},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.71238898038469},"tag":2},{"id":54,"mainPositions":[{"sample":true,"main":true,"tag":"gRjJpIOG","cid":108,"metaData":{"name":"【简欧  英伦系列】欧式   500-600mm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.212419684575316},"location":{"x":-993,"y":319,"z":0},"locationInHouse":{"x":-3593,"y":-3100,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2360,"dy":401,"dz":401},"modelId":16962,"skuId":153300,"categoryId":327,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"gRjJpIOG","cid":108,"metaData":{"name":"【简欧  英伦系列】欧式   500-600mm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.212419684575316},"location":{"x":-993,"y":319,"z":0},"locationInHouse":{"x":-3593,"y":-3100,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2360,"dy":401,"dz":401},"modelId":16962,"skuId":153300,"categoryId":327,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":false,"tag":"BUuWEPX1","cid":99,"metaData":{"name":"电视","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.07076562260427},"location":{"x":-983,"y":477,"z":850},"locationInHouse":{"x":-3583,"y":-2942,"z":850},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1371,"dy":84,"dz":84},"modelId":16912,"skuId":0,"categoryId":132,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":120,"y":-102,"z":850},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.858345938028954},"isDrModel":true}],"center":{"x":-993,"y":319,"z":0},"bound":{"x":-2173,"y":-868,"dx":2360,"dy":2376},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.212419684575316},"tag":1},{"id":40,"mainPositions":[{"sample":true,"main":true,"tag":"vAPKFQli","cid":118,"metaData":{"name":"余颢凌-情迷地中海-玄关柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"location":{"x":2724,"y":-4062,"z":0},"locationInHouse":{"x":124,"y":-7482,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2000,"dy":360,"dz":360},"modelId":24590,"skuId":0,"categoryId":537,"isCustom":true,"spuId":8100439,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"positions":[{"sample":true,"main":true,"tag":"vAPKFQli","cid":118,"metaData":{"name":"余颢凌-情迷地中海-玄关柜","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"location":{"x":2724,"y":-4062,"z":0},"locationInHouse":{"x":124,"y":-7482,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":2000,"dy":360,"dz":360},"modelId":24590,"skuId":0,"categoryId":537,"isCustom":true,"spuId":8100439,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"center":{"x":2724,"y":-4062,"z":0},"bound":{"x":1724,"y":-4971,"dx":1999,"dy":1817},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"tag":0},{"id":99,"mainPositions":[{"sample":false,"main":false,"tag":"DCaRXj8I","cid":-1,"metaData":{"name":"UNKNOWN","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":0,"y":0,"z":0},"locationInHouse":{"x":0,"y":0,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1,"dy":1,"dz":1},"modelId":0,"skuId":0,"categoryId":0,"isCustom":false,"spuId":0,"mark":0},"isDrModel":false}],"positions":[{"sample":false,"main":false,"tag":"sFfMDUts","cid":173,"metaData":{"name":"[修正名称]\u003d【喜舍  】现代  塑料 塑料吸顶灯  [原先名称]\u003d【天路行  】现代  塑料 塑料吸顶灯","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-2567,"y":-131,"z":2600},"locationInHouse":{"x":-5167,"y":-3551,"z":2600},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":400,"dy":400,"dz":400},"modelId":23661,"skuId":158075,"categoryId":457,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-2567,"y":-131,"z":2600},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":false,"main":false,"tag":"Cg3TqaSP","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2313,"y":-2659,"z":560},"locationInHouse":{"x":-4913,"y":-6079,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":179,"dy":151,"dz":151},"modelId":16921,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-2313,"y":-2659,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"HQi02x8S","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-553,"y":-1259,"z":570},"locationInHouse":{"x":-3153,"y":-4679,"z":570},"scale":{"x":0.978,"y":0.718,"z":0.737},"size":{"dx":432,"dy":241,"dz":241},"modelId":16914,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-553,"y":-1259,"z":570},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":false,"main":false,"tag":"GVn4hEai","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-853,"y":-1209,"z":490},"locationInHouse":{"x":-3453,"y":-4629,"z":490},"scale":{"x":1.0,"y":1.0,"z":1.462},"size":{"dx":217,"dy":218,"dz":218},"modelId":16924,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-853,"y":-1209,"z":490},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"kOaNMYyq","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":6.051747862571702},"location":{"x":-1133,"y":-1299,"z":490},"locationInHouse":{"x":-3733,"y":-4719,"z":490},"scale":{"x":0.906481,"y":0.999564,"z":0.646},"size":{"dx":306,"dy":216,"dz":216},"modelId":16925,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1133,"y":-1299,"z":490},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":6.051747862571702},"isDrModel":true},{"sample":false,"main":false,"tag":"YsYJyGrZ","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.296630477778912},"location":{"x":-1613,"y":-2049,"z":10},"locationInHouse":{"x":-4213,"y":-5469,"z":10},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":468,"dy":485,"dz":485},"modelId":16926,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1613,"y":-2049,"z":10},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.296630477778912},"isDrModel":true},{"sample":false,"main":false,"tag":"abJ4TM7O","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":573,"y":-2869,"z":560},"locationInHouse":{"x":-2026,"y":-6289,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":560,"dy":291,"dz":291},"modelId":16928,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":573,"y":-2869,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"5vga1vBI","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2403,"y":-2789,"z":130},"locationInHouse":{"x":-5003,"y":-6209,"z":130},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":265,"dy":208,"dz":208},"modelId":16931,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":1},"relativeLocation":{"x":-2403,"y":-2789,"z":130},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"xrdLvRTj","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2583,"y":-2639,"z":550},"locationInHouse":{"x":-5183,"y":-6059,"z":550},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":136,"dy":132,"dz":132},"modelId":16918,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-2583,"y":-2639,"z":550},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"SvZ9Jpxs","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":603,"y":-2859,"z":130},"locationInHouse":{"x":-1996,"y":-6279,"z":130},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":265,"dy":208,"dz":208},"modelId":16931,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":2},"relativeLocation":{"x":603,"y":-2859,"z":130},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"ZeV2wrIy","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-1753,"y":309,"z":560},"locationInHouse":{"x":-4353,"y":-3110,"z":560},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":216,"dy":216,"dz":216},"modelId":16905,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1753,"y":309,"z":560},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"Zz1ttyib","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.0354133288802956},"location":{"x":-403,"y":289,"z":320},"locationInHouse":{"x":-3003,"y":-3130,"z":320},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":172,"dy":237,"dz":237},"modelId":16909,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-403,"y":289,"z":320},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.0354133288802956},"isDrModel":true},{"sample":false,"main":false,"tag":"cdl1nV81","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":5.884755627372201},"location":{"x":-1393,"y":329,"z":320},"locationInHouse":{"x":-3993,"y":-3090,"z":320},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":212,"dy":336,"dz":336},"modelId":16906,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-1393,"y":329,"z":320},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":5.884755627372201},"isDrModel":true},{"sample":false,"main":false,"tag":"fwpVHPpZ","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-213,"y":269,"z":540},"locationInHouse":{"x":-2813,"y":-3150,"z":540},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":289,"dy":320,"dz":320},"modelId":16963,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":1},"relativeLocation":{"x":-213,"y":269,"z":540},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"pDuChmyC","metaData":{"name":"摆件","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":2820,"y":-3812,"z":1020},"locationInHouse":{"x":220,"y":-7232,"z":1020},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":289,"dy":320,"dz":320},"modelId":16963,"skuId":0,"categoryId":87,"isCustom":false,"spuId":0,"mark":2},"relativeLocation":{"x":2820,"y":-3812,"z":1020},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":true,"main":false,"tag":"hU5IaADb","cid":670,"metaData":{"name":"定制背景墙","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-943,"y":411,"z":80},"locationInHouse":{"x":-3543,"y":-3009,"z":80},"scale":{"x":0.952154,"y":0.091743,"z":0.909795},"size":{"dx":4201,"dy":218,"dz":218},"modelId":24384,"skuId":162233,"categoryId":710,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-943,"y":411,"z":80},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"lblpxEC6","cid":677,"metaData":{"name":"艾佳定制吊顶","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"location":{"x":-883,"y":-1399,"z":2340},"locationInHouse":{"x":-3483,"y":-4819,"z":2340},"scale":{"x":0.988924,"y":0.968223,"z":1.0},"size":{"dx":3792,"dy":4028,"dz":4028},"modelId":24626,"skuId":164590,"categoryId":709,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-883,"y":-1399,"z":2340},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":2.035359923118577},"isDrModel":true},{"sample":true,"main":false,"tag":"nwXBFOob","cid":733,"metaData":{"name":"艾佳定制 石材波打线-10cm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-823,"y":-2919,"z":0},"locationInHouse":{"x":-3423,"y":-6339,"z":0},"scale":{"x":1.380434,"y":1.273076,"z":1.0},"size":{"dx":100,"dy":3000,"dz":3000},"modelId":24447,"skuId":162147,"categoryId":703,"isCustom":false,"spuId":0,"mark":1},"relativeLocation":{"x":-823,"y":-2919,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":true,"main":false,"tag":"5SP2avVz","cid":733,"metaData":{"name":"艾佳定制 石材波打线-10cm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"location":{"x":-823,"y":341,"z":0},"locationInHouse":{"x":-3423,"y":-3079,"z":0},"scale":{"x":1.366746,"y":1.273076,"z":1.0},"size":{"dx":100,"dy":3000,"dz":3000},"modelId":24447,"skuId":162147,"categoryId":703,"isCustom":false,"spuId":0,"mark":2},"relativeLocation":{"x":-823,"y":341,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":4.247787237088353},"isDrModel":true},{"sample":true,"main":false,"tag":"059dkj6G","cid":733,"metaData":{"name":"艾佳定制 石材波打线-10cm","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-2663,"y":-1289,"z":0},"locationInHouse":{"x":-5263,"y":-4709,"z":0},"scale":{"x":1.430801,"y":1.135553,"z":1.0},"size":{"dx":100,"dy":3000,"dz":3000},"modelId":24447,"skuId":162147,"categoryId":703,"isCustom":false,"spuId":0,"mark":3},"relativeLocation":{"x":-2663,"y":-1289,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"lPSNFPMf","cid":89,"metaData":{"name":"装饰花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":896,"y":279,"z":730},"locationInHouse":{"x":-1703,"y":-3140,"z":730},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":535,"dy":461,"dz":461},"modelId":16910,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":896,"y":279,"z":730},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true},{"sample":false,"main":false,"tag":"zjx4NSH0","cid":89,"metaData":{"name":"装饰花艺","rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"location":{"x":-903,"y":-1509,"z":550},"locationInHouse":{"x":-3503,"y":-4929,"z":550},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":321,"dy":357,"dz":357},"modelId":16936,"skuId":0,"categoryId":82,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":-903,"y":-1509,"z":550},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":true}],"center":{"x":0,"y":0,"z":0},"bound":{"x":-2943,"y":-3836,"dx":3999,"dy":5094},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":3}],"usageId":1,"roomName":"客厅","walls":[{"scid":12,"wallPoints":[{"x":-1930,"y":1945},{"x":-1930,"y":2145},{"x":1730,"y":2145},{"x":1930,"y":1945}]},{"scid":1,"wallPoints":[{"x":1930,"y":1945},{"x":1730,"y":2145},{"x":1730,"y":3235},{"x":1930,"y":3215}]},{"scid":12,"wallPoints":[{"x":-1930,"y":-2145},{"x":-1930,"y":-1945},{"x":2170,"y":-1945},{"x":1970,"y":-2145}]},{"scid":16,"wallPoints":[{"x":-2130,"y":-1945},{"x":-2130,"y":1945},{"x":-1930,"y":1945},{"x":-1930,"y":-1945}]}],"windows":[],"doors":[{"scid":16,"points":[{"x":-2133,"y":-910},{"x":-2133,"y":1289},{"x":-1932,"y":1289},{"x":-1932,"y":-910}],"type":2,"horizontalFlip":false,"verticalFlip":false,"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":1.5707956610046239}}],"areaPoints":[{"x":-1930,"y":1945},{"x":1930,"y":1945},{"x":1930,"y":1985},{"x":1932,"y":1984},{"x":1934,"y":1984},{"x":1940,"y":1983},{"x":1942,"y":1982},{"x":1947,"y":1980},{"x":1949,"y":1979},{"x":1954,"y":1976},{"x":1956,"y":1975},{"x":1960,"y":1971},{"x":1961,"y":1969},{"x":1964,"y":1964},{"x":1965,"y":1962},{"x":1967,"y":1957},{"x":1968,"y":1955},{"x":1969,"y":1949},{"x":1969,"y":1947},{"x":1970,"y":1945},{"x":1970,"y":-1945},{"x":-1930,"y":-1945},{"x":-1930,"y":1945}],"roomId":0}'
    # test_json_data = '{"functionZones":[{"id":48,"positions":[{"sample":true,"main":true,"tag":"JfudZDnP","cid":40,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":1080,"y":85,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1238,"dy":1934,"dz":1934},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false},{"sample":true,"main":false,"tag":"JMV66PL1","cid":330,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":-134,"y":798,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1047,"dy":510,"dz":510},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":1215,"y":-712,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":1080,"y":85,"z":0},"bound":{"x":-658,"y":-881,"dx":2358,"dy":1934},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"tag":0},{"id":38,"positions":[{"sample":true,"main":true,"tag":"6ollMzLi","cid":120,"metaData":{"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"location":{"x":-1441,"y":628,"z":0},"scale":{"x":1.0,"y":1.0,"z":1.0},"size":{"dx":1202,"dy":884,"dz":884},"modelId":0,"skuId":0,"isCustom":false,"spuId":0,"mark":0},"relativeLocation":{"x":0,"y":0,"z":0},"relativeRotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"isDrModel":false}],"center":{"x":-1441,"y":628,"z":0},"bound":{"x":-2042,"y":186,"dx":1202,"dy":884},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":3.141592653589793},"tag":1},{"id":99,"positions":[],"center":{"x":0,"y":0,"z":0},"bound":{"x":0,"y":0,"dx":0,"dy":0},"rotate":{"xAxis":0.0,"yAxis":0.0,"zAxis":0.0},"tag":2}],"usageId":4,"roomName":"儿童房","walls":[{"scid":14,"wallPoints":[{"x":1842,"y":-1062},{"x":2089,"y":-1062},{"x":1842,"y":-1608},{"x":2089,"y":-1608}]},{"scid":1,"wallPoints":[{"x":1706,"y":1400},{"x":2130,"y":1400},{"x":1706,"y":-166},{"x":2130,"y":-166}]},{"scid":12,"wallPoints":[{"x":2054,"y":1411},{"x":2207,"y":1411},{"x":2054,"y":-6534},{"x":2207,"y":-6534}]},{"scid":11,"wallPoints":[{"x":-6319,"y":1393},{"x":2182,"y":1393},{"x":-6319,"y":1063},{"x":2182,"y":1063}]},{"scid":11,"wallPoints":[{"x":-2373,"y":1411},{"x":-2052,"y":1411},{"x":-2373,"y":306},{"x":-2052,"y":306}]},{"scid":11,"wallPoints":[{"x":-2143,"y":377},{"x":-2052,"y":377},{"x":-2143,"y":-1191},{"x":-2052,"y":-1191}]},{"scid":11,"wallPoints":[{"x":-2362,"y":-1056},{"x":1995,"y":-1056},{"x":-2362,"y":-1192},{"x":1995,"y":-1192}]},{"scid":11,"wallPoints":[{"x":-970,"y":-1062},{"x":-825,"y":-1062},{"x":-970,"y":-2827},{"x":-825,"y":-2827}]}],"windows":[{"scid":1,"points":[{"x":2033,"y":-1063},{"x":2033,"y":-181},{"x":2199,"y":-181},{"x":2199,"y":-1063}],"type":0,"horizontalFlip":false,"verticalFlip":false}],"doors":[{"scid":11,"points":[{"x":-1942,"y":-1185},{"x":-1942,"y":-1031},{"x":-1066,"y":-1031},{"x":-1066,"y":-1185}],"type":0,"horizontalFlip":true,"verticalFlip":true,"rotate":{"zAxis":90.0}}],"areaPoints":[{"x":1995,"y":-1063},{"x":1995,"y":-1056},{"x":-2053,"y":-1056},{"x":-2053,"y":306},{"x":-2053,"y":376},{"x":-2053,"y":1063},{"x":1706,"y":1063},{"x":1706,"y":-166},{"x":2053,"y":-166},{"x":2053,"y":-1063},{"x":1995,"y":-1063}],"roomId":344042}'

    ck.init_room(room_str=test_json_data, roomTypeId=1)

    # 房间初始化 房间门窗墙信息
    print('ck.data_infos:', ck.data_infos)

    # 显示户型图
    show_img(ck.roomMats[0], "room")
    ######################################################

    ### 原始 #####
    # result_data = ck.gen_data(enhance=False)
    #
    # # 生成中间状态图
    # # for i, furniture in enumerate(result_data["furniture_mats"]):
    # #     show_img(furniture[0], "furniture:{}".format(i))
    #
    # # 生成最后布局图
    # show_img(result_data["0"]["mid_states"][-1][0], "last_mid_states")


    # # for angleIndex in range(4):
    # #     print(result_data[str(angleIndex * 90)]["labels"])
    # #     for i, data in enumerate(result_data[str(angleIndex * 90)]["mid_states"]):
    # #         fig, ax = plt.subplots()
    # #         img = display_state(data[0])
    # #         ax.imshow(img)
    # #         if i > 0:
    # #             res = ck.label2InputGrids(inputSegs=64, label=result_data[str(angleIndex * 90)]["labels"][i - 1])
    # #             plt.text(x=res["centerY"], y=res["centerX"], s=str(str(angleIndex * 90) + str(i)))
    # #             ax.scatter(y=res["centerX"], x=res["centerY"], c="r")
    # #         plt.show()
    ###########################################
    # 输入layout 检测碰撞 并生成img
    result_data = ck.check_data(layout=[618,425,116])
    # 碰撞结果 result_data[0] = [True,{}]/[False,{}]
    print('碰撞结果：',result_data[0])
    # 生成最后布局图
    if result_data[0]== True:
        # data_return['0']['mid_states'][-1][0]
        last_img= result_data[-1]["0"]["mid_states"][-1][0]
        show_img(last_img,"last_mid_states")

    pass
