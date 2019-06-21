'''
对原始数据进行碰撞检测，删除碰撞数据，生成后续训练数据
'''

import pandas as pd
import numpy as np
from checkCrash import Crash_check
import json
import csv

# 读取文件
path = '../data/evaluate.csv'
df = pd.read_csv(path, header=0, encoding='gbk', low_memory=False)
jsonpath = r"F:\pycharm\工作\GA\GA_functionZone\data\0\1-room-json.json"
data = json.load(open(jsonpath, "r", encoding="utf-8"))
room_str = json.dumps(data, ensure_ascii=False)


# 获取列信息
zids = df['zids']
dxs = df['dxs']
dys = df['dys']
rotates = df['rotate']
layout_pairs = df['layout_pairs']
room_json = df['room_json']


# zid字符转数值型
zids = [json.loads(i) for i in zids]
dxs = [json.loads(i) for i in dxs]
dys = [json.loads(i) for i in dys]
rotates = [json.loads(i) for i in rotates]
layout_pairs = [json.loads(i) for i in layout_pairs]

size = 16  # 棋牌是16*16格子


# 通过位置索引计算x，y，rotate
def get_position_rotate(position_index, size=16):
    # 方向 0,1,2,3 分别 0，90,180,270
    rot = position_index // (size * size)
    ij = position_index % (size * size)
    cen_y = ij // size
    cen_x = ij % size
    return [cen_x, cen_y, rot]


# # test
# print(get_position_rotate(365,size))  #[13, 6, 1]


# 计算位置索引
def caculate_position_index(cen_x, cen_y, rot, size):
    position_index = cen_x + cen_y * size + rot * size * size
    return position_index

# # test
# print(caculate_position_index(13,6,1,size)) # 356


def function_id_check():
    data_list = [('zid', 'layout', 'genes', 'dx', 'dy', 'rotate','image'), ]
    genes = []
    all_genes = []
    count = 0
    for k in range(df.shape[0]):
        zid = zids[k]
        dx = dxs[k]
        dy = dys[k]
        rotate = rotates[k]

        for j in range(len(layout_pairs[k]) - 1):
            layout = layout_pairs[k][j]
            check = Crash_check.check_data(layout,zid,label_grid_num=16, image_grid_num=64, max_room_size=8000, debug=False,
               objGranularity=500, get_hitMask_grid_num=64, room_str=room_str)
            if check[0] == True:
                image = check[-1]
                # 定义 genes
                for position_index in layout:
                    gene = get_position_rotate(position_index, size)
                    genes.append(gene)
                # genes 一个房间基因型
                all_genes.append(genes)
                genes = []
                one_gene = all_genes[count]
                data = (zid, layout, one_gene, dx, dy, rotate,image)
                data_list.append(data)
                count += 1

    with open(r'F:\pycharm\工作\GA\GA_functionZone\data\evaluate_check.csv', 'w', newline='') as t_file:
        csv_writer = csv.writer(t_file)
        for line in data_list:
            csv_writer.writerow(line)
        t_file.close()


if __name__ == '__main__':
    import time
    st = time.clock()
    #function_id_check()
    result = get_position_rotate(635)
    print("x:{},y:{},r:{}".format(result[0],result[1],result[2]))
    ed = time.clock()
    print('use:{}s'.format(ed - st))

# check = Crash_check.check_data(zid, layout, dx, dy, debug=True, objGranularity=500,
#                image_grid_num=128, get_hitMask_grid_num=16)
# print(check)
import pandas as pd
import numpy as np
from checkCrash import Crash_check
import json
import csv

# 读取文件
path = '../data/evaluate.csv'
df = pd.read_csv(path, header=0,encoding='gbk', low_memory=False)
jsonpath = r"F:\pycharm\工作\GA\GA_functionZone\data\0\1-room-json.json"
data = json.load(open(jsonpath, "r", encoding="utf-8"))
room_str = json.dumps(data, ensure_ascii=False)

# 获取列信息
zids = df['zids']
dxs = df['dxs']
dys = df['dys']
rotates = df['rotate']
layout_pairs = df['layout_pairs']
room_json = df['room_json']


# zid字符转数值型
zids = [json.loads(i) for i in zids]
dxs = [json.loads(i) for i in dxs]
dys = [json.loads(i) for i in dys]
rotates = [json.loads(i) for i in rotates]
layout_pairs = [json.loads(i) for i in layout_pairs]

size = 16  # 棋牌是16*16格子


# 通过位置索引计算x，y，rotate
def get_position_rotate(position_index, size=16):
    # 方向 0,1,2,3 分别 0，90,180,270
    rot = position_index // (size * size)
    ij = position_index % (size * size)
    cen_y = ij // size
    cen_x = ij % size
    return [cen_x, cen_y, rot]


# # test
# print(get_position_rotate(365,size))  #[13, 6, 1]


# 计算位置索引
def caculate_position_index(cen_x, cen_y, rot, size):
    position_index = cen_x + cen_y * size + rot * size * size
    return position_index

# # test
# print(caculate_position_index(13,6,1,size)) # 356


def function_id_check():
    data_list = [('zid', 'layout', 'genes', 'dx', 'dy', 'rotate','image'), ]
    genes = []
    all_genes = []
    count = 0
    for k in range(df.shape[0]):
        zid = zids[k]
        dx = dxs[k]
        dy = dys[k]
        rotate = rotates[k]

        for j in range(len(layout_pairs[k]) - 1):
            layout = layout_pairs[k][j]
            check = Crash_check.check_data(layout,zid,label_grid_num=16, image_grid_num=64, max_room_size=8000, debug=False,
               objGranularity=500, get_hitMask_grid_num=64, room_str=room_str)
            if check[0] == True:
                image = check[-1]
                # 定义 genes
                for position_index in layout:
                    gene = get_position_rotate(position_index, size)
                    genes.append(gene)
                # genes 一个房间基因型
                all_genes.append(genes)
                genes = []
                one_gene = all_genes[count]
                data = (zid, layout, one_gene, dx, dy, rotate,image)
                data_list.append(data)
                count += 1

    with open(r'F:\pycharm\工作\GA\GA_functionZone\data\evaluate_check.csv', 'w', newline='') as t_file:
        csv_writer = csv.writer(t_file)
        for line in data_list:
            csv_writer.writerow(line)
        t_file.close()


if __name__ == '__main__':
    import time
    st = time.clock()
    #function_id_check()
    result = get_position_rotate(635)
    print("x:{},y:{},r:{}".format(result[0],result[1],result[2]))
    ed = time.clock()
    print('use:{}s'.format(ed - st))

# check = Crash_check.check_data(zid, layout, dx, dy, debug=True, objGranularity=500,
#                image_grid_num=128, get_hitMask_grid_num=16)
# print(check)
