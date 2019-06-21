'''
加入roomjson信息 进行碰撞检测
生成check_data\evaluate_check_roomjson.csv
'''

import pandas as pd
import numpy as np
from checkCrash import Crash_check
import json
import csv

# 读取文件
path = '../data/evaluate.csv'
df = pd.read_csv(path, header=0, encoding='gbk', low_memory=False)
room_strs = [json.dumps(json.loads(i)) for i in df.room_json.tolist()]
# jsonpath = r"F:\pycharm\工作\GA\GA_functionZone\data\0\1-room-json.json"
# data = json.load(open(jsonpath, "r", encoding="utf-8"))
# room_str = json.dumps(data, ensure_ascii=False)

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
    data_list = [('zids', 'layouts', 'genes', 'dxs', 'dys', 'rotates','image','room_json'), ]
    genes = []
    all_genes = []
    count = 0
    for k in range(df.shape[0]):
        zid = zids[k]
        dx = dxs[k]
        dy = dys[k]
        rotate = rotates[k]
        room_str = room_strs[k]

        for j in range(len(layout_pairs[k]) - 1):
            layout = layout_pairs[k][j]

            check = Crash_check.check_data(layout,zid,dx,dy,label_grid_num=16, image_grid_num=64, max_room_size=8000, debug=True,
               objGranularity=500, get_hitMask_grid_num=64, room_str=room_str)
            if check==False:
                continue
            else:
                print('第{}行：True'.format(k))
                print('zid:',zid)
                print('layout:',layout)

                image = np.array(check[-1]).tolist()

                # 定义 genes
                for position_index in layout:
                    gene = get_position_rotate(position_index, size)
                    genes.append(gene)
                # genes 一个房间基因型
                all_genes.append(genes)
                genes = []
                one_gene = all_genes[count]
                data = (zid, layout, one_gene, dx, dy, rotate, image,room_str )
                data_list.append(data)
                count += 1
                with open(r'F:\pycharm\工作\GA\GA_functionZone\check_data\evaluate_check_roomjson.csv', 'w',
                          newline='') as t_file:
                    csv_writer = csv.writer(t_file)
                    for line in data_list:
                        csv_writer.writerow(line)
                    t_file.close()




if __name__ == '__main__':
    function_id_check()


