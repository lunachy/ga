"""
数据解析 以及 格式生成
"""
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import copy
import random
from utils.record_utils import int64_feature, float64_feature, bytes_feature


def parse_numpy_data(count_value_list: list, image_width: int, image_height: int):
    """
    解析numpy 数据
    :param: count_value_list 待解析的数据
    :param: image_width  数据的宽度
    :param: image_height  数据的长度
    :return:  返回解析后的numpy数据格式
    """
    start_index = 0
    values = []
    mat = np.zeros(image_width * image_height)
    for count, value in count_value_list:
        mat[start_index: start_index+count] = value
        start_index += count
        values.append(value)
    mat = np.reshape(mat, (image_width, image_height))
    return mat


def parse_numpy_feature_data(data: str, image_width: int, image_height: int, ids: list=["zid", "cid", "mid", "scid"]):
    """
    :param data:  数据解析
    :return:  返回所有的数据
    :param image_width  图片宽度
    :param image_height  图片高度
    :return zid_image
             cid_image
             scid_image
             mid_image
    """
    data_json = json.loads(data)
    if len(ids) == 1:
        return parse_numpy_data(count_value_list=data_json[ids[0]], image_width=image_width, image_height=image_height)
    if len(ids) == 2:
        return parse_numpy_data(count_value_list=data_json[ids[0]], image_width=image_width, image_height=image_height), \
               parse_numpy_data(count_value_list=data_json[ids[1]], image_width=image_width, image_height=image_height)
    if len(ids) == 3:
        return parse_numpy_data(count_value_list=data_json[ids[0]], image_width=image_width, image_height=image_height), \
               parse_numpy_data(count_value_list=data_json[ids[1]], image_width=image_width, image_height=image_height), \
               parse_numpy_data(count_value_list=data_json[ids[2]], image_width=image_width, image_height=image_height)
    if len(ids) == 4:
        return parse_numpy_data(count_value_list=data_json[ids[0]], image_width=image_width, image_height=image_height), \
               parse_numpy_data(count_value_list=data_json[ids[1]], image_width=image_width, image_height=image_height), \
               parse_numpy_data(count_value_list=data_json[ids[2]], image_width=image_width, image_height=image_height), \
               parse_numpy_data(count_value_list=data_json[ids[3]], image_width=image_width, image_height=image_height)


def parse_line_data(line: str):
    """
    解析一行的数据
    :param line:
    :return:
    """
    line = line.strip()
    line_info = line.split("\t")
    return line_info[0].split("|"), line_info[1]


def get_frame_orientation_wall_distance(img: np.ndarray, image_width: int, image_height: int, default_value: int,
                                        valid_zones: list=[38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55]):
    # 睡眠区 视听区 储衣区):
    distance_mask = np.ones((image_width, image_height)) * default_value
    for zone_id in valid_zones:
        xs, ys = np.where(img == zone_id)
        if len(xs) > 0:
            unique_xs = np.unique(xs)
            unique_ys = np.unique(ys)
            for x in unique_xs:  # 遍历每一行
                row = img[x]
                x_ids = np.where(xs == x)[0]
                x_y_ids = ys[x_ids]
                small_y = np.min(x_y_ids)
                max_y = np.max(x_y_ids)
                small_walls = row[: small_y]
                small_walls_indexs = np.where((small_walls >= 33) & (small_walls <= 35))[0]
                if len(small_walls_indexs) == 0:
                     small_walls_indexs = np.where(small_walls == 32)[0]
                if len(small_walls_indexs) > 0:
                    small_distances = small_walls_indexs - small_y
                    distance_mask[x][small_y] = np.min(np.abs(small_distances)-1)
                max_walls = row[max_y:]
                max_walls_indexs = np.where((max_walls >= 33) & (max_walls <= 35))[0]
                if len(max_walls_indexs) == 0:
                    max_walls_indexs = np.where(max_walls == 32)[0]
                if len(max_walls_indexs) > 0:
                    max_distances = max_walls_indexs  # 直接就是距离
                    distance_mask[x][max_y] = np.min(np.abs(max_distances)-1)
            for y in unique_ys:  # 遍历每一行
                col = img[:, y]
                y_ids = np.where(ys == y)[0]
                y_x_ids = xs[y_ids]
                small_x = np.min(y_x_ids)
                max_x= np.max(y_x_ids)
                small_walls = col[: small_x]
                small_walls_indexs = np.where((small_walls >= 33) & (small_walls <= 35))[0]
                if len(small_walls_indexs) == 0:
                     small_walls_indexs = np.where(small_walls == 32)[0]
                if len(small_walls_indexs) > 0:
                    small_distances = small_walls_indexs - small_x
                    distance_mask[small_x][y] = np.min(np.abs(small_distances)-1)
                max_walls = col[max_x:]
                max_walls_indexs = np.where((max_walls >= 33) & (max_walls <= 35))[0]
                if len(max_walls_indexs) == 0:
                    max_walls_indexs = np.where(max_walls == 32)[0]
                if len(max_walls_indexs) > 0:
                    max_distances = max_walls_indexs # 直接就是距离
                    distance_mask[max_x][y] = np.min(np.abs(max_distances)-1)
    return distance_mask


def parse_data_single_tf_record(data_empty_path: str,
                                data_zone_path: str,
                                data_zone_context_path: str,
                                image_width: int,
                                image_height: int,
                                record_train_path: str,
                                record_test_path: str,
                                dtype=np.int32,
                                ignore_zids: list=["98", "99"],
                                valid_zids: list=None,
                                limit: int=100000,
                                seed: int=200):
    """
    :param data_empty_path:   空房间数据
    :param data_zone_path:   功能区数据
    :param data_zone_context_path:   功能区上下文数据
    :param image_width:
    :param image_height:
    :param record_train_path: 训练集路径
    :param record_test_path: 测试集路径
    :param debug:
    :param dtype:
    :param ignore_zids: 生成数据时 需要忽略掉的zid
    :param valid_zids:  仅生成有效的zid数据
    :return:
    """
    random.seed(seed)
    # 由功能区上下文当中 找出所有有效的数据 name
    data_zone_context_lines = open(data_zone_context_path, "r", encoding="utf-8").readlines()
    data_zone_context_lines = list(set(data_zone_context_lines))
    data_zone_context_lines = [parse_line_data(line) for line in data_zone_context_lines]

    # 功能区上下文 summary 数据生成
    zone_context_summary_dict = {}
    for info, data_str in data_zone_context_lines:
        if "{0}_{1}".format(info[0], info[2]) not in zone_context_summary_dict:
            zone_context_summary_dict["{0}_{1}".format(info[0], info[2])] = [[info, data_str]]
        else:
            zone_context_summary_dict["{0}_{1}".format(info[0], info[2])].append([info, data_str])

    data_empty_lines = open(data_empty_path, "r", encoding="utf-8").readlines()
    data_empty_lines = list(set(data_empty_lines))
    data_empty_lines = [parse_line_data(line) for line in data_empty_lines]
    # 空房间 summary 数据生成
    empty_summary_dict = {}
    for info, data_str in data_empty_lines:
        if "{0}_{1}".format(info[0], info[2]) not in zone_context_summary_dict:  # 仅有空房间 无功能区上下文的数据 过滤
            continue
        if "{0}_{1}".format(info[0], info[2]) not in empty_summary_dict:
            empty_summary_dict["{0}_{1}".format(info[0], info[2])] = [[info, data_str]]
        else:
            empty_summary_dict["{0}_{1}".format(info[0], info[2])].append([info, data_str])

    data_zone_lines = open(data_zone_path, "r", encoding="utf-8").readlines()
    data_zone_lines = list(set(data_zone_lines))
    data_zone_lines = [parse_line_data(line) for line in data_zone_lines]
    # 功能区 summary 数据生成 功能区数据 没有角度
    zone_summary_dict = {}
    for info, data_str in data_zone_lines:
        if "{0}_{1}".format(info[0], info[2]) not in zone_context_summary_dict:  # 仅有功能区 无功能区上下文的数据 过滤
            continue
        if "{0}_{1}".format(info[0], info[2]) not in zone_summary_dict:
            zone_summary_dict["{0}_{1}".format(info[0], info[2])] = [[info, data_str]]
        else:
            zone_summary_dict["{0}_{1}".format(info[0], info[2])].append([info, data_str])
    print("zone_context_summary:{0}, empty_summary:{1}, zone_summary:{2}".format(len(zone_context_summary_dict.keys()), len(empty_summary_dict.keys()), len(zone_summary_dict.keys()) * 4))
    names = list(set(zone_context_summary_dict.keys()))
    line_indexs = list(set([name.split("_")[0] for name in names]))
    random.shuffle(line_indexs)
    test_nums = min(888, len(names) * 0.1)  # 测试集数据不能超过888
    test_line_indexs = line_indexs[: test_nums]
    print("部分测试集:{0}".format(test_line_indexs[: 10]))
    train_names = list(filter(lambda x: x.split("_")[0] not in test_line_indexs, names))
    test_names = list(filter(lambda x: x.split("_")[0] in test_line_indexs, names))

    train_test = [name.split("_")[0] for name in train_names]
    test_test = [name.split("_")[0] for name in test_names]
    print("测试集与训练集重复数据:{0}".format(set(train_test).intersection(set(test_test))))
    paths = [record_train_path, record_test_path]
    train_indexs = list(range(len(train_names)))
    test_indexs = list(range(len(test_names)))
    # 训练样本打乱顺序  房间粒度的打乱顺序
    random.shuffle(train_indexs)
    all_indexs = [train_indexs, test_indexs]
    all_names = [train_names, test_names]

    for i in range(2):
        path = paths[i]
        writer = tf.python_io.TFRecordWriter(path)
        now_indexs = all_indexs[i]
        now_names = all_names[i]
        print("生成数据路径:{0}, indexs:{1}, names:{2}".format(paths[i], len(now_indexs), len(now_names)))
        for index in tqdm(now_indexs[: limit]):
            name = now_names[index]
            # 空房间
            if name not in empty_summary_dict or name not in zone_summary_dict:  # 数据出现缺失的情况
                continue
            empty_param, empty_data_str = empty_summary_dict[name][0]
            # 空房间数据
            empty_img_zid, empty_img_cid, empty_img_mid, empty_img_scid = parse_numpy_feature_data(data=empty_data_str,
                                                                                                   image_width=image_width,
                                                                                                   image_height=image_height)
            # 功能区上下文
            zone_context_summary = zone_context_summary_dict[name]
            # 过滤掉无效的功能区id
            if valid_zids is None:
                zone_context_summary = list(filter(lambda x: x[0][3] not in ignore_zids, zone_context_summary))
            else:
                zone_context_summary = list(filter(lambda x: x[0][3] in valid_zids, zone_context_summary))
            # 功能区上下文 排序
            zone_context_summary = list(sorted(zone_context_summary, key=lambda x: x[0][4]))
            zone_context_images = [
                parse_numpy_feature_data(data=summary[1], image_width=image_width, image_height=image_height)
                for summary in zone_context_summary]
            # n步的结果 作为n+1步的输入
            zone_context_zid_images = [empty_img_zid] + [img[0] for img in zone_context_images]
            zone_context_cid_images = [empty_img_cid] + [img[1] for img in zone_context_images]
            zone_context_mid_images = [empty_img_mid] + [img[2] for img in zone_context_images]
            zone_context_scid_images = [empty_img_scid] + [img[3] for img in zone_context_images]

            # 根据上下文 生成距离
            zone_context_distance_zid_images = [get_frame_orientation_wall_distance(img=img, image_width=image_width, image_height=image_height, default_value=98) for img in zone_context_zid_images]

            # 功能区数据
            zone_name = name.split("_")[0]
            zone_name = "{0}_0".format(zone_name)  # 功能区仅有0度的数据
            zone_summary = zone_summary_dict[zone_name]

            labels = [int(summary[0][5]) for summary in zone_context_summary]

            # 过滤掉无效的功能区
            if valid_zids is None:
                zone_summary = list(filter(lambda x: x[0][3] not in ignore_zids, zone_summary))
            else:
                zone_summary = list(filter(lambda x: x[0][3] in valid_zids, zone_context_summary))
            # 功能区数据 排序
            zone_summary = list(sorted(zone_summary, key=lambda x: x[0][4]))
            zone_images = [parse_numpy_feature_data(data=summary[1], image_width=image_width, image_height=image_height)
                           for summary in zone_summary]

            zone_zid_images = [img[0] for img in zone_images]
            zone_cid_images = [img[1] for img in zone_images]
            zone_mid_images = [img[2] for img in zone_images]
            zone_scid_images = [img[3] for img in zone_images]

            for i in range(len(labels)):
                label = labels[i]
                zone_zid_image = zone_zid_images[i]
                zone_cid_image = zone_cid_images[i]
                zone_mid_image = zone_mid_images[i]
                zone_scid_image = zone_scid_images[i]
                zone_context_zid_image = zone_context_zid_images[i]
                zone_context_cid_image = zone_context_cid_images[i]
                zone_context_scid_image = zone_context_scid_images[i]
                zone_context_mid_image = zone_context_mid_images[i]
                zone_context_distance_zid_image = zone_context_distance_zid_images[i]

                zone_zid_image_byte = tf.compat.as_bytes(np.array(zone_zid_image, dtype=dtype).tostring())
                zone_cid_image_byte = tf.compat.as_bytes(np.array(zone_cid_image, dtype=dtype).tostring())
                zone_mid_image_byte = tf.compat.as_bytes(np.array(zone_mid_image, dtype=dtype).tostring())
                zone_scid_image_byte = tf.compat.as_bytes(np.array(zone_scid_image, dtype=dtype).tostring())
                zones_context_zid_image_byte = tf.compat.as_bytes(np.array(zone_context_zid_image, dtype=dtype).tostring())
                zones_context_cid_image_byte = tf.compat.as_bytes(np.array(zone_context_cid_image, dtype=dtype).tostring())
                zones_context_scid_image_byte = tf.compat.as_bytes(np.array(zone_context_scid_image, dtype=dtype).tostring())
                zones_context_mid_image_byte = tf.compat.as_bytes(np.array(zone_context_mid_image, dtype=dtype).tostring())
                zone_context_distance_zid_image_byte = tf.compat.as_bytes(np.array(zone_context_distance_zid_image, dtype=dtype).tostring())
                tf_zone_zid_image = bytes_feature(zone_zid_image_byte)
                tf_zone_cid_image = bytes_feature(zone_cid_image_byte)
                tf_zone_mid_image = bytes_feature(zone_mid_image_byte)
                tf_zone_scid_image = bytes_feature(zone_scid_image_byte)
                tf_zone_context_zid_image = bytes_feature(zones_context_zid_image_byte)
                tf_zone_context_cid_image = bytes_feature(zones_context_cid_image_byte)
                tf_zone_context_scid_image = bytes_feature(zones_context_scid_image_byte)
                tf_context_mid_image = bytes_feature(zones_context_mid_image_byte)
                tf_zone_context_distance_zid_image = bytes_feature(zone_context_distance_zid_image_byte)
                tf_label = int64_feature(label)
                feature = {"zone_zid_image": tf_zone_zid_image,
                           "zone_cid_image": tf_zone_cid_image,
                           "zone_mid_image": tf_zone_mid_image,
                           # "zone_scid_image": tf_zone_scid_image,
                           "zone_context_zid_image": tf_zone_context_zid_image,
                           "zone_context_cid_image": tf_zone_context_cid_image,
                           "zone_context_mid_image": tf_context_mid_image,
                           "zone_context_scid_image": tf_zone_context_scid_image,
                           "zone_context_distance_zid_image": tf_zone_context_distance_zid_image,
                           "label": tf_label}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        writer.close()


def parse_data_sequence_tf_record(data_empty_path: str,
                                 data_zone_path: str,
                                 data_zone_context_path: str,
                                 room_json_path: str,
                                 image_width: int,
                                 image_height: int,
                                 record_train_path: str,
                                 record_test_path: str,
                                 dtype=np.int32,
                                 ignore_zids: list=["98", "99"],
                                 valid_zids: list=None,
                                 max_length: int=5,
                                 reverse: bool=False,
                                 limit: int=100000,
                                 seed: int=200):
    """
    序列的数据生成
    :param data_empty_path:
    :param data_zone_path:
    :param data_zone_context_path:
    :param room_json_path:  room_json字段
    :param image_width:
    :param image_height:
    :param record_path:
    :param dtype:
    :param ignore_zids:
    :param valid_zids:
    :param max_length: 序列的最大长度 超出此长度 取前 max_length个数据 不足此长度 进行填充
    :param reverse: 生成的序列是否进行翻转
    :return:
    """
    random.seed(seed)
    # 由功能区上下文当中 找出所有有效的数据 name
    data_zone_context_lines = open(data_zone_context_path, "r", encoding="utf-8").readlines()
    data_zone_context_lines = list(set(data_zone_context_lines))
    data_zone_context_lines = [parse_line_data(line) for line in data_zone_context_lines]

    # 功能区上下文 summary 数据生成
    zone_context_summary_dict = {}
    for info, data_str in data_zone_context_lines:
        if "{0}_{1}".format(info[0], info[2]) not in zone_context_summary_dict:
            zone_context_summary_dict["{0}_{1}".format(info[0], info[2])] = [[info, data_str]]
        else:
            zone_context_summary_dict["{0}_{1}".format(info[0], info[2])].append([info, data_str])

    data_empty_lines = open(data_empty_path, "r", encoding="utf-8").readlines()
    data_empty_lines = list(set(data_empty_lines))
    data_empty_lines = [parse_line_data(line) for line in data_empty_lines]

    # 空房间 summary 数据生成
    empty_summary_dict = {}
    for info, data_str in data_empty_lines:
        if "{0}_{1}".format(info[0], info[2]) not in zone_context_summary_dict:  # 仅有空房间 无功能区上下文的数据 过滤
            continue
        if "{0}_{1}".format(info[0], info[2]) not in empty_summary_dict:
            empty_summary_dict["{0}_{1}".format(info[0], info[2])] = [[info, data_str]]
        else:
            empty_summary_dict["{0}_{1}".format(info[0], info[2])].append([info, data_str])

    data_zone_lines = open(data_zone_path, "r", encoding="utf-8").readlines()
    data_zone_lines = list(set(data_zone_lines))
    data_zone_lines = [parse_line_data(line) for line in data_zone_lines]
    # 功能区 summary 数据生成 功能区数据 没有角度
    zone_summary_dict = {}
    for info, data_str in data_zone_lines:
        if "{0}_{1}".format(info[0], info[2]) not in zone_context_summary_dict:  # 仅有功能区 无功能区上下文的数据 过滤
            continue
        if "{0}_{1}".format(info[0], info[2]) not in zone_summary_dict:
            zone_summary_dict["{0}_{1}".format(info[0], info[2])] = [[info, data_str]]
        else:
            zone_summary_dict["{0}_{1}".format(info[0], info[2])].append([info, data_str])

    # room_json数据
    data_room_lines = open(room_json_path, "r", encoding="utf-8").readlines()

    print("zone_context_summary:{0}, empty_summary:{1}, zone_summary:{2}".format(
        len(zone_context_summary_dict.keys()), len(empty_summary_dict.keys()), len(zone_summary_dict.keys()) * 4))
    names = list(set(zone_context_summary_dict.keys()))
    line_indexs = list(set([name.split("_")[0] for name in names]))
    random.shuffle(line_indexs)
    test_nums = min(888, len(names) * 0.1)  # 测试集数据不能超过888
    test_line_indexs = line_indexs[: test_nums]
    print("部分测试集:{0}".format(test_line_indexs[: 10]))
    train_names = list(filter(lambda x: x.split("_")[0] not in test_line_indexs, names))
    test_names = list(filter(lambda x: x.split("_")[0] in test_line_indexs, names))

    train_test = [name.split("_")[0] for name in train_names]
    test_test = [name.split("_")[0] for name in test_names]
    print("测试集与训练集重复数据:{0}".format(set(train_test).intersection(set(test_test))))
    paths = [record_train_path, record_test_path]
    train_indexs = list(range(len(train_names)))
    test_indexs = list(range(len(test_names)))
    # 训练样本打乱顺序  房间粒度的打乱顺序
    random.shuffle(train_indexs)
    all_indexs = [train_indexs, test_indexs]
    all_names = [train_names, test_names]

    for i in range(2):
        path = paths[i]
        writer = tf.python_io.TFRecordWriter(path)
        now_indexs = all_indexs[i]
        now_names = all_names[i]
        print("生成数据路径:{0}, indexs:{1}, names:{2}".format(paths[i], len(now_indexs), len(now_names)))
        for index in tqdm(now_indexs[: limit]):
            name = now_names[index]
            # 空房间
            if name not in empty_summary_dict or name not in zone_summary_dict:  # 数据出现缺失的情况
                continue
            empty_param, empty_data_str = empty_summary_dict[name][0]
            # 空房间数据
            empty_img_zid, empty_img_cid, empty_img_mid, empty_img_scid = parse_numpy_feature_data(data=empty_data_str,
                                                                                                   image_width=image_width,
                                                                                                   image_height=image_height)
            # 功能区上下文
            zone_context_summary = zone_context_summary_dict[name]
            # 过滤掉无效的功能区id
            if valid_zids is None:
                zone_context_summary = list(filter(lambda x: x[0][3] not in ignore_zids, zone_context_summary))
            else:
                zone_context_summary = list(filter(lambda x: x[0][3] in valid_zids, zone_context_summary))
            # 功能区上下文 排序
            zone_context_summary = list(sorted(zone_context_summary, key=lambda x: x[0][4]))
            zone_context_images = [
                parse_numpy_feature_data(data=summary[1], image_width=image_width, image_height=image_height)
                for summary in zone_context_summary]
            # n步的结果 作为n+1步的输入
            zone_context_zid_images = [empty_img_zid] + [img[0] for img in zone_context_images]
            zone_context_cid_images = [empty_img_cid] + [img[1] for img in zone_context_images]
            zone_context_mid_images = [empty_img_mid] + [img[2] for img in zone_context_images]
            zone_context_scid_images = [empty_img_scid] + [img[3] for img in zone_context_images]

            # 功能区数据
            zone_name = name.split("_")[0]
            zone_name = "{0}_0".format(zone_name)  # 功能区仅有0度的数据
            zone_summary = zone_summary_dict[zone_name]

            labels = [int(summary[0][5]) for summary in zone_context_summary]

            # 过滤掉无效的功能区
            if valid_zids is None:
                zone_summary = list(filter(lambda x: x[0][3] not in ignore_zids, zone_summary))
            else:
                zone_summary = list(filter(lambda x: x[0][3] in valid_zids, zone_context_summary))
            # 功能区数据 排序
            zone_summary = list(sorted(zone_summary, key=lambda x: x[0][4]))
            zone_images = [parse_numpy_feature_data(data=summary[1], image_width=image_width, image_height=image_height)
                           for summary in zone_summary]
            # 根据功能区上下文 生成距离图
            zone_context_distance_zid_images = [get_frame_orientation_wall_distance(img=img, image_width=image_width, image_height=image_height, default_value=98) for img in zone_context_zid_images]

            zone_zid_images = [img[0] for img in zone_images]
            zone_cid_images = [img[1] for img in zone_images]
            zone_mid_images = [img[2] for img in zone_images]
            zone_scid_images = [img[3] for img in zone_images]

            # label=0 预留给mask
            labels = [label+1 for label in labels]

            labels = mask_sequence(labels, reverse=reverse, max_length=max_length)
            zone_context_zid_images = mask_sequence(zone_context_zid_images, reverse=reverse, h=image_height,
                                                    w=image_width, max_length=max_length)
            zone_context_distance_zid_images = mask_sequence(zone_context_distance_zid_images, reverse=reverse, h=image_height,
                                                             w=image_width, max_length=max_length)
            zone_context_cid_images = mask_sequence(zone_context_cid_images, reverse=reverse, h=image_height,
                                                    w=image_width, max_length=max_length)
            zone_context_mid_images = mask_sequence(zone_context_mid_images, reverse=reverse, h=image_height,
                                                    w=image_width, max_length=max_length)
            zone_context_scid_images = mask_sequence(zone_context_scid_images, reverse=reverse, h=image_height,
                                                    w=image_width, max_length=max_length)
            zone_zid_images = mask_sequence(zone_zid_images, reverse=reverse, h=image_height, w=image_width,
                                            max_length=max_length)
            zone_cid_images = mask_sequence(zone_cid_images, reverse=reverse, h=image_height, w=image_width,
                                            max_length=max_length)
            zone_mid_images = mask_sequence(zone_mid_images, reverse=reverse, h=image_height, w=image_width,
                                            max_length=max_length)
            zone_scid_images = mask_sequence(zone_scid_images, reverse=reverse, h=image_height, w=image_width,
                                            max_length=max_length)

            room_json_str = data_room_lines[int(empty_param[0]) - 1]  # 数据下标问题
            room_json_byte = bytes(room_json_str, encoding="utf8")

            zone_zid_images_byte = tf.compat.as_bytes(np.array(zone_zid_images, dtype=dtype).tostring())
            zone_cid_images_byte = tf.compat.as_bytes(np.array(zone_cid_images, dtype=dtype).tostring())
            zone_mid_images_byte = tf.compat.as_bytes(np.array(zone_mid_images, dtype=dtype).tostring())
            zone_scid_images_byte = tf.compat.as_bytes(np.array(zone_scid_images, dtype=dtype).tostring())
            zones_context_zid_images_byte = tf.compat.as_bytes(np.array(zone_context_zid_images, dtype=dtype).tostring())
            zones_context_cid_images_byte = tf.compat.as_bytes(np.array(zone_context_cid_images, dtype=dtype).tostring())
            zones_context_scid_images_byte = tf.compat.as_bytes(np.array(zone_context_scid_images, dtype=dtype).tostring())
            zones_context_mid_images_byte = tf.compat.as_bytes(np.array(zone_context_mid_images, dtype=dtype).tostring())
            zone_context_distance_zid_images_byte = tf.compat.as_bytes(np.array(zone_context_distance_zid_images, dtype=dtype).tostring())
            tf_zone_zid_images = bytes_feature(zone_zid_images_byte)
            tf_zone_cid_images = bytes_feature(zone_cid_images_byte)
            tf_zone_mid_images = bytes_feature(zone_mid_images_byte)
            tf_zone_scid_images = bytes_feature(zone_scid_images_byte)
            tf_zone_context_zid_images = bytes_feature(zones_context_zid_images_byte)
            tf_zone_context_cid_images = bytes_feature(zones_context_cid_images_byte)
            tf_zone_context_scid_images = bytes_feature(zones_context_scid_images_byte)
            tf_context_mid_images = bytes_feature(zones_context_mid_images_byte)
            tf_zone_context_distance_zid_images = bytes_feature(zone_context_distance_zid_images_byte)

            labels_byte = tf.compat.as_bytes(np.array(labels, dtype=dtype).tostring())
            tf_labels = bytes_feature(labels_byte)

            tf_room_json = bytes_feature(room_json_byte)

            feature = {"zone_zid_images": tf_zone_zid_images,
                       "zone_cid_images": tf_zone_cid_images,
                       "zone_mid_images": tf_zone_mid_images,
                       # "zone_scid_images": tf_zone_scid_images,
                       "zone_context_zid_images": tf_zone_context_zid_images,
                       "zone_context_cid_images": tf_zone_context_cid_images,
                       "zone_context_scid_images": tf_zone_context_scid_images,
                       "zone_context_mid_images": tf_context_mid_images,
                       "zone_context_distance_zid_images": tf_zone_context_distance_zid_images,
                       "labels": tf_labels,
                       "room_json": tf_room_json}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


def mask_sequence(states: list, max_length: int, reverse: bool, h: int = None, w: int = None):
    """
    填充数据
    :param states:
    :param max_length:
    :param reverse:
    :param h
    :param w
    :return:
    """
    states = copy.deepcopy(states)
    if h is None:
        if len(states) > max_length:  # 超出部分
            states = states[: max_length]
        if reverse:
            states = [0] * (max_length - len(states)) + states
        else:
            states = states + [0] * (max_length - len(states))
    else:
        if len(states) > max_length:  # 超出部分
            states = states[: max_length]
        if reverse:
            states = [[[0] * h] * w] * (max_length - len(states)) + states
        else:
            states = states + [[[0] * h] * w] * (max_length - len(states))
    return states

