#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:   ga_optimizer.py
@author: chengyong
@time:   2019/6/15 10:10
@desc:   Genetic Algorithm to optimize layout
"""

from math import sin, cos, pi
import sys
import os
import tensorflow as tf
import pandas as pd
import json
import numpy as np
from checkCrash.CrashChecker import CrashChecker
# from Evalutor_Model import evalutor_predict
from show_img import show_img
from Layout_Service.graph.APIForRpcServer import LayoutServerAPI
from Evalutor_Model.graph.BasicEvalutorModel import BasicRegressionModel

from gafp import GAFP

ROOMTYPEID = 2
LABEL_IMG_CRASH = []

image_width = 64
image_height = 64
furniture_nums = 5000
label_size = 16 * 16 * 4 + 1
furniture_width = 64
furniture_height = 64
furniture_embedding_size = 4  # 向量维度
max_length = 8
single_cnn_layout_model = BasicRegressionModel(image_width=image_width,
                                               image_height=image_height,
                                               furniture_nums=furniture_nums,
                                               label_size=label_size,
                                               furniture_width=furniture_width,
                                               furniture_height=furniture_height,
                                               furniture_embedding_size=furniture_embedding_size,
                                               max_length=max_length)

# 载入模型
saver = tf.train.Saver()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
config = tf.ConfigProto()  # 支持多个gpu的sess
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

single_cnn_layout_model.build_cnn_graph(trainable=False, filter_list=[64, 128, 256, 512, 512],
                                        keep_prob=None, normal_model="batch_norm")
single_cnn_layout_model.build_regression_graph(trainable=False, keep_prob=None,
                                               filter_list=[64, 128, 256, 512, 512],
                                               normal_model="batch_norm")
single_cnn_layout_model.initializer(sess=sess)
path = './Evalutor_Model/model/evalutor_train.ckpt'
saver.restore(sess, path)
print('Loading success...')


def predict(img):
    prediction = sess.run(single_cnn_layout_model.output_a,
                          feed_dict={single_cnn_layout_model.room_a: img})
    return prediction[0][0]


def population(conf_path, model_path, room_str, size):
    layout_model = LayoutServerAPI(configPath=conf_path)
    layout_model.intModelTrans(model_path=model_path)
    return layout_model.predictAPI(room_str, top_k=size, room_type_id=ROOMTYPEID, model_type=0)


# Define fitness function.
def fitness(indv: list):
    # x, *_ = indv
    # return round(x * sin(2 * pi * x) + x * cos(2 * pi * x), 2)
    ret, embedding = ck.checkFullPathEx(indv)
    ck.room_finish()
    if ret == 0:
        _img = ck.buildEmbeddingMat(embedding)
        individual_img = np.array(_img[0]).reshape([-1, ck.numSegs, ck.numSegs])
        return predict(individual_img)
    else:
        return None


if '__main__' == __name__:
    json_path = './data/100_bedroom.json'
    conf_path = './checkCrash/config.props'
    model_path = "./Layout_Service/model/model-55"
    # df = pd.read_csv(json_path, header=None)
    total_row = -1
    no_crash_row = -1

    # epoch of evolution
    generation_count = 10
    # population size
    popu_size = 8
    # evolution size, maybe double of population size
    evolu_size = 12
    top1_list, top4_list, top8_list = [], [], []
    ck = CrashChecker(conf_path)
    with open(json_path) as f:
        for line in f:
            if no_crash_row == 8:
                break
            total_row += 1
            print('*' * 88)
            print('total_row: ', total_row)
            top1_list.append(0)
            top4_list.append(0)
            top8_list.append(0)

            ret = ck.init_room(roomTypeId=ROOMTYPEID, room_str=line)
            if ret != 0:
                ck.room_finish()
                continue
            # print('ck.data_infos: ', ck.data_infos)

            # check if room_str crash
            true_labels = [x['label'] for x in ck.data_infos]
            # print('true_labels: ', true_labels)
            ret, embedding = ck.checkFullPathEx(true_labels)
            ck.room_finish()
            if ret == 0:
                no_crash_row += 1
                img = ck.buildEmbeddingMat(embedding)[0]
                ck.room_finish()
                show_img(img, 'row_{}'.format(total_row))
                print('true_labels precision: ', predict(img.reshape([-1, ck.numSegs, ck.numSegs])))
            else:
                continue

            # generate popu_size individuals
            populations = population(conf_path, model_path, line, popu_size)
            print('populations size: ', popu_size, '    initial populations: ', populations)

            # TODO: Duplicate removal(if needed?)
            gafp = GAFP(populations)
            for i in range(generation_count):
                local_indvs = []
                # _scores = [fitness(x) for x in gafp.population]
                # sorted_layouts = sorted(layouts, key=lambda x: x[1], reverse=True)
                # best_indv = max(gafp.population, key=lambda x: _scores[gafp.population.index(x)])

                # Fill the new population until popu_size.
                while len(local_indvs) < evolu_size:
                    # Select father and mother.
                    parents = gafp.tournament_selection(fitness)
                    # parents = gafp.roulette_wheel_selection(fitness)
                    # Crossover.
                    children = gafp.cross(*parents)
                    for child in children:
                        # Mutation and check collision.
                        mutate_child = gafp.mutate(child)
                        ret = ck.checkFullPath(mutate_child)
                        ck.room_finish()
                        if ret == 0:
                            # Collect children.
                            local_indvs.append(mutate_child)

                zip_indvs_scores = list(zip(local_indvs, map(fitness, local_indvs)))
                # sort indvs and get top popu_size
                zip_indvs_scores = list(filter(lambda x: x[1], zip_indvs_scores))
                sorted_indvs_scores = sorted(zip_indvs_scores, key=lambda x: x[1], reverse=True)[:popu_size]

                unzip_indvs_scores = list(zip(*sorted_indvs_scores))
                gafp.population, _scores = unzip_indvs_scores[0], unzip_indvs_scores[1]

                print('generation_count: ', i)
                # print('     gafp.population: ', gafp.population)
                print('     scores: ', _scores)
                # print('     best_score: ', _scores[0])
            # break

            if true_labels == gafp.population[:1]:
                top1_list[total_row] = 1

            for _layout in gafp.population[:4]:
                if _layout == true_labels:
                    top4_list[total_row] = 1
                    break

            for _layout in gafp.population[:8]:
                if _layout == true_labels:
                    top8_list[total_row] = 1
                    break

        top1_precision = sum(top1_list) / len(top1_list)
        top4_precision = sum(top4_list) / len(top4_list)
        top8_precision = sum(top4_list) / len(top4_list)
        print('top1: ', top1_list, 'top1_precision: ', top1_precision)
        print('top4: ', top4_list, 'top4_precision: ', top4_precision)
        print('top8: ', top8_list, 'top8_precision: ', top8_precision)
