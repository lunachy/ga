# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    APIForRpcServer
    Description :    
    Author      :    zhaowen
    date        :    2019/5/30
____________________________________________________________________
    Change Activity:
                        2019/5/30:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'
import sys
sys.path.append(r"F:\pycharm\WorkSpace\GA\GA_functionZone")

#from graph.TransformerLayoutModel import LayoutTransformer_vgg
from Layout_Service.graph.TransformerCnnPicx import LayoutTransformer_vgg

import os
import tensorflow as tf

from tqdm import tqdm
from Layout_Service.utils.Hyper_parameter import Hyperparams as hp
# from data.StateDataGens import SingleGlobalStateFurnitureDataSets
hp.max_length = 5
hp.maxlen = 5
hp.image_width = 64
hp.image_height = 64
hp.use_cnn_loss = False
hp.use_cnn = False
hp.use_cnn_predict_predict = False
hp.furniture_width = 64
hp.furniture_height = 64
hp.furniture_embedding_size = 4
image_width = hp.image_width
image_height = hp.image_height
furniture_nums = hp.furniture_nums
label_size = hp.label_size
furniture_width = hp.furniture_width
furniture_height = hp.furniture_height
furniture_embedding_size = hp.furniture_embedding_size
max_length = hp.max_length
keep_prob = 1.0

class LayoutServerAPI():

    def __init__(self, configPath: str = "../checkCrash/config.props"):
        self.configPath = configPath

    def intModel(self, model_path: str = r"d:workspace/checkpoint/transformer_model_nocnn/model-105"):
        self.model_path = model_path
        model_path = self.model_path
        configPath = self.configPath
        tf.reset_default_graph()
        transformer_layout_model = LayoutTransformer_vgg(
            image_width=image_width,
            image_height=image_height,
            furniture_nums=furniture_nums,
            label_size=label_size,
            furniture_width=furniture_width,
            furniture_height=furniture_height,
            furniture_embedding_size=furniture_embedding_size,
            max_length=max_length, configPath=configPath)

        sess = transformer_layout_model.create_sess()

        transformer_layout_model.build_cnn_graph(
            trainable=False,
            filter_list=[64, 128, 256, 512, 512],
            normal_model="norm",
            keep_prob=keep_prob, mask=True)

        var_to_restore_cnn = tf.global_variables()
        transformer_layout_model.initializer(sess=sess)
        transformer_layout_model.build_cnn_graph(
            trainable=False,
            filter_list=[64, 128, 256, 512, 512],
            normal_model="norm",
            keep_prob=keep_prob, mask=True)
        transformer_layout_model.build_attention_graph(trainable=False, )

        vars = tf.global_variables()
        var_to_restore_rnn = [val for val in vars if val not in var_to_restore_cnn]
        var_to_init = var_to_restore_rnn
        sess.run(tf.variables_initializer(var_to_init))
        vars = tf.global_variables()
        var_to_restore_rnn = [val for val in vars if val not in var_to_restore_cnn]
        var_to_init = var_to_restore_rnn
        sess.run(tf.initialize_variables(var_to_init))
        transformer_layout_model.restore(sess=sess,
                                         model_path=model_path)
        self.model = transformer_layout_model

    def intModelTrans(self, model_path: str = r"d:workspace/checkpoint/transformer_model_nocnn/model-105"):
        self.model_path = model_path
        model_path = self.model_path
        configPath = self.configPath
        tf.reset_default_graph()

        transformer_layout_model = LayoutTransformer_vgg(
            image_width=image_width,
            image_height=image_height,
            furniture_nums=furniture_nums,
            label_size=label_size,
            furniture_width=furniture_width,
            furniture_height=furniture_height,
            furniture_embedding_size=furniture_embedding_size,
            max_length=max_length, configPath=configPath)

        sess = transformer_layout_model.create_sess()

        var_to_restore_cnn = tf.global_variables()
        transformer_layout_model.initializer(sess=sess)
        transformer_layout_model.build_attention_graph(trainable=False, )
        vars = tf.global_variables()
        var_to_restore_rnn = [val for val in vars if val not in var_to_restore_cnn]
        var_to_init = var_to_restore_rnn
        sess.run(tf.variables_initializer(var_to_init))
        vars = tf.global_variables()
        var_to_restore_rnn = [val for val in vars if val not in var_to_restore_cnn]
        var_to_init = var_to_restore_rnn
        sess.run(tf.initialize_variables(var_to_init))
        transformer_layout_model.restore(sess=sess,
                                         model_path=model_path)
        self.model = transformer_layout_model

    # result needs no checkFullPath.
    def predictAPI(self, room_json: str, top_k: int, room_type_id: int, model_type: int = 0,
                   type_pred: str = "predict",save_log=False):

        result = self.model.predict_route(room_json, top_k=top_k, room_type_id=room_type_id,save_log=save_log)
        if type_pred == "predict":
            return result
        else:
            return result, self.model.clash_checker.data_infos
