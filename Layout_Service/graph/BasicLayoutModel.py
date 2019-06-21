import tensorflow as tf
import numpy as np
import os
from checkCrash.CrashChecker import CrashChecker
from Layout_Service.beam_search.BeamSearchBase import CrashSingleStepBeamSearch

from Layout_Service.utils.graph_utils import *


class SingleBasicLayoutModel(object):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int=10,
                 tf_dtype: str="32"):
        """
        :param image_width:  图像宽
        :param image_height:  图像高
        :param furniture_nums:  家具数目
        :param label_size:
        :param furniture_width:  家具图像宽
        :param furniture_height:  家具图像高
        :param furniture_embedding_size:  家具的嵌入维数
        :param max_length:  序列最大的长度
        :param dtype:  数据类型 32 还是16
        """
        self.image_width = image_width
        self.image_height = image_height
        self.furniture_nums = furniture_nums
        self.label_size = label_size
        self.furniture_width = furniture_width
        self.furniture_height = furniture_height
        self.furniture_embedding_size = furniture_embedding_size
        self.max_length = max_length
        assert tf_dtype in ["16", "32"]
        self.tf_dtype = tf_dtype

        self.room_zid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_zid", dtype=tf.int32)
        self.room_cid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_cid", dtype=tf.int32)
        self.room_mid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_mid", dtype=tf.int32)
        self.room_scid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_scid", dtype=tf.int32)

        self.room_distance = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_distance", dtype=tf.int32)

        self.zone_zid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="zone_zid", dtype=tf.int32)
        self.zone_cid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="zone_cid", dtype=tf.int32)
        self.zone_mid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="zone_mid", dtype=tf.int32)

        self.target = tf.placeholder(shape=(None,), name="target", dtype=tf.int32)
        self.target_one_hot = tf.one_hot(self.target, self.label_size, 1, 0)
        # 样本权重 用于计算loss todo

        # 评分值
        self.layout_marks = tf.placeholder(shape=(None, ), name="layout_marks", dtype=tf.float32)

        # 用于学习率的衰减
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        # 功能区 zid cid mid 输入

        # cnn 损失部分  基础特征
        self.cnn_loss = NotImplemented
        self.cnn_acc = NotImplemented
        self.cnn_train_op = NotImplemented

        # rnn 损失部分  布局
        self.rnn_loss = NotImplemented
        self.rnn_acc = NotImplemented
        self.rnn_train_op = NotImplemented

        # attention
        self.attention_loss = NotImplemented
        self.attention_acc = NotImplemented
        self.attention_train_op = NotImplemented

        # regression 损失部分  评分
        self.regression_loss = NotImplemented
        self.regression_acc = NotImplemented
        self.regression_train_op = NotImplemented

        # cnn 特征共享层
        self.infrastructure_cnn_feature = NotImplemented
        self.furniture_cnn_feature = NotImplemented
        # cnn 特征共享层 (未展开)
        self.middle_state_cnn_not_flatten = NotImplemented
        self.furniture_cnn_not_flatten = NotImplemented
        # cnn 输出结果分布
        self.cnn_output_distribute = NotImplemented

        # rnn 输出结果分布
        self.rnn_output_distribute = NotImplemented

        # attention 输出结果分布
        self.attention_output_distribute = NotImplemented

        # regression 输出结果
        self.regression_output = NotImplemented

        # 学习率 衰减
        self.lr_rate = tf.maximum(0.01, tf.train.exponential_decay(0.001, self.global_step, 10240, 0.001))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)

        # l2 损失函数部分
        self.cnn_l2_loss = NotImplemented
        self.attention_l2_loss = NotImplemented
        self.rnn_l2_loss = NotImplemented
        self.regression = NotImplemented

        # 日志记录部分
        self.cnn_train_write = NotImplemented
        self.cnn_test_write = NotImplemented
        self.cnn_train_summary = NotImplemented
        self.cnn_test_summary = NotImplemented
        self.cnn_train_summary_op = NotImplemented
        self.cnn_test_summary_op = NotImplemented

        self.attention_train_write = NotImplemented
        self.attention_test_write = NotImplemented
        self.attention_train_summary = NotImplemented
        self.attention_test_summary = NotImplemented
        self.attention_train_summary_op = NotImplemented
        self.attention_test_summary_op = NotImplemented

        self.rnn_train_write = NotImplemented
        self.rnn_test_write = NotImplemented
        self.rnn_train_summary = NotImplemented
        self.rnn_test_summary = NotImplemented
        self.rnn_train_summary_op = NotImplemented
        self.rnn_test_summary_op = NotImplemented

        self.regression_train_write = NotImplemented
        self.regression_test_write = NotImplemented
        self.regression_train_summary = NotImplemented
        self.regression_test_summary = NotImplemented
        self.regression_train_summary_op = NotImplemented
        self.regression_test_summary_op = NotImplemented

        self.sess = NotImplemented
        self.clash_checker = NotImplemented

        self.multi_gpu_train_op = NotImplemented
        self.multi_gpu_acc = NotImplemented
        self.multi_gpu_loss = NotImplemented

        self.id_list = NotImplemented

    def create_cnn_summary(self, log_path: str, sess: tf.Session):
        """
        创建 cnn的日志记录
        :param log_path:
        :return:
        """
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        cnn_train_write = tf.summary.FileWriter("{0}/train".format(log_path), sess.graph)
        cnn_test_write = tf.summary.FileWriter("{0}/test".format(log_path))
        cnn_train_summary = [tf.summary.scalar("acc", self.cnn_acc),
                             tf.summary.scalar("loss", self.cnn_loss)]
        cnn_test_summary = [tf.summary.scalar("acc", self.cnn_acc),
                            tf.summary.scalar("loss", self.cnn_loss)]
        cnn_train_summary_op = tf.summary.merge(cnn_train_summary)
        cnn_test_summary_op = tf.summary.merge(cnn_test_summary)

        self.cnn_train_write = cnn_train_write
        self.cnn_test_write = cnn_test_write
        self.cnn_train_summary = cnn_train_summary
        self.cnn_test_summary = cnn_test_summary
        self.cnn_train_summary_op = cnn_train_summary_op
        self.cnn_test_summary_op = cnn_test_summary_op

    def create_attention_summary(self, log_path: str, sess: tf.Session):
        """
        创建 attention的日志记录
        :param log_path:
        :return:
        """
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        attention_train_write = tf.summary.FileWriter("{0}/train".format(log_path), sess.graph)
        attention_test_write = tf.summary.FileWriter("{0}/test".format(log_path))
        attention_train_summary = [tf.summary.scalar("acc", self.attention_acc),
                                   tf.summary.scalar("loss", self.attention_loss)]
        attention_test_summary = [tf.summary.scalar("acc", self.attention_acc),
                            tf.summary.scalar("loss", self.attention_loss)]
        attention_train_summary_op = tf.summary.merge(attention_train_summary)
        attention_test_summary_op = tf.summary.merge(attention_test_summary)

        self.attention_train_write = attention_train_write
        self.attention_test_write = attention_test_write
        self.attention_train_summary = attention_train_summary
        self.attention_test_summary = attention_test_summary
        self.attention_train_summary_op = attention_train_summary_op
        self.attention_test_summary_op = attention_test_summary_op

    def create_rnn_summary(self, log_path: str, sess: tf.Session):
        """
        创建 rnn的日志记录
        :param log_path:
        :return:
        """
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        rnn_train_write = tf.summary.FileWriter("{0}/train".format(log_path), sess.graph)
        rnn_test_write = tf.summary.FileWriter("{0}/test".format(log_path))
        rnn_train_summary = [tf.summary.scalar("acc", self.rnn_acc),
                                   tf.summary.scalar("loss", self.rnn_loss)]
        rnn_test_summary = [tf.summary.scalar("acc", self.rnn_acc),
                                  tf.summary.scalar("loss", self.rnn_loss)]
        rnn_train_summary_op = tf.summary.merge(rnn_train_summary)
        rnn_test_summary_op = tf.summary.merge(rnn_test_summary)

        self.rnn_train_write = rnn_train_write
        self.rnn_test_write = rnn_test_write
        self.rnn_train_summary = rnn_train_summary
        self.rnn_test_summary = rnn_test_summary
        self.rnn_train_summary_op = rnn_train_summary_op
        self.rnn_test_summary_op = rnn_test_summary_op

    def create_regression_summary(self, log_path: str, sess: tf.Session):
        """
        创建回归的日志记录
        :param log_path:
        :return:
        """
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        regression_train_write = tf.summary.FileWriter("{0}/train".format(log_path), sess.graph)
        regression_test_write = tf.summary.FileWriter("{0}/test".format(log_path))
        regression_train_summary = [tf.summary.scalar("acc", self.regression_acc),
                                    tf.summary.scalar("loss", self.regression_loss)]
        regression_test_summary = [tf.summary.scalar("acc", self.regression_acc),
                                   tf.summary.scalar("loss", self.regression_loss)]
        regression_train_summary_op = tf.summary.merge(regression_train_summary)
        regression_test_summary_op = tf.summary.merge(regression_test_summary)

        self.regression_train_write = regression_train_write
        self.regression_test_write = regression_test_write
        self.regression_train_summary = regression_train_summary
        self.regression_test_summary = regression_test_summary
        self.regression_train_summary_op = regression_train_summary_op
        self.regression_test_summary_op = regression_test_summary_op

    def create_sess(self):
        """
        创建会话
        :return:
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()  # 支持多个gpu的sess
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.sess = sess
        return sess

    def initializer(self, sess: tf.Session, var_list: list=None):
        """
        初始化全局变量
        :param sess:
        :return:
        """
        if var_list is None:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        else:
            sess.run([tf.variables_initializer(var_list), tf.local_variables_initializer()])

    def restore(self, sess: tf.Session, model_path: str, var_list: list=None):
        """
        sess 加载预训练的权重
        :param model_path 模型文件
        :return:
        """
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, model_path)
        return sess

    def build_cnn_graph(self, trainable: bool=True):
        """
        建立cnn图 需压实现 cnn_loss cnn_acc cnn_train_op
        :param trainable:
        :return:
        """
        raise NotImplementedError

    def build_rnn_graph(self, trainable: bool=True):
        """
        建立rnn图 需要实现 rnn_loss rnn_acc rnn_train_op
        :param trainable:
        :return:
        """
        raise NotImplementedError

    def build_attention_graph(self, trainable: bool=True):
        """
        建立attention图 需要实现 attention_loss attention_acc attention_train_op
        :param trainable:
        :return:
        """
        raise NotImplemented

    def build_regression_graph(self, trainable: bool=True):
        """
        建立 raise NotImplemented图 需要实现 regression_loss regression_acc regression_train_op
        :param trainable:
        :return:
        """
        raise NotImplemented

    def predict_logits(self, sess: tf.Session,
                             logits_tensor: tf.Tensor,
                             feed_dict: dict):
        """
        预测结果
        :param sess:  会话
        :param logits_tensor:  输出的tensor
        :param feed_dict:  传入的数据
        :return:
        """
        return sess.run(logits_tensor, feed_dict=feed_dict)

    def predict_route(self, room_json: str, top_k: int, room_type_id: int, mask_label: bool=False):
        """
        预测全路径结果  单步模型 todo 支持序列分类
        :param room_json:
        :param top_k 搜索宽度
        :param room_type_id 房屋类型
        :param mask_label 是否进行mask操作
        :return:
        """
        id_list = self.id_list
        room_zid = self.room_zid if "zid" in id_list else None
        room_cid = self.room_cid if "cid" in id_list else None
        room_mid = self.room_mid if "mid" in id_list else None
        room_scid = self.room_scid if "scid" in id_list else None

        zone_zid = self.zone_zid if "zid" in id_list else None
        zone_cid = self.zone_cid if "cid" in id_list else None
        zone_mid = None  # 没有功能区mid数据

        input_tensors = [room_zid, room_cid, room_mid, room_scid, zone_zid, zone_cid, zone_mid]
        input_tensors = list(filter(lambda x: x is not None, input_tensors))  # 过滤掉 没有使用到的tensor
        return CrashSingleStepBeamSearch(sess=self.sess, output_distribute=self.cnn_output_distribute,
                                         input_tensors=input_tensors,
                                         clash_checker=self.clash_checker,  # 碰撞检测部分 只共享一个数据
                                         mask_label=mask_label,
                                         search_width=top_k,
                                         room_type_id=room_type_id,
                                         room_str=room_json,
                                         image_grid_num=64,
                                         use_id_list=self.id_list).run_beam_search()

    def load_clash_checker(self, configPath: str):
        """
        加载碰撞检测模型
        :param configPath:  碰撞检测加载
        :return:
        """
        self.clash_checker = CrashChecker(configPath=configPath)

    @property
    def train_param_nums(self):
        train_param_sum = get_train_param_nums()
        print("当前模型除embedding共有训练参数:{0}".format(train_param_sum))
        return train_param_sum

    def build_multi_gpu_cnn_tensor_graph(self, gpu_nums: int,
                                         labels: tf.Tensor,
                                         room_zid: tf.Tensor,
                                         room_cid: tf.Tensor,
                                         room_mid: tf.Tensor,
                                         room_scid: tf.Tensor,
                                         zone_zid: tf.Tensor,
                                         zone_cid: tf.Tensor,
                                         zone_mid: tf.Tensor,
                                         training: tf.Tensor,
                                         mask: bool = False,
                                         trainable: bool = True,
                                         keep_prob: int = None):
        """
        多个gpu运算
        :param gpu_nums:  gpu数目
        :param labels:
        :param mask:
        :param trainable:
        :param keep_prob:
        :return:
        """
        # 不同的gpu上面数据切分
        labels_list = tf.split(labels, gpu_nums)
        room_zid_list = tf.split(room_zid, gpu_nums) if room_zid is not None else None
        room_cid_list = tf.split(room_cid, gpu_nums) if room_cid is not None else None
        room_mid_list = tf.split(room_mid, gpu_nums) if room_mid is not None else None
        room_scid_list = tf.split(room_scid, gpu_nums) if room_scid is not None else None
        zone_zid_list = tf.split(zone_zid, gpu_nums) if room_zid is not None else None
        zone_cid_list = tf.split(zone_cid, gpu_nums) if zone_cid is not None else None
        zone_mid_list = tf.split(zone_mid, gpu_nums) if zone_mid is not None else None
        acc_list = []
        loss_list = []
        grads_list = []
        for i in range(gpu_nums):
            with tf.device('/gpu:%s' % i):
                with tf.name_scope('%s_%s' % ('tower', i)):
                    cnn_loss, accuracy = self.build_cnn_tensor_graph(labels=labels_list[i],
                                                                     room_zid=room_zid_list[
                                                                         i] if room_zid_list is not None else None,
                                                                     room_cid=room_cid_list[
                                                                         i] if room_cid_list is not None else None,
                                                                     room_mid=room_mid_list[
                                                                         i] if room_mid_list is not None else None,
                                                                     room_scid=room_scid_list[
                                                                         i] if room_scid_list is not None else None,
                                                                     zone_zid=zone_zid_list[
                                                                         i] if zone_zid_list is not None else None,
                                                                     zone_cid=zone_cid_list[
                                                                         i] if zone_cid_list is not None else None,
                                                                     zone_mid=zone_mid_list[
                                                                         i] if zone_mid_list is not None else None,
                                                                     trainable=trainable,
                                                                     mask=mask,
                                                                     multi_gpu=True,
                                                                     keep_prob=keep_prob,
                                                                     training=training)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        with tf.variable_scope("loss", reuse=True):
                            # 计算梯度
                            grads = self.optimizer.compute_gradients(cnn_loss)
                            grads_list.append(grads)
                            acc_list.append(accuracy)
                            loss_list.append(cnn_loss)
        # 平均损失 平均准确率
        mean_loss = tf.reduce_mean(loss_list, 0)
        mean_acc = tf.reduce_mean(acc_list, 0)
        mean_grads = average_gradients(grads_list)
        train_op = self.optimizer.apply_gradients(mean_grads, global_step=self.global_step)

        self.multi_gpu_train_op = train_op
        self.multi_gpu_loss = mean_loss
        self.multi_gpu_train_op = mean_acc
