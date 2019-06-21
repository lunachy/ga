import tensorflow as tf
import numpy as np
import os

from Evalutor_Model.utils.graph_utils import FullConnect, MaxPool, Basic2dConv, MultiFilter2dConv, GetTokenEmbedding


class BasicLayoutModel(object):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int=10):
        """
        :param image_width:  图像宽
        :param image_height:  图像高
        :param furniture_nums:  家具数目
        :param label_size:
        :param furniture_width:  家具图像宽
        :param furniture_height:  家具图像高
        :param furniture_embedding_size:  家具的嵌入维数
        :param max_length:  序列最大的长度
        """
        self.image_width = image_width
        self.image_height = image_height
        self.furniture_nums = furniture_nums
        self.label_size = label_size
        self.furniture_width = furniture_width
        self.furniture_height = furniture_height
        self.furniture_embedding_size = furniture_embedding_size
        self.max_length = max_length

        # 生成 placeholder
        # 户型图
        infrastructure = tf.placeholder(shape=(None, self.max_length, self.image_height, self.image_width), name="infrastructure", dtype=tf.int32)
        # 家具图
        furniture = tf.placeholder(shape=(None, self.max_length, self.furniture_height, self.furniture_width), name="furniture", dtype=tf.int32)
        # 标签
        target = tf.placeholder(shape=(None, self.max_length, ), name="target", dtype=tf.int32)
        target_one_hot = tf.one_hot(target, self.label_size, 1, 0)
        # 样本权重 用于计算loss todo

        # 家具长宽 (id)
        furniture_dx_id = tf.placeholder(shape=(None, self.max_length, ), name="furniture_dx_id", dtype=tf.int32)
        furniture_dy_id = tf.placeholder(shape=(None, self.max_length, ), name="furniture_dy_id", dtype=tf.int32)

        # 家具长宽 真实数据
        furniture_dx_value = tf.placeholder(shape=(None, self.max_length, ), name="furniture_dx_value", dtype=tf.float32)
        furniture_dy_value = tf.placeholder(shape=(None, self.max_length, ), name="furniture_dy_value", dtype=tf.float32)

        # 家具cid
        furniture_cids = tf.placeholder(shape=(None, self.max_length, ), name="furniture_cids", dtype=tf.int32)

        # 用于学习率的衰减
        self.global_step = tf.Variable(0, trainable=False)
        self.infrastructure = infrastructure
        self.furniture = furniture
        self.target = target
        self.target_one_hot = target_one_hot
        self.furniture_dx_id = furniture_dx_id
        self.furniture_dy_id = furniture_dy_id
        self.furniture_dx_value = furniture_dx_value
        self.furniture_dy_value = furniture_dy_value
        self.furniture_cids = furniture_cids

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
        # cnn 输出结果分布
        self.cnn_output_distribute = NotImplemented

        # rnn 输出结果分布
        self.rnn_output_distribute = NotImplemented

        # attention 输出结果分布
        self.attention_output_distribute = NotImplemented

        # regression 输出结果
        self.regression_output = NotImplemented

        # 学习率 衰减
        self.lr_rate = tf.maximum(0.001, tf.train.exponential_decay(0.01, self.global_step, 10240, 0.001))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)

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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def initializer(self, sess: tf.Session):
        """
        初始化全局变量
        :param sess:
        :return:
        """
        sess.run(tf.global_variables_initializer())

    def restore(self, sess: tf.Session, model_path: str):
        """
        sess 加载预训练的权重
        :param model_path 模型文件
        :return:
        """
        saver = tf.train.Saver()
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
        # 生成 placeholder
        # 生成 placeholder
        # 户型图
        room = tf.placeholder(shape=(None, self.image_height, self.image_width), name="infrastructure",
                                        dtype=tf.int32)
        # 家具图
        furniture = tf.placeholder(shape=(None, self.furniture_height, self.furniture_width), name="furniture",
                                   dtype=tf.int32)
        # 标签
        target = tf.placeholder(shape=(None,), name="target", dtype=tf.int32)
        target_one_hot = tf.one_hot(target, self.label_size, 1, 0)
        # 样本权重 用于计算loss todo

        # 家具长宽 (id)
        furniture_dx_id = tf.placeholder(shape=(None,), name="furniture_dx_id", dtype=tf.int32)
        furniture_dy_id = tf.placeholder(shape=(None,), name="furniture_dy_id", dtype=tf.int32)

        # 家具长宽 真实数据
        furniture_dx_value = tf.placeholder(shape=(None,), name="furniture_dx_value", dtype=tf.float32)
        furniture_dy_value = tf.placeholder(shape=(None,), name="furniture_dy_value", dtype=tf.float32)

        # 家具cid
        furniture_cids = tf.placeholder(shape=(None,), name="furniture_cids", dtype=tf.int32)

        # 评分值
        layout_marks = tf.placeholder(shape=(None, ), name="layout_marks", dtype=tf.float32)

        # 用于学习率的衰减
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.room = room
        self.furniture = furniture
        self.target = target
        self.target_one_hot = target_one_hot
        self.furniture_dx_id = furniture_dx_id
        self.furniture_dy_id = furniture_dy_id
        self.furniture_dx_value = furniture_dx_value
        self.furniture_dy_value = furniture_dy_value
        self.furniture_cids = furniture_cids
        self.layout_marks = layout_marks

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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def initializer(self, sess: tf.Session, var_list: list=None):
        """
        初始化全局变量
        :param sess:
        :return:
        """
        if var_list is None:
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.variables_initializer(var_list))

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
