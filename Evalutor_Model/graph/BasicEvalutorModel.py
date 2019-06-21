import tensorflow as tf
from Evalutor_Model.graph.CNNGlobalLayoutModel import GlobalVgg16Cnn
from Evalutor_Model.utils.graph_utils import tf_mse_loss, tf_mae_loss, tf_huber_loss, GetTokenEmbedding, tf_alpha_beta_weight, \
    FullConnect, Vgg16Layer, tf_alpha_beta, Basic2dConv


class BasicRegressionModel(GlobalVgg16Cnn):
    """
    回归模型
    特征:
        1 cnn布局特征
        2 特征工程部分
    """
    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int, furniture_width: int,
                 furniture_height: int, furniture_embedding_size: int, max_length: int=10):
        """
        :param image_width:  状态图宽
        :param image_height:  状态图高
        :param furniture_nums:  家具数目
        :param label_size:  单步推断位置 标签数目
        :param furniture_width:   家具图宽
        :param furniture_height:   家具图高
        :param furniture_embedding_size:   家具embedding_size
        :param max_length:  长度
        """
        # cnn 特征
        super(BasicRegressionModel, self).__init__(image_width=image_width,
                                                   image_height=image_height,
                                                   furniture_nums=furniture_nums,
                                                   label_size=label_size,
                                                   furniture_width=furniture_width,
                                                   furniture_height=furniture_height,
                                                   furniture_embedding_size=furniture_embedding_size,
                                                   max_length=max_length)
        self.room_a = tf.placeholder(name="room_a", shape=(None, image_height, image_width), dtype=tf.int32)
        self.room_b = tf.placeholder(name="room_b", shape=(None, image_height, image_width), dtype=tf.int32)

    def build_regression_graph(self, trainable: bool=True, keep_prob: int=None, filter_list: list=[64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm"):

        """
        回归模型
        :param trainable:
        :param keep_prob:
        :return:
        """
        self.build_regression_tensor_graph(layout_marks=self.layout_marks, trainable=trainable, keep_prob=keep_prob,
                                           filter_list=filter_list, normal_model=normal_model, room_a=self.room_a,
                                           room_b=self.room_b)

    def build_regression_tensor_graph(self, layout_marks: tf.Tensor, room_a: tf.Tensor, room_b: tf.Tensor,
                                      trainable: bool=True, keep_prob: int=None, filter_list: list=[64, 128, 256, 512, 512],
                                      normal_model: str="batch_norm"):
        # 卷积层特征
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        room_a_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_a)
        room_b_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_b)

        room_a_pool5, room_a_l2_loss = Vgg16Layer(input_tensor=room_a_embedding,
                                                      name="state",
                                                      trainable=trainable,
                                                      filter_list=filter_list,
                                                      normal_model=normal_model,
                                                      dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        room_b_pool5, room_b_l2_loss = Vgg16Layer(input_tensor=room_b_embedding,
                                                      name="state",
                                                      trainable=trainable,
                                                      filter_list=filter_list,
                                                      normal_model=normal_model,
                                                      dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # 先经过卷积层 降低通道
        room_a_feature, state_a_l2_loss = Basic2dConv(x=room_a_pool5, d_out=32, name="state_reduction",
                                                          active=tf.nn.relu, trainable=trainable)
        _, height, width, c = room_a_feature.get_shape()
        
        room_a_feature = tf.reshape(tensor=room_a_feature, shape=[-1, height * width * c])

        fc1_a, fc1_a_l2_loss = FullConnect(x=room_a_feature, out=1024 * 2, name="regression_fc1",
                                               active=tf.nn.relu, trainable=trainable, keep_prob=keep_prob,
                                               dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        output_a, output_a_l2_loss = FullConnect(x=fc1_a, out=1, name="regression_fc2", trainable=trainable,
                                                     dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        
        room_b_feature, state_b_l2_loss = Basic2dConv(x=room_b_pool5, d_out=32, name="state_reduction",
                                                            active=tf.nn.relu, trainable=trainable)

        room_b_feature = tf.reshape(tensor=room_b_feature, shape=[-1, height * width * c])
        
        fc1_b, fc1_b_l2_loss = FullConnect(x=room_b_feature, out=1024 * 2, name="regression_fc1",
                                                 active=tf.nn.relu, trainable=trainable, keep_prob=keep_prob,
                                                 dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        output_b, output_b_l2_loss = FullConnect(x=fc1_b, out=1, name="regression_fc2", trainable=trainable,
                                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # 差值
        diff = abs(output_a) - abs(output_b)
        # [b, 1] -> [b, ]
        l2_loss = self.cnn_l2_loss + state_a_l2_loss + fc1_a_l2_loss + output_a_l2_loss + state_b_l2_loss + \
                  fc1_b_l2_loss + output_b_l2_loss
        mse_loss = tf_mse_loss(logits=diff, targets=layout_marks)

        regression_loss = mse_loss + l2_loss * 1e-3
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(regression_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.regression_train_op = train_op
            self.regression_loss = regression_loss

        else:
            self.output_a = output_a
            self.output_b = output_b
        #  均方误差
        self.regression_loss = mse_loss
        self.output_a = output_a  # room a 值
        self.output_b = output_b  # room b 值
