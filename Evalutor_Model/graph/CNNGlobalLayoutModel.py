"""
cnn 版本的布局
"""
import tensorflow as tf
from Evalutor_Model.graph.CnnLayoutModel import SingleBasicLayoutModel
from Evalutor_Model.utils.graph_utils import GetTokenEmbedding, tf_alpha_beta_weight, FullConnect, Vgg16Layer, \
    tf_alpha_beta


class GlobalVgg16Cnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16Cnn, self).__init__(image_width=image_width,
                                             image_height=image_height,
                                             furniture_nums=furniture_nums,
                                             label_size=label_size,
                                             furniture_width=furniture_width,
                                             furniture_height=furniture_height,
                                             furniture_embedding_size=furniture_embedding_size,
                                             max_length=max_length,
                                             tf_dtype=tf_dtype)

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = (64, 128, 256, 512, 512),
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = (64, 128, 256, 512, 512),
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程展开后的特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图过程展开后的特征
        # [b, h//32, w//32, 512]
        self.middle_state_cnn_not_flatten = state_pool5
        self.furniture_cnn_not_flatten = furniture_pool5
        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = furniture_l2_loss + state_l2_loss


class GlobalVgg16CnnV2(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16CnnV2, self).__init__(image_width=image_width,
                                               image_height=image_height,
                                               furniture_nums=furniture_nums,
                                               label_size=label_size,
                                               furniture_width=furniture_width,
                                               furniture_height=furniture_height,
                                               furniture_embedding_size=furniture_embedding_size,
                                               max_length=max_length,
                                               tf_dtype=tf_dtype)

        room_cid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_cid", dtype=tf.int32)
        room_zid = tf.placeholder(shape=(None, self.image_height, self.image_width), name="room_zid", dtype=tf.int32)

        furniture_cid = tf.placeholder(shape=(None, self.furniture_height, self.furniture_width), name="furniture_cid",
                                       dtype=tf.int32)
        furniture_zid = tf.placeholder(shape=(None, self.furniture_height, self.furniture_width), name="furniture_zid",
                                       dtype=tf.int32)

        self.room_cid = room_cid
        self.room_zid = room_zid

        self.furniture_cid = furniture_cid
        self.furniture_zid = furniture_zid

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_cid_state=self.room_cid,
                                    room_zid_state=self.room_zid, furniture_cid_state=self.furniture_cid,
                                    trainable=trainable, furniture_zid_state=self.furniture_zid,
                                    filter_list=filter_list, normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_cid_state: tf.Tensor,
                               room_zid_state: tf.Tensor,
                               furniture_cid_state: tf.Tensor,
                               furniture_zid_state: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_cid_state:  房间cid
        :param room_zid_state:  房间zid
        :param furniture_cid_state: 家具cid
        :param furniture_zid_state:  家具zid
        :return:
        """
        lookup_cid_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                             num_units=self.furniture_embedding_size,
                                             zero_pad=True,
                                             scope="cid_embedding",
                                             trainable=trainable,
                                             dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_zid_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                             num_units=self.furniture_embedding_size,
                                             zero_pad=True,
                                             scope="zid_embedding",
                                             trainable=trainable,
                                             dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_cid_embedding = tf.nn.embedding_lookup(params=lookup_cid_table, ids=furniture_cid_state)
        furniture_zid_embedding = tf.nn.embedding_lookup(params=lookup_zid_table, ids=furniture_zid_state)
        furniture_embedding = tf.concat(values=(furniture_cid_embedding, furniture_zid_embedding), axis=-1)
        # [b, h, w] -> [b, h, w, e]
        room_cid_embedding = tf.nn.embedding_lookup(params=lookup_cid_table, ids=room_cid_state)
        room_zid_embedding = tf.nn.embedding_lookup(params=lookup_zid_table, ids=room_zid_state)
        room_embedding = tf.concat(values=(room_cid_embedding, room_zid_embedding), axis=-1)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程展开后的特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图过程展开后的特征
        # [b, h//32, w//32, 512]
        self.middle_state_cnn_not_flatten = state_pool5
        self.furniture_cnn_not_flatten = furniture_pool5
        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = furniture_l2_loss + state_l2_loss


class GlobalVgg16CnnMulti(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10):
        super(GlobalVgg16CnnMulti, self).__init__(image_width=image_width,
                                                  image_height=image_height,
                                                  furniture_nums=furniture_nums,
                                                  label_size=label_size,
                                                  furniture_width=furniture_width,
                                                  furniture_height=furniture_height,
                                                  furniture_embedding_size=furniture_embedding_size,
                                                  max_length=max_length)

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding, name="furniture",
                                                        trainable=trainable, filter_list=filter_list,
                                                        normal_model=normal_model)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding, name="state", trainable=trainable,
                                                filter_list=filter_list, normal_model=normal_model)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]  直接相乘 特征融合
        concat_feature = tf.multiply(x=room, y=furniture)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu, trainable=trainable,
                                       keep_prob=keep_prob)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数

        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            max_grad_norm = 10  # 梯度计算
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), max_grad_norm)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16CnnPlus(SingleBasicLayoutModel):
    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10):
        super(GlobalVgg16CnnPlus, self).__init__(image_width=image_width,
                                                 image_height=image_height,
                                                 furniture_nums=furniture_nums,
                                                 label_size=label_size,
                                                 furniture_width=furniture_width,
                                                 furniture_height=furniture_height,
                                                 furniture_embedding_size=furniture_embedding_size,
                                                 max_length=max_length)

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding, name="furniture",
                                                        trainable=trainable, filter_list=filter_list,
                                                        normal_model=normal_model)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding, name="state", trainable=trainable,
                                                filter_list=filter_list, normal_model=normal_model)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]  直接相乘 特征融合
        concat_feature = tf.add(x=room, y=furniture)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu, trainable=trainable,
                                       keep_prob=keep_prob)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数

        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            max_grad_norm = 10  # 梯度计算
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), max_grad_norm)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16CnnMultiPlus(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10):
        super(GlobalVgg16CnnMultiPlus, self).__init__(image_width=image_width,
                                                      image_height=image_height,
                                                      furniture_nums=furniture_nums,
                                                      label_size=label_size,
                                                      furniture_width=furniture_width,
                                                      furniture_height=furniture_height,
                                                      furniture_embedding_size=furniture_embedding_size,
                                                      max_length=max_length)

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope= "furniture_embedding",
                                         trainable=trainable)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding, name="furniture",
                                                        trainable=trainable, filter_list=filter_list,
                                                        normal_model=normal_model)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding, name="state", trainable=trainable,
                                                filter_list=filter_list, normal_model=normal_model)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]  直接相乘 特征融合
        add_feature = tf.add(x=room, y=furniture)
        multi_feature = tf.multiply(x=room, y=furniture)
        concat_feature = tf.concat(values=(add_feature, multi_feature, furniture, room), axis=-1)  # 乘法 加法 单独 等特征相连
        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu, trainable=trainable,
                                       keep_prob=keep_prob)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数

        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            max_grad_norm = 10  # 梯度计算
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), max_grad_norm)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16AlphaBetaCnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16AlphaBetaCnn, self).__init__(image_width=image_width,
                                                      image_height=image_height,
                                                      furniture_nums=furniture_nums,
                                                      label_size=label_size,
                                                      furniture_width=furniture_width,
                                                      furniture_height=furniture_height,
                                                      furniture_embedding_size=furniture_embedding_size,
                                                      max_length=max_length,
                                                      tf_dtype=tf_dtype)

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要

        linear = tf_alpha_beta(values=(room, furniture), name="linear", trainable=trainable)

        fc1, fc1_l2_loss = FullConnect(x=linear, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16AlphaBetaWeightCnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16AlphaBetaWeightCnn, self).__init__(image_width=image_width,
                                                            image_height=image_height,
                                                            furniture_nums=furniture_nums,
                                                            label_size=label_size,
                                                            furniture_width=furniture_width,
                                                            furniture_height=furniture_height,
                                                            furniture_embedding_size=furniture_embedding_size,
                                                            max_length=max_length,
                                                            tf_dtype=tf_dtype)

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        linear = tf_alpha_beta_weight(values=(room, furniture), name="linear", trainable=trainable)

        fc1, fc1_l2_loss = FullConnect(x=linear, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16LabelEncodeCnn(SingleBasicLayoutModel):
    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16LabelEncodeCnn, self).__init__(image_width=image_width,
                                                        image_height=image_height,
                                                        furniture_nums=furniture_nums,
                                                        label_size=label_size,
                                                        furniture_width=furniture_width,
                                                        furniture_height=furniture_height,
                                                        furniture_embedding_size=furniture_embedding_size,
                                                        max_length=max_length,
                                                        tf_dtype=tf_dtype)
        label_room = tf.placeholder(shape=(None, self.image_height, self.image_width), name="label_room",
                                    dtype=tf.int32)
        self.label_room = label_room

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room, label_room=self.label_room,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               label_room: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_label_table = GetTokenEmbedding(vocab_size=self.label_size,
                                               num_units=self.furniture_embedding_size,
                                               zero_pad=False,
                                               scope="label_embedding",
                                               trainable=trainable,
                                               dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        label_room_embedding = tf.nn.embedding_lookup(params=lookup_label_table, ids=label_room)
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)
        room_embedding = tf.concat(values=(label_room_embedding, room_embedding), axis=-1)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16AngleEncodeCnn(SingleBasicLayoutModel):
    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16AngleEncodeCnn, self).__init__(image_width=image_width,
                                                        image_height=image_height,
                                                        furniture_nums=furniture_nums,
                                                        label_size=label_size,
                                                        furniture_width=furniture_width,
                                                        furniture_height=furniture_height,
                                                        furniture_embedding_size=furniture_embedding_size,
                                                        max_length=max_length,
                                                        tf_dtype=tf_dtype)
        angle_state = tf.placeholder(shape=(None, self.image_height, self.image_width), name="angle_state",
                                     dtype=tf.int32)
        self.angle_state = angle_state

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room, angle_state=self.angle_state,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               angle_state: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_angle_table = GetTokenEmbedding(vocab_size=30,
                                               num_units=8,
                                               zero_pad=False,
                                               scope="angle_embedding",
                                               trainable=trainable,
                                               dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        angle_embedding = tf.nn.embedding_lookup(params=lookup_angle_table, ids=angle_state)
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)
        room_embedding = tf.concat(values=(angle_embedding, room_embedding), axis=-1)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16CenterAngleDistanceCnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16CenterAngleDistanceCnn, self).__init__(image_width=image_width,
                                                                image_height=image_height,
                                                                furniture_nums=furniture_nums,
                                                                label_size=label_size,
                                                                furniture_width=furniture_width,
                                                                furniture_height=furniture_height,
                                                                furniture_embedding_size=furniture_embedding_size,
                                                                max_length=max_length,
                                                                tf_dtype=tf_dtype)
        angle_state = tf.placeholder(shape=(None, self.image_height, self.image_width), name="angle_state",
                                     dtype=tf.int32)
        distance_state = tf.placeholder(shape=(None, self.image_height, self.image_width), name="distance_state",
                                        dtype=tf.int32)
        self.angle_state = angle_state
        self.distance_state = distance_state

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room, angle_state=self.angle_state,
                                    furniture_feature=self.furniture, distance_state=self.distance_state,
                                    trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               angle_state: tf.Tensor,
                               distance_state: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_angle_table = GetTokenEmbedding(vocab_size=361,
                                               num_units=self.furniture_embedding_size,
                                               zero_pad=False,
                                               scope="angle_embedding",
                                               trainable=trainable,
                                               dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_distance_table = GetTokenEmbedding(vocab_size=2048,
                                                  num_units=self.furniture_embedding_size,
                                                  zero_pad=False,
                                                  scope="distance_embedding",
                                                  trainable=trainable,
                                                  dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        angle_embedding = tf.nn.embedding_lookup(params=lookup_angle_table, ids=angle_state)
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)
        # [b, h, w] -> [b, h, w, e]
        distance_embedding = tf.nn.embedding_lookup(params=lookup_distance_table, ids=distance_state)
        # 房间特征
        room_embedding = tf.concat(values=(angle_embedding, room_embedding, distance_embedding), axis=-1)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16CenterDistanceCnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16CenterDistanceCnn, self).__init__(image_width=image_width,
                                                           image_height=image_height,
                                                           furniture_nums=furniture_nums,
                                                           label_size=label_size,
                                                           furniture_width=furniture_width,
                                                           furniture_height=furniture_height,
                                                           furniture_embedding_size=furniture_embedding_size,
                                                           max_length=max_length,
                                                           tf_dtype=tf_dtype)
        distance_state = tf.placeholder(shape=(None, self.image_height, self.image_width), name="distance_state",
                                        dtype=tf.int32)
        self.distance_state = distance_state

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target,
                                    room_state=self.room,
                                    furniture_feature=self.furniture,
                                    distance_state=self.distance_state,
                                    trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512],
                                    normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               distance_state: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_distance_table = GetTokenEmbedding(vocab_size=2048,
                                                  num_units=8,
                                                  zero_pad=False,
                                                  scope="distance_embedding",
                                                  trainable=trainable,
                                                  dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)
        # [b, h, w] -> [b, h, w, e]
        distance_embedding = tf.nn.embedding_lookup(params=lookup_distance_table, ids=distance_state)
        # 房间特征
        room_embedding = tf.concat(values=(room_embedding, distance_embedding), axis=-1)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16WindowFurnitureCnn(SingleBasicLayoutModel):
    """
    输入  3 个家具图  用于预测第一个
    """

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32", window_length: int = 3):
        super(GlobalVgg16WindowFurnitureCnn, self).__init__(image_width=image_width,
                                                            image_height=image_height,
                                                            furniture_nums=furniture_nums,
                                                            label_size=label_size,
                                                            furniture_width=furniture_width,
                                                            furniture_height=furniture_height,
                                                            furniture_embedding_size=furniture_embedding_size,
                                                            max_length=max_length,
                                                            tf_dtype=tf_dtype)
        window_furniture = tf.placeholder(shape=(None, window_length, self.image_height, self.image_width),
                                          name="distance_state", dtype=tf.int32)
        self.window_furniture = window_furniture
        self.window_length = window_length

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target,
                                    room_state=self.room,
                                    window_furniture_feature=self.window_furniture,
                                    trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512],
                                    normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               window_furniture_feature: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, window, h, w] -> [b, window, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=window_furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)
        # [b, window, h, w, e] -> [b*window, h, w, e]
        furniture_embedding = tf.reshape(tensor=furniture_embedding,
                                         shape=(-1, self.image_height, self.image_width, self.furniture_embedding_size))
        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b*window, h//32, w//32, 128] -> [b*window, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")
        # [b*window, h//32*w//32*128] -> [b, window*h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture, shape=[-1, self.window_length * height * width * c])
        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16AngleEncode2Cnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32"):
        super(GlobalVgg16AngleEncode2Cnn, self).__init__(image_width=image_width,
                                                         image_height=image_height,
                                                         furniture_nums=furniture_nums,
                                                         label_size=label_size,
                                                         furniture_width=furniture_width,
                                                         furniture_height=furniture_height,
                                                         furniture_embedding_size=furniture_embedding_size,
                                                         max_length=max_length,
                                                         tf_dtype=tf_dtype)
        angle_state = tf.placeholder(shape=(None, self.image_height, self.image_width), name="angle_state",
                                     dtype=tf.int32)
        self.angle_state = angle_state

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room, angle_state=self.angle_state,
                                    furniture_feature=self.furniture, trainable=trainable,
                                    filter_list=[64, 128, 256, 512, 512], normal_model=normal_model,
                                    keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               angle_state: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        lookup_angle_table = GetTokenEmbedding(vocab_size=361,
                                               num_units=2,
                                               zero_pad=False,
                                               scope="angle_embedding",
                                               trainable=trainable,
                                               dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        angle_embedding = tf.nn.embedding_lookup(params=lookup_angle_table, ids=angle_state)
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        angle_pool5, angle_l2_loss = Vgg16Layer(input_tensor=angle_embedding,
                                                name="angle",
                                                trainable=trainable,
                                                filter_list=[8, 8, 8, 8, 8],
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16, )

        _, height, width, c = furniture_pool5.get_shape()
        _, height2, width2, c2 = angle_pool5.get_shape()

        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        angle = tf.reshape(tensor=angle_pool5, shape=[-1, height2 * width2 * c2],
                           name="angle_cnn_feature")
        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, angle, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss + angle_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss


class GlobalVgg16DistributionCnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32", zero_num: int = 7):
        super(GlobalVgg16DistributionCnn, self).__init__(image_width=image_width,
                                                         image_height=image_height,
                                                         furniture_nums=furniture_nums,
                                                         label_size=label_size,
                                                         furniture_width=furniture_width,
                                                         furniture_height=furniture_height,
                                                         furniture_embedding_size=furniture_embedding_size,
                                                         max_length=max_length,
                                                         tf_dtype=tf_dtype)

        grid_distribution = tf.placeholder(shape=(None, self.image_height, self.image_width, zero_num),
                                           dtype=tf.float32)
        self.zero_num = zero_num
        self.grid_distribution = grid_distribution

    def build_cnn_graph(self, trainable: bool = True,
                        filter_list: list = [64, 128, 256, 512, 512],
                        normal_model: str = "batch_norm",
                        keep_prob: int = None):
        self.build_cnn_tensor_graph(labels=self.target, room_state=self.room,
                                    furniture_feature=self.furniture, grid_distribution=self.grid_distribution,
                                    trainable=trainable, filter_list=[64, 128, 256, 512, 512],
                                    normal_model=normal_model, keep_prob=keep_prob)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture_feature: tf.Tensor,
                               grid_distribution: tf.Tensor,
                               trainable: bool = True,
                               filter_list: list = [64, 128, 256, 512, 512],
                               normal_model: str = "batch_norm",
                               keep_prob: int = None):
        """
        创建 cnn 布局结果
        :param trainable:
        :param labels:
        :param room_state:
        :param furniture_feature: 家具图
        :return:
        """
        lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
                                         num_units=self.furniture_embedding_size,
                                         zero_pad=True,
                                         scope="furniture_embedding",
                                         trainable=trainable,
                                         dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # b: batch_size, h: height, w: width, e: embedding_size
        # [b, h, w] -> [b, h, w, e]
        furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
        # [b, h, w] -> [b, h, w, e]
        room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_state)

        furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding,
                                                        name="furniture",
                                                        trainable=trainable,
                                                        filter_list=filter_list,
                                                        normal_model=normal_model,
                                                        dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # 合并概率分布 [b, h, w, e+zero_num]
        room_embedding = tf.concat(values=[room_embedding, grid_distribution], axis=-1)

        state_pool5, state_l2_loss = Vgg16Layer(input_tensor=room_embedding,
                                                name="state",
                                                trainable=trainable,
                                                filter_list=filter_list,
                                                normal_model=normal_model,
                                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        _, height, width, c = furniture_pool5.get_shape()
        # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        room = tf.reshape(tensor=state_pool5, shape=[-1, height * width * c],
                          name="room_cnn_feature")
        # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
        furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c],
                               name="furniture_cnn_feature")

        room_cnn_feature = room
        furniture_cnn_feature = furniture

        # 共享特征层
        self.middle_state_cnn_feature = room_cnn_feature  # 中间过程特征
        self.furniture_cnn_feature = furniture_cnn_feature  # 家具图特征

        # 卷积层特征 部分数据 不需要
        # [b, (h//32)*(w//32)*72*2]
        concat_feature = tf.concat(values=[room, furniture], axis=-1)

        fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024 * 2, name="cnn_fc1", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)

        fc2, fc2_l2_loss = FullConnect(x=fc1, out=512 * 2, name="cnn_fc2", active=tf.nn.relu,
                                       trainable=trainable, keep_prob=keep_prob,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, label_size]
        fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable,
                                       dtype=tf.float32 if self.tf_dtype == "32" else tf.float16)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        l2_loss = furniture_l2_loss + state_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
        # [b, label_size] -> [b, label_size]
        batch_target = labels
        istarget = tf.to_float(tf.not_equal(batch_target, 0))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
        accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        if trainable:
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
            train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        self.cnn_l2_loss = l2_loss
