"""
cnn 版本的布局
"""
from Layout_Service.graph.CnnLayoutModel import SingleBasicLayoutModel
from Layout_Service.utils.graph_utils import *


class GlobalVgg16Cnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32", id_list: list=["zid"]):

        super(GlobalVgg16Cnn, self).__init__(image_width=image_width,
                                             image_height=image_height,
                                             furniture_nums=furniture_nums,
                                             label_size=label_size,
                                             furniture_width=furniture_width,
                                             furniture_height=furniture_height,
                                             furniture_embedding_size=furniture_embedding_size,
                                             max_length=max_length,
                                             tf_dtype=tf_dtype)
        self.id_list = id_list

    def build_cnn_graph(self, trainable: bool,
                        mask: bool,
                        training: tf.Tensor,
                        keep_prob: int=None,
                        ):
        """

        :param trainable:
        :param mask:
        :param training:
        :param keep_prob:
        :param id_list:  模型使用的id
        :return:
        """
        id_list = self.id_list
        room_zid = self.room_zid if "zid" in id_list else None
        room_cid = self.room_cid if "cid" in id_list else None
        room_mid = self.room_mid if "mid" in id_list else None
        room_scid = self.room_scid if "scid" in id_list else None
        room_distance = self.room_distance if "distance" in id_list else None
        zone_zid = self.zone_zid if "zid" in id_list else None
        zone_cid = self.zone_cid if "cid" in id_list else None
        zone_mid = self.zone_mid if "mid" in id_list else None

        self.build_cnn_tensor_graph(labels=self.target,
                                    room_zid=room_zid,
                                    room_cid=room_cid,
                                    room_mid=room_mid,
                                    room_scid=room_scid,
                                    zone_zid=zone_zid,
                                    zone_cid=zone_cid,
                                    zone_mid=zone_mid,
                                    room_distance=room_distance,
                                    mask=mask,
                                    trainable=trainable,
                                    keep_prob=keep_prob,
                                    training=training)

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_zid: tf.Tensor,
                               room_cid: tf.Tensor,
                               room_mid: tf.Tensor,
                               room_scid: tf.Tensor,
                               room_distance: tf.Tensor,
                               zone_zid: tf.Tensor,
                               zone_cid: tf.Tensor,
                               zone_mid: tf.Tensor,
                               training: tf.Tensor,
                               mask: bool = False,
                               trainable: bool = True,
                               keep_prob: int = None, multi_gpu: bool=False):
        """

        :param labels:   输入label
        :param room_zid:
        :param room_cid:
        :param room_mid:
        :param room_scid:
        :param zone_zid:
        :param zone_cid:
        :param zone_mid:
        :param training:  是否训练 用于控制 batch_norm
        :param mask:
        :param trainable:
        :param keep_prob:
        :param multi_gpu:
        :return:
        """
        room_embedding_list = EmbeddingLayers(input_tensor_list=[room_zid, room_cid, room_mid, room_scid, room_distance],
                                              vocab_size_list=[200, self.furniture_nums, 200, 200, 200],
                                              num_unit_list=[self.furniture_embedding_size]*4,
                                              scopes=["zid_table", "cid_table", "mid_table", "scid_table", "distance_table"],
                                              trainable=trainable,
                                              dtype=tf.float32)

        zone_embedding_list = EmbeddingLayers(input_tensor_list=[zone_zid, zone_cid, zone_mid],
                                              vocab_size_list=[200, self.furniture_nums, 200],
                                              num_unit_list=[self.furniture_embedding_size]*4,
                                              scopes=["zid_table", "cid_table", "mid_table"],
                                              trainable=trainable,
                                              dtype=tf.float32)
        room_concat_embedding = tf.concat(values=room_embedding_list, axis=-1)

        zone_concat_embedding = tf.concat(values=zone_embedding_list, axis=-1)
        room_pool5 = Vgg16Layer(input_tensor=room_concat_embedding,
                                name="room_vgg16",
                                trainable=trainable,
                                filter_list=[64, 128, 256, 512, 512],
                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16,
                                training=training)

        zone_pool5 = Vgg16Layer(input_tensor=zone_concat_embedding,
                                name="zone_vgg16",
                                trainable=trainable,
                                filter_list=[64, 128, 256, 512, 512],
                                dtype=tf.float32 if self.tf_dtype == "32" else tf.float16,
                                training=training)

        concat_feature = tf.concat(values=[room_pool5, zone_pool5], axis=-1)
        # 使用卷积网络 表达注意力机制 使用两层卷积 减少全连接的使用
        compress_1 = Basic2dConv(concat_feature, 256,  'compress_1', active=None, trainable=trainable, dtype=tf.float32)
        compress_1 = BatchNormal(x=compress_1, name="compress_bn1", trainable=trainable, dtype=tf.float32, training=training)
        compress_1 = tf.nn.relu(compress_1)
        compress_2 = Basic2dConv(compress_1, 256, 'compress_2', active=None, trainable=trainable, dtype=tf.float32)
        compress_2 = BatchNormal(x=compress_2, name="compress_bn2", trainable=trainable, dtype=tf.float32, training=training)
        compress_2 = tf.nn.relu(compress_2)
        _, height, width, c = compress_2.get_shape()
        feature = tf.reshape(tensor=compress_2, shape=[-1, height * width * c], name="room_cnn_feature")
        fc3 = FullConnect(x=feature, out=self.label_size, name="cnn_fc", trainable=trainable, dtype=tf.float32)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        batch_target = labels
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        l2_loss = get_trainable_l2_loss() + get_variable_l2_loss(room_embedding_list)*1e-1 + get_variable_l2_loss(zone_embedding_list)*1e-1
        if mask:
            istarget = tf.to_float(tf.not_equal(batch_target, 0))
            cnn_loss = tf.reduce_mean(cnn_loss_ * istarget) + l2_loss*1e-3
        else:
            cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss*1e-3
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        if mask:
            correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
            accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        else:
            correct = tf.cast(tf.equal(batch_target, cnn_predict), "float")
            accuracy = tf.reduce_mean(correct, name="cnn_accuracy")
        if trainable and not multi_gpu:  # 多gpu部分 不再此处生成
            tvars = tf.trainable_variables()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
                train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy

        if multi_gpu:  # 仅对多个gpu训练的时候 才会返回结果
            return cnn_loss, accuracy


class InceptionResnetCnn(SingleBasicLayoutModel):

    def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
                 furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10,
                 tf_dtype: str = "32", id_list: list = ["zid"]):

        super(InceptionResnetCnn, self).__init__(image_width=image_width,
                                                 image_height=image_height,
                                                 furniture_nums=furniture_nums,
                                                 label_size=label_size,
                                                 furniture_width=furniture_width,
                                                 furniture_height=furniture_height,
                                                 furniture_embedding_size=furniture_embedding_size,
                                                 max_length=max_length,
                                                 tf_dtype=tf_dtype)
        self.id_list = id_list

    def build_cnn_tensor_graph(self, labels: tf.Tensor,
                               room_zid: tf.Tensor,
                               room_cid: tf.Tensor,
                               room_mid: tf.Tensor,
                               room_scid: tf.Tensor,
                               room_distance: tf.Tensor,
                               zone_zid: tf.Tensor,
                               zone_cid: tf.Tensor,
                               zone_mid: tf.Tensor,
                               training: tf.Tensor,
                               room_distribute: tf.Tensor,
                               mask: bool = False,
                               trainable: bool = True,
                               keep_prob: int = None, multi_gpu: bool=False):
        """

        :param labels:   输入label
        :param room_zid:
        :param room_cid:
        :param room_mid:
        :param room_scid:
        :param zone_zid:
        :param zone_cid:
        :param zone_mid:
        :param training:  是否训练 用于控制 batch_norm
        :param mask:
        :param trainable:
        :param keep_prob:
        :param multi_gpu:
        :return:
        """
        room_embedding_list = EmbeddingLayers(input_tensor_list=[room_zid, room_cid, room_mid, room_scid, room_distance],
                                              vocab_size_list=[200, self.furniture_nums, 200, 200, 200],
                                              num_unit_list=[self.furniture_embedding_size]*4,
                                              scopes=["zid_table", "cid_table", "mid_table", "scid_table", "distance_table"],
                                              trainable=trainable,
                                              dtype=tf.float32)

        zone_embedding_list = EmbeddingLayers(input_tensor_list=[zone_zid, zone_cid, zone_mid],
                                              vocab_size_list=[200, self.furniture_nums, 200],
                                              num_unit_list=[self.furniture_embedding_size]*4,
                                              scopes=["zid_table", "cid_table", "mid_table"],
                                              trainable=trainable,
                                              dtype=tf.float32)
        zone_concat_embedding = tf.concat(values=zone_embedding_list, axis=-1)

        zone_extract_feature = deep_conv(input=zone_concat_embedding, name="zone_extract_dcn", ksize_list=[3, 3],
                                         stride_list=[1, 1], filter_list=[32, 64],
                                         trainable=trainable, dtype=tf.float32, active=tf.nn.relu, batch_norm=True,
                                         training=training)

        zone_extract_wide_feature = wide_conv(input=zone_extract_feature, name="zone_extract_wide",
                                              ksize_lists=[[1], [3], [5]], stride_lists=[[1], [1], [1]],
                                              filter_lists=[[32], [64], [128]], trainable=trainable, dtype=tf.float32,
                                              active=tf.nn.relu, batch_norm=True,
                                              training=training)
        zone_extract_wide_feature = deep_conv(input=zone_extract_wide_feature, name="zone_reduce_dcn", ksize_list=[3],
                                              stride_list=[2], filter_list=[224], trainable=trainable, dtype=tf.float32,
                                              active=tf.nn.relu, batch_norm=True,
                                              training=training)
        zone_extract_resnet_feature = resnet_block(input=zone_extract_wide_feature, name="zone_extract_resnet",
                                                   ksize_list=[3, 3],
                                                   filter_list=[112, 224],
                                                   trainable=trainable, dtype=tf.float32, active=tf.nn.relu,
                                                   batch_norm=True, training=training)

        room_concat_embedding = tf.concat(values=room_embedding_list, axis=-1)
        room_concat_embedding = tf.concat(values=(room_distance, room_concat_embedding), axis=-1)
        room_extract_feature = deep_conv(input=room_concat_embedding, name="room_extract_dcn", ksize_list=[3, 3],
                                         stride_list=[1, 1],
                                         filter_list=[32, 64],
                                         trainable=trainable, dtype=tf.float32, active=tf.nn.relu, batch_norm=True,
                                         training=training)

        room_extract_wide_feature = wide_conv(input=room_extract_feature, name="room_extract_wide",
                                              ksize_lists=[[1], [3], [5]], stride_lists=[[1], [1], [1]],
                                              filter_lists=[[32], [64], [128]], trainable=trainable, dtype=tf.float32,
                                              active=tf.nn.relu, batch_norm=True,
                                              training=training)
        room_extract_wide_feature = deep_conv(input=room_extract_wide_feature, name="room_reduce_dcn", ksize_list=[3],
                                              stride_list=[2], filter_list=[224], trainable=trainable, dtype=tf.float32,
                                              active=tf.nn.relu, batch_norm=True,
                                              training=training)
        room_extract_resnet_feature = resnet_block(input=room_extract_wide_feature, name="room_extract_resnet",
                                                   ksize_list=[3, 3], filter_list=[112, 224],
                                                   trainable=trainable, dtype=tf.float32, active=tf.nn.relu,
                                                   batch_norm=True, training=training)

        feature = tf.concat(values=(zone_extract_resnet_feature, room_extract_resnet_feature), axis=-1)

        feature = wide_conv(input=feature, name="channel_reduce_1",
                            ksize_lists=[[1], [3], [5]], stride_lists=[[1], [1], [1]],
                            filter_lists=[[256], [512], [256]], trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True,
                            training=training)

        feature = deep_conv(input=feature, name="concat_cnn_1", ksize_list=[3], stride_list=[2],
                            filter_list=[self.label_size],
                            trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True, training=training)

        feature = resnet_block(input=feature, name="concat_resnet_1",
                               ksize_list=[3, 3], filter_list=[int(self.label_size / 2), self.label_size],
                               trainable=trainable, dtype=tf.float32, active=tf.nn.relu,
                               batch_norm=True, training=training)

        feature = wide_conv(input=feature, name="channel_reduce_2",
                            ksize_lists=[[1], [3], [5]], stride_lists=[[1], [1], [1]],
                            filter_lists=[[256], [512], [256]], trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True,
                            training=training)

        feature = deep_conv(input=feature, name="concat_cnn_2", ksize_list=[3], stride_list=[2],
                            filter_list=[self.label_size],
                            trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True, training=training)

        feature = resnet_block(input=feature, name="concat_resnet_2",
                               ksize_list=[3, 3], filter_list=[int(self.label_size / 2), self.label_size],
                               trainable=trainable, dtype=tf.float32, active=tf.nn.relu,
                               batch_norm=True, training=training)

        feature = wide_conv(input=feature, name="channel_reduce_3",
                            ksize_lists=[[1], [3], [5]], stride_lists=[[1], [1], [1]],
                            filter_lists=[[256], [512], [256]], trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True,
                            training=training)

        feature = deep_conv(input=feature, name="concat_cnn_3", ksize_list=[3], stride_list=[2],
                            filter_list=[self.label_size],
                            trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True, training=training)

        feature = resnet_block(input=feature, name="concat_resnet_3",
                               ksize_list=[3, 3], filter_list=[int(self.label_size / 2), self.label_size],
                               trainable=trainable, dtype=tf.float32, active=tf.nn.relu,
                               batch_norm=True, training=training)

        feature = wide_conv(input=feature, name="channel_reduce_4",
                            ksize_lists=[[1], [3], [5]], stride_lists=[[1], [1], [1]],
                            filter_lists=[[256], [512], [256]], trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True,
                            training=training)

        feature = deep_conv(input=feature, name="concat_cnn_4", ksize_list=[3], stride_list=[2],
                            filter_list=[self.label_size],
                            trainable=trainable, dtype=tf.float32,
                            active=tf.nn.relu, batch_norm=True, training=training)

        feature = resnet_block(input=feature, name="concat_resnet_4",
                               ksize_list=[3, 3], filter_list=[int(self.label_size / 2), self.label_size],
                               trainable=trainable, dtype=tf.float32, active=tf.nn.relu,
                               batch_norm=True, training=training)
        feature = tf.nn.avg_pool(feature, (1, 2, 2, 1), (1, 1, 1, 1), padding='VALID', name="a")
        _, height, width, c = feature.get_shape()
        feature = tf.reshape(tensor=feature, shape=[-1, height * width * c], name="full_feature")
        fc3 = FullConnect(x=feature, out=self.label_size, name="cnn_fc", trainable=trainable, dtype=tf.float32)
        # [b, ] -> [b, label_size]
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        # 损失函数 以及 acc 等
        # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
        # [b, ]
        batch_target = labels
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
        l2_loss = get_trainable_l2_loss() + get_variable_l2_loss(room_embedding_list) * 1e-1 + get_variable_l2_loss(
            zone_embedding_list) * 1e-1
        if mask:
            istarget = tf.to_float(tf.not_equal(batch_target, 0))
            cnn_loss = tf.reduce_mean(cnn_loss_ * istarget) + l2_loss * 1e-3
        else:
            cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3
        # [b, label_size]
        cnn_score = tf.nn.softmax(fc3, name="cnn_score")
        # [b, label_size] -> [b, label_size]
        self.cnn_output_distribute = cnn_score
        # [b, label_size] -> [b, ]
        cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        # (目标 且预测正确) / (目标数目)
        if mask:
            correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
            accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
        else:
            correct = tf.cast(tf.equal(batch_target, cnn_predict), "float")
            accuracy = tf.reduce_mean(correct, name="cnn_accuracy")
        if trainable and not multi_gpu:  # 多gpu部分 不再此处生成
            tvars = tf.trainable_variables()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), 10)
                train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.cnn_train_op = train_op

        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy

        if multi_gpu:  # 仅对多个gpu训练的时候 才会返回结果
            return cnn_loss, accuracy