"""
cnn 布局模块
"""
import tensorflow as tf
from Layout_Service.graph.BasicLayoutModel import SingleBasicLayoutModel
# from utils.graph_utils import GetTokenEmbedding, Basic2dConv, MultiFilter2dConv, MaxPool, FullConnect, BatchNormal, DeformableSquareCnn, Basic2dDilatedConv, Vgg16Layer
#
#
# class CnnLayoutModel(BasicLayoutModel):
#     def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
#                  furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int=10):
#         """
#         :param image_width:
#         :param image_height:
#         :param furniture_nums:
#         :param label_size:
#         :param furniture_width:
#         :param furniture_height:
#         :param furniture_embedding_size:
#         :param max_length:
#         """
#         super(CnnLayoutModel, self).__init__(image_width=image_width,
#                                              image_height=image_height,
#                                              furniture_nums=furniture_nums,
#                                              label_size=label_size,
#                                              furniture_width=furniture_width,
#                                              furniture_height=furniture_height,
#                                              furniture_embedding_size=furniture_embedding_size,
#                                              max_length=max_length)
#
#     def build_cnn_graph(self, trainable: bool=True):
#         """
#         创建图
#         :param trainable:
#         :return:
#         """
#         lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
#                                              num_units=self.furniture_embedding_size,
#                                              zero_pad=True,
#                                              scope="furniture_embedding",
#                                              trainable=trainable)
#         # b: batch_size, h: height, w: width, e: embedding_size
#         # [b, l, h, w] -> [b, l, h, w, e]
#         furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.furniture)
#         # [b, l, h, w, e] -> [b*l, h, w, e]
#         furniture_embedding_reshape = tf.reshape(tensor=furniture_embedding, shape=(-1, self.image_height, self.image_width, self.furniture_embedding_size))
#         # [b, l, h, w] -> [b, l, h, w, e]
#         room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.room)
#         # [b, l, h, w, e] -> [b*l, h, w, e]
#         room_embedding_reshape = tf.reshape(tensor=room_embedding, shape=(-1, self.image_height, self.image_width, self.furniture_embedding_size))
#
#         # 普通 cnn 卷积 (128, 128) -> (64, 64)
#         room_cnn1, in_cnn1_l2_loss = Basic2dConv(x=room_embedding_reshape, d_out=72, ksize=(3, 3), name="room_cnn1", active=tf.nn.relu, trainable=trainable)
#         room_cnn1 = BatchNormal(x=room_cnn1, name="room_cnn1_bn", trainable=trainable)
#         room_pool1 = MaxPool(x=room_cnn1, name="room_pool1")
#
#         furniture_cnn1, f_cnn1_l2_loss = Basic2dConv(x=furniture_embedding_reshape, d_out=72, ksize=(3, 3), name="furniture_cnn1", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn1 = BatchNormal(x=furniture_cnn1, name="furniture_cnn1_bn", trainable=trainable)
#         furniture_pool1 = MaxPool(x=furniture_cnn1, name="furniture_pool1")
#         # 普通 cnn 卷积 (64, 64) -> (32, 32)
#         room_cnn2, in_cnn2_l2_loss = Basic2dConv(x=room_pool1, d_out=72, ksize=(3, 3), name="room_cnn2", active=tf.nn.relu, trainable=trainable)
#         room_cnn2 = BatchNormal(x=room_cnn2, name="room_cnn2_bn", trainable=trainable)
#         room_pool2 = MaxPool(x=room_cnn2, name="room_pool2")
#
#         furniture_cnn2, f_cnn2_l2_loss = Basic2dConv(x=furniture_pool1, d_out=72, ksize=(3, 3), name="furniture_cnn2", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn2 = BatchNormal(x=furniture_cnn2, name="furniture_cnn2_bn", trainable=trainable)
#         furniture_pool2 = MaxPool(x=furniture_cnn2, name="furniture_pool2")
#         # 普通 cnn 卷积  (32, 32) -> (16, 16)
#         room_cnn3, in_cnn3_l2_loss = Basic2dConv(x=room_pool2, d_out=72, ksize=(3, 3), name="room_cnn3", active=tf.nn.relu, trainable=trainable)
#         room_cnn3 = BatchNormal(x=room_cnn3, name="room_cnn3_bn", trainable=trainable)
#         room_pool3 = MaxPool(x=room_cnn3, name="room_pool3")
#
#         furniture_cnn3, f_cnn3_l2_loss = Basic2dConv(x=furniture_pool2, d_out=72, ksize=(3, 3), name="furniture_cnn3", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn3 = BatchNormal(x=furniture_cnn3, name="furniture_cnn3_bn", trainable=trainable)
#         furniture_pool3 = MaxPool(x=furniture_cnn3, name="furniture_pool3")
#
#         # 多种核尺寸 cnn 卷积  (32, 32) -> (16, 16)
#         d_outs = [24, 24, 24]
#         ksizes = [[3, 3], [5, 5], [7, 7]]
#         strides = [[1, 1], [1, 1], [1, 1]]
#
#         room_cnn4, in_cnn4_l2_loss = MultiFilter2dConv(x=room_pool3, d_outs=d_outs, ksizes=ksizes, strides=strides, active=tf.nn.relu, name="room_cnn4", trainable=trainable)
#         room_cnn4 = tf.concat(values=room_cnn4, axis=-1)
#         room_pool4 = MaxPool(x=room_cnn4, name="room_pool4")
#
#         furniture_cnn4, f_cnn4_l2_loss = MultiFilter2dConv(x=furniture_pool3, d_outs=d_outs, ksizes=ksizes, strides=strides, active=tf.nn.relu, name="furniture_cnn4", trainable=trainable)
#         furniture_cnn4 = tf.concat(values=furniture_cnn4, axis=-1)
#         furniture_pool4 = MaxPool(x=furniture_cnn4, name="furniture_pool4")
#
#         # 多种核尺寸 cnn 卷积  (16, 16) -> (8, 8)
#         d_outs = [24, 24, 24]
#         ksizes = [[3, 3], [5, 5], [7, 7]]
#         strides = [[1, 1], [1, 1], [1, 1]]
#
#         room_cnn5, in_cnn5_l2_loss = MultiFilter2dConv(x=room_pool4, d_outs=d_outs, ksizes=ksizes, strides=strides, active=tf.nn.relu, name="room_cnn5", trainable=trainable)
#         room_cnn5 = tf.concat(values=room_cnn5, axis=-1)
#         # [b*l, h//32, w//32, 72]
#         room_pool5 = MaxPool(x=room_cnn5, name="room_pool5")
#
#         furniture_cnn5, f_cnn5_l2_loss = MultiFilter2dConv(x=furniture_pool4, d_outs=d_outs, ksizes=ksizes, strides=strides, active=tf.nn.relu, name="furniture_cnn5", trainable=trainable)
#         furniture_cnn5 = tf.concat(values=furniture_cnn5, axis=-1)
#         furniture_pool5 = MaxPool(x=furniture_cnn5, name="furniture_pool5")
#
#         height, weight = int(self.image_height/32), int(self.image_width/32)
#         # 卷积层特征 [b*l, h//32, w//32, 72] -> [b, l, h//32*w//32*72]
#         room_cnn_feature = tf.reshape(tensor=room_pool5, shape=[-1, self.max_length, height*weight*72], name="room_cnn_feature")
#         # [b, l, h//32, w//32, 72] -> [b*l, (h//32)*(w//32)*72]
#         room = tf.reshape(tensor=room_pool5, shape=[-1, height*weight*72])
#         # [b*l, h//32, w//32, 72] -> [b, l, h//32*w//32*72]
#         furniture_cnn_feature = tf.reshape(tensor=furniture_pool5, shape=[-1, self.max_length, height * weight * 72], name="furniture_cnn_feature")
#         # 共享特征层
#         self.room_cnn_feature = room_cnn_feature
#         self.furniture_cnn_feature = furniture_cnn_feature
#
#         # 卷积层特征
#
#         # [b, l, h//32, w//32, 72] -> [b*l, (h//32)*(w//32)*72]
#         furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height*weight*72])
#         # [b*l, (h//32)*(w//32)*72*2]
#         concat_feature = tf.concat(values=[room, furniture], axis=-1)
#
#         fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024, name="cnn_fc1", active=tf.nn.relu, trainable=trainable)
#
#         fc2, fc2_l2_loss = FullConnect(x=fc1, out=512, name="cnn_fc2", active=tf.nn.relu, trainable=trainable)
#         # [b*l, label_size]
#         fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
#
#         # [b, l, label_size] -> [b*l, label_size]
#         labels = tf.reshape(tensor=self.target_one_hot, shape=[-1, self.label_size])
#         # 损失函数 以及 acc 等
#         cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3)
#         # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
#         l2_loss = in_cnn1_l2_loss + f_cnn1_l2_loss + in_cnn2_l2_loss + f_cnn2_l2_loss + in_cnn3_l2_loss + f_cnn3_l2_loss + \
#                   in_cnn4_l2_loss + f_cnn4_l2_loss + in_cnn5_l2_loss + f_cnn5_l2_loss
#         # [b, l] -> [b*l, ]
#         batch_target = tf.reshape(tensor=self.target, shape=(-1,))
#         istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
#         cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss*1e-3  # l2 损失函数
#         # [b*l, label_size]
#         cnn_score = tf.nn.softmax(fc3, name="cnn_score")
#         # [b*l, label_size] -> [b, l, label_size]
#         self.cnn_output_distribute = tf.reshape(tensor=cnn_score, shape=[-1, self.max_length, self.label_size])
#         # [b*l, label_size] -> [b*l, ]
#         cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
#         # (目标 且预测正确) / (目标数目)
#         correct = tf.cast(tf.equal(batch_target, cnn_predict), "float")*istarget/(tf.reduce_sum(istarget))
#         accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
#         if trainable:
#             grads_and_vars = self.optimizer.compute_gradients(cnn_loss)
#             train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
#             self.cnn_train_op = train_op
#
#         self.cnn_loss = cnn_loss
#         self.cnn_acc = accuracy
#
#
# class SingleCnnLayoutModel(SingleBasicLayoutModel):
#
#     def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
#                  furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int=10):
#         """
#         :param image_width:
#         :param image_height:
#         :param furniture_nums:
#         :param label_size:
#         :param furniture_width:
#         :param furniture_height:
#         :param furniture_embedding_size:
#         :param max_length:
#         """
#         super(SingleCnnLayoutModel, self).__init__(image_width=image_width,
#                                              image_height=image_height,
#                                              furniture_nums=furniture_nums,
#                                              label_size=label_size,
#                                              furniture_width=furniture_width,
#                                              furniture_height=furniture_height,
#                                              furniture_embedding_size=furniture_embedding_size,
#                                              max_length=max_length)
#
#     def build_cnn_graph(self, trainable: bool=True):
#         """
#         创建图
#         :param trainable:
#         :return:
#         """
#         self.build_cnn_tensor_graph(room_feature=self.room,
#                                     furniture_feature=self.furniture,
#                                     labels=self.target_one_hot,
#                                     trainable=trainable)
#
#     def build_cnn_tensor_graph(self, room_feature: tf.Tensor, furniture_feature: tf.Tensor, labels: tf.Tensor, trainable: bool=True):
#         """
#         tensorflow data api 的训练方式
#         :param room_feature:  [b, h, w]
#         :param furniture_feature:  [b, h, w]
#         :param labels:  [b]
#         :param trainable:
#         :return:
#         """
#         lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
#                                              num_units=self.furniture_embedding_size,
#                                              zero_pad=True,
#                                              scope="furniture_embedding",
#                                              trainable=trainable)
#         # b: batch_size, h: height, w: width, e: embedding_size
#         # [b, h, w] -> [b, h, w, e]
#         furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
#         # [b, h, w] -> [b, h, w, e]
#         room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_feature)
#
#         # 普通 cnn 卷积 (128, 128) -> (64, 64)
#         room_cnn1, in_cnn1_l2_loss = Basic2dConv(x=room_embedding, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn1", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn1 = BatchNormal(x=room_cnn1, name="room_cnn1_bn", trainable=trainable)
#         room_pool1 = MaxPool(x=room_cnn1, name="room_pool1")
#
#         furniture_cnn1, f_cnn1_l2_loss = Basic2dConv(x=furniture_embedding, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn1", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn1 = BatchNormal(x=furniture_cnn1, name="furniture_cnn1_bn", trainable=trainable)
#         furniture_pool1 = MaxPool(x=furniture_cnn1, name="furniture_pool1")
#         # 普通 cnn 卷积 (64, 64) -> (32, 32)
#         room_cnn2, in_cnn2_l2_loss = Basic2dConv(x=room_pool1, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn2", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn2 = BatchNormal(x=room_cnn2, name="room_cnn2_bn", trainable=trainable)
#         room_pool2 = MaxPool(x=room_cnn2, name="room_pool2")
#
#         furniture_cnn2, f_cnn2_l2_loss = Basic2dConv(x=furniture_pool1, d_out=72, ksize=(3, 3), name="furniture_cnn2",
#                                                      active=tf.nn.relu, trainable=trainable)
#         furniture_cnn2 = BatchNormal(x=furniture_cnn2, name="furniture_cnn2_bn", trainable=trainable)
#         furniture_pool2 = MaxPool(x=furniture_cnn2, name="furniture_pool2")
#         # 普通 cnn 卷积  (32, 32) -> (16, 16)
#         room_cnn3, in_cnn3_l2_loss = Basic2dConv(x=room_pool2, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn3", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn3 = BatchNormal(x=room_cnn3, name="room_cnn3_bn", trainable=trainable)
#         room_pool3 = MaxPool(x=room_cnn3, name="room_pool3")
#
#         furniture_cnn3, f_cnn3_l2_loss = Basic2dConv(x=furniture_pool2, d_out=72, ksize=(3, 3), name="furniture_cnn3",
#                                                      active=tf.nn.relu, trainable=trainable)
#         furniture_cnn3 = BatchNormal(x=furniture_cnn3, name="furniture_cnn3_bn", trainable=trainable)
#         furniture_pool3 = MaxPool(x=furniture_cnn3, name="furniture_pool3")
#
#         # 多种核尺寸 cnn 卷积  (32, 32) -> (16, 16)
#         d_outs = [24, 24, 24]
#         ksizes = [[3, 3], [5, 5], [7, 7]]
#         strides = [[1, 1], [1, 1], [1, 1]]
#
#         room_cnn4, in_cnn4_l2_loss = MultiFilter2dConv(x=room_pool3, d_outs=d_outs, ksizes=ksizes,
#                                                                  strides=strides, active=tf.nn.relu,
#                                                                  name="room_cnn4", trainable=trainable)
#         room_cnn4 = tf.concat(values=room_cnn4, axis=-1)
#         room_pool4 = MaxPool(x=room_cnn4, name="room_pool4")
#
#         furniture_cnn4, f_cnn4_l2_loss = MultiFilter2dConv(x=furniture_pool3, d_outs=d_outs, ksizes=ksizes,
#                                                            strides=strides, active=tf.nn.relu, name="furniture_cnn4",
#                                                            trainable=trainable)
#         furniture_cnn4 = tf.concat(values=furniture_cnn4, axis=-1)
#         furniture_pool4 = MaxPool(x=furniture_cnn4, name="furniture_pool4")
#
#         # 多种核尺寸 cnn 卷积  (16, 16) -> (8, 8)
#         d_outs = [24, 24, 24]
#         ksizes = [[3, 3], [5, 5], [7, 7]]
#         strides = [[1, 1], [1, 1], [1, 1]]
#
#         room_cnn5, in_cnn5_l2_loss = MultiFilter2dConv(x=room_pool4, d_outs=d_outs, ksizes=ksizes,
#                                                                  strides=strides, active=tf.nn.relu,
#                                                                  name="room_cnn5", trainable=trainable)
#         room_cnn5 = tf.concat(values=room_cnn5, axis=-1)
#         # [b, h//32, w//32, 72]
#         room_pool5 = MaxPool(x=room_cnn5, name="room_pool5")
#
#         furniture_cnn5, f_cnn5_l2_loss = MultiFilter2dConv(x=furniture_pool4, d_outs=d_outs, ksizes=ksizes,
#                                                            strides=strides, active=tf.nn.relu, name="furniture_cnn5",
#                                                            trainable=trainable)
#         furniture_cnn5 = tf.concat(values=furniture_cnn5, axis=-1)
#         furniture_pool5 = MaxPool(x=furniture_cnn5, name="furniture_pool5")
#
#         height, weight = int(self.image_height / 32), int(self.image_width / 32)
#         # 卷积层特征 [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         room = tf.reshape(tensor=room_pool5, shape=[-1, height * weight * 72],
#                                     name="room_cnn_feature")
#
#         # [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * weight * 72], name="furniture_cnn_feature")
#
#         room_cnn_feature = room
#         furniture_cnn_feature = furniture
#
#         # 共享特征层
#         self.room_cnn_feature = room_cnn_feature
#         self.furniture_cnn_feature = furniture_cnn_feature
#
#         # 卷积层特征
#
#         # [b, (h//32)*(w//32)*72*2]
#         concat_feature = tf.concat(values=[room, furniture], axis=-1)
#
#         fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024, name="cnn_fc1", active=tf.nn.relu,
#                                        trainable=trainable)
#
#         fc2, fc2_l2_loss = FullConnect(x=fc1, out=512, name="cnn_fc2", active=tf.nn.relu, trainable=trainable)
#         # [b, label_size]
#         fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
#
#         # [b, ] -> [b, label_size]
#         target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
#         # 损失函数 以及 acc 等
#         cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
#         # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
#         l2_loss = in_cnn1_l2_loss + f_cnn1_l2_loss + in_cnn2_l2_loss + f_cnn2_l2_loss + in_cnn3_l2_loss + f_cnn3_l2_loss + \
#                   in_cnn4_l2_loss + f_cnn4_l2_loss + in_cnn5_l2_loss + f_cnn5_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
#         # [b, ]
#         batch_target = labels
#         istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
#         cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
#         # [b, label_size]
#         cnn_score = tf.nn.softmax(fc3, name="cnn_score")
#         # [b, label_size] -> [b, label_size]
#         self.cnn_output_distribute = cnn_score
#         # [b, label_size] -> [b, ]
#         cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
#         # (目标 且预测正确) / (目标数目)
#         correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
#         accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
#         if trainable:
#             grads_and_vars = self.optimizer.compute_gradients(cnn_loss)
#             train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
#             self.cnn_train_op = train_op
#         self.cnn_l2_loss = l2_loss
#         self.cnn_loss = cnn_loss
#         self.cnn_acc = accuracy
#
#
# class SingleBasicCnn(SingleBasicLayoutModel):
#     """
#     普通卷积 (3*3)
#     """
#
#     def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
#                  furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10):
#         """
#         :param image_width:
#         :param image_height:
#         :param furniture_nums:
#         :param label_size:
#         :param furniture_width:
#         :param furniture_height:
#         :param furniture_embedding_size:
#         :param max_length:
#         """
#         super(SingleBasicCnn, self).__init__(image_width=image_width,
#                                                    image_height=image_height,
#                                                    furniture_nums=furniture_nums,
#                                                    label_size=label_size,
#                                                    furniture_width=furniture_width,
#                                                    furniture_height=furniture_height,
#                                                    furniture_embedding_size=furniture_embedding_size,
#                                                    max_length=max_length)
#
#     def build_cnn_graph(self, trainable: bool = True):
#         """
#         创建图
#         :param trainable:
#         :return:
#         """
#         lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
#                                              num_units=self.furniture_embedding_size,
#                                              zero_pad=True,
#                                              scope="furniture_embedding",
#                                              trainable=trainable)
#         # b: batch_size, h: height, w: width, e: embedding_size
#         # [b, h, w] -> [b, h, w, e]
#         furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.furniture)
#         # [b, h, w] -> [b, h, w, e]
#         room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.room)
#
#         # 普通 cnn 卷积 (128, 128) -> (64, 64)
#         room_cnn1, in_cnn1_l2_loss = Basic2dConv(x=room_embedding, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn1", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn1 = BatchNormal(x=room_cnn1, name="room_cnn1_bn", trainable=trainable)
#         room_pool1 = MaxPool(x=room_cnn1, name="room_pool1")
#
#         furniture_cnn1, f_cnn1_l2_loss = Basic2dConv(x=furniture_embedding, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn1", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn1 = BatchNormal(x=furniture_cnn1, name="furniture_cnn1_bn", trainable=trainable)
#         furniture_pool1 = MaxPool(x=furniture_cnn1, name="furniture_pool1")
#         # 普通 cnn 卷积 (64, 64) -> (32, 32)
#         room_cnn2, in_cnn2_l2_loss = Basic2dConv(x=room_pool1, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn2", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn2 = BatchNormal(x=room_cnn2, name="room_cnn2_bn", trainable=trainable)
#         room_pool2 = MaxPool(x=room_cnn2, name="room_pool2")
#
#         furniture_cnn2, f_cnn2_l2_loss = Basic2dConv(x=furniture_pool1, d_out=72, ksize=(3, 3), name="furniture_cnn2",
#                                                      active=tf.nn.relu, trainable=trainable)
#         furniture_cnn2 = BatchNormal(x=furniture_cnn2, name="furniture_cnn2_bn", trainable=trainable)
#         furniture_pool2 = MaxPool(x=furniture_cnn2, name="furniture_pool2")
#         # 普通 cnn 卷积  (32, 32) -> (16, 16)
#         room_cnn3, in_cnn3_l2_loss = Basic2dConv(x=room_pool2, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn3", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn3 = BatchNormal(x=room_cnn3, name="room_cnn3_bn", trainable=trainable)
#         room_pool3 = MaxPool(x=room_cnn3, name="room_pool3")
#
#         furniture_cnn3, f_cnn3_l2_loss = Basic2dConv(x=furniture_pool2, d_out=72, ksize=(3, 3), name="furniture_cnn3",
#                                                      active=tf.nn.relu, trainable=trainable)
#         furniture_cnn3 = BatchNormal(x=furniture_cnn3, name="furniture_cnn3_bn", trainable=trainable)
#         furniture_pool3 = MaxPool(x=furniture_cnn3, name="furniture_pool3")
#
#         # 普通 cnn 卷积 (16, 16) -> (8, 8)
#         room_cnn4, in_cnn4_l2_loss = Basic2dConv(x=room_pool3, d_out=72, ksize=(3, 3),
#                                                            active=tf.nn.relu, name="room_cnn4", trainable=trainable)
#         room_pool4 = MaxPool(x=room_cnn4, name="room_pool4")
#
#         furniture_cnn4, f_cnn4_l2_loss = Basic2dConv(x=furniture_pool3, d_out=72, ksize=(3, 3),
#                                                      active=tf.nn.relu, name="furniture_cnn4",
#                                                      trainable=trainable)
#         furniture_pool4 = MaxPool(x=furniture_cnn4, name="furniture_pool4")
#
#         # 普通 cnn 卷积  (16, 16) -> (8, 8)
#         room_cnn5, in_cnn5_l2_loss = Basic2dConv(x=room_pool4, d_out=72, ksize=(3, 3),
#                                                            active=tf.nn.relu, name="room_cnn5",
#                                                            trainable=trainable)
#         # [b, h//32, w//32, 72]
#         room_pool5 = MaxPool(x=room_cnn5, name="room_pool5")
#
#         furniture_cnn5, f_cnn5_l2_loss = Basic2dConv(x=furniture_pool4, d_out=72, ksize=(3, 3),
#                                                      active=tf.nn.relu, name="furniture_cnn5",
#                                                      trainable=trainable)
#         furniture_pool5 = MaxPool(x=furniture_cnn5, name="furniture_pool5")
#
#         height, weight = int(self.image_height / 32), int(self.image_width / 32)
#         # 卷积层特征 [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         room = tf.reshape(tensor=room_pool5, shape=[-1, height * weight * 72],
#                                     name="room_cnn_feature")
#
#         # [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * weight * 72], name="furniture_cnn_feature")
#
#         room_cnn_feature = room
#         furniture_cnn_feature = furniture
#
#         # 共享特征层
#         self.room_cnn_feature = room_cnn_feature
#         self.furniture_cnn_feature = furniture_cnn_feature
#
#         # 卷积层特征
#
#         # [b, (h//32)*(w//32)*72*2]
#         concat_feature = tf.concat(values=[room, furniture], axis=-1)
#
#         fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024, name="cnn_fc1", active=tf.nn.relu,
#                                        trainable=trainable)
#
#         fc2, fc2_l2_loss = FullConnect(x=fc1, out=512, name="cnn_fc2", active=tf.nn.relu, trainable=trainable)
#         # [b, label_size]
#         fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
#
#         # [b, label_size] -> [b, label_size]
#         labels = self.target_one_hot
#         # 损失函数 以及 acc 等
#         cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3)
#         # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
#         l2_loss = in_cnn1_l2_loss + f_cnn1_l2_loss + in_cnn2_l2_loss + f_cnn2_l2_loss + in_cnn3_l2_loss + f_cnn3_l2_loss + \
#                   in_cnn4_l2_loss + f_cnn4_l2_loss + in_cnn5_l2_loss + f_cnn5_l2_loss
#         # [b, ]
#         batch_target = self.target
#         istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
#         cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
#         # [b, label_size]
#         cnn_score = tf.nn.softmax(fc3, name="cnn_score")
#         # [b, label_size] -> [b, label_size]
#         self.cnn_output_distribute = cnn_score
#         # [b, label_size] -> [b, ]
#         cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
#         # (目标 且预测正确) / (目标数目)
#         correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
#         accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
#         if trainable:
#             grads_and_vars = self.optimizer.compute_gradients(cnn_loss)
#             train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
#             self.cnn_train_op = train_op
#
#         self.cnn_loss = cnn_loss
#         self.cnn_acc = accuracy
#
#
# class SingleDilatedCnn(SingleBasicLayoutModel):
#     """
#        空洞卷积 (3*3) rate = 2
#     """
#
#     def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
#                  furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int = 10):
#         """
#         :param image_width:
#         :param image_height:
#         :param furniture_nums:
#         :param label_size:
#         :param furniture_width:
#         :param furniture_height:
#         :param furniture_embedding_size:
#         :param max_length:
#         """
#         super(SingleDilatedCnn, self).__init__(image_width=image_width,
#                                                image_height=image_height,
#                                                furniture_nums=furniture_nums,
#                                                label_size=label_size,
#                                                furniture_width=furniture_width,
#                                                furniture_height=furniture_height,
#                                                furniture_embedding_size=furniture_embedding_size,
#                                                max_length=max_length)
#
#     def build_cnn_graph(self, trainable: bool = True):
#         """
#         创建图
#         :param trainable:
#         :return:
#         """
#         lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
#                                              num_units=self.furniture_embedding_size,
#                                              zero_pad=True,
#                                              scope="furniture_embedding",
#                                              trainable=trainable)
#         # b: batch_size, h: height, w: width, e: embedding_size
#         # [b, h, w] -> [b, h, w, e]
#         furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.furniture)
#         # [b, h, w] -> [b, h, w, e]
#         room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.room)
#
#         # 空洞 cnn 卷积 (128, 128) -> (64, 64)
#         room_cnn1, in_cnn1_l2_loss = Basic2dDilatedConv(x=room_embedding, d_out=72, ksize=(3, 3),
#                                                                   name="room_cnn1", active=tf.nn.relu, rate=2,
#                                                                   trainable=trainable)
#         room_cnn1 = BatchNormal(x=room_cnn1, name="room_cnn1_bn", trainable=trainable)
#         room_pool1 = MaxPool(x=room_cnn1, name="room_pool1")
#
#         furniture_cnn1, f_cnn1_l2_loss = Basic2dDilatedConv(x=furniture_embedding, d_out=72, ksize=(3, 3),
#                                                             name="furniture_cnn1", active=tf.nn.relu, rate=2, trainable=trainable)
#         furniture_cnn1 = BatchNormal(x=furniture_cnn1, name="furniture_cnn1_bn", trainable=trainable)
#         furniture_pool1 = MaxPool(x=furniture_cnn1, name="furniture_pool1")
#         # 空洞 cnn 卷积 (64, 64) -> (32, 32)
#         room_cnn2, in_cnn2_l2_loss = Basic2dDilatedConv(x=room_pool1, d_out=72, ksize=(3, 3),
#                                                                   name="room_cnn2", active=tf.nn.relu, rate=2,
#                                                                   trainable=trainable)
#         room_cnn2 = BatchNormal(x=room_cnn2, name="room_cnn2_bn", trainable=trainable)
#         room_pool2 = MaxPool(x=room_cnn2, name="room_pool2")
#
#         furniture_cnn2, f_cnn2_l2_loss = Basic2dDilatedConv(x=furniture_pool1, d_out=72, ksize=(3, 3), name="furniture_cnn2",
#                                                             active=tf.nn.relu, trainable=trainable, rate=2)
#         furniture_cnn2 = BatchNormal(x=furniture_cnn2, name="furniture_cnn2_bn", trainable=trainable)
#         furniture_pool2 = MaxPool(x=furniture_cnn2, name="furniture_pool2")
#         # 普通 cnn 卷积  (32, 32) -> (16, 16)
#         room_cnn3, in_cnn3_l2_loss = Basic2dDilatedConv(x=room_pool2, d_out=72, ksize=(3, 3),
#                                                                   name="room_cnn3", active=tf.nn.relu,
#                                                                   trainable=trainable, rate=2)
#         room_cnn3 = BatchNormal(x=room_cnn3, name="room_cnn3_bn", trainable=trainable)
#         room_pool3 = MaxPool(x=room_cnn3, name="room_pool3")
#
#         furniture_cnn3, f_cnn3_l2_loss = Basic2dDilatedConv(x=furniture_pool2, d_out=72, ksize=(3, 3), name="furniture_cnn3",
#                                                             active=tf.nn.relu, trainable=trainable, rate=2)
#         furniture_cnn3 = BatchNormal(x=furniture_cnn3, name="furniture_cnn3_bn", trainable=trainable)
#         furniture_pool3 = MaxPool(x=furniture_cnn3, name="furniture_pool3")
#
#         # 普通 cnn 卷积 (16, 16) -> (8, 8)
#         room_cnn4, in_cnn4_l2_loss = Basic2dDilatedConv(x=room_pool3, d_out=72, ksize=(3, 3),
#                                                                   active=tf.nn.relu, name="room_cnn4",
#                                                                   trainable=trainable, rate=2)
#         room_pool4 = MaxPool(x=room_cnn4, name="room_pool4")
#
#         furniture_cnn4, f_cnn4_l2_loss = Basic2dDilatedConv(x=furniture_pool3, d_out=72, ksize=(3, 3),
#                                                             active=tf.nn.relu, name="furniture_cnn4",
#                                                             trainable=trainable, rate=2)
#         furniture_pool4 = MaxPool(x=furniture_cnn4, name="furniture_pool4")
#
#         # 普通 cnn 卷积  (16, 16) -> (8, 8)
#         room_cnn5, in_cnn5_l2_loss = Basic2dDilatedConv(x=room_pool4, d_out=72, ksize=(3, 3),
#                                                                   active=tf.nn.relu, name="room_cnn5",
#                                                                   trainable=trainable, rate=2)
#         # [b, h//32, w//32, 72]
#         room_pool5 = MaxPool(x=room_cnn5, name="room_pool5")
#
#         furniture_cnn5, f_cnn5_l2_loss = Basic2dDilatedConv(x=furniture_pool4, d_out=72, ksize=(3, 3),
#                                                             active=tf.nn.relu, name="furniture_cnn5",
#                                                             trainable=trainable, rate=2)
#         furniture_pool5 = MaxPool(x=furniture_cnn5, name="furniture_pool5")
#
#         height, weight = int(self.image_height / 32), int(self.image_width / 32)
#         # 卷积层特征 [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         room = tf.reshape(tensor=room_pool5, shape=[-1, height * weight * 72],
#                                     name="room_cnn_feature")
#
#         # [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * weight * 72], name="furniture_cnn_feature")
#
#         room_cnn_feature = room
#         furniture_cnn_feature = furniture
#
#         # 共享特征层
#         self.room_cnn_feature = room_cnn_feature
#         self.furniture_cnn_feature = furniture_cnn_feature
#
#         # 卷积层特征
#
#         # [b, (h//32)*(w//32)*72*2]
#         concat_feature = tf.concat(values=[room, furniture], axis=-1)
#
#         fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024, name="cnn_fc1", active=tf.nn.relu,
#                                        trainable=trainable)
#
#         fc2, fc2_l2_loss = FullConnect(x=fc1, out=512, name="cnn_fc2", active=tf.nn.relu, trainable=trainable)
#         # [b, label_size]
#         fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
#
#         # [b, label_size] -> [b, label_size]
#         labels = self.target_one_hot
#         # 损失函数 以及 acc 等
#         cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3)
#         # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
#         l2_loss = in_cnn1_l2_loss + f_cnn1_l2_loss + in_cnn2_l2_loss + f_cnn2_l2_loss + in_cnn3_l2_loss + f_cnn3_l2_loss + \
#                   in_cnn4_l2_loss + f_cnn4_l2_loss + in_cnn5_l2_loss + f_cnn5_l2_loss
#         # [b, ]
#         batch_target = self.target
#         istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
#         cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
#         # [b, label_size]
#         cnn_score = tf.nn.softmax(fc3, name="cnn_score")
#         # [b, label_size] -> [b, label_size]
#         self.cnn_output_distribute = cnn_score
#         # [b, label_size] -> [b, ]
#         cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
#         # (目标 且预测正确) / (目标数目)
#         correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
#         accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
#         if trainable:
#             grads_and_vars = self.optimizer.compute_gradients(cnn_loss)
#             train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
#             self.cnn_train_op = train_op
#
#         self.cnn_loss = cnn_loss
#         self.cnn_acc = accuracy
#
#
# class SingleDeformableSquareCnn(SingleBasicLayoutModel):
#     """
#     可变卷积实现
#     """
#     def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
#                  furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int=10):
#         """
#         :param image_width:
#         :param image_height:
#         :param furniture_nums:
#         :param label_size:
#         :param furniture_width:
#         :param furniture_height:
#         :param furniture_embedding_size:
#         :param max_length:
#         """
#         super(SingleDeformableSquareCnn, self).__init__(image_width=image_width,
#                                              image_height=image_height,
#                                              furniture_nums=furniture_nums,
#                                              label_size=label_size,
#                                              furniture_width=furniture_width,
#                                              furniture_height=furniture_height,
#                                              furniture_embedding_size=furniture_embedding_size,
#                                              max_length=max_length)
#
#     def build_cnn_graph(self, trainable: bool = True):
#         """
#         创建图
#         :param trainable:
#         :return:
#         """
#         lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
#                                              num_units=self.furniture_embedding_size,
#                                              zero_pad=True,
#                                              scope="furniture_embedding",
#                                              trainable=trainable)
#         # b: batch_size, h: height, w: width, e: embedding_size
#         # [b, h, w] -> [b, h, w, e]
#         room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.room)
#         # 户型图
#         # 普通 cnn 卷积 生成特征图 (128, 128) -> (64, 64)
#         room_cnn1, in_cnn1_l2_loss = Basic2dConv(x=room_embedding, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn1", active=tf.nn.relu,
#                                                            trainable=trainable)
#         room_cnn1 = BatchNormal(x=room_cnn1, name="room_cnn1_bn", trainable=trainable)
#         room_pool1 = MaxPool(x=room_cnn1, name="room_pool1")
#
#         # 可变 cnn 卷积 (64, 64) -> (32, 32)
#         d_room_cnn2, d_in_cnn2_l2_loss = DeformableSquareCnn(x=room_pool1,
#                                                                        name="d_room_cnn2",
#                                                                        trainable=trainable)
#         room_cnn2, in_cnn2_l2_loss = Basic2dConv(x=d_room_cnn2, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn2", active=tf.nn.relu,
#                                                            trainable=trainable)
#
#         room_cnn2 = BatchNormal(x=room_cnn2, name="room_cnn2_bn", trainable=trainable)
#         room_pool2 = MaxPool(x=room_cnn2, name="room_pool2")
#
#         # 可变 cnn 卷积  (32, 32) -> (16, 16)
#         d_room_cnn3, d_in_cnn3_l2_loss = DeformableSquareCnn(x=room_pool2,
#                                                                        name="d_room_cnn3",
#                                                                        trainable=trainable)
#
#         room_cnn3, in_cnn3_l2_loss = Basic2dConv(x=d_room_cnn3, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn3", active=tf.nn.relu,
#                                                            trainable=trainable)
#
#         room_cnn3 = BatchNormal(x=room_cnn3, name="room_cnn3_bn", trainable=trainable)
#         room_pool3 = MaxPool(x=room_cnn3, name="room_pool3")
#
#         # 可变 cnn 卷积  (32, 32) -> (8, 8)
#         d_room_cnn4, d_in_cnn4_l2_loss = DeformableSquareCnn(x=room_pool3,
#                                                                        name="d_room_cnn4",
#                                                                        trainable=trainable)
#
#         room_cnn4, in_cnn4_l2_loss = Basic2dConv(x=d_room_cnn4, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn4", active=tf.nn.relu,
#                                                            trainable=trainable)
#
#         room_cnn4 = BatchNormal(x=room_cnn4, name="room_cnn4bn", trainable=trainable)
#         room_pool4 = MaxPool(x=room_cnn4, name="room_pool4")
#
#         # 可变 cnn 卷积 (8, 8) -> (4, 4)
#         d_room_cnn5, d_in_cnn5_l2_loss = DeformableSquareCnn(x=room_pool4,
#                                                                        name="d_room_cnn5",
#                                                                        trainable=trainable)
#
#         room_cnn5, in_cnn5_l2_loss = Basic2dConv(x=d_room_cnn5, d_out=72, ksize=(3, 3),
#                                                            name="room_cnn5", active=tf.nn.relu,
#                                                            trainable=trainable)
#
#         room_cnn5 = BatchNormal(x=room_cnn5, name="room_cnn5bn", trainable=trainable)
#         room_pool5 = MaxPool(x=room_cnn5, name="room_pool5")
#
#         # 家具
#         furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=self.furniture)
#         # 普通 cnn 卷积 生成特征图 (128, 128) -> (64, 64)
#         furniture_cnn1, f_cnn1_l2_loss = Basic2dConv(x=furniture_embedding, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn1", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn1 = BatchNormal(x=furniture_cnn1, name="furniture_cnn1_bn", trainable=trainable)
#         furniture_pool1 = MaxPool(x=furniture_cnn1, name="furniture_pool1")
#         # 可变 cnn 卷积 生成特征图 (64, 64) -> (32, 32)
#         d_furniture_cnn2, d_f_cnn2_l2_loss = DeformableSquareCnn(x=furniture_pool1,
#                                                                  name="d_furniture_cnn2",
#                                                                  trainable=trainable)
#         furniture_cnn2, f_cnn2_l2_loss = Basic2dConv(x=d_furniture_cnn2, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn2", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn2 = BatchNormal(x=furniture_cnn2, name="furniture_cnn2_bn", trainable=trainable)
#         furniture_pool2 = MaxPool(x=furniture_cnn2, name="furniture_pool2")
#         # 可变 cnn 卷积 生成特征图 (32, 32) -> (16, 16)
#         d_furniture_cnn3, d_f_cnn3_l2_loss = DeformableSquareCnn(x=furniture_pool2,
#                                                                  name="d_furniture_cnn3",
#                                                                  trainable=trainable)
#         furniture_cnn3, f_cnn3_l2_loss = Basic2dConv(x=d_furniture_cnn3, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn3", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn3 = BatchNormal(x=furniture_cnn3, name="furniture_cnn3_bn", trainable=trainable)
#         furniture_pool3 = MaxPool(x=furniture_cnn3, name="furniture_pool3")
#         # 可变 cnn 卷积 生成特征图 (16, 16) -> (8, 8)
#         d_furniture_cnn4, d_f_cnn4_l2_loss = DeformableSquareCnn(x=furniture_pool3,
#                                                                  name="d_furniture_cnn4",
#                                                                  trainable=trainable)
#         furniture_cnn4, f_cnn4_l2_loss = Basic2dConv(x=d_furniture_cnn4, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn4", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn4 = BatchNormal(x=furniture_cnn4, name="furniture_cnn4_bn", trainable=trainable)
#         furniture_pool4 = MaxPool(x=furniture_cnn4, name="furniture_pool4")
#         # 可变 cnn 卷积 生成特征图 (8, 8) -> (4, 4)
#         d_furniture_cnn5, d_f_cnn5_l2_loss = DeformableSquareCnn(x=furniture_pool4,
#                                                                  name="d_furniture_cnn5",
#                                                                  trainable=trainable)
#         furniture_cnn5, f_cnn5_l2_loss = Basic2dConv(x=d_furniture_cnn5, d_out=72, ksize=(3, 3),
#                                                      name="furniture_cnn5", active=tf.nn.relu, trainable=trainable)
#         furniture_cnn5 = BatchNormal(x=furniture_cnn5, name="furniture_cnn5_bn", trainable=trainable)
#         furniture_pool5 = MaxPool(x=furniture_cnn5, name="furniture_pool5")
#
#         height, weight = int(self.image_height / 32), int(self.image_width / 32)
#         # 卷积层特征 [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         room = tf.reshape(tensor=room_pool5, shape=[-1, height * weight * 72],
#                                     name="room_cnn_feature")
#
#         # [b, h//32, w//32, 72] -> [b, h//32*w//32*72]
#         furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * weight * 72], name="furniture_cnn_feature")
#
#         room_cnn_feature = room
#         furniture_cnn_feature = furniture
#
#         # 共享特征层
#         self.room_cnn_feature = room_cnn_feature
#         self.furniture_cnn_feature = furniture_cnn_feature
#
#         # 卷积层特征
#
#         # [b, (h//32)*(w//32)*72*2]
#         concat_feature = tf.concat(values=[room, furniture], axis=-1)
#
#         fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024, name="cnn_fc1", active=tf.nn.relu,
#                                        trainable=trainable)
#
#         fc2, fc2_l2_loss = FullConnect(x=fc1, out=512, name="cnn_fc2", active=tf.nn.relu, trainable=trainable)
#         # [b, label_size]
#         fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
#
#         # [b, label_size] -> [b, label_size]
#         labels = self.target_one_hot
#         # 损失函数 以及 acc 等
#         cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3)
#         # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
#         l2_loss = in_cnn1_l2_loss + f_cnn1_l2_loss + in_cnn2_l2_loss + f_cnn2_l2_loss + in_cnn3_l2_loss + f_cnn3_l2_loss + \
#                   in_cnn4_l2_loss + f_cnn4_l2_loss + in_cnn5_l2_loss + f_cnn5_l2_loss + d_in_cnn2_l2_loss + \
#                   d_in_cnn3_l2_loss + d_in_cnn4_l2_loss + d_in_cnn5_l2_loss + d_f_cnn2_l2_loss + d_f_cnn3_l2_loss + \
#                   d_f_cnn4_l2_loss + d_f_cnn5_l2_loss
#
#         # [b, ]
#         batch_target = self.target
#         istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
#         cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
#         # [b, label_size]
#         cnn_score = tf.nn.softmax(fc3, name="cnn_score")
#         # [b, label_size] -> [b, label_size]
#         self.cnn_output_distribute = cnn_score
#         # [b, label_size] -> [b, ]
#         cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
#         # (目标 且预测正确) / (目标数目)
#         correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
#         accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
#         if trainable:
#             grads_and_vars = self.optimizer.compute_gradients(cnn_loss)
#             train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
#             self.cnn_train_op = train_op
#         self.cnn_loss = cnn_loss
#         self.cnn_acc = accuracy
#
#
# class SingleVgg16Cnn(SingleBasicLayoutModel):
#     """
#     vgg16 模型
#     """
#     def __init__(self, image_width: int, image_height: int, furniture_nums: int, label_size: int,
#                  furniture_width: int, furniture_height: int, furniture_embedding_size: int, max_length: int=10):
#         """
#         :param image_width:
#         :param image_height:
#         :param furniture_nums:
#         :param label_size:
#         :param furniture_width:
#         :param furniture_height:
#         :param furniture_embedding_size:
#         :param max_length:
#         """
#         super(SingleVgg16Cnn, self).__init__(image_width=image_width,
#                                              image_height=image_height,
#                                              furniture_nums=furniture_nums,
#                                              label_size=label_size,
#                                              furniture_width=furniture_width,
#                                              furniture_height=furniture_height,
#                                              furniture_embedding_size=furniture_embedding_size,
#                                              max_length=max_length)
#
#     def build_cnn_graph(self, trainable: bool = True, filter_list: list = [64, 128, 256, 512, 512]):
#         """
#         创建图
#         :param trainable:
#         :return:
#         """
#         self.build_cnn_tensor_graph(room_feature=self.room,
#                                     furniture_feature=self.furniture,
#                                     labels=self.target,
#                                     trainable=trainable,
#                                     filter_list=filter_list)
#
#     def build_cnn_tensor_graph(self, room_feature: tf.Tensor, furniture_feature: tf.Tensor, labels: tf.Tensor,
#                                trainable: bool = True, filter_list: list = [64, 128, 256, 512, 512]):
#         """
#         :param room_feature:  [b, h, w]
#         :param furniture_feature:  [b, h, w]
#         :param labels:  [b]
#         :param trainable:  是否训练
#         :param filter_list:
#         :return
#         """
#         lookup_table = GetTokenEmbedding(vocab_size=self.furniture_nums,
#                                              num_units=self.furniture_embedding_size,
#                                              zero_pad=True,
#                                              scope="furniture_embedding",
#                                              trainable=trainable)
#         # b: batch_size, h: height, w: width, e: embedding_size
#         # [b, h, w] -> [b, h, w, e]
#         furniture_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=furniture_feature)
#         # [b, h, w] -> [b, h, w, e]
#         room_embedding = tf.nn.embedding_lookup(params=lookup_table, ids=room_feature)
#
#         furniture_pool5, furniture_l2_loss = Vgg16Layer(input_tensor=furniture_embedding, name="furniture", trainable=trainable, filter_list=filter_list)
#         room_pool5, room_l2_loss = Vgg16Layer(input_tensor=room_embedding, name="room", trainable=trainable, filter_list=filter_list)
#         _, height, width, c = furniture_pool5.get_shape()
#
#         # 卷积层特征 [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
#         room = tf.reshape(tensor=room_pool5, shape=[-1, height * width * c], name="room_cnn_feature")
#         # [b, h//32, w//32, 128] -> [b, h//32*w//32*128]
#         furniture = tf.reshape(tensor=furniture_pool5, shape=[-1, height * width * c], name="furniture_cnn_feature")
#
#         room_cnn_feature = room
#         furniture_cnn_feature = furniture
#
#         # 共享特征层
#         self.room_cnn_feature = room_cnn_feature
#         self.furniture_cnn_feature = furniture_cnn_feature
#
#         # 卷积层特征 部分数据 不需要
#         # [b, (h//32)*(w//32)*72*2]
#         concat_feature = tf.concat(values=[room, furniture], axis=-1)
#
#         fc1, fc1_l2_loss = FullConnect(x=concat_feature, out=1024*2, name="cnn_fc1", active=tf.nn.relu,
#                                        trainable=trainable)
#
#         fc2, fc2_l2_loss = FullConnect(x=fc1, out=512*2, name="cnn_fc2", active=tf.nn.relu, trainable=trainable)
#         # [b, label_size]
#         fc3, fc3_l2_loss = FullConnect(x=fc2, out=self.label_size, name="cnn_fc3", trainable=trainable)
#         # [b, ] -> [b, label_size]
#         target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
#         # 损失函数 以及 acc 等
#         # 权重类别衰减  (0数目出现次数较多 权重小一些 其余权重大一些)
#         # [b, ]
#         l2_loss = furniture_l2_loss + room_l2_loss + fc1_l2_loss + fc2_l2_loss + fc3_l2_loss
#         # [b, label_size] -> [b, label_size]
#         batch_target = labels
#         istarget = tf.to_float(tf.not_equal(batch_target, 0))  # 1 表示目标 0 表示填充
#         cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=fc3)
#         cnn_loss = tf.reduce_mean(cnn_loss_) + l2_loss * 1e-3  # l2 损失函数
#         # [b, label_size]
#
#         cnn_score = tf.nn.softmax(fc3, name="cnn_score")
#         # [b, label_size] -> [b, label_size]
#         self.cnn_output_distribute = cnn_score
#         # [b, label_size] -> [b, ]
#         cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
#         # (目标 且预测正确) / (目标数目)
#         correct = tf.cast(tf.equal(batch_target, cnn_predict), "float") * istarget / (tf.reduce_sum(istarget))
#         accuracy = tf.reduce_sum(correct, name="cnn_accuracy")
#         if trainable:
#             tvars = tf.trainable_variables()
#             max_grad_norm = 10  # 梯度计算
#             grads, global_norm = tf.clip_by_global_norm(tf.gradients(cnn_loss, tvars), max_grad_norm)
#             train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
#             self.cnn_train_op = train_op
#
#         self.cnn_loss = cnn_loss
#         self.cnn_acc = accuracy
#         self.cnn_l2_loss = l2_loss
#
