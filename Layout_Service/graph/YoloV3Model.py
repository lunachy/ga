import tensorflow as tf
from utils import common
from utils.graph_utils import EmbeddingLayers
import tensorflow.contrib.slim as slim
from graph.CnnLayoutModel import SingleBasicLayoutModel


DECAY_STEPS = 100
DECAY_RATE = 0.9
CARDINALITY = 16
DEEPTH = 4
DECAY_STEPS = 100
DECAY_RATE = 0.9


# todo 改成加入分布的形式
class darknet53(object):
    """network for performing feature extraction"""

    def __init__(self, room_zid: tf.Tensor,
                       room_cid: tf.Tensor,
                       room_mid: tf.Tensor,
                       room_scid: tf.Tensor,
                       room_distance: tf.Tensor,
                       zone_zid: tf.Tensor,
                       zone_cid: tf.Tensor,
                       zone_mid: tf.Tensor,
                       furniture_nums: int,
                       furniture_embedding_size: int):
        self.furniture_nums = furniture_nums
        self.furniture_embedding_size = furniture_embedding_size
        self.outputs = self.forward(room_zid=room_zid, room_cid=room_cid, room_mid=room_mid, room_scid=room_scid,
                                    room_distance=room_distance, zone_zid=zone_zid, zone_cid=zone_cid, zone_mid=zone_mid)

    def transform_layer(self, inputs, scope):
        with tf.name_scope(scope):
            inputs = common._conv2d_fixed_padding(inputs, DEEPTH, 1, strides=1, layer_name="trans_1")
            inputs = common._conv2d_fixed_padding(inputs, DEEPTH, 3, strides=1, layer_name="trans_2")
            return inputs

    def transition_layer(self, inputs, filters, scope):
        with tf.name_scope(scope):
            inputs = common._conv2d_fixed_padding(inputs, filters, 1, strides=1, layer_name="trans_3")
            return inputs

    def split_layer(self, inputs, layer_name):
        with tf.name_scope(layer_name):
            layers_lst = []
            for i in range(CARDINALITY):
                splits = self.transform_layer(inputs, scope=layer_name + '_splitN_' + str(i))
                layers_lst.append(splits)
            return tf.concat(layers_lst, axis=3)

    def _ResNeXt_block(self, inputs, filters, layer_num, name=None):
        """
        implement residuals block in darknet53
        """
        shortcut = inputs
        x = self.split_layer(inputs, layer_name='split_layer_' + layer_num + '_1')
        inputs = self.transition_layer(x, filters * 2, scope='trans_layer_' + layer_num + '_1')
        outputs = inputs + shortcut
        return outputs

    def _darknet53_block(self, inputs, filters, name=None):
        """
        implement residuals block in darknet53
        """
        shortcut = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1, 1, name + '_conv1')
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3, 1, name + '_conv2')
        inputs = inputs + shortcut
        return inputs

    def _block(self, inputs, filter, layer_name):
        with tf.name_scope(layer_name):
            inputs = common._conv2d_fixed_padding(inputs, filter[0], 1, 1, layer_name + '_conv1')
            inputs = common._conv2d_fixed_padding(inputs, filter[1], 3, 1, layer_name + '_conv2')
            inputs = common._conv2d_fixed_padding(inputs, 64, 3, 2, layer_name + '_conv3')
            return inputs

    def _extract_block(self, inputs, layer_name):
        with tf.name_scope(layer_name):
            inputs = common._conv2d_fixed_padding(inputs=inputs, filters=32, kernel_size=3, strides=1,
                                                  layer_name=layer_name+'_conv1')
            for i in range(2):
                name = layer_name + 'res_block1' + str(i)
                inputs = self._darknet53_block(inputs, 16, name)

            inputs = common._conv2d_fixed_padding(inputs=inputs, filters=128, kernel_size=3, strides=2,
                                                  layer_name=layer_name + '_conv2')

            for i in range(4):
                name = layer_name + 'res_block2' + str(i)
                inputs = self._darknet53_block(inputs, 64, name)

            inputs = common._conv2d_fixed_padding(inputs=inputs, filters=512, kernel_size=3, strides=2,
                                                  layer_name=layer_name + '_conv3')
            for i in range(4):
                name = layer_name + 'res_block3' + str(i)
                inputs = self._darknet53_block(inputs, 256, name)

            inputs = common._conv2d_fixed_padding(inputs=inputs, filters=128, kernel_size=3, strides=2,
                                                  layer_name=layer_name + '_conv4')

            for i in range(2):
                name = layer_name + 'res_block4' + str(i)
                inputs = self._darknet53_block(inputs, 64, name)

            inputs = common._conv2d_fixed_padding(inputs=inputs, filters=4, kernel_size=3, strides=2,
                                                  layer_name=layer_name + '_conv5')

            inputs = tf.reshape(inputs, (-1, 1, 1, 64))
            inputs = tf.tile(inputs, multiples=[1, 16, 16, 1])
            return inputs

    def forward(self, room_zid: tf.Tensor,
                      room_cid: tf.Tensor,
                      room_mid: tf.Tensor,
                      room_scid: tf.Tensor,
                      zone_zid: tf.Tensor,
                      zone_cid: tf.Tensor,
                      zone_mid: tf.Tensor,
                      room_distance: tf.Tensor=None):
        room_embedding_list = EmbeddingLayers(input_tensor_list=[room_zid, room_cid, room_mid, room_scid, room_distance],
                                              vocab_size_list=[200, self.furniture_nums, 200, 200, 200],
                                              num_unit_list=[self.furniture_embedding_size]*6,
                                              scopes=["zid_table", "cid_table", "mid_table", "scid_table", "distance_table"],
                                              trainable=True,
                                              dtype=tf.float32)

        room_embedding = tf.concat(values=room_embedding_list, axis=-1)

        zone_embedding_list = EmbeddingLayers(input_tensor_list=[zone_zid, zone_cid, zone_mid],
                                              vocab_size_list=[200, self.furniture_nums, 200],
                                              num_unit_list=[self.furniture_embedding_size]*3,
                                              scopes=["zid_table", "cid_table", "scid_table"],
                                              trainable=True,
                                              dtype=tf.float32)
        zone_embedding = tf.concat(values=zone_embedding_list, axis=-1)

        zone_feature = self._extract_block(zone_embedding, 'zone_extract')
        room_feature = self._block(room_embedding, [16, 32], 'room')
        for i in range(2):
            name = 'res_block1' + str(i)
            room_feature = self._darknet53_block(room_feature, 32, name)
        room_feature = common._conv2d_fixed_padding(inputs=room_feature, filters=64, kernel_size=3,
                                                    strides=2, layer_name="forward_a")
        inputs = tf.concat([zone_feature, room_feature], axis=3, name='concat_zone_room')
        for i in range(8):
            name = 'res_block2' + str(i)
            inputs = self._darknet53_block(inputs, 64, name)
        route_1 = inputs
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=512, kernel_size=3, strides=2,
                                              layer_name="forward_b")
        for i in range(8):
            name = 'res_block3' + str(i)
            inputs = self._darknet53_block(inputs, 256, name)
        route_2 = inputs
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=1024, kernel_size=3, strides=2,
                                              layer_name="forward_c")
        for i in range(4):
            name = 'res_block4' + str(i)
            inputs = self._darknet53_block(inputs, 512, name)

        return route_1, route_2, inputs


class yolov3(SingleBasicLayoutModel):

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 furniture_nums: int,
                 label_size: int,
                 furniture_width: int,
                 furniture_height: int,
                 furniture_embedding_size: int,
                 max_length: int = 10,
                 tf_dtype=tf.float32,
                 batch_norm_decay=0.9,
                 leaky_relu=0.1,
                 trainable: bool = False,
                 id_list: list=["zid"]):

        super(yolov3, self).__init__(image_width=image_width,
                                     image_height=image_height,
                                     furniture_nums=furniture_nums,
                                     label_size=label_size,
                                     furniture_width=furniture_width,
                                     furniture_height=furniture_height,
                                     furniture_embedding_size=furniture_embedding_size,
                                     max_length=max_length,
                                     tf_dtype="32")
        self.id_list = id_list
        self.image_width = image_width
        self.image_height = image_height
        self.furniture_nums = furniture_nums
        self.label_size = label_size
        self.furniture_width = furniture_width
        self.furniture_height = furniture_height
        self.furniture_embedding_size = furniture_embedding_size
        self.max_length = max_length
        self.tf_dtype = tf_dtype
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self.trainable = trainable
        self.global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.lr_rate = tf.train.exponential_decay(0.001, self.global_step, decay_steps=DECAY_STEPS,
                                                  decay_rate=DECAY_RATE, staircase=True)
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)

    def _yolo_block(self, inputs: tf.Tensor, filters: int, layer_name: str):
        """

        :param inputs:
        :param filters:
        :param layer_name:
        :return:
        """
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=filters * 1, kernel_size=1,
                                              layer_name=layer_name + "_a")
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=filters * 2, kernel_size=3,
                                              layer_name=layer_name + "_b")
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=filters * 1, kernel_size=1,
                                              layer_name=layer_name + "_c")
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=filters * 2, kernel_size=3,
                                              layer_name=layer_name + "_d")
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=filters * 1, kernel_size=1,
                                              layer_name=layer_name + "_e")
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs=inputs, filters=filters * 2, kernel_size=3,
                                              layer_name=layer_name + "_f")
        return route, inputs

    @staticmethod
    def _upsample(inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsampled')

        return inputs

    def forward(self, room_zid: tf.Tensor,
                      room_cid: tf.Tensor,
                      room_mid: tf.Tensor,
                      room_scid: tf.Tensor,
                      room_distance: tf.Tensor,
                      zone_zid: tf.Tensor,
                      zone_cid: tf.Tensor,
                      zone_mid: tf.Tensor,
                ):
        """
        Creates YOLO v3 model.
        """
        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': self.trainable,
            'fused': None,
        }
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding]):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, inputs = darknet53(room_zid=room_zid,
                                                         room_cid=room_cid,
                                                         room_mid=room_mid,
                                                         room_scid=room_scid,
                                                         zone_zid=zone_zid,
                                                         zone_cid=zone_cid,
                                                         zone_mid=zone_mid,
                                                         room_distance=room_distance,
                                                         furniture_nums=self.furniture_nums,
                                                         furniture_embedding_size=self.furniture_embedding_size).outputs
                    print("yolo:{0}".format(inputs))
                with tf.variable_scope('yolo-v3'):
                    route, inputs = self._yolo_block(inputs=inputs, filters=512, layer_name="route1")
                    inputs = common._conv2d_fixed_padding(inputs, 512, 1, layer_name="route1_conv")
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)
                    route, inputs = self._yolo_block(inputs=inputs, filters=256, layer_name="route2")
                    inputs = common._conv2d_fixed_padding(inputs, 64, 1, layer_name="route2_conv")
                    upsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3, name='concat_2')
                    route, inputs = self._yolo_block(inputs=inputs, filters=4, layer_name="route3")
                    feature_map_3 = tf.identity(route, name='feature_map_3')
            return feature_map_3

    def predict(self, feature_maps: tf.Tensor, labels: tf.Tensor):
        labels = tf.reshape(labels, (-1, 1))
        target_one_hot = tf.one_hot(labels, self.label_size, 1, 0)
        self.batch_target = labels
        self.istarget = tf.to_float(tf.not_equal(self.batch_target, 0))
        feature_maps = tf.reshape(tf.transpose(feature_maps, perm=[0, 3, 1, 2]), (-1, self.label_size))
        cnn_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_one_hot, logits=feature_maps)
        cnn_loss = tf.reduce_mean(cnn_loss_)
        cnn_score = tf.nn.softmax(feature_maps, name="cnn_score")
        self.cnn_output_distribute = cnn_score
        self.cnn_predict = tf.argmax(cnn_score, 1, name="cnn_prediction", output_type=tf.int32)
        self.cnn_predict = tf.reshape(self.cnn_predict, (-1, 1))
        correct = tf.cast(tf.equal(self.batch_target, self.cnn_predict), "float")
        accuracy = tf.reduce_mean(correct, name="cnn_accuracy")
        self.cnn_loss = cnn_loss
        self.cnn_acc = accuracy
        return self.cnn_loss, self.cnn_acc

    def build_cnn_graph(self,
                        trainable: bool,
                        mask: bool,
                        training: tf.Tensor=None,
                        keep_prob: int=None):
        id_list = self.id_list

        room_zid = self.room_zid if "zid" in id_list else None
        room_cid = self.room_cid if "cid" in id_list else None
        room_mid = self.room_mid if "mid" in id_list else None
        room_scid = self.room_scid if "scid" in id_list else None
        room_distance = self.room_distance if "distance" in id_list else None

        zone_zid = self.zone_zid if "zid" in id_list else None
        zone_cid = self.zone_cid if "cid" in id_list else None
        zone_mid = self.zone_mid if "mid" in id_list else None

        feature = self.forward(room_zid=room_zid,
                               room_cid=room_cid,
                               room_mid=room_mid,
                               room_scid=room_scid,
                               room_distance=room_distance,
                               zone_zid=zone_zid,
                               zone_cid=zone_cid,
                               zone_mid=zone_mid)
        loss, acc = self.predict(feature, self.target)
