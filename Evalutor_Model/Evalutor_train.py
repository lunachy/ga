'''
Evalutor_train.py
对搭建好的网络进行训练，并保存训练参数，以便下次使用

'''
# 导入文件
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from Evalutor_Model.graph.BasicEvalutorModel import BasicRegressionModel

# 变量声明
n_class = 6
batch_size = 10  # 每个batch要放多少张图片
capacity = 200  # 一个队列最大多少
max_step = 10000
learning_rate = 0.0001  # 一般小于0.0001

# 训练数据及标签
path = '../data/evalauter_train.csv'
df = pd.read_csv(path, header=0)
room_a = df['room_a']
room_b = df['room_b']
label = df['scores'].tolist()
# label归一化处理 x∗=(x−min)/(max−min)
# label = [(i - min(label)) / (max(label) - min(label)) for i in label.tolist()]
room_a = [json.loads(a) for a in room_a.tolist()]
room_b = [json.loads(b) for b in room_b.tolist()]
label = np.array(label)


# room_a_batch = np.array(room_a[: 32], dtype=np.int32)
# room_b_batch = np.array(room_b[: 32], dtype=np.int32)
# label_batch = np.array(label[: 32])


def shuffle_set(room_a, room_b, label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    room_a = [room_a[row] for row in train_row]
    room_a = [room_a[row] for row in train_row]
    label = [label[row] for row in train_row]
    return room_a, room_b, label


def Get_batch(room_a, room_b, label, batch_size, now_batch, total_batch):
    '''
    :param image: 训练集
    :param label: label
    :param batch_size: batch_size
    :param now_batch: 当前epoch
    :param total_batch:epotch数
    :return:
    '''
    if now_batch < total_batch - 1:
        room_a_batch = room_a[now_batch * batch_size:(now_batch + 1) * batch_size]
        room_b_batch = room_b[now_batch * batch_size:(now_batch + 1) * batch_size]
        label_batch = label[now_batch * batch_size:(now_batch + 1) * batch_size]

    else:
        room_a_batch = room_a[now_batch * batch_size:]
        room_b_batch = room_b[now_batch * batch_size:]
        label_batch = label[now_batch * batch_size:]

    return room_a_batch, room_b_batch, label_batch


image_width = 64
image_height = 64
furniture_nums = 5000
label_size = 16 * 16 * 4 + 1
furniture_width = 64
furniture_height = 64
furniture_embedding_size = 4  # 向量维度
max_length = 8
epochs = 10000
save_step = 100
logs_train_dir = 'F:\pycharm\WorkSpace\GA\GA_functionZone\Evalutor_Model'

single_cnn_layout_model = BasicRegressionModel(image_width=image_width,
                                               image_height=image_height,
                                               furniture_nums=furniture_nums,
                                               label_size=label_size,
                                               furniture_width=furniture_width,
                                               furniture_height=furniture_height,
                                               furniture_embedding_size=furniture_embedding_size,
                                               max_length=max_length)


# 训练数据及标签
def train():
    loss_list = []
    logs_train_dir = r'./model/'
    sess = single_cnn_layout_model.create_sess()
    saver = tf.train.Saver()
    # var_to_restore = [val for val in cnn_variables]
    single_cnn_layout_model.build_cnn_graph(trainable=False, filter_list=[64, 128, 256, 512, 512],
                                            keep_prob=None, normal_model="batch_norm")
    cnn_variables = tf.global_variables()
    single_cnn_layout_model.build_regression_graph(trainable=True, keep_prob=None, filter_list=[64, 128, 256, 512, 512],
                                                   normal_model="batch_norm")

    single_cnn_layout_model.initializer(sess=sess)
    print("CNN 特征参数导入。。。")
    single_cnn_layout_model.restore(sess=sess, model_path=r'./model-20', var_list=cnn_variables)
    print("CNN 特征参数导入完成。")
    print("--" * 20)

    train_op = single_cnn_layout_model.regression_train_op
    train_loss = single_cnn_layout_model.regression_loss

    for step in np.arange(epochs):
        room_a_shuff, room_b_shuff, label_shuff = shuffle_set(room_a, room_b, label)
        # total_batch = len(label_shuff)/batch_size
        if len(label_shuff) % batch_size == 0:
            total_batch = len(label_shuff) / batch_size
        else:
            total_batch = int(len(label_shuff) / batch_size) + 1

        for now_batch in range(total_batch):
            room_a_batch, room_b_batch, label_batch = Get_batch(room_a_shuff, room_b_shuff, label_shuff, batch_size,
                                                                now_batch, total_batch)
            # room_a_batch, room_b_batch, label_batch = get_batch(room_a, room_b, label, batch_size, capacity)
            # # _, tra_loss = sess.run([train_op, train_loss])

            feed_dict = {single_cnn_layout_model.room_a: room_a_batch,
                         single_cnn_layout_model.room_b: room_b_batch,
                         single_cnn_layout_model.layout_marks: label_batch}

            _, loss = sess.run([train_op, train_loss], feed_dict=feed_dict)
            loss_list.append(loss)

        loss = sum(loss_list) / len(loss_list)
        print("step:{0}, loss:{1}".format(step, loss))

    # 保存最后一次网络参数
    checkpoint_path = os.path.join(logs_train_dir, 'evalutor_train.ckpt')
    saver.save(sess, checkpoint_path)


if __name__ == '__main__':
    train()
