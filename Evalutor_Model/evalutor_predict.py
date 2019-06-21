'''
功能：评分器的测试与预测
2019/5/27  @ZD
'''
import pandas as pd
import tensorflow as tf
import json
import numpy as np
from Evalutor_Model.graph.BasicEvalutorModel import BasicRegressionModel
import os
import pysnooper

image_width = 64
image_height = 64
furniture_nums = 5000
label_size = 16 * 16 * 4 + 1
furniture_width = 64
furniture_height = 64
furniture_embedding_size = 4  # 向量维度
max_length = 8


def path_file():
    file_path = 'data/evalauter_train.csv'
    # path = r'..\data\evalute_train.csv'
    if os.path.exists('./' + file_path):
        path = r'./' + file_path
    elif os.path.exists('../' + file_path):
        path = r'../' + file_path
    else:
        path = r'../data/evalauter_train.csv'
    return path


path = path_file()
df = pd.read_csv(path, header=0)
room_a = df['room_a']
room_b = df['room_b']
label = df['scores']

# #label归一化处理 x∗=(x−min)/(max−min)
# label = [(i - min(label)) / (max(label) - min(label)) for i in label.tolist()]
room_a = [json.loads(a) for a in room_a.tolist()]
room_b = [json.loads(b) for b in room_b.tolist()]
label = np.array(label)


# 从测试集中随机挑选一张图片看测试结果
def get_one_room(room_a, room_b, label):
    num = len(room_a)
    index = np.random.randint(0, num)
    img_a = np.array(room_a[index]).reshape([-1, 64, 64])
    img_b = np.array(room_b[index]).reshape([-1, 64, 64])
    img_label = label[index]
    return img_a, img_b, img_label


# print(get_one_room(room_a,room_b,label))


# def get_much_room(rooma,roomb,lab):
#     room_a = []
#     room_b = []
#     label = []
#     for i in range(10):
#         img_a,img_b,img_label = get_one_room(rooma,roomb,lab)
#         img_a = img_a.reshape([64,64])
#         img_b = img_b.reshape([64,64])
#         room_a.append(img_a)
#         room_b.append(img_b)
#         label.append(img_label)
#     return room_a,room_b,label


def pre_test(room_a, room_b, label):
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
    sess = tf.Session()
    single_cnn_layout_model.build_cnn_graph(trainable=False, filter_list=[64, 128, 256, 512, 512],
                                            keep_prob=None, normal_model="batch_norm")
    single_cnn_layout_model.build_regression_graph(trainable=False, keep_prob=None,
                                                   filter_list=[64, 128, 256, 512, 512],
                                                   normal_model="batch_norm")

    single_cnn_layout_model.initializer(sess=sess)
    saver.restore(sess, './model/evalutor_train.ckpt')
    print('Loading success...')

    feed_dict = {single_cnn_layout_model.room_a: room_a,
                 single_cnn_layout_model.room_b: room_b,
                 single_cnn_layout_model.layout_marks: label
                 }
    output_a = single_cnn_layout_model.output_a
    output_b = single_cnn_layout_model.output_b
    output_label = single_cnn_layout_model.layout_marks
    output_loss = single_cnn_layout_model.regression_loss
    out_a, out_b, out_label, mse_loss = sess.run([output_a, output_b, output_label, output_loss], feed_dict=feed_dict)
    print('测试集均方误差为：', mse_loss)
    return mse_loss


def predict(img_a):
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    prediction = sess.run(single_cnn_layout_model.output_a, feed_dict={single_cnn_layout_model.room_a: img_a})
    prediction = abs(prediction).reshape([1, -1]).tolist()[0]
    prediction = [round(i, 2) for i in prediction]
    return prediction


if __name__ == '__main__':
    ####predict####
    # room_a,b,l = get_one_room(room_a,room_b,label)
    # room_a.shape =[-1,64,64]

    # room_a = room_a[:30]
    # print(predict(room_a))

    ####test####
    room_a = room_a[:30]
    room_b = room_b[:30]
    label = label[:30]
    pre_test(room_a, room_b, label)
