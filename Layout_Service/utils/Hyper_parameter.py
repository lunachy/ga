# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    Hyper_parameter
    Description :    
    Author      :    zhaowen
    date        :    2019/4/15
____________________________________________________________________
    Change Activity:
                        2019/4/15:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'


class Hyperparams:
    single = False
    batch_size = 10
    lr = 0.01
    maxlen = 5
    hidden_units = 512
    num_blocks = 2
    num_heads = 4
    dropout_rate = 0.1  # 0.2
    use_single = False
    ebd_size = 6
    sinusoid = False
    top_k = 3
    angle = 90
    length = 8000

    image_width = 128
    image_height = 128
    furniture_nums = 5000
    label_size = 16 * 16 * 4 + 1
    label_grid = 16
    furniture_width = 128
    furniture_height = 128
    furniture_embedding_size = 8
    max_length = 5
    epochs = 500
    save_step = 2
    keep_prob = 0.5

    use_cnn_reshape = False
    use_cnn_predict_encoder = True
    use_cnn_predict_predict = True
    use_cnn = True
    use_DarkNet=False

    use_cnn_loss = False

    train_furniture_path_list = [
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-0.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-0.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-0.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-0.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-90.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-90.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-90.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-90.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-180.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-180.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-180.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-180.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-270.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-270.0\1",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-270.0\0",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-270.0\1"]

    test_furniture_path_list = [
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-0.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-0.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-90.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-90.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-180.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-180.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-主卧-270.0\2",
        r"E:\zhaowen\src\智能布局\git_ihome\layout\python\deeplearning_layerout\dataset\128_128_16_16_data\layout_cnn-次卧-270.0\2",
        ]
