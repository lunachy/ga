"""
速度快的评估指标函数
"""
import numpy as np
import tensorflow as tf


def beam_search_list_label(prior_score: np.ndarray,
                           prior_label_list: dict,
                           now_score: np.ndarray,
                           top_k: int = 3,
                           unique: bool=False,
                           mask_label: int=None):
    """
     beam search 搜索  执行效率不高  排序阶段会取出所有的数据 进行排序
    :param prior_score  前一过程对应分值  shape (b, )
    :param prior_label_list [[0, 1, 2]]
    :param now_score  当前的分值  shape (b, target_size)
    :param top_k  取的top_k数据
    :param unique  单条路径中 是否允许重复
    :param debug
    :param mask_label  mask的label数据
    :return:
    """
    if prior_score is None:  # 首次执行beam_search
        multi_score = now_score
        top_k_indexs = np.argsort(a=-multi_score, axis=-1)[:, : top_k]  # (b, target_size) -> (b, top_k)
        prior_label_list = []
        prior_score = []
        for b in range(now_score.shape[0]):
            label_lists = []
            for k in range(top_k):  # top_k
                label = top_k_indexs[b][k]
                label_lists.append([label])
                score = multi_score[b][label]
                prior_score.append(score)
            prior_label_list.append(label_lists)
        return prior_label_list, prior_score
    else:  # 非首次执行beam_search
        assert len(prior_score) == len(now_score)
        multi_score = []
        for i in range(len(prior_score)):
            multi_score.append(prior_score[i] * now_score[i])
        multi_score = np.array(multi_score)
        new_prior_label_list = []
        new_prior_score = []
        point = 0  # 指针位置
        for label_lists in prior_label_list:  # 每一个id数据 都要选择 top_k数据
            id_label_list_score = []  # 每一个id样本的 label与score
            for label_list in label_lists:
                id_brach_multi_score = multi_score[point]
                id_brach_top_index = np.argsort(a=-id_brach_multi_score)[: top_k*3]  # 多获取一些样本 后续需要去重 可能去掉
                for t in id_brach_top_index:
                    id_brach_multi_score_t = id_brach_multi_score[t]
                    id_label_list_score.append([label_list+[t], id_brach_multi_score_t])
                point += 1
            label_list_score = list(sorted(id_label_list_score, key=lambda x: x[1], reverse=True))  # 排序
            if unique:  # 保证样本无重复
                if mask_label is None:  # mask_label 不允许重复
                    label_list_score = list(filter(lambda x: len(set(x[0])) == len(x[0]), label_list_score))
                else:  # mask_label 允许重复
                    label_list_score_ = []
                    for label_list, score in id_label_list_score:
                        label_list_ = list(filter(lambda x: x != mask_label, label_list))
                        if len(set(label_list_)) == len(label_list_):
                            label_list_score_.append([label_list, score])
                    label_list_score = label_list_score_
            top_k_label_list_score = label_list_score[: top_k]
            label_lists = []
            for label_list, score in top_k_label_list_score:
                label_lists.append(label_list)
                new_prior_score.append(score)
            new_prior_label_list.append(label_lists)
        return new_prior_label_list, new_prior_score


class BeamSearchBase(object):

    def __init__(self, top_k: int, furniture_dxs: list, furniture_cids: list, furniture_dys: list,
                       state_encode: list, height: int, width: int, length: int, grid_size: int,
                 angle_size: int):
        """

        :param top_k:
        :param furniture_dxs:
        :param furniture_cids:
        :param furniture_dys:
        :param state_encode:
        :param height:
        :param width:
        :param length:
        :param grid_size:
        :param angle_size:
        """
        self.top_k = top_k
        self.furniture_dxs = furniture_dxs
        self.furniture_dys = furniture_dys
        self.furniture_cids = furniture_cids
        self.state_encode = state_encode
        self.height = height
        self.width = width
        self.length = length
        self.grid_size = grid_size
        self.angle_size = angle_size

        self.sess = NotImplemented
        self.output = NotImplemented
        self.middle_values = NotImplemented  # 中间状态

    def set_sess(self, sess: tf.Session, output: tf.Tensor):
        self.sess = sess
        self.output = output

    def run(self, tensors: list, values: list):
        feed_dict = dict(zip(tensors, values))
        return self.sess.run(self.output, feed_dict=feed_dict)

    def middle_states(self):  # 中间状态图
        NotImplemented


def top_k_sequence_acc(squence_predicts: list, squence_targets: list, top_k: int=1, ignore_label: int=0):
    """
    计算序列的完全正确的acc数值
    :param squence_predicts:   [batch_size, all_top, step_length]
    :param squence_targets:   [batch_size, step_length]
    :param ignore_label:  当前序列全为此值 不计正确与错误
    :return:
    """
    assert len(squence_predicts) == len(squence_targets)
    top_k_acc = 0  # top_k 准确次数
    top_k_error = 0  # top_k 错误次数
    for i in range(len(squence_predicts)):
        squence_target = squence_targets[i]
        if squence_target in squence_predicts[i][: top_k]:
            top_k_acc += 1
        else:
            top_k_error += 1
    return top_k_acc / (top_k_acc + top_k_error + 1e-10)


def beam_search_list_label_test1():
    now_score = np.array([[0.3, 0.4, 0.5]])
    prior_score = None
    prior_label_list = None
    prior_label_list, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_list=prior_label_list, now_score=now_score, top_k=2)
    print("首次beamsearch结果:{0},   {1}".format(prior_label_list, prior_score))
    now_score2 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    prior_label_list, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_list=prior_label_list, now_score=now_score2, top_k=2, unique=True)
    print("第二次beamsearch结果:{0},   {1}".format(prior_label_list, prior_score))
    now_score3 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    prior_label_list, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_list=prior_label_list, now_score=now_score3, top_k=2, unique=True)
    print("第三次beamsearch结果:{0},   {1}".format(prior_label_list, prior_score))

