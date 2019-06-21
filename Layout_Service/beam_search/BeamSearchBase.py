import tensorflow as tf
import numpy as np
import copy
from checkCrash.CrashChecker import CrashChecker, buildEmbeddingMat


class BeamSearchBase(object):

    def __init__(self, sess: tf.Session, output_distribute: tf.Tensor, input_tensors: tf.Tensor):
        """
        beam search
        :param sess:  创建的会话
        :param output_distribute:  输出的tensor
        :param input_tensors:  输入的tensor
        """
        self.sess = sess
        self.output_distribute = output_distribute
        self.input_tensors = input_tensors

    def run_predict_step(self, values: list):
        """
        预测某一步骤的 logstis返回值
        :param values:
        :return:
        """
        feed_dict = dict(zip(self.input_tensors, values))
        return self.sess.run(self.output_distribute, feed_dict=feed_dict)


class CrashSingleStepBeamSearch(BeamSearchBase):

    def __init__(self, sess: tf.Session, output_distribute: tf.Tensor, input_tensors: list,
                 clash_checker: CrashChecker, search_width: int, room_type_id: int,
                 room_str: str, image_grid_num: int, use_id_list: list=["zid"], mask_label: bool=False):
        """
        碰撞检测
        :param sess:
        :param output_distribute:
        :param input_tensors:  输入的tensor  注 room_tensor 在前面 furniture_tensor在后面 按照 zid cid mid scid 顺序排列
        :param clash_checker:  碰撞检测接口
        :param mask_label 0 是否有效
        :param search_width  搜索宽度
        :param room_type_id 房间类型
        :param use_id_list 模型使用到的id类型 例如 zid cid mid scid 有几个id 就有几个
               room有 zid cid mid scid
               家具有 zid cid mid
        :param
        """
        BeamSearchBase.__init__(self, sess=sess, output_distribute=output_distribute, input_tensors=input_tensors)
        self.clash_checker = clash_checker
        self.mask_label = mask_label
        self.search_width = search_width
        self.image_grid_num = image_grid_num
        # 初始化房间
        self.clash_checker.init_room(roomTypeId=room_type_id, room_str=room_str)
        self.id_dict = {"zid": 0, "cid": 1, "mid": 2, "scid": 3}  # zid cid mid scid 下标位置映射字典
        self.use_id_list = use_id_list
        roomMats = self.clash_checker.roomMats

        init_states_dict = {}
        for use_id in use_id_list:
            use_index = self.id_dict[use_id]
            init_states_dict[use_id] = roomMats[use_index]
        self.init_states_dict = init_states_dict  # 初始化状态
        furniture_dict_list = []
        for furniture_mat in self.clash_checker.furniture_mats:
            furniture_mat_dict = {}
            for use_id in use_id_list:
                use_index = self.id_dict[use_id]
                furniture_mat_dict[use_id] = furniture_mat[use_index]
            furniture_dict_list.append(furniture_mat_dict)
        self.furniture_dict_list = furniture_dict_list

    def step_search(self, step: int, values: list, input_indexs: list=None,
                    input_ids: list=None, input_scores: list=None, debug: bool=False):
        """
        执行 某一步骤的 search  (不同的模型实现的方式不同)
        :param step 当前步数
        :param values 输入的特征
        :param input_indexs 之前路径所有的label
        :param input_ids 上一步骤的路径id 与输入的values对应
        :param input_scores 前一过程的分值
        :return: 成功:
                      crash_predict_indexs 预测的路径,
                      crash_imd_state_images 路径对应的中间状态图,
                      out_ids  输出的链路id
                      predict_scores  输出的链路预测分值
                  错误:
                      None, None, None, None

        """
        # 获取当前步骤的得分值
        predict_score = self.run_predict_step(values=values)

        if self.mask_label:  # 去除 label=0的影响
            # [batch_size, k]
            k = min(16, 2 * self.search_width)
            temp_k = k+1
            predict_indexs_ = np.argsort(-predict_score, axis=1)[:, : temp_k] - 1  # 0维度顺序不变
            predict_indexs = []  # 预测的label
            predict_scores = []  # 预测的score
            for i, predict_index in enumerate(predict_indexs_):
                predict_index_ = list(filter(lambda index: index > 0, predict_index))
                predict_indexs.append(predict_index_[: k])
                predict_scores.append([predict_score[i][index+1] for index in predict_index_])  # 当心mask影响
            predict_indexs = np.array(predict_indexs)
            predict_scores = np.array(predict_scores)
            if debug:
                print("strp:{0}, indexs:{1}".format(step, predict_indexs))
        else:
            k = min(16, 2 * self.search_width)
            predict_indexs = np.argsort(-predict_score, axis=1)[:, : k]
            predict_scores = []
            for i, predict_index in enumerate(predict_indexs):
                predict_scores.append([predict_score[i][index] for index in predict_index])
            predict_scores = np.array(predict_scores)
            if debug:
                print("strp:{0}, indexs:{1}".format(step, predict_indexs))
        if step == 0:  # 当前步骤是第一步
            try:
                crash_predict_indexs = []  # 碰撞检测成功的 label
                crash_imd_state_dict_list = []  # 碰撞检测返回的 中间状态图
                out_ids = []  # 记录 id 用于作为下一步的碰撞检测输入
                ind = -1  # 初始化的id
                predict_indexs = predict_indexs[0]  # [top_k+n]
                for index in predict_indexs:
                    ret, hit_mask = self.clash_checker.detect_crash_label({"label": int(index)}, index=0, tag=0)
                    if ret == 0:  # 不碰撞
                        crash_predict_indexs.append([index])
                        crash_imd_state_image = buildEmbeddingMat(EmbeddingArray=hit_mask, numSegs=self.image_grid_num)  # 待输入的家具  zid cid mid scid
                        crash_imd_state_dict = {}
                        for use_id in self.use_id_list:
                            use_index = self.id_dict[use_id]
                            crash_imd_state_dict[use_id] = crash_imd_state_image[use_index]
                        crash_imd_state_dict_list.append(crash_imd_state_dict)
                        ind += 1
                        out_ids.append(ind)
                    if len(out_ids) >= self.search_width:  # 已经找到前 k 个数据 提前退出
                        break
                predict_scores = predict_scores[0]
            except Exception as e:
                print(e)
                crash_predict_indexs = None
                crash_imd_state_dict_list = None
                out_ids = None
                predict_scores = None
            finally:
                self.clash_checker.step_finash()  # 当前部署结束
            return crash_predict_indexs, crash_imd_state_dict_list, out_ids, predict_scores
        else:  # 当前步骤不是第一步
            try:
                input_indexs = copy.deepcopy(input_indexs)
                # 先排序
                all_now_infos = []
                for k in range(len(predict_indexs)):  # [top_k, top_k]
                    now_index = input_indexs[k]
                    # shape [top_k, ]
                    scores = predict_scores[k] * input_scores[k]  # 当前分支乘以上一步分值作为最终得分
                    indexs = predict_indexs[k]
                    for score, index in zip(scores, indexs):
                        all_now_infos.append([now_index + [index], score, k])
                all_now_infos = list(sorted(all_now_infos, key=lambda x: x[1], reverse=True))  # 结果排序
                # 碰撞检测
                crash_predict_indexs = []
                predict_scores = []
                out_ids = []
                crash_imd_state_dict_list = []  # 碰撞检测返回的 中间状态图
                ind = -1
                for all_now_info in all_now_infos:
                    indexs = all_now_info[0]
                    score = all_now_info[1]
                    k = all_now_info[2]  # 第k条路径
                    ret, hit_mask = self.clash_checker.detect_crash_label({"label": int(indexs[-1])},
                                                       index=input_ids[k],  # 第k条路径
                                                       tag=step)
                    if ret == 0:
                        ind += 1
                        crash_imd_state_image = buildEmbeddingMat(EmbeddingArray=hit_mask, numSegs=self.image_grid_num)
                        self.temp = crash_imd_state_image  # 临时debug 用的数据

                        crash_imd_state_dict = {}
                        for use_id in self.use_id_list:
                            use_index = self.id_dict[use_id]
                            crash_imd_state_dict[use_id] = crash_imd_state_image[use_index]
                        crash_imd_state_dict_list.append(crash_imd_state_dict)
                        # 下标数值 一一对应
                        out_ids.append(ind)
                        crash_predict_indexs.append(indexs)
                        predict_scores.append(score)
                    if len(out_ids) >= self.search_width:  # 已经找到前 k 个数据 提前退出
                        break
            except Exception as e:
                print(e)
                crash_predict_indexs = None
                crash_imd_state_dict_list = None
                out_ids = None
                predict_scores = None
            finally:
                self.clash_checker.step_finash()  # 当前部署结束
            return crash_predict_indexs, crash_imd_state_dict_list, out_ids, predict_scores

    def run_beam_search(self, t1: bool=False, t2: bool=False):
        """
        执行beam search 操作 中间状态由碰撞检测生成
        :return:
        """
        try:
            furniture_dict_list = self.furniture_dict_list  # 输入的家具
            init_states_dict = self.init_states_dict  # 输入的初始状态
            # 准备room输入数据
            room_input_values = []
            for use_id in self.use_id_list:
                room_input_values.append([init_states_dict[use_id]])
            # 准备furniture输入数据
            input_furniture_lists = []
            for i in range(len(furniture_dict_list)):
                input_furniture_list = []
                furniture_dict = furniture_dict_list[i]
                for use_id in self.use_id_list:
                    if use_id == "scid" or use_id == "mid":  # 家具没有 scid mid
                        continue
                    input_furniture_list.append([furniture_dict[use_id]])
                input_furniture_lists.append(input_furniture_list)
            for step, input_furniture_value in enumerate(input_furniture_lists):
                if step == 0:
                    room_values = room_input_values  # zid cid scid mid 等一个或者多个
                    furniture_values = input_furniture_value  # zid cid scid 等一个或者多个
                    if t1:  # 仅debug时候生效
                        return room_values, furniture_values
                    assert type(room_values) == list and type(furniture_values) == list
                    crash_predict_indexs, crash_imd_state_dict_list, out_ids, predict_scores = \
                        self.step_search(step=step, values=room_values+furniture_values, debug=False)
                    if out_ids is None or len(out_ids) == 0:  # 当前所有结果都有碰撞 提前结束 返回结果值
                        return []
                else:
                    room_values = []
                    for use_id in self.use_id_list:
                        values = []
                        for crash_imd_state_dict in crash_imd_state_dict_list:
                            values.append(crash_imd_state_dict[use_id])
                        room_values.append(values)
                    furniture_values = []
                    for input_furniture in input_furniture_value:
                        furniture_values.append(input_furniture*len(crash_imd_state_dict_list))
                    if t2:  # 仅debug时候生效
                        return room_values, furniture_values
                    ori_crash_predict_indexs = copy.deepcopy(crash_predict_indexs)  # 上一步结果
                    crash_predict_indexs, crash_imd_state_dict_list, out_ids, predict_scores = \
                        self.step_search(values=room_values+furniture_values,
                                         step=1, input_indexs=crash_predict_indexs, input_scores=predict_scores, input_ids=out_ids)
                    if out_ids is None or len(out_ids) == 0:  # 当前所有结果都有碰撞 提前结束 并且返回上一步结果
                        return ori_crash_predict_indexs
        except Exception as e:
            print(e)
            crash_predict_indexs = []
        finally:
            self.clash_checker.room_finish()
        return crash_predict_indexs


class CrashMultiStepBeamSearch(BeamSearchBase):  # todo 序列模型的beam search
    pass


