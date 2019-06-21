"""
评估指标
"""
import tensorflow as tf
import numpy as np
import copy
from data.old.StateFurniture import get_room_state, get_layerout_last_state
from checkCrash.CrashCheck import CrashCheck, check_one_step


def top_k_acc(targets: np.ndarray, predicts: np.ndarray, top_k: int=4, ignore_label: int=None):
    """
    计算 top_k 准确率  (单步准确率计算)
    :param targets:  真实目标   (batch_size, )
    :param predicts:  预测结果  (batch_size, label_size)
    :param top_k:
    :param ignore_label:  忽略计算的target数据
    :return:
    """
    top_k_acc = 0  # top_k 准确次数
    top_k_error = 0  # top_k 错误次数
    assert len(targets) == len(predicts)
    top_k_predicts = np.argsort(-predicts, axis=1)[:, : top_k]
    for index in range(len(targets)):
        if ignore_label is not None and targets[index] == 0:
            continue
        else:
            if targets[index] in top_k_predicts[index]:
                top_k_acc += 1
            else:
                top_k_error += 1
    return top_k_acc/(top_k_acc + top_k_error + 1e-10)


def top_k_sequence_acc(squence_predicts: np.ndarray, squence_targets: np.ndarray, top_k: int=1, ignore_label: int=0):
    """
    计算序列的完全正确的acc数值
    :param squence_predicts:   [batch_size, all_top, step_length]
    :param squence_targets:   [batch_size, step_length]
    :param ignore_label:  当前序列全为此值 不计正确与错误
    :return:
    """
    assert len(squence_predicts) == len(squence_targets)
    assert squence_predicts.shape[1] >= top_k
    batch_size, step_length = squence_targets.shape
    top_k_acc = 0  # top_k 准确次数
    top_k_error = 0  # top_k 错误次数
    for i in range(batch_size):
        squence_target = squence_targets[i]
        if squence_target.tolist() == [ignore_label]*step_length:  # 整条路全为填充的 不进行计算
            continue
        if squence_target.tolist() in squence_predicts[i][: top_k].tolist():
            top_k_acc += 1
        else:
            top_k_error += 1
    return top_k_acc / (top_k_acc + top_k_error + 1e-10)


def beam_search_label(prior_label_score: np.ndarray,
                      now_score: np.ndarray,
                      target_size: int,
                      top_k: int=3,
                      keep_top_k: int=None,
                      unique: bool=False,
                      debug: bool=False,
                      mask_label: int=None):
    """
    # todo 无效label概率值过滤  这个可以在前面过滤 不在此处进行
    执行 beam search操作 返回当前最佳 label和score 以及记录之前过程的label
    :param prior_score:  [b, [label1, label2, label3,......score]]
    :param now_score:  [b, label_size*top_k]
    :param target_size:  目标数目
    :param top_k:  搜索的宽度
    :param keep_top_k:  每一次beamsearch 是否保留前 keep_top_k 个数据结果  todo 支持
    :param unique: 是否允许 重复label存在 用于过滤已经占用的位置
    :param debug:
    :param mask_label: mask无效的label数据
    :return:
    """
    now_score = now_score.copy()
    if prior_label_score is None:  # 首次搜索  仅选择top_k个数据结果
        # [b, target_size] -> [b, top_k]
        top_k_new_label = np.argsort(-now_score, axis=1)[:, : top_k]
        new_prior_label_score = []  # [b*top_k, [label, score]]
        for i in range(top_k_new_label.shape[0]):  # b
            for j in range(top_k_new_label.shape[1]):  # top_k
                label = top_k_new_label[i][j]
                score = now_score[i][label]
                new_prior_label_score.append([label, score])
        new_prior_label_score = np.array(new_prior_label_score)
        return new_prior_label_score
    else:  # 非首次搜索  搜索全部结果  然后使用乘积的最终分值排序 选择top_k结果  (排除自己为自己的情况)
        # [b*top_k, [label1, label2, ..., score]]
        b_top_k = prior_label_score.shape[0]
        b = int(b_top_k / top_k)  # batch_size 数目
        assert b_top_k % top_k == 0  # 保证整除
        # [b*top_k, [label1, label2, label3, ......]]  [b*top_k, target_size]
        all_prior_label_score = []
        for i in range(b_top_k):  # 遍历每一个样本数目
            for j in range(target_size):
                score = now_score[i][j]
                prior_score = prior_label_score[i][-1]
                new_score = score * prior_score  # 上一步得分与本次的乘积 当做最终结果
                prior_labels = prior_label_score[i][: -1].tolist()  # 上一步所有的label
                prior_labels.extend([j, new_score])  # [label1, label2, ...., score]
                all_prior_label_score.append(prior_labels)
        new_prior_label_score = np.array(all_prior_label_score)
        new_prior_label_score_ = []
        assert len(new_prior_label_score) == b*top_k*target_size
        for i in range(b):  # batch
            # [top_k*target_size, [label1, label2, ....score]]
            b_new_label_score = []
            for j in range(top_k):  # top_k
                for k in range(target_size):  # target_size
                    index = i*top_k*target_size + j*target_size + k  # 小心数据格式
                    if debug:
                        print("第:{0}样本, top:{1}样本, label:{2}, 数据:{3}".format(i, j, k, new_prior_label_score[index]))
                    b_new_label_score.append(new_prior_label_score[index])
            assert len(b_new_label_score) == 1*top_k*target_size
            if unique:  # 过滤重复  小心label很小的情况 易报错
                if mask_label is not None:
                    b_new_label_score_ = []
                    for new_label_score in b_new_label_score:
                        no_mask_label = list(filter(lambda x: x != mask_label, new_label_score[: -1]))  # 选出非mask_label
                        if len(no_mask_label) == len(set(no_mask_label)):
                            b_new_label_score_.append(new_label_score)
                    b_new_label_score = b_new_label_score_
                else:
                    b_new_label_score = list(filter(lambda x: len(set(x[: -1])) == len(x[: -1]), b_new_label_score))
            b_new_label_score = np.array(b_new_label_score)
            b_new_score = b_new_label_score[:, -1]
            ids = np.argsort(-b_new_score)[: top_k]
            b_new_label_score = b_new_label_score[ids, :]
            new_prior_label_score_.extend(b_new_label_score)
        return np.array(new_prior_label_score_)


def beam_search_list_label(prior_score: np.ndarray,
                           prior_label_dict: dict,
                           now_score: np.ndarray,
                           top_k: int = 3,
                           unique: bool=False,
                           mask_label: int=None):
    """
     beam search 搜索  执行效率不高  排序阶段会取出所有的数据 进行排序
    :param prior_score  前一过程对应分值  shape (b, )
    :param prior_label_dict {id, [0, 1, 2]}
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
        prior_label_dict = {}
        prior_score = []
        for b in range(now_score.shape[0]):
            label_lists = []
            score_lists = []
            for k in range(top_k):  # top_k
                label = top_k_indexs[b][k]
                label_lists.append([label])
                score = multi_score[b][label]
                score_lists.append(score)
                prior_score.append(score)
            prior_label_dict[b] = label_lists
        return prior_label_dict, prior_score
    else:  # 非首次执行beam_search
        assert len(prior_score) == len(now_score)
        multi_score = []
        for i in range(len(prior_score)):
            multi_score.append(prior_score[i] * now_score[i])
        multi_score = np.array(multi_score)
        new_prior_label_dict = {}
        new_prior_score = []
        point = 0  # 指针位置
        for id, label_lists in prior_label_dict.items():  # 每一个id数据 都要选择 top_k数据
            id_label_list_score = []  # 每一个id样本的 label与score
            for label_list in label_lists:
                id_brach_multi_score = multi_score[point]
                id_brach_top_index = np.argsort(a=-id_brach_multi_score)[: top_k*3]  # 多获取一些样本 后续需要去重 可能去掉
                for t in id_brach_top_index:
                    id_brach_multi_score_t = id_brach_multi_score[t]
                    id_label_list_score.append([label_list+[t], id_brach_multi_score_t])
                point += 1
            id_label_list_score = list(sorted(id_label_list_score, key=lambda x: x[1], reverse=True))  # 排序
            if unique:  # 保证样本无重复
                if mask_label is None:  # mask_label 不允许重复
                    id_label_list_score = list(filter(lambda x: len(set(x[0])) == len(x[0]), id_label_list_score))
                else:  # mask_label 允许重复
                    id_label_list_score_ = []
                    for label_list, score in id_label_list_score:
                        label_list_ = list(filter(lambda x: x != mask_label, label_list))
                        if len(set(label_list_)) == len(label_list_):
                            id_label_list_score_.append([label_list, score])
                    id_label_list_score = id_label_list_score_
            top_k_id_label_list_score = id_label_list_score[: top_k]
            id_label_lists = []
            for label_list, score in top_k_id_label_list_score:
                id_label_lists.append(label_list)
                new_prior_score.append(score)
            new_prior_label_dict[id] = id_label_lists
        return new_prior_label_dict, new_prior_score


def beam_search_global_list_cnn_run(start_state_images: list,
                                    furniture_images: list,
                                    furniture_dxs: list,
                                    furniture_dys: list,
                                    furniture_cids: list,
                                    width: int,
                                    height: int,
                                    grid_size: int,
                                    top_k: int,
                                    output_distribute: tf.Tensor,
                                    room_state: tf.Tensor,
                                    furniture: tf.Tensor,
                                    sess: tf.Session,
                                    target_size: int,
                                    length: int=8000,
                                    mask_label: int=0):
    """
    执行beamsearch cnn 执行结果  建议单个样本 单个样本进行
    :param start_state_images: 户型图 [batch_size, height, width]
    :param furniture_images: 家具图 [batch_size, length, height, width]
    :param furniture_dxs:  家具dx [batch_size, length]
    :param furniture_dys:  家具dy [batch_size, length]
    :param furniture_cids:  家具cid [batch_size, length]
    :param width:  图片宽
    :param height:  图片长
    :param grid_size:
    :param top_k:   搜索半径
    :param output_distribute:  模型输出结果 tensor
    :param room_state:  模型布局状态 tensor
    :param furniture:  模型家具图状态 tensor
    :param sess:  Session
    :param target_size: 输入label的数目
    :param length:  长度8000
    :param mask_label:  是否使用 masklabel 如果使用， 还原状态的时候需要对label进行 -1 处理
    :return:
    """
    furniture_dxs = copy.deepcopy(furniture_dxs)
    furniture_dys = copy.deepcopy(furniture_dys)
    furniture_cids = copy.deepcopy(furniture_cids)
    batch_size, step_length = np.shape(furniture_dxs)[:2]
    new_prior_label_dict = None  # 前一步dict数据
    prior_label_score = None  # 前一步分值
    prior_state_images = None  # 前一步状态图
    for i in range(step_length):
        if i == 0:  # 计算初始结果 并执行 beamsearch
            prior_state_images = np.array(start_state_images)
            furniture_image = np.array(furniture_images)[:, i, :, :]
            #  (b, target_size)
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)
            prior_state_images = mask_state_image(prior_state_images=prior_state_images,
                                                  furniture_image=furniture_image)

            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})
            # 当前结果  当前结果分值
            new_prior_label_dict, prior_label_score = beam_search_list_label(prior_score=None,
                                                                             prior_label_dict=None,
                                                                             now_score=predict_scores,
                                                                             top_k=top_k,
                                                                             mask_label=mask_label,
                                                                             unique=True)

        else:  # 非首次执行结果
            furniture_image = np.array(furniture_images)[:, i, :, :]
            furniture_image = np.repeat(furniture_image, repeats=top_k, axis=0)
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)
            prior_state_images = mask_state_image(prior_state_images=prior_state_images,
                                                  furniture_image=furniture_image)  # bug待修订
            # return furniture_image, prior_state_images
            # 前一时刻状态 当前布局家具 当前布局的家具 需要重复 top_k次
            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})

            new_prior_label_dict, prior_label_score = beam_search_list_label(prior_score=prior_label_score,
                                                                             prior_label_dict=new_prior_label_dict,
                                                                             now_score=predict_scores,
                                                                             top_k=top_k,
                                                                             mask_label=mask_label,
                                                                             unique=True)
        # 更新中间状态图
        prior_state_images_ = []
        for id, label_lists in new_prior_label_dict.items():
            for label_list in label_lists:  # 每一个中间状态图 都对应一个 label_list
                if mask_label is not None:
                    ori_label_list = [label - 1 for label in label_list]
                else:
                    ori_label_list = label_list
                now_length = len(ori_label_list)  # 当前路径长度  使用之前的长度 获取最终状态
                j_state = get_layerout_last_state(ori_labels=ori_label_list,
                                                  furniture_dxs=furniture_dxs[id][: now_length],
                                                  furniture_cids=furniture_cids[id][: now_length],
                                                  furniture_dys=furniture_dys[id][: now_length],
                                                  state_encode=start_state_images[id],
                                                  height=height,
                                                  width=width,
                                                  length=length,
                                                  grid_size=grid_size)
                prior_state_images_.append(j_state)
        prior_state_images = np.array(prior_state_images_)
    return prior_state_images, new_prior_label_dict, prior_label_score


def beam_search_global_list_crash_cnn_run(start_state_images: list,
                                          furniture_images: list,
                                          furniture_dxs: list,
                                          furniture_dys: list,
                                          furniture_cids: list,
                                          width: int,
                                          height: int,
                                          grid_size: int,
                                          top_k: int,
                                          output_distribute: tf.Tensor,
                                          room_state: tf.Tensor,
                                          furniture: tf.Tensor,
                                          sess: tf.Session,
                                          target_size: int,
                                          room_str: str,
                                          length: int=8000,
                                          split_angle: int=90,
                                          mask_label: int=0):
    """
    执行带碰撞检测的 beamsearch  此处限定 仅支持当个房间样本的 beamsearch
    :param start_state_images:
    :param furniture_images:
    :param furniture_dxs:
    :param furniture_dys:
    :param furniture_cids:
    :param width:
    :param height:
    :param grid_size:
    :param top_k:
    :param output_distribute:
    :param room_state:
    :param furniture:
    :param sess:
    :param target_size:
    :param room_str: 房间户型字符串
    :param length:
    :param mask_label:
    :return:
    """
    furniture_dxs = copy.deepcopy(furniture_dxs)
    furniture_dys = copy.deepcopy(furniture_dys)
    furniture_cids = copy.deepcopy(furniture_cids)
    batch_size, step_length = np.shape(furniture_dxs)[:2]
    assert batch_size == 1  # 仅能保证单个样本正确
    new_prior_label_dict = None  # 前一步dict数据
    prior_label_score = None  # 前一步分值
    prior_state_images = None  # 前一步状态图

    # 初始化碰撞检测
    furniture_list = []
    for cid, dx, dy in zip(furniture_cids[0], furniture_dxs[0], furniture_dys[0]):
        furniture_list.append({"cid": int(cid), "dx": int(dx), "dy": int(dy)})
    crash_check = CrashCheck(model_dir='', room_str=room_str, furniture_list=furniture_list,
                             x_grid_num=16, y_grid_num=16, split_angle=90, max_room_size=8000, topk=top_k)
    results = []
    for i in range(step_length):
        if i == 0:  # 计算初始结果 并执行 beamsearch
            prior_state_images = np.array(start_state_images)
            furniture_image = np.array(furniture_images)[:, i, :, :]
            #  (b, target_size)
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)
            prior_state_images = mask_state_image(prior_state_images=prior_state_images,
                                                  furniture_image=furniture_image)

            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})
            # 当前结果  当前结果分值
            new_prior_label_dict, prior_label_score = beam_search_list_label(prior_score=None,
                                                                             prior_label_dict=None,
                                                                             now_score=predict_scores,
                                                                             top_k=top_k*3,
                                                                             mask_label=mask_label,
                                                                             unique=True)
            # 碰撞检测
            label_lists = [x[0] for x in new_prior_label_dict[0]]
            results = check_one_step(crash_check,
                                     label_lists=label_lists,
                                     furniture=furniture_list[i],
                                     pre_results=None,
                                     mask_label=mask_label,
                                     add_mask=False)
            if len(results) == 0:  # 碰撞检测无结果 直接返回
                return None, None
            # 满足条件的 label
            valid_label = [result["label"] for result in results]
            label_lists_ = []
            # 过滤无效的数据 首次的状态图肯定满足 直接过滤
            prior_label_score_ = []
            for id, label_lists in new_prior_label_dict.items():
                for h in range(len(label_lists)):
                    label_list = label_lists[h]
                    if label_list[-1] in valid_label:
                        label_lists_.append(label_list)
                        prior_label_score_.append(prior_label_score[h])
            new_prior_label_dict = {0: label_lists_}
            prior_label_score = prior_label_score_
        else:  # 非首次执行结果
            furniture_image = np.array(furniture_images)[:, i, :, :]
            furniture_image = np.repeat(furniture_image, repeats=len(prior_state_images), axis=0)  #
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)  可以不用此步骤
            prior_state_images = mask_state_image(prior_state_images=prior_state_images,
                                                  furniture_image=furniture_image)
            # 前一时刻状态 当前布局家具 当前布局的家具 需要重复 top_k次
            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})

            new_prior_label_dict, prior_label_score = beam_search_list_label(prior_score=prior_label_score,
                                                                             prior_label_dict=new_prior_label_dict,
                                                                             now_score=predict_scores,
                                                                             top_k=top_k,
                                                                             mask_label=mask_label,
                                                                             unique=True)
            # 碰撞检测
            label_lists = []
            for result in results:
                result_label = result["label"]
                label_list = []
                for route_label_list in new_prior_label_dict[0]:  # 仅有一个样本
                    if result_label == route_label_list[-2]:  # 保证上一步的路径一样
                        label_list.append(route_label_list[-1])
                label_lists.append(label_list)

            old_results = copy.deepcopy(results)
            results = check_one_step(crash_check,
                                     label_lists=label_lists,
                                     furniture=furniture_list[i],
                                     pre_results=results,
                                     mask_label=mask_label,
                                     add_mask=False)
            if len(results) == 0:
                return new_prior_label_dict, prior_label_score
            # 满足条件的 label  满足碰撞检测的label
            valid_label_last_two_list = []
            for result in results:
                for old_result in old_results:
                    if old_result["ind"] == result["in_ind"]:  # 路径id号一致
                        valid_label_last_two_list.append([old_result["label"], result["label"]])
            # 过滤无效的数据 中间状态图过滤
            prior_label_score_ = []
            label_lists_ = []
            for id, label_lists in new_prior_label_dict.items():
                for h in range(len(label_lists)):
                    label_list = label_lists[h]
                    if label_list[-2: ] in valid_label_last_two_list:  # 当前路径 label 有效
                        # 当前路径由有效路径传递
                        label_lists_.append(label_list)
                        prior_label_score_.append(prior_label_score[h])
            # # 更新过滤后的结果数据
            new_prior_label_dict = {0: label_lists_}
            prior_label_score = prior_label_score_
        # 更新中间状态图
        prior_state_images_ = []
        for id, label_lists in new_prior_label_dict.items():
            for label_list in label_lists:  # 每一个中间状态图 都对应一个 label_list
                if mask_label is not None:
                    ori_label_list = [label - 1 for label in label_list]
                else:
                    ori_label_list = label_list
                now_length = len(ori_label_list)  # 当前路径长度  使用之前的长度 获取最终状态
                j_state = get_layerout_last_state(ori_labels=ori_label_list, furniture_dxs=furniture_dxs[id][: now_length],
                                                  furniture_cids=furniture_cids[id][: now_length],
                                                  furniture_dys=furniture_dys[id][: now_length],
                                                  state_encode=start_state_images[id],
                                                  height=height,
                                                  width=width,
                                                  length=length,
                                                  grid_size=grid_size)
                prior_state_images_.append(j_state)
        prior_state_images = np.array(prior_state_images_)
    return prior_state_images, new_prior_label_dict, prior_label_score


def beam_search_global_cnn_run_old(start_state_images: list,
                                   furniture_images: list,
                                   furniture_dxs: list,
                                   furniture_dys: list,
                                   furniture_cids: list,
                                   width: int,
                                   height: int,
                                   grid_size: int,
                                   top_k: int,
                                   output_distribute: tf.Tensor,
                                   room_state: tf.Tensor,
                                   furniture: tf.Tensor,
                                   sess: tf.Session,
                                   target_size: int,
                                   length: int=8000,
                                   mask_label: int=0):
    """
    执行beamsearch cnn 执行结果
    :param start_state_images: 户型图 [batch_size, height, width]
    :param furniture_images: 家具图 [batch_size, length, height, width]
    :param furniture_dxs:  家具dx [batch_size, length]
    :param furniture_dys:  家具dy [batch_size, length]
    :param furniture_cids:  家具cid [batch_size, length]
    :param width:  图片宽
    :param height:  图片长
    :param grid_size:
    :param top_k:   搜索半径
    :param output_distribute:  模型输出结果 tensor
    :param room_state:  模型布局状态 tensor
    :param furniture:  模型家具图状态 tensor
    :param sess:  Session
    :param target_size: 输入label的数目
    :param length:  长度8000
    :param mask_label:  是否使用 masklabel 如果使用， 还原状态的时候需要对label进行 -1 处理
    :return:
    """
    # 部署的长度
    batch_size, step_length = np.shape(furniture_dxs)[:2]
    prior_label_score = None  # 前一步骤的分值记录
    prior_state_images = None  # 前一状态的户型图状态记录
    # all_temp_image = copy.deepcopy(start_state_images)  # 用于debug数据使用
    for i in range(step_length):
        if i == 0:  # 计算初始结果 并执行 beamsearch
            # [batch_size, height, width]
            prior_state_images = np.array(start_state_images)
            furniture_image = np.array(furniture_images)[:, i, :, :]
            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)
            prior_state_images = mask_state_image(prior_state_images=prior_state_images,
                                                  furniture_image=furniture_image)

            prior_label_score = beam_search_label(prior_label_score=prior_label_score,
                                                  now_score=predict_scores,
                                                  target_size=target_size,
                                                  top_k=top_k,
                                                  unique=True,
                                                  mask_label=mask_label)
        else:  # 计算非初始结果 并执行 beamsearch
            # [batch_size, ...] -> [batch_size*top_k, ...]  batch_size维度 复制即 axis=0 每一个样本复制
            furniture_image = np.array(furniture_images)[:, i, :, :]
            furniture_image = np.repeat(furniture_image, repeats=top_k, axis=0)
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)
            prior_state_images = mask_state_image(prior_state_images=prior_state_images, furniture_image=furniture_image)  # bug待修订
            # return furniture_image, prior_state_images
            # 前一时刻状态 当前布局家具 当前布局的家具 需要重复 top_k次
            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})
            # 执行 beamsearch  返回搜索的结果
            prior_label_score = beam_search_label(prior_label_score=prior_label_score,
                                                  now_score=predict_scores,
                                                  target_size=target_size,
                                                  top_k=top_k,
                                                  unique=True,
                                                  mask_label=mask_label)
        # 更新中间状态
        labels = prior_label_score[:, -2]  # 上一步生成的label
        prior_state_images_ = []
        for j in range(batch_size):  # batch  每一个样本有 top_k个状态
            for k in range(top_k):  # top_k
                furniture_cid = furniture_cids[j][i]
                furniture_dx = furniture_dxs[j][i]
                furniture_dy = furniture_dys[j][i]
                if i == 0:  # prior_state_images [batch_size, image_height, image_width]
                    state = prior_state_images[j]
                else:  # prior_state_images [batch_size*top_k, image_height, image_width]
                    state = prior_state_images[j*top_k+k]  # bug 有问题
                if mask_label is not None:
                    label = labels[j * top_k + k] - 1
                j_state = get_room_state(h=height,
                                         w=width,
                                         furniture_cid=furniture_cid,
                                         state=state,
                                         label=label,
                                         furniture_dx=furniture_dx,
                                         furniture_dy=furniture_dy,
                                         length=length,
                                         grid_size=grid_size)
                prior_state_images_.append(j_state)
        prior_state_images = np.array(prior_state_images_)
    return np.reshape(prior_label_score, newshape=[batch_size, top_k, step_length+1])


def beam_search_global_cnn_run(start_state_images: list,
                               furniture_images: list,
                               furniture_dxs: list,
                               furniture_dys: list,
                               furniture_cids: list,
                               width: int,
                               height: int,
                               grid_size: int,
                               top_k: int,
                               output_distribute: tf.Tensor,
                               room_state: tf.Tensor,
                               furniture: tf.Tensor,
                               sess: tf.Session,
                               target_size: int,
                               length: int=8000,
                               mask_label: int=0):
    """
    执行beamsearch cnn 执行结果
    :param start_state_images: 户型图 [batch_size, height, width]
    :param furniture_images: 家具图 [batch_size, length, height, width]
    :param furniture_dxs:  家具dx [batch_size, length]
    :param furniture_dys:  家具dy [batch_size, length]
    :param furniture_cids:  家具cid [batch_size, length]
    :param width:  图片宽
    :param height:  图片长
    :param grid_size:
    :param top_k:   搜索半径
    :param output_distribute:  模型输出结果 tensor
    :param room_state:  模型布局状态 tensor
    :param furniture:  模型家具图状态 tensor
    :param sess:  Session
    :param target_size: 输入label的数目
    :param length:  长度8000
    :param mask_label:  是否使用 masklabel 如果使用， 还原状态的时候需要对label进行 -1 处理
    :return:
    """
    # 部署的长度
    batch_size, step_length = np.shape(furniture_dxs)[:2]
    prior_label_score = None  # 前一步骤的分值记录
    prior_state_images = None  # 前一状态的户型图状态记录
    # all_temp_image = copy.deepcopy(start_state_images)  # 用于debug数据使用
    for i in range(step_length):
        if i == 0:  # 计算初始结果 并执行 beamsearch
            # [batch_size, height, width]
            prior_state_images = np.array(start_state_images)
            furniture_image = np.array(furniture_images)[:, i, :, :]

            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})

            prior_label_score = beam_search_label(prior_label_score=prior_label_score,
                                                  now_score=predict_scores,
                                                  target_size=target_size,
                                                  top_k=top_k,
                                                  unique=True,
                                                  mask_label=mask_label)
        else:  # 计算非初始结果 并执行 beamsearch
            # [batch_size, ...] -> [batch_size*top_k, ...]  batch_size维度 复制即 axis=0 每一个样本复制
            furniture_image = np.array(furniture_images)[:, i, :, :]
            furniture_image = np.repeat(furniture_image, repeats=top_k, axis=0)
            # 根据 furniture_image 对 prior_state_images 修订  (训练样本集 没有此类数据)
            prior_state_images = mask_state_image(prior_state_images=prior_state_images, furniture_image=furniture_image)  # bug待修订
            # return furniture_image, prior_state_images
            # 前一时刻状态 当前布局家具 当前布局的家具 需要重复 top_k次
            predict_scores = sess.run(output_distribute,
                                      feed_dict={room_state: prior_state_images, furniture: furniture_image})
            # 执行 beamsearch  返回搜索的结果
            prior_label_score = beam_search_label(prior_label_score=prior_label_score,
                                                  now_score=predict_scores,
                                                  target_size=target_size,
                                                  top_k=top_k,
                                                  unique=True,
                                                  mask_label=mask_label)
        # 中间状态更新
        label_lists = prior_label_score[:, :-1]  # 直接根据 label_list 保证结果正确
        prior_state_images_ = []
        for b in range(batch_size):
            for k in range(top_k):
                now_length = i+1
                if mask_label is not None:
                    ori_label_list = [label-1 for label in label_lists[b*top_k+k]]
                else:
                    ori_label_list = label_lists[b*top_k+k]
                j_state = get_layerout_last_state(ori_labels=ori_label_list,
                                                  furniture_dxs=furniture_dxs[b][: now_length],
                                                  furniture_cids=furniture_cids[b][: now_length],
                                                  furniture_dys=furniture_dys[b][: now_length],
                                                  state_encode=start_state_images[b],
                                                  height=height,
                                                  width=width,
                                                  length=length,
                                                  grid_size=grid_size)
                prior_state_images_.append(j_state)
        prior_state_images = np.array(prior_state_images_)
    return np.reshape(prior_label_score, newshape=[batch_size, top_k, step_length+1])


def mask_state_image(prior_state_images: np.ndarray, furniture_image: np.ndarray):
    """
    :param prior_state_images:  状态
    :param furniture_image:   家具
    :return:
    """
    assert len(prior_state_images) == len(furniture_image)
    prior_state_images_ = []
    for i in range(len(furniture_image)):
        if np.sum(furniture_image[i]) == 0:
            prior_state_images_.append(np.zeros_like(prior_state_images[i]))
        else:
            prior_state_images_.append(prior_state_images[i])
    return np.array(prior_state_images_)


def beam_search_label_test1():
    """
    batch_size = 1 的自测试结果  测试成功
    :return:
    """
    now_score = np.array([[0.3, 0.4, 0.5]])
    a = beam_search_label(prior_label_score=None, now_score=now_score, target_size=3, top_k=2, unique=True, debug=False)
    print("第一次结果，每一个样本选取top_k:\n{0}".format(a))
    now_score2 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    b = beam_search_label(prior_label_score=a, now_score=now_score2, target_size=3, top_k=2, unique=True, debug=False)
    print("第二次结果:\n{0}".format(b))
    now_score3 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    c = beam_search_label(prior_label_score=b, now_score=now_score3, target_size=3, top_k=2, unique=True, debug=False)
    print("第三次结果:\n{0}".format(c))


def beam_search_label_test2():
    """
    batch_size = 3 的自测试结果  测试成功
    :return:
    """
    now_score = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.1, 0.2, 0.3]])
    a = beam_search_label(prior_label_score=None, now_score=now_score, target_size=3, top_k=2, unique=True, debug=True)
    print("第一次结果，每一个样本选取top_k:\n{0}".format(a))
    now_score2 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    b = beam_search_label(prior_label_score=a, now_score=now_score2, target_size=3, top_k=2, unique=True, debug=False)
    print("第二次结果:\n{0}".format(b))
    now_score3 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    c = beam_search_label(prior_label_score=b, now_score=now_score3, target_size=3, top_k=2, unique=True, debug=False)
    print("第三次结果:\n{0}".format(c))


def beam_search_list_label_test1():
    now_score = np.array([[0.3, 0.4, 0.5]])
    prior_score = None
    prior_label_dict = None
    prior_label_dict, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_dict=prior_label_dict, now_score=now_score, top_k=2)
    print("首次beamsearch结果:{0},   {1}".format(prior_label_dict, prior_score))
    now_score2 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    prior_label_dict, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_dict=prior_label_dict, now_score=now_score2, top_k=2, debug=False, unique=True)
    print("第二次beamsearch结果:{0},   {1}".format(prior_label_dict, prior_score))
    now_score3 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    prior_label_dict, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_dict=prior_label_dict, now_score=now_score3, top_k=2, debug=False, unique=True)
    print("第三次beamsearch结果:{0},   {1}".format(prior_label_dict, prior_score))


def beam_search_list_label_test2():
    now_score = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    prior_score = None
    prior_label_dict = None
    prior_label_dict, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_dict=prior_label_dict,
                                                           now_score=now_score, top_k=2)
    print("首次beamsearch结果:{0},   {1}".format(prior_label_dict, prior_score))
    now_score2 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    prior_label_dict, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_dict=prior_label_dict,
                                                           now_score=now_score2, top_k=2, debug=False, unique=True)
    print("第二次beamsearch结果:{0},   {1}".format(prior_label_dict, prior_score))
    now_score3 = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.25, 0.25]])
    prior_label_dict, prior_score = beam_search_list_label(prior_score=prior_score, prior_label_dict=prior_label_dict,
                                                           now_score=now_score3, top_k=2, debug=False, unique=True)
    print("第三次beamsearch结果:{0},   {1}".format(prior_label_dict, prior_score))

