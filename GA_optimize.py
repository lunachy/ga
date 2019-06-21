# coding=utf-8
'''
输入一个房间的基因型，生成最优布局topk
'''
from random import randint, random
import numpy as np
import random
import sys
from operator import add
from functools import reduce
from checkCrash.CrashChecker import CrashChecker
from Evalutor_Model import evalutor_predict
from show_img import show_img
from Layout_Service.graph.APIForRpcServer import LayoutServerAPI


class GA_optimize():
    number = 0

    def __init__(self, gene, zid, room_str, grid_min, grid_max, min_d, max_d, count, target, evolution_num):
        self.gene = gene  # 输入功能区基因型
        self.zid = zid  # 输入功能区id
        self.room_str = room_str  # 房间户型图信息
        self.grid_min = grid_min  # 格子编号 划分成16 *16 格子
        self.grid_max = grid_max
        self.min_d = min_d  # 方向编号
        self.max_d = max_d
        self.count = count  # 种群大小
        self.target = target  # 目标分数
        self.evolution_num = evolution_num  # 进化代数

    def population(self):

        configPath = "./checkCrash/config.props"
        layout_model = LayoutServerAPI(configPath=configPath)
        model_path = "./Layout_Service/model/model-55"
        layout_model.intModelTrans(model_path=model_path)
        pop_layouts = layout_model.predictAPI(self.room_str, top_k=30, room_type_id=1, model_type=0)
        print('pop_layouts:', pop_layouts)

        pop_genes = []
        pop_imgs = []
        for lay in pop_layouts:
            check = ck.check_data(lay)
            if check[0]:
                print(1111)
                img = check[-1]['0']['mid_states'][-1][0]
                gene = [get_position_rotate(one_lay) for one_lay in lay]
                pop_imgs.append(img)
                pop_genes.append(gene)

        print('len(pop_genes)', len(pop_genes))
        return pop_genes, pop_imgs

    def fitness(self, individual_img, target):
        print('打分器的输入中。。。')
        # print('individual_shape_0:',np.array(individual_img[0]).shape)
        individual_img = np.array(individual_img).reshape([-1, 64, 64])
        score = evalutor_predict.predict(individual_img)
        fit = list(map(lambda x: abs(target - x), score))
        return fit

    def grade(self, pop_img, target):

        print('第{}代种群生成评分列表中。。。'.format(self.number + 1))
        fit_list = self.fitness(pop_img, target)
        summed = reduce(add, fit_list, 0)
        grade_fitness = summed / len(fit_list)
        grade_fitness = round(grade_fitness, 2)

        return grade_fitness, fit_list

    def evolve(self, pop_gene, pop_img, target, retain=0.8, random_select=0.05, mutate=0.1):
        # 原始种群的相关信息
        print('第{}代种群进化中。。。'.format(self.number + 1))
        # parents
        # 种群进化的分数列表
        fit_list = self.grade(pop_img, target)[-1]
        # 将生成的childern加入到种群信息中
        graded_all = [[fit_list[i], pop_gene[i], np.array(pop_img[i]).tolist()] for i in range(len(pop_gene))]
        print('graded_all finished!!')
        # 排序后的基因型 genes
        graded = [x[1] for x in sorted(graded_all)]
        # 排序后的img信息
        graded_img = [x[2] for x in sorted(graded_all)]
        # 按照retain比例保留 —> parents
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]
        # parents img 信息
        parents_graded_img = graded_img[:retain_length]

        # promote genetic diversity( randomly add other individuals)
        for individual in graded[retain_length:]:
            if random_select > random.random():
                index = graded[retain_length:].index(individual)
                parents_graded_img.append(graded_img[retain_length:][index])
                parents.append(individual)

        # 随机突变 parents 中一些个体
        for individual in parents:
            individual_1 = individual

            if mutate > random.random():
                index_indiv = parents.index(individual)
                count_ = 0
                while True:
                    print('开始突变......')
                    # 选择功能区突变位置
                    pos_to_mutate = randint(0, len(individual_1) - 1)
                    print('突变前', individual_1)  # 突变前

                    ########方案四：对功能区位置的偏移##########
                    # start mutate #

                    if count_ > 50:
                        print(count_)
                        print('超过阈值，未能生成突变个体。')
                        break

                    # 突变一个功能区
                    elif pos_to_mutate == 0:
                        # 偏移20个单位以内
                        range_ = randint(1, 16)
                        # 选择突变一个功能区
                        mutate_list = individual_1[pos_to_mutate]
                        x = randint(0, 2)
                        if x == 2:
                            mutate_list[x] = randint(self.min_d, self.max_d)
                        # 一个功能区的某个位点
                        else:
                            mutate_list[x] = abs(mutate_list[x] - range_)

                        print('一个位点突变后：', individual_1)  # 突变后
                        print('进行碰撞检测。。。')

                    ## 突变两个功能区
                    elif pos_to_mutate == 1:
                        # 偏移20个单位以内
                        range_ = randint(1, 16)
                        mutate_list = [random.choice(individual_1) for _ in range(2)]
                        x = randint(0, 2)
                        if x == 2:
                            mutate_list[0][x] = randint(self.min_d, self.max_d)
                            mutate_list[1][x] = randint(self.min_d, self.max_d)
                        else:
                            mutate_list[0][x] = abs(mutate_list[0][x] - range_)
                            mutate_list[1][x] = abs(mutate_list[1][x] - range_)
                        print('两位点突变后：', individual_1)  # 突变后
                        print('进行碰撞检测。。。')
                    else:
                        # 超出设定的三个功能区
                        print('*' * 10, '功能区选择超过设定阈值，重新选择突变位点', '*' * 10)
                        continue

                    # end mutate #

                    ###加入碰撞检测###
                    child_img = self.room_check(individual_1)
                    child_img = np.array(child_img).reshape([64, 64])
                    count_ += 1
                    if child_img.tolist() == []:
                        print('*' * 20, '有碰撞重新突变生成个体...', '*' * 20)
                    else:
                        parents[index_indiv] = individual_1
                        parents_graded_img[index_indiv] = child_img
                        print('#' * 10, '突变个体已生成:{}'.format(individual_1), '#' * 10)
                        break

        # crossover
        parents_length = len(parents)
        desired_length = count - parents_length
        children = []
        children_img = []
        iter = 0
        while len(children) < desired_length:
            # 从parents genes中挑选 父本gene索引 母本gene索引
            male = randint(0, parents_length - 1)
            female = randint(0, parents_length - 1)
            if male != female:
                # 父本母本genes的选择
                male = parents[male]
                female = parents[female]
                print('male:', male)
                print('female:', female)
                # crossover  产生子代（父本母本各0.5继承）
                half = int(len(male) / 2)
                # child 基因型
                child = male[:half] + female[half:]
                print('child:', child)

                ## 产生的child调用碰撞检测 ##
                child_img = self.room_check(child)
                # child_img = np.array(child_img).reshape([-1, 64, 64])
                iter += 1
                print('第{0}代进化第{1}次产生子代'.format(self.number + 1, iter))
                if iter == 20:
                    print('子代产生已达到循环上线！！')
                    break
                elif child_img.tolist() == []:
                    print('*' * 20, 'crossover有碰撞，重新生成子代...', '*' * 20)
                    continue
                children.append(child)
                children_img.append(child_img)
        parents.extend(children)
        parents_graded_img.extend(children_img)
        # print('parents',parents)
        self.number += 1
        print('parents已生成！第{}代种群进化完成'.format(self.number + 1))
        return parents, parents_graded_img

    def evolution_fitness(self):
        generation_num = 0
        evolution_list = []
        pop_layouts = []
        pop_genes, pop_imgs = self.population()
        fitness_history = []
        for i in range(evolution_num):
            # pop 进化后的种群基因型genes(变异，交叉，择优)
            pop_genes, pop_imgs = self.evolve(pop_genes, pop_imgs, target)
            print("第%s代进化完成。" % (i + 1))
            fitness_history.append(self.grade(pop_imgs, target)[0])

        _, fit_list = self.grade(pop_imgs, target)
        # evolution_population = pop_genes
        # 对最后一代进行排序 取前十
        evolution_fit, evolution_img, evolution_population = evolution_grad(pop_genes, fit_list, pop_imgs, 10)

        pop_layouts = [[caculate_position_index(lay[0], lay[1], lay[2]) for lay in one] for one in
                       evolution_population]

        # duplicate removal of pop_layouts
        _pop_layouts = []
        for pop_layout in pop_layouts:
            if pop_layout not in _pop_layouts:
                _pop_layouts.append(pop_layout)
        pop_layouts = _pop_layouts

        print('种群进化后前10分数:\n', list(map(lambda x: round(abs(self.target - x), 2), evolution_fit)))
        print('种群进化后:\n', pop_layouts)

        # 图片显示
        evolution_img = np.array(evolution_img).reshape([-1, 64, 64])
        for i in range(evolution_img.shape[0]):
            show_img(evolution_img[i], 'score:{}'.format(round(abs(self.target - evolution_fit[i]), 2)))

        for evolution_fitness in fitness_history:
            evolution_list.append(evolution_fitness)
        print('进化适应度为:\n', evolution_list)
        self.number += 1
        return evolution_list, pop_layouts

    # 获取房间的信息 layout,zid,dx,dy,img 并碰撞检测 并输出img信息
    def room_check(self, room):

        layout = [caculate_position_index(room[i][0], room[i][1], room[i][2], 16) for i in range(len(self.zid))]
        print('碰撞检测中。。。')
        check = ck.check_data(layout)
        if check[0] == False:
            return []
        elif check[0] == True:
            img = check[-1]['0']['mid_states'][-1][0]
            print('*' * 20, '已生成新个体！！！！', '*' * 20)
            return img


# 使其输入分数 基因型 进行排列生成前n个图
def evolution_grad(gene, score, img, n):
    print(len(gene), len(score), len(img))
    print(type(gene), type(score), type(img))
    print(gene[0], score[0])
    grad = [[score[i], gene[i], np.array(img[i]).tolist()] for i in range(len(gene))]
    grad_ = []
    # 去重
    for i in grad:
        if i not in grad_:
            grad_.append(i)

    graded_all = sorted(grad_)
    grad_score = [round(x[0], 2) for x in graded_all]
    grad_gene = [x[1] for x in graded_all]
    grad_img = [x[2] for x in graded_all]
    # 取前n个
    grad_score = grad_score[:n]
    grad_img = grad_img[:n]

    return grad_score, grad_img, grad_gene


# 计算位置索引
def caculate_position_index(cen_x, cen_y, rot, size=16):
    position_index = cen_x + cen_y * size + rot * size * size
    return position_index


def get_position_rotate(position_index, size=16):
    # 方向 0,1,2,3 分别 0，90,180,270
    rot = position_index // (size * size)
    ij = position_index % (size * size)
    cen_y = ij // size
    cen_x = ij % size
    return [cen_x, cen_y, rot]


if __name__ == "__main__":
    # 格子编号16*16
    grid_min = 0
    grid_max = 16
    # 方向编号 1-4
    min_d = 0
    max_d = 3
    # 生成 种群大小
    count = 100
    # 目标分数
    target = 2
    # 进化的代数
    evolution_num = 5

    #######test_model:一个房间的基因型########
    import pandas as pd
    import json

    path = './data/GA_data.csv'
    df = pd.read_csv(path, header=0)
    # n = 20,49,53
    ck = CrashChecker("./checkCrash/config.props")
    n = 0
    top1_list, top4_list, top8_list = [], [], []
    # for n in range(len(df)):
    zid = json.loads(df.zids.iloc[n])
    gene = json.loads(df.genes.iloc[n])
    layout = json.loads(df.layouts.iloc[n])
    room_str = json.dumps(json.loads(df.room_json.iloc[n]))

    ck.init_room(roomTypeId=1, room_str=room_str)
    print('ck.data_infos:', ck.data_infos)
    print('*' * 20)
    data_infos = ck.data_infos
    sample_layout = []
    tag_list = []
    for data in data_infos:
        zid_ = data['zid']
        tag_ = data['tag']
        label_ = data['label']
        if zid_ in [98, 99]:
            continue
        tag_list.append(tag_)
        sample_layout.append(label_)
    print('sample_layout: ', sample_layout)

    check = ck.check_data(sample_layout)
    print('*334')  # check = [True, {'0': {'tags': [0, 99],
    if check[0] == False:
        print('输入样本有碰撞{}。'.format(n))
    elif check[0] == True:
        img = check[-1]['0']['mid_states'][-1][0]
        show_img(img, 'true_Sample:{},{},{}'.format(zid, sample_layout, n))
        print(evalutor_predict.predict(img))
        sys.exit()

    print('layout', layout)
    check = ck.check_data(layout)
    if check[0] == False:
        print('输入样本有碰撞{}。'.format(n))
    elif check[0] == True:
        img = check[-1]['0']['mid_states'][-1][0]
        show_img(img, 'test_Sample:{},{}'.format(zid, layout, n))

    # sys.exit()
    #################################################################
    test = GA_optimize(gene, zid, room_str, grid_min, grid_max, min_d, max_d, count, target, evolution_num)
    evolution_list, pop_layouts = test.evolution_fitness()

    for one in pop_layouts[:1]:
        if one == sample_layout:
            top1_list.append(1)
            break
    top1_list.append(0)

    for one in pop_layouts[:4]:
        if one == sample_layout:
            top4_list.append(1)
            break
    top4_list.append(0)

    for one in pop_layouts[:8]:
        if one == sample_layout:
            top8_list.append(1)
            break
    top8_list.append(0)

    print('top1:', top1_list)
    print('top4:', top4_list)
    print('top8:', top8_list)
