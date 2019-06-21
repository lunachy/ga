#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:   gafp.py
@author: chengyong
@time:   2019/6/18 11:09
@desc:   Genetic Algorithm Framework in Python
"""

from random import sample, random
from itertools import accumulate
from bisect import bisect_right
import pysnooper


class GAFP:
    def __init__(self, population, tournament_size=2, chromosome_length=10, pc=0.5, pm=0.1):
        """
        :param population: Population where the selection operation occurs.
        :type population: list of labels

        :param tournament_size: Individual number in one tournament
        :type tournament_size: int

        :param chromosome_length: length of chromosome.
        :type chromosome_length: int

        :param pc: The probability of crossover (usaully between 0.25 ~ 1.0)
        :type pc: float in (0.0, 1.0]

        :param pe: Gene exchange probability.
        :type pe: float in range (0.0, 1.0]

        :param pm: The probability of mutation (usually between 0.001 ~ 0.1)
        :type pm: float in range (0.0, 1.0]
        """

        assert 0.0 < pc < 1, 'Invalid crossover probability'
        assert 0.0 < pm < 1, 'Invalid mutation probability'
        assert len(population) % 2 == 0, 'Population size must be an even number'

        self.population = population
        self.tournament_size = tournament_size
        self.chromosome_length = chromosome_length
        self.pc = pc
        self.pm = pm

    def tournament_selection(self, fitness_func):
        """
        Select a pair of parent using Tournament strategy.
        :return: Selected parents (a father and a mother)
        """

        # Check validity of tournament size.
        if self.tournament_size >= len(self.population):
            msg = 'Tournament size({}) is larger than population size({})'
            raise ValueError(msg.format(self.tournament_size, len(self.population)))

        # Pick winners of two groups as parent.
        competitors_1 = sample(self.population, self.tournament_size)
        competitors_2 = sample(self.population, self.tournament_size)
        father = max(competitors_1, key=lambda x: fitness_func(x))
        mother = max(competitors_2, key=lambda x: fitness_func(x))

        return father, mother

    # @pysnooper.snoop()
    def roulette_wheel_selection(self, fitness_func):
        """
        Selection operator with fitness proportionate selection(FPS) or
        so-called roulette-wheel selection implementation.
        :return: Selected parents (a father and a mother)
        """
        # Normalize fitness values for all individuals.
        fits = [fitness_func(_p) for _p in self.population]
        min_fit = min(fits)
        fits = [(i - min_fit) for i in fits]

        # Create roulette wheel.
        sum_fit = sum(fits)
        wheel = list(accumulate([i / sum_fit for i in fits]))

        # Select a father and a mother.
        father_idx = bisect_right(wheel, random())
        father = self.population[father_idx]
        mother_idx = (father_idx + 1) % len(wheel)
        mother = self.population[mother_idx]

        return father, mother

    def cross(self, father, mother):
        """
        :return: children
        Cross chromsomes of parent using uniform crossover method.
        """
        # do_cross = True if random() <= self.pc else False
        # if not do_cross:
        #     return father, mother

        # get binary data and extend to fixed length
        chrom1 = [bin(_i)[2:] for _i in father]
        chrom2 = [bin(_i)[2:] for _i in mother]
        for i in range(len(father)):
            chrom1[i] = list('0' * (self.chromosome_length - len(chrom1[i])) + chrom1[i])
            chrom2[i] = list('0' * (self.chromosome_length - len(chrom2[i])) + chrom2[i])

        assert len(chrom1[0]) == len(chrom2[0]) == self.chromosome_length

        # cross --- bit exchange
        for i in range(len(chrom1)):
            for j in range(self.chromosome_length):
                if self.pc > random():
                    chrom1[i][j], chrom2[i][j] = chrom2[i][j], chrom1[i][j]
            # chrom1[i] = ''.join(chrom1[i])
            # chrom2[i] = ''.join(chrom2[i])

        # sample result --> [['0', '0', '0', '1', '1', '1', '0', '0', '1', '0'], ['0', '1', '1', '0', '1', '1', '0', '0', '1', '1']]
        return chrom1, chrom2

    def mutate(self, indv):
        ''' Mutation operator with Flip Bit mutation implementation.

        :param individual: The individual on which crossover operation occurs
        :type individual: :obj:`gaft.components.IndividualBase`

        :return: A mutated individual
        :rtype: :obj:`gaft.components.IndividualBase`
        '''
        for i in range(len(indv)):
            for j in range(self.chromosome_length):
                if self.pm > random():
                    indv[i][j] = str(int(indv[i][j]) ^ 1)
            indv[i] = int(''.join(indv[i]), 2)
        # sample result --> [586, 435]
        return indv


def checkFullPath(a):
    return 0


if __name__ == '__main__':
    populatons = [[114, 435], [833, 967], [83, 567], [133, 17], [173, 117], [113, 197]]
    popu_size = len(populatons)
    gafp = GAFP(populatons)
    local_indvs = []
    while len(local_indvs) < popu_size:
        parents = gafp.tournament_selection(max)
        children = gafp.cross(*parents)
        for child in children:
            mutate_child = gafp.mutate(child)
            if checkFullPath(mutate_child) == 0:
                local_indvs.append(mutate_child)
    local_indvs = local_indvs[:popu_size]
    print(local_indvs)
