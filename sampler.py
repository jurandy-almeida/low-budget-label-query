from __future__ import print_function
import os
import os.path
import numpy as np

import torch
from torch import nn

import torchvision.transforms as transforms

from heapq import heapify, heappop 

from scorer import Scorer

class Sampler():
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def select(self):
        raise NotImplementedError


class Random(Sampler):
    def __init__(self, ratio=0.1):
        super(Random, self).__init__(ratio=ratio)

    def select(self, size):
        sample_size  = int(self.ratio * size)
        selected_indices = np.random.choice(size, size=sample_size, replace=False)
        return selected_indices


class Score(Sampler):
    def __init__(self, scorer, ratio=0.1):
        super(Score, self).__init__(ratio=ratio)

        if not isinstance(scorer, Scorer):
             raise RuntimeError('A scorer must be provided')

        self.scorer = scorer

    def select(self):
        raise NotImplementedError

class TopRank(Score):
    def __init__(self, scorer, ratio=0.1):
        super(TopRank, self).__init__(scorer=scorer, ratio=ratio)

    def select(self, probs):
        size = probs.shape[0]
        sample_size  = int(self.ratio * size)

        scores = -self.scorer.compute(probs)

        selected_indices = np.argsort(scores)[:sample_size]
        return selected_indices


class Uniform(Score):
    def __init__(self, scorer, ratio=0.1):
        super(Uniform, self).__init__(scorer=scorer, ratio=ratio)

    def select(self, probs):
        size = probs.shape[0]
        sample_size  = int(self.ratio * size)

        scores = self.scorer.compute(probs)

        minval = scores.min()
        maxval = scores.max()
        intervals = np.arange(minval, maxval, (maxval - minval) / sample_size)
        bins = np.digitize(scores, intervals)

        unique_bins = np.unique(bins)

        pq_intra = dict()
        for i in unique_bins:
            pq_intra[i] = []

        for i in unique_bins:
            bin_indices = np.where(bins == i)[0]
            bin_scores = scores[bin_indices]
            pq_intra[i].extend(zip(bin_scores, bin_indices))

        for i in unique_bins:
            heapify(pq_intra[i])

        selected_indices = []
        while len(selected_indices) < sample_size:
            pq_inter = []
            for i in unique_bins:
                if len(pq_intra[i]) > 0:
                    pq_inter.append(heappop(pq_intra[i]))

            if len(selected_indices) + len(pq_inter) < sample_size:
                selected_indices.extend(list(list(zip(*pq_inter))[1]))
            else:
                heapify(pq_inter)
                while len(selected_indices) < sample_size:
                    score, index = heappop(pq_inter)
                    selected_indices.append(index)

        return selected_indices

