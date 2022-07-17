import sys
import numpy as np

class Scorer():
    def compute(self, probs):
        raise NotImplementedError

class Energy(Scorer):
    def compute(self, probs):
        return np.sum(probs ** 2, axis=-1)

class Entropy(Scorer):
    def compute(self, probs):
        return -np.sum(probs * np.log(np.clip(probs, sys.float_info.epsilon, 1.0)), axis=-1)

class Consistency(Scorer):
    def compute(self, probs):
        return np.sum(np.var(probs, axis=-2), axis=-1)

class Consensus(Scorer):
    def compute(self, probs):
        return -np.max(np.mean(probs, axis=-2), axis=-1)

