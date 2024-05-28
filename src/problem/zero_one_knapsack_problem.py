from src.problem.base_problem import BaseProblem
from src.types_ import *
import numpy as np


class ZeroOneKnapsackProblem(BaseProblem):
    def __init__(self, dimension: int = 10, **kwargs):
        self.dimension: int = dimension
        index_shuffle: NpArray = np.arange(dimension, dtype=np.int32)
        np.random.shuffle(index_shuffle)
        self.weights: NpArray = np.sort(np.random.random(size=dimension))[index_shuffle]
        self.values: NpArray = np.sort(np.random.random(size=dimension))[index_shuffle]
        self.max_weight: float = np.max([np.min(self.weights), (np.random.random() * 0.6 + 0.2) * np.sum(self.weights)])
        max_unit_value: float = float(np.max([self.values[i] / self.weights[i] for i in range(dimension)]))
        super(BaseProblem, self).__init__()

    def evaluate(self, solution: NpArray) -> float:
        quality: float = 0
        remaining_weight: float = self.max_weight
        for index in range(self.dimension):
            if remaining_weight < self.weights[index]:
                break
            if solution[index] == 1:
                remaining_weight -= self.weights[index]
                quality += self.values[index]
        return quality


if __name__ == '__main__':
    for _ in range(100):
        p = ZeroOneKnapsackProblem(10)
        p.evaluate(np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0]))
