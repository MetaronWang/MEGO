from src.problem.base_problem import BaseProblem
from src.types_ import *
import numpy as np

class MatchMaxProblem(BaseProblem):
    def __init__(self, dimension: int = 10, **kwargs):
        self.dimension: int = dimension
        self.match_solution: NpArray = np.random.randint(2, size=dimension)
        super(BaseProblem, self).__init__()

    def evaluate(self, solution: NpArray) -> float:
        return self.dimension - np.sum(np.bitwise_xor(np.array(solution, dtype=np.int32), self.match_solution))


if __name__ == '__main__':
    for _ in range(100):
        p = MatchMaxProblem(10)
        print(p.match_solution)
