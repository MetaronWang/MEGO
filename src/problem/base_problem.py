from src.types_ import *


class BaseProblem():
    def __init__(self, dimension: int = 10, **kwargs):
        self.dimension = dimension
        super(BaseProblem).__init__()

    def evaluate(self, solution: Union[NpArray, List[int]]) -> float:
        raise NotImplementedError
