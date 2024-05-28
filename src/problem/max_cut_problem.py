import random
import networkx as nx
from src.problem.base_problem import BaseProblem
from src.types_ import *


class MaxCutProblem(BaseProblem):
    '''
    Generate the connected grpah by networkX. The #edge is randomly chosen in range from 0.2*D^2 to 0.4*D^2, and
    the minimal value of it is D+1, The max size of the cut set is randomly chosen in range from 0.2*D to 0.4*D
    '''

    def __init__(self, dimension: int = 10):
        self.dimension: int = dimension
        self.max_k: int = random.randint(int(self.dimension * 0.2), int(self.dimension * 0.4))
        self.edge_num: int = int(max(self.dimension + 1, self.dimension ** 2 * (random.random() * 0.2 + 0.2)))
        self.network: Graph = nx.generators.random_graphs.dense_gnm_random_graph(self.dimension, self.edge_num)
        while not nx.is_connected(self.network):
            self.network: Graph = nx.generators.random_graphs.dense_gnm_random_graph(self.dimension, self.edge_num)
        super(BaseProblem).__init__()

    def evaluate(self, solution: NpArray) -> float:
        S: Set[int] = set()
        T: Set[int] = set()
        for index in range(self.dimension):
            if solution[index] == 1 and len(S) < self.max_k:
                S.add(index)
            else:
                T.add(index)
        return nx.cut_size(self.network, S, T)


if __name__ == '__main__':
    for _ in range(10):
        p = MaxCutProblem(dimension=10)
        for _ in range(20):
            s = np.random.randint(2, size=10)
            print(s, p.evaluate(s))
