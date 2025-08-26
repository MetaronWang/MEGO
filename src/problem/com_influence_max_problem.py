import random
import sys
import os
import time

from src.problem.base_problem import BaseProblem
from os import path
from pathlib import Path
from types_ import *

sys.path.append(str(Path(path.dirname(path.abspath(__file__)), "../../lib/com_imp")))

import ComIMP

_root_dir = Path(path.dirname(path.abspath(__file__)), "../../data/dataset/com_imp")

dataset_dir_dic: Dict[str, Tuple[Path, str, int]] = {
    "Wiki": (Path(_root_dir, "Wiki"), "Wiki_WC", 7115),
    "Epinions": (Path(_root_dir, "Epinions"), "Epinions_WC", 75879),  # 1.146s, 10000, 32
    "LiveJournal": (Path(_root_dir, "LiveJournal"), "LiveJournal_WC", 4847571),  # 66s, 10000, 32
    "Twitter": (Path(_root_dir, "Twitter"), "Twitter_WC", 41652230),  # 973.74s, 1000. 8
    "Facebook": (Path(_root_dir, "Facebook"), "Facebook_WC", 4039),  # 0.20s, 10000, 32
    "Flixster": (Path(_root_dir, "Flixster"), "Flixster_WC", 2523386),  # 23.9s, 10000, 32
}

dataset_directed_dic: Dict[str, bool] = {
    "Wiki": True,
    "Epinions": True,
    "LiveJournal": True,
    "Twitter": True,
    "Facebook": False,
    "Flixster": False,
}

candidate_gpas: List[List[float]] = [
    [0.5, 0.75, 0.5, 0.75],
    [0.5, 0.25, 0.5, 0.25]
]


def get_seed_path(dataset_name: str, seed_name: str) -> str:
    graph_dir = dataset_dir_dic[dataset_name][0]
    seed_path = os.path.join(graph_dir, "seed")
    seed_path = os.path.join(seed_path, seed_name)
    # seed_path = dataset_dir_dic[dataset_name][0] + 'seed/' + seed_name
    return seed_path


def load_seeds(seed_path: str) -> list:
    seeds = []
    with open(seed_path, 'r') as f:
        for line in f:
            seeds.append(int(line.strip()))
    return seeds


class ComInfluenceMaxProblem(BaseProblem):
    def __init__(self, dimension: int, sample_num: int = 1000, thread_num: int = 32,
                 **kwargs):
        super().__init__(dimension, **kwargs)
        self.dimension: int = dimension
        self.dataset: str = np.random.choice(["Wiki", "Facebook"])
        self.graph_name: str = dataset_dir_dic[self.dataset][1]
        self.graph_dir: str = str(dataset_dir_dic[self.dataset][0]) + "/"
        self.directed: bool = dataset_directed_dic[self.dataset]
        self.node_num: int = dataset_dir_dic[self.dataset][2]
        self.max_k: int = random.randint(int(self.dimension * 0.2), int(self.dimension * 0.6))  # max_k
        self.gap: List[float] = candidate_gpas[0] if random.random() < 0.5 else candidate_gpas[1]
        self.sample_num: int = sample_num
        self.thread_num: int = thread_num
        self.seeds_b: List[int] = list(np.random.choice(range(self.node_num), size=self.max_k, replace=False))
        self.candidate_seeds: List[int] = list(
            np.random.choice(range(self.node_num), size=self.dimension, replace=False))

    def evaluate(self, solution: Union[NpArray, List[int]]):
        seeds: List[int] = []
        for index in range(self.dimension):
            if len(seeds) >= self.max_k:
                break
            if solution[index] == 1:
                seeds.append(self.candidate_seeds[index])
        strategy = ComIMP.Strategy(self.graph_dir, self.graph_name, self.directed, self.gap,
                                   self.seeds_b, self.sample_num, self.thread_num, True)
        com_exp = strategy.evaluate(seeds)
        return com_exp


if __name__ == '__main__':
    problem = ComInfluenceMaxProblem(50)
    print(problem.dataset)
    for _ in range(100):
        solution = np.random.randint(2, size=50)
        c1 = problem.evaluate(solution)
        c2 = problem.evaluate(solution)
        print(c1, c2, (c1-c2)/c2)
