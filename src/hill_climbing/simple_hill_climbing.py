import os
import pickle
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.types_ import *


class SimpleHillClimbing:
    def __init__(self, eval_func: Callable[[NpArray], float], candidate_init: List[NpArray] = None,
                 known_x: List[NpArray] = None, max_eval: int = 6400, dim: int = 40):
        self.eval_func: Callable[[NpArray], float] = eval_func
        self.dim: int = dim
        self.max_eval: int = max_eval
        self.candidate_init: List[NpArray] = candidate_init if candidate_init is not None else []
        self.best_y: float = -1e15
        self.best_x: NpArray = None
        self.eval_time = len(known_x) if known_x is not None else 0
        self.quality_table: Dict[str, float] = {}
        known_x.reverse()
        self.quality_step_history: List[float] = [self.eval_func(solution) for solution in known_x]
        self.quality_update_history: List[float] = []

    def get_quality(self, solution: NpArray) -> float:
        if self.eval_time > self.max_eval:
            return -1e15
        solution_str = "".join([str(i) for i in solution])
        if solution_str in self.quality_table:
            quality = self.quality_table[solution_str]
        else:
            quality = self.eval_func(solution)
            self.quality_table[solution_str] = quality
        return quality

    def add_quality_record(self, solution: NpArray, quality: float):
        solution_str = "".join([str(i) for i in solution])
        self.quality_table[solution_str] = quality

    def get_init_solution(self):
        need_eval = False
        if len(self.candidate_init) == 0:
            return_solution = np.random.randint(2, size=self.dim, dtype=np.int32)
            need_eval = True
        else:
            return_solution = np.array(self.candidate_init[0], dtype=np.int32)
            self.candidate_init = self.candidate_init[1:]
        return_quality = self.get_quality(return_solution)
        self.quality_update_history.append(return_quality)
        if return_quality > self.best_y:
            self.best_y = return_quality
            self.best_x = return_solution
        if need_eval:
            self.eval_time += 1
            self.quality_step_history.append(return_quality)
        return return_solution, return_quality

    def run(self):
        np.random.seed(int(time.time() * 1000) % 10000)
        random.seed(int(time.time() * 1000) % 10000)
        while self.eval_time < self.max_eval:
            current_solution, current_quality = self.get_init_solution()
            while True:
                candidate_solutions = [deepcopy(current_solution) for index in range(self.dim)]
                candidate_qualities = []
                for index in range(self.dim):
                    candidate_solutions[index][index] = current_solution[index] ^ 1
                    quality = self.get_quality(candidate_solutions[index])
                    self.eval_time += 1
                    self.quality_step_history.append(quality)
                    candidate_qualities.append(quality)
                max_index = np.argmax(candidate_qualities)
                if candidate_qualities[max_index] > current_quality:
                    current_solution, current_quality = candidate_solutions[max_index], candidate_qualities[max_index]
                    self.quality_update_history.append(candidate_qualities[max_index])
                    if candidate_qualities[max_index] >= self.best_y:
                        self.best_y = candidate_qualities[max_index]
                        self.best_x = candidate_solutions[max_index]
                    # print(current_quality, self.best_y)
                else:
                    break


if __name__ == '__main__':
    os.chdir(Path(os.path.dirname(os.path.abspath(__file__)), "../"))
    problem = pickle.load(open("../data/problem_instance/valid/compiler_args_selection_problem_5_60/problem.pkl", "rb"))
    simple_hill_climbing = SimpleHillClimbing(eval_func=problem.evaluate, dim=problem.dimension,
                                              candidate_init=[[1 for _ in range(60)], [0 for _ in range(60)]])
    simple_hill_climbing.run()
