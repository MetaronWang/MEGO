import random
from copy import deepcopy

import numpy as np

from src.types_ import *


class SimpleBitGA():
    def __init__(self, eval_func: Callable[[NpArray], float], pop_size: int = 32, mut_prob: float = 0.001,
                 dim: int = 30, max_iter: int = 50, max_eval: int = 500, elite=True, elite_k: int = None):
        self.eval_func: Callable[[NpArray], float] = eval_func
        self.pop_size: int = pop_size
        self.mut_prob: float = mut_prob
        self.dim: int = dim
        self.max_iter: int = max_iter
        self.max_eval: int = max_eval
        self.elite = elite
        self.elite_k = elite_k if elite_k is not None else pop_size
        self.population: List[NpArray] = []
        self.population_strs: Set[str] = set()
        self.population_quality: List[float] = []
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.population_history: List[List[float]] = []
        self.step_history: List[float] = []

    def get_quality(self, solution: NpArray) -> float:
        solution_str = "".join([str(i) for i in solution])
        if self.eval_time > self.max_eval:
            return -1e15
        self.eval_time += 1
        if solution_str in self.quality_table:
            quality = self.quality_table[solution_str]
        else:
            quality = self.eval_func(solution)
            self.quality_table[solution_str] = quality
        self.step_history.append(quality)
        return quality

    def add_quality_record(self, solution: NpArray, quality: float):
        solution_str = "".join([str(i) for i in solution])
        self.quality_table[solution_str] = quality

    def add_solution_to_population(self, solution) -> bool:
        solution_str = "".join([str(i) for i in solution])
        if solution_str not in self.population_strs:
            self.population.append(solution)
            self.population_strs.add(solution_str)
            quality = self.get_quality(solution)
            self.population_quality.append(quality)
            self.add_quality_record(solution, quality)
            return True
        else:
            return False

    def update_population(self, new_population: NpArray):
        if not self.elite:
            for index in range(min(self.pop_size, len(new_population))):
                new_solution = new_population[index]
                new_solution_str = "".join([str(i) for i in new_solution])
                if new_solution_str not in self.population_strs:
                    old_solution_str = "".join([str(i) for i in self.population[index]])
                    self.population[index] = new_solution
                    self.population_strs.remove(old_solution_str)
                    self.population_strs.add(new_solution_str)
                    quality = self.get_quality(new_solution)
                    self.population_quality[index] = quality
                    self.add_quality_record(new_solution, quality)
        else:
            all_solution = deepcopy(self.population)
            all_quality = deepcopy(self.population_quality)
            all_population_str = deepcopy(self.population_strs)
            for new_solution in new_population:
                new_solution_str = "".join([str(i) for i in new_solution])
                if new_solution_str not in all_population_str:
                    all_solution.append(new_solution)
                    quality = self.get_quality(new_solution)
                    self.add_quality_record(new_solution, quality)
                    all_quality.append(quality)
                    all_population_str.add(new_solution_str)
            top_k_index = set(list(np.argsort(all_quality)[::-1][:self.elite_k]))
            while len(top_k_index) < self.pop_size:
                top_k_index.add(random.randint(0, len(all_solution) - 1))
            top_k_index = list(top_k_index)
            self.population = [all_solution[index] for index in top_k_index]
            self.population_quality = [all_quality[index] for index in top_k_index]
        shuffle_index = list(range(self.pop_size))
        random.shuffle(shuffle_index)
        self.population = [self.population[index] for index in shuffle_index]
        self.population_quality = [self.population_quality[index] for index in shuffle_index]
        self.population_strs = {"".join([str(i) for i in solution]) for solution in self.population}

    def initial_population(self, initial_solutions: List[NpArray]):
        self.population: List[NpArray] = []
        self.population_strs: Set[str] = set()
        for index in range(min(len(initial_solutions), self.pop_size)):
            self.add_solution_to_population(initial_solutions[index])
        while len(self.population) < self.pop_size:
            self.add_solution_to_population(np.random.randint(2, size=self.dim))
        shuffle_index = list(range(self.pop_size))
        random.shuffle(shuffle_index)
        self.population = [self.population[index] for index in shuffle_index]
        self.population_quality = [self.population_quality[index] for index in shuffle_index]
        self.population_history.append(deepcopy(self.population_quality))

    def mutation(self):
        mask = (np.random.rand(self.pop_size, self.dim) < self.mut_prob)
        temp = np.array(self.population)
        temp ^= mask
        return temp

    def crossover(self):
        new_population = []
        for _ in range(self.pop_size // 2):
            i_1, i_2 = np.random.randint(0, self.pop_size, 2)
            n1, n2 = np.random.randint(0, self.dim, 2)
            if n1 > n2:
                n1, n2 = n2, n1
            p1, p2 = self.population[i_1].copy(), self.population[i_2].copy()
            seg1, seg2 = p1[n1:n2].copy(), p2[n1:n2].copy()
            p1[n1:n2], p2[n1:n2] = seg2, seg1
            new_population.append(p1)
            new_population.append(p2)
        return np.array(new_population)

    def run(self):
        for _ in range(self.max_iter):
            if self.eval_time > self.max_eval:
                break
            self.update_population(self.mutation())
            self.update_population(self.crossover())
            self.population_history.append(deepcopy(self.population_quality))
            if self.eval_time > self.max_eval:
                break
        best_x = max(self.quality_table.keys(), key=lambda item: self.quality_table[item])
        best_y = self.quality_table[best_x]
        return best_x, best_y
