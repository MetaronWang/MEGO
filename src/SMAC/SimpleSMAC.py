import os
import pickle
import random
import time
from pathlib import Path

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.initial_design.sobol_design import SobolInitialDesign

from src.problem.base_problem import BaseProblem
from src.types_ import *


class MyBitProblem():
    def __init__(self, eval_func: Callable[[NpArray], float], dim: int, max_eval: int, init_solutions: List):
        self.eval_func: Callable[[NpArray], float] = eval_func
        self.max_eval: int = max_eval
        self.eval_time: int = 0
        self.dim: int = dim
        self.init_solutions = init_solutions
        self.step_history = []
        self.init_solution_strs = set()
        self.quality_table: Dict[str, float] = {}
        self.init()

    def init(self):
        for solution in self.init_solutions:
            solution_str = "".join([str(i) for i in solution])
            eval_value = self.eval_func(solution)
            self.quality_table[solution_str] = eval_value
            self.init_solution_strs.add(solution_str)
            self.eval_time += 1
            self.step_history.append(eval_value)

    def evaluate(self, config: Configuration, seed: int = 0):
        solution = np.array([config[str(index)] for index in range(self.dim)], dtype=np.int32)
        solution_str = "".join([str(i) for i in solution])
        if self.eval_time >= self.max_eval:
            eval_value = -1e15
        else:
            if solution_str in self.quality_table.keys():
                eval_value = self.quality_table[solution_str]
            else:
                eval_value = self.eval_func(solution)
                self.quality_table[solution_str] = eval_value
        if solution_str not in self.init_solution_strs:
            self.eval_time += 1
            self.step_history.append(eval_value)
        else:
            self.init_solution_strs.remove(solution_str)
        return -eval_value


def create_SMAC(problem: BaseProblem, max_eval: int, initial_solutions: List):
    my_problem = MyBitProblem(eval_func=problem.evaluate, dim=problem.dimension, max_eval=max_eval,
                              init_solutions=initial_solutions)
    configspace = ConfigurationSpace(space={str(index): [0, 1] for index in range(problem.dimension)})
    np.random.seed(int(time.time() * 1000) % 10000)
    random.seed(int(time.time() * 1000) % 10000)
    scenario = Scenario(configspace, n_trials=max_eval, seed=int(time.time() * 1000) % 10000)
    configs = [Configuration(
        configuration_space=configspace,
        values={str(index): int(solution[index]) for index in range(problem.dimension)},
        origin='Initial Solution'
    ) for solution in initial_solutions]
    smac = HPOFacade(scenario, my_problem.evaluate, overwrite=True, initial_design=SobolInitialDesign(
        scenario=scenario,
        n_configs=None,
        n_configs_per_hyperparameter=10,
        max_ratio=0.25 if len(initial_solutions) == 0 else 0,
        additional_configs=configs,
    ), logging_level=50)
    incumbent = smac.optimize()
    x = [int(incumbent[str(index)]) for index in range(problem.dimension)]
    return x, my_problem


if __name__ == '__main__':
    os.chdir(Path(os.path.dirname(os.path.abspath(__file__)), "../"))
    problem_instance = pickle.load(
        open("../data/problem_instance/valid/anchor_selection_problem_11_100/problem.pkl", "rb"))
    solutions = [np.random.randint(2, size=100, dtype=np.int32) for _ in range(10)]
    x, my_problem = create_SMAC(problem=problem_instance, max_eval=800, initial_solutions=[])
    print(x)
    print(my_problem.step_history)
    print(problem_instance.evaluate(x))
    print(max(my_problem.step_history))
    print(len(my_problem.quality_table))
    print(my_problem.eval_time)
