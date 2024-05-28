import datetime
import logging
import os
import pickle
import random
import time
from copy import deepcopy
from multiprocessing import Manager, Process
from pathlib import Path

from experiment_problem import load_problem_instance
from src.SMARTEST.SMARTEST import SMARTEST
from src.problem.compiler_args_selectionProblem import CompilerArgsSelectionProblem
from types_ import *


def start_SMARTEST(problem_root_dir: str, target_instance_path: str, pop_size: int = 100, crossover_prob: float = 0.8,
                   max_eval: int = 800, elite_rate: float = 0.1):
    target_root_dir = Path(problem_root_dir, "valid", target_instance_path)
    problem_instance: CompilerArgsSelectionProblem = load_problem_instance(problem_dir=target_root_dir)
    np.random.seed(int(time.time() * 1000) % 10000)
    random.seed(int(time.time() * 1000) % 10000)
    smartest = SMARTEST(eval_func=problem_instance.evaluate, pop_size=pop_size, crossover_prob=crossover_prob,
                        dim=problem_instance.dimension, max_eval=max_eval, elite_rate=elite_rate)
    best_x, best_y = smartest.run()
    pop_history: List[List[float]] = [population_quality for population_quality in smartest.population_history]
    return best_x, best_y, pop_history, smartest.eval_time, smartest.step_history


def start_SMARTEST_task(result_dict, problem_root_dir: str, target_instance_path: str, task_index: int = 0,
                        pop_sizes=None, crossover_prob: float = 0.8, max_eval: int = 800, elite_rate: float = 0.1):
    if pop_sizes is None:
        pop_sizes = [32, 100]
    result: Dict[str, Dict[str, Union[NpArray, float]]] = {}
    for pop_size in pop_sizes:
        x, y, pop_history, eval_time, step_history = start_SMARTEST(problem_root_dir=problem_root_dir,
                                                                    target_instance_path=target_instance_path,
                                                                    pop_size=pop_size, crossover_prob=crossover_prob,
                                                                    max_eval=max_eval, elite_rate=elite_rate
                                                                    )
        result["smartest_{}".format(pop_size)] = {
            "x": list(x),
            "y": y,
            "pop_history": pop_history,
            "step_history": step_history,
            "eval_time": eval_time
        }
    result_dict["{}_{}".format(target_instance_path, task_index)] = deepcopy(result)
    print("Finish", target_instance_path, task_index, datetime.datetime.now())


def hold_best(sequence: List[Union[int, float]]) -> List[Union[int, float]]:
    new_sequence = [sequence[0]]
    for i in range(1, len(sequence)):
        new_sequence.append(max(new_sequence[i - 1], sequence[i]))
    return new_sequence


def statisitc_SMARTEST(max_eval: int = 800, pop_sizes: int = None, max_cut: int = 800):
    if pop_sizes is None:
        pop_sizes = [32, 100]
    log_path = Path("../SMARTEST_logs", "result_{}.pkl".format(max_eval))
    all_result = pickle.load(open(log_path, "rb"))
    problem_list = set(["_".join(key.split("_")[:-1]) for key in all_result.keys()])
    for target_instance_path in problem_list:
        print(target_instance_path, end="\t")
        results = [all_result["{}_{}".format(target_instance_path, task_index)] for task_index in
                   range(30)]
        for pop_size in pop_sizes:
            smartest_data = [max(result["smartest_{}".format(pop_size)]["step_history"][:max_cut]) for result in
                             results]
            print("{:.2f}±{:.2f}".format(np.mean(smartest_data), np.std(smartest_data)), end="\t")
        print()


def run_SMARTEST_multi(problem_root_dir: str = "../data/problem_instance", pop_sizes: int = None,
                       crossover_prob: float = 0.8, max_eval: int = 800, elite_rate: float = 0.1):
    if pop_sizes is None:
        pop_sizes = [32, 100]
    if not os.path.exists(Path("../SMARTEST_logs")):
        os.mkdir(Path("../SMARTEST_logs"))
    log_path = Path("../SMARTEST_logs", "result_{}.pkl".format(max_eval))
    last_result = pickle.load(open(log_path, "rb")) if os.path.exists(log_path) else {}
    manager = Manager()
    all_result = manager.dict(last_result)
    max_parallel_num = 150
    useful_gpu = [0, 1, 2, 3, 4, 5, 6, 7]
    task_args = {
        "result_dict": all_result,
        "problem_root_dir": "../data/problem_instance",
        "pop_sizes": pop_sizes, "crossover_prob": crossover_prob,
        "max_eval": max_eval, "elite_rate": elite_rate
    }
    task_num = 0
    process_list = []
    task_list = []
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        for task_index in range(30):
            if "{}_{}".format(target_instance_path, task_index) in all_result.keys():
                print("{}_{}".format(target_instance_path, task_index))
                continue
            if "compiler_args_selection_problem" in target_instance_path:
                task_list.append((target_instance_path, task_index))
    random.shuffle(task_list)
    for element in task_list:
        target_instance_path, task_index = element
        task_args["target_instance_path"] = target_instance_path
        task_args["task_index"] = task_index
        task_num += 1
        while True:
            alive_num = sum([1 if process.is_alive() else 0 for process in process_list])
            if alive_num < max_parallel_num:
                break
        for process in process_list[:]:
            if not process.is_alive():
                process.join()
                process.close()
                process_list.remove(process)
        print(task_num, alive_num, target_instance_path, task_index,
              datetime.datetime.now())
        p = Process(target=start_SMARTEST_task, kwargs=task_args)
        p.start()
        process_list.append(p)
    [p.join() for p in process_list]
    pickle.dump(dict(all_result), open(log_path, "wb"))


if __name__ == '__main__':
    os.chdir(Path(os.path.dirname(os.path.abspath(__file__)), "../"))
    logging.disable(logging.INFO)
    start = time.time()
    run_SMARTEST_multi(max_eval=1600)
    print("-----------max_cut=800--------------")
    statisitc_SMARTEST(max_eval=1600, max_cut=800)
    print("-----------max_cut=1600--------------")
    statisitc_SMARTEST(max_eval=1600, max_cut=1600)
    print(start - time.time())
