import datetime
import gc
import logging
import os
import pickle
import time
from copy import deepcopy
from multiprocessing import Manager, Process
from pathlib import Path

import torch

from decoder_mapping import DecoderMapping
from types_ import *


def get_initial_solution_by_surrogate_mapping(problem_root_dir: str, target_instance_path: str, eval_sample_num=128,
                                              initial_solution_num: int = 16, gpu_index: int = 0,
                                              randomly: bool = False) -> Tuple[float, int]:
    positive_correlation_dps: List[DecoderMapping] = []
    known_x: List[NpArray] = []
    task_args = {
        "model_name": "SurrogateVAE",
        "config_name": "surrogate_vae_mapping",
        "sample_num": eval_sample_num,
        "gpu_index": gpu_index,
        "target_problem_dir": Path(problem_root_dir, "valid", target_instance_path),
        "target_index": int(target_instance_path.split("_")[-2])
    }
    for source_instance_path in os.listdir(Path(problem_root_dir, "train")):
        task_args["source_index"] = int(source_instance_path.split("_")[-2])
        task_args["source_problem_dir"] = Path(problem_root_dir, "train", source_instance_path)
        dp = DecoderMapping(**deepcopy(task_args))
        correlation = dp.evaluate_correlation()
        if correlation[0] > 0 and correlation[1] > 0:
            positive_correlation_dps.append(dp)
        else:
            del dp

    know_x_quality, initial_solution, solution_strings = [], [], set()
    for dp in positive_correlation_dps:
        dp.load_mapping_model(sample_num=eval_sample_num)
        solutions = dp.get_topk_target_solution(k=initial_solution_num)
        for solution in solutions:
            solution_str = "".join([str(bit) for bit in solution])
            if solution_str not in solution_strings:
                solution_strings.add(solution_str)
                known_x.append(solution)
                know_x_quality.append(dp.target_instance.evaluate(solution))
    sorted_index = np.argsort(know_x_quality)[::-1]
    initial_solution = [known_x[index] for index in sorted_index]
    best_quality = know_x_quality[sorted_index[0]]
    for dp in positive_correlation_dps:
        del dp
    gc.collect()
    return best_quality, len(know_x_quality)


def initial_solution_task(result_dict, problem_root_dir: str, target_instance_path: str, gpu_index: int = 0,
                          eval_sample_num: int = 128, initial_solution_nums=None, task_index: int = 0,
                          ):
    if initial_solution_nums is None:
        initial_solution_nums = [2, 4]
    result: Dict[float, Tuple[float, int]] = {}
    for initial_solution_num in initial_solution_nums:
        a = get_initial_solution_by_surrogate_mapping(problem_root_dir=problem_root_dir,
                                                      target_instance_path=target_instance_path,
                                                      initial_solution_num=initial_solution_num,
                                                      gpu_index=gpu_index, eval_sample_num=eval_sample_num,
                                                      randomly=False)
        result[initial_solution_num] = a
    result_dict["{}_{}_{}".format(target_instance_path, eval_sample_num, task_index)] = deepcopy(result)
    print("Finish", target_instance_path, task_index, datetime.datetime.now())


def async_save_result(result_dict, log_path):
    while True:
        time.sleep(600)
        store_value = dict(deepcopy(result_dict))
        pickle.dump(store_value, open(log_path, "wb"))


def run_initial_solution_multi(eval_sample_num: int = 64,
                               problem_root_dir: str = "../data/problem_instance",
                               initial_solution_nums=None):
    if initial_solution_nums is None:
        initial_solution_nums = [2, 4]
    if not os.path.exists(Path("../initial_logs")):
        os.mkdir(Path("../initial_logs"))
    log_path = Path("../initial_logs", "mapSamplNum-{}-{}.pkl".format(eval_sample_num, initial_solution_nums))
    last_result = pickle.load(open(log_path, "rb")) if os.path.exists(log_path) else {}
    manager = Manager()
    all_result = manager.dict(last_result)
    max_parallel_num = 120
    useful_gpu = [0, 1, 2, 3, 4, 5, 6, 7]
    task_args = {
        "result_dict": all_result,
        "problem_root_dir": "../data/problem_instance",
        "eval_sample_num": eval_sample_num,
        "initial_solution_nums": initial_solution_nums,
    }
    task_num = 0
    process_list = []
    async_save_process = Process(target=async_save_result, args=(all_result, deepcopy(log_path),))
    async_save_process.start()
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        for task_index in range(30):
            if "{}_{}_{}".format(target_instance_path, eval_sample_num, task_index) in all_result.keys():
                continue
            task_args["target_instance_path"] = target_instance_path
            task_args["task_index"] = task_index
            task_args["gpu_index"] = useful_gpu[task_num % len(useful_gpu)]
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
            print(task_num, alive_num, task_args["gpu_index"], target_instance_path, task_index,
                  datetime.datetime.now())
            p = Process(target=initial_solution_task, kwargs=task_args)
            p.start()
            process_list.append(p)

    [p.join() for p in process_list]
    async_save_process.terminate()
    async_save_process.join()
    pickle.dump(dict(all_result), open(log_path, "wb"))
    async_save_process.close()


if __name__ == '__main__':
    logging.disable(logging.INFO)
    start = time.time()
    run_initial_solution_multi(eval_sample_num=64, problem_root_dir="../data/problem_instance")

    print(start - time.time())
