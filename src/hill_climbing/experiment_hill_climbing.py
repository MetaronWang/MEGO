import datetime
import gc
import logging
import os
import pickle
import random
import time
from copy import deepcopy
from multiprocessing import Manager, Process
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pds
import seaborn as sns
from scipy.stats import ranksums

from decoder_mapping import DecoderMapping
from experiment_problem import load_problem_instance, load_problem_data, load_sample_indices
from src.hill_climbing.simple_hill_climbing import SimpleHillClimbing
from types_ import *


def get_initial_solution_by_surrogate_mapping(problem_root_dir: str, target_instance_path: str, eval_sample_num=128,
                                              initial_solution_num: int = 16, gpu_index: int = 0,
                                              randomly: bool = False) -> List[NpArray]:
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

    if not randomly:
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
    else:
        initial_solution, solution_strings = [], set()
        generate_results: List[NpArray] = [
            dp.get_random_k_target_solution(k=initial_solution_num) for dp in positive_correlation_dps]
        for result in generate_results:
            for solution in result:
                solution_str = "".join([str(bit) for bit in solution])
                if solution_str not in solution_strings:
                    solution_strings.add(solution_str)
                    initial_solution.append(solution)
        random.shuffle(initial_solution)
    for dp in positive_correlation_dps:
        del dp
    gc.collect()
    return initial_solution


def start_hill_climbing(problem_root_dir: str, target_instance_path: str, target_dimension: int,
                        eval_sample_num: int = 128,
                        candidate_init=None, max_eval: int = 500000):
    target_root_dir = Path(problem_root_dir, "valid", target_instance_path)
    problem_instance = load_problem_instance(problem_dir=target_root_dir)
    if candidate_init is None:
        known_x = []
    else:
        target_x, _ = load_problem_data(problem_dir=target_root_dir)
        target_indices = load_sample_indices(problem_dir=target_root_dir, sample_num=eval_sample_num)
        known_x = candidate_init + list(target_x[target_indices])
    np.random.seed(int(time.time() * 1000) % 10000)
    random.seed(int(time.time() * 1000) % 10000)
    shc = SimpleHillClimbing(eval_func=problem_instance.evaluate, dim=target_dimension, known_x=known_x,
                             max_eval=max_eval, candidate_init=candidate_init)

    shc.run()
    return shc.best_x, shc.best_y, shc.quality_update_history, shc.eval_time, shc.quality_step_history, len(known_x)


def hill_climbing_task(result_dict, problem_root_dir: str, target_instance_path: str, gpu_index: int = 0,
                       eval_sample_num: int = 128, max_eval: int = 500, initial_solution_nums=None, task_index: int = 0,
                       ):
    if initial_solution_nums is None:
        initial_solution_nums = [4]
    result: Dict[str, Dict[str, Union[NpArray, float]]] = {}
    target_dimension = int(target_instance_path.split("_")[-1])
    for initial_solution_num in initial_solution_nums:
        a = get_initial_solution_by_surrogate_mapping(problem_root_dir=problem_root_dir,
                                                      target_instance_path=target_instance_path,
                                                      initial_solution_num=initial_solution_num,
                                                      gpu_index=gpu_index, eval_sample_num=eval_sample_num,
                                                      randomly=False)
        x, y, quality_history, eval_time, step_history, len_known = start_hill_climbing(
            problem_root_dir=problem_root_dir,
            target_instance_path=target_instance_path,
            target_dimension=target_dimension,
            candidate_init=a,
            eval_sample_num=eval_sample_num,
            max_eval=max_eval)
        if "compiler_args" in target_instance_path:
            print(target_instance_path, "Max Map Finished", datetime.datetime.now())
        result["max_map_{}".format(initial_solution_num)] = {
            "x": list(x),
            "y": y,
            "quality_history": quality_history,
            "step_history": step_history,
            "eval_time": eval_time,
            "len_known": len_known
        }
        # a = get_initial_solution_by_surrogate_mapping(problem_root_dir=problem_root_dir,
        #                                               target_instance_path=target_instance_path,
        #                                               initial_solution_num=initial_solution_num,
        #                                               gpu_index=gpu_index, eval_sample_num=eval_sample_num,
        #                                               randomly=True)
        # x, y, quality_history, eval_time, step_history, len_known = start_hill_climbing(
        #     problem_root_dir=problem_root_dir,
        #     target_instance_path=target_instance_path,
        #     target_dimension=target_dimension,
        #     candidate_init=a,
        #     eval_sample_num=eval_sample_num,
        #     max_eval=max_eval)
        # if "compiler_args" in target_instance_path:
        #     print(target_instance_path, "Random Map Finished", datetime.datetime.now())
        # result["random_map_{}".format(initial_solution_num)] = {
        #     "x": list(x),
        #     "y": y,
        #     "quality_history": quality_history,
        #     "step_history": step_history,
        #     "eval_time": eval_time,
        #     "len_known": len_known
        # }
    x, y, quality_history, eval_time, step_history, len_known = start_hill_climbing(problem_root_dir=problem_root_dir,
                                                                                    target_instance_path=target_instance_path,
                                                                                    target_dimension=target_dimension,
                                                                                    candidate_init=None,
                                                                                    eval_sample_num=eval_sample_num,
                                                                                    max_eval=max_eval)
    if "compiler_args" in target_instance_path:
        print(target_instance_path, "Random Initial Finished", datetime.datetime.now())
    result["random_initial"] = {
        "x": list(x),
        "y": y,
        "quality_history": quality_history,
        "step_history": step_history,
        "eval_time": eval_time,
        "len_known": len_known
    }
    result_dict["{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index)] = deepcopy(result)
    print("Finish", target_instance_path, task_index, datetime.datetime.now())


def async_save_result(result_dict, log_path):
    while True:
        time.sleep(600)
        store_value = dict(deepcopy(result_dict))
        pickle.dump(store_value, open(log_path, "wb"))


def run_hill_climbing_multi(eval_sample_num: int = 64, max_eval: int = 6400,
                            problem_root_dir: str = "../data/problem_instance"):
    if not os.path.exists(Path("../HillClimbing_logs")):
        os.mkdir(Path("../HillClimbing_logs"))
    log_path = Path("../HillClimbing_logs", "mapSamplNum-{}maxEval-{}.pkl".format(eval_sample_num, max_eval))
    last_result = pickle.load(open(log_path, "rb")) if os.path.exists(log_path) else {}
    manager = Manager()
    all_result = manager.dict(last_result)
    max_parallel_num = 150
    useful_gpu = [0, 1, 2, 3, 4, 5, 6, 7]
    task_args = {
        "result_dict": all_result,
        "problem_root_dir": "../data/problem_instance",
        "eval_sample_num": eval_sample_num,
        "initial_solution_nums": [2, 4],
        "max_eval": max_eval
    }
    task_num = 0
    compiler_process_list, other_process_list = [], []
    compiler_task_list, other_task_list = [], []
    compiler_finish_num, other_finish_num = 0, 0
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        for task_index in range(30):
            if "{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index) in all_result.keys():
                print("{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index))
                continue
            if "compiler_args" in target_instance_path:
                compiler_task_list.append((target_instance_path, task_index))
            else:
                other_task_list.append((target_instance_path, task_index))
    random.shuffle(compiler_task_list)
    random.shuffle(other_task_list)
    async_save_process = Process(target=async_save_result, args=(all_result, deepcopy(log_path),))
    async_save_process.start()
    while len(compiler_task_list) > 0 or len(other_task_list) > 0:
        for element in compiler_task_list[:]:
            target_instance_path, task_index = element
            task_args["target_instance_path"] = target_instance_path
            task_args["task_index"] = task_index
            task_args["gpu_index"] = useful_gpu[task_num % len(useful_gpu)]
            if len(other_task_list) > 0:
                compiler_max_parallel_num = max_parallel_num * 3 // 4
            else:
                compiler_max_parallel_num = max_parallel_num - len(other_task_list)
            alive_num = sum([1 if process.is_alive() else 0 for process in compiler_process_list])
            if alive_num >= compiler_max_parallel_num:
                break
            for process in compiler_process_list[:]:
                if not process.is_alive():
                    process.join()
                    process.close()
                    compiler_finish_num += 1
                    compiler_process_list.remove(process)
            task_num += 1
            print(task_num, "Compiler Task", alive_num, len(compiler_task_list), compiler_finish_num,
                  task_args["gpu_index"], target_instance_path, task_index, datetime.datetime.now())
            p = Process(target=hill_climbing_task, kwargs=task_args)
            p.start()
            compiler_process_list.append(p)
            compiler_task_list.remove(element)

        for element in other_task_list[:]:
            target_instance_path, task_index = element
            task_args["target_instance_path"] = target_instance_path
            task_args["task_index"] = task_index
            task_args["gpu_index"] = useful_gpu[task_num % len(useful_gpu)]
            if len(compiler_task_list) > 0:
                other_max_parallel_num = max_parallel_num // 4
            else:
                other_max_parallel_num = max_parallel_num - len(compiler_task_list)
            alive_num = sum([1 if process.is_alive() else 0 for process in other_process_list])
            if alive_num >= other_max_parallel_num:
                break
            for process in other_process_list[:]:
                if not process.is_alive():
                    process.join()
                    process.close()
                    other_finish_num += 1
                    other_process_list.remove(process)
            task_num += 1
            print(task_num, "Other Task", alive_num, len(other_task_list), other_finish_num, task_args["gpu_index"],
                  target_instance_path, task_index, datetime.datetime.now())
            p = Process(target=hill_climbing_task, kwargs=task_args)
            p.start()
            other_process_list.append(p)
            other_task_list.remove(element)
        time.sleep(0.1)
    [p.join() for p in compiler_process_list]
    [p.join() for p in other_process_list]
    async_save_process.terminate()
    async_save_process.join()
    pickle.dump(dict(all_result), open(log_path, "wb"))
    async_save_process.close()


def print_distribution_step_curve(curve_data, save_folder, title, file_name, x_name):
    plt.clf()
    plt.figure(figsize=[8, 6])
    data = {
        x_name: [],
        "Initial Type": [],
        "Max Fitness": []
    }
    for initial_type in curve_data.keys():
        type_data = curve_data[initial_type]
        max_step = min(len(step) for step in type_data)
        for task_data in type_data:
            for index in range(max_step):
                if task_data[index] > -1e8:
                    data[x_name].append(index)
                    data["Max Fitness"].append(task_data[index])
                    data["Initial Type"].append(initial_type)

    df = pds.DataFrame(data)
    g = sns.lineplot(df, x=x_name, y="Max Fitness", hue="Initial Type", style="Initial Type", markers=False,
                     errorbar='sd', linewidth='1')
    # g.legend(prop={'family': 'Times New Roman', 'weight': 'bold'})
    ax = g.axes
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    # [label.set_fontname('Times New Roman') for label in labels]
    plt.title(title)
    plt.savefig(Path(save_folder, "{}.jpg".format(file_name)), dpi=1000)
    # plt.show(dpi=1000)
    plt.close()


def hold_best(sequence: List[Union[int, float]]) -> List[Union[int, float]]:
    new_sequence = [sequence[0]]
    for i in range(1, len(sequence)):
        new_sequence.append(max(new_sequence[i - 1], sequence[i]))
    return new_sequence


def output_multi_statistic(eval_sample_num: int = 64, max_eval: int = 6400, cut_eval=800,
                           problem_root_dir: str = "../data/problem_instance"):
    print("\n\n\n------------------\n\n\n\n\n")
    print("eval_sample_num={}, max_eval={}".format(eval_sample_num, max_eval))
    all_result = pickle.load(
        open(Path("../HillClimbing_logs", "mapSamplNum-{}maxEval-{}.pkl".format(eval_sample_num, max_eval)), "rb"))
    if not os.path.exists(Path("../HillClimbing_logs", "pics")):
        os.mkdir(Path("../HillClimbing_logs", "pics"))
    save_folder = Path("../HillClimbing_logs", "pics", "mapSamplNum-{}maxEval-{}".format(eval_sample_num, max_eval))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        # if "compiler_args" in target_instance_path:
        #     continue
        print("{}\t{}".format(target_instance_path, ""), end="\t")
        results = [all_result["{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index)]
                   for task_index in range(30)]
        all_update_data, all_step_data = {}, {}
        random_initial_data = [result["random_initial"]["y"] for result in results]
        all_update_data["random_initial"] = [hold_best(result["random_initial"]["quality_history"]) for result in
                                             results]
        all_step_data["random_initial"] = [hold_best(result["random_initial"]["step_history"][:cut_eval]) for result in
                                           results]
        for initial_type in ["max_map"]:
            for initial_solution_num in [2, 4]:
                map_data = [result["{}_{}".format(initial_type, initial_solution_num)]["y"] for result in results]
                print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
                if ranksums(map_data, random_initial_data, alternative="greater")[1] < 0.05:
                    print("↑", end="\t")
                elif ranksums(map_data, random_initial_data, alternative="less")[1] < 0.05:
                    print("↓", end="\t")
                else:
                    print("→", end="\t")
                all_update_data["{}_{}".format(initial_type, initial_solution_num)] = [
                    hold_best(result["{}_{}".format(initial_type, initial_solution_num)]["quality_history"]) for
                    result in results]
                all_step_data["{}_{}".format(initial_type, initial_solution_num)] = [
                    hold_best(result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][:cut_eval]) for
                    result in results]
        print("{:.2f}±{:.2f}".format(np.mean(random_initial_data), np.std(random_initial_data)))
        for initial_solution_num in [2, 4]:
            population_curve_data = {
                "random_initial": all_update_data["random_initial"],
                "max_map": all_update_data["max_map_{}".format(initial_solution_num)],
                # "random_map": all_update_data["random_map_{}".format(initial_solution_num)]
            }
            print_distribution_step_curve(curve_data=population_curve_data, save_folder=save_folder,
                                          file_name="{}_{}_update".format(target_instance_path,
                                                                          initial_solution_num),
                                          title="{}, and the map solution num is {},\n"
                                                "with max_step={},\n"
                                                "while sample_num={} in decoder mapping".format(
                                              target_instance_path, initial_solution_num,
                                              max_eval, eval_sample_num
                                          ), x_name="Update Times")
            step_curve_data = {
                "random_initial": all_step_data["random_initial"],
                "max_map": all_step_data["max_map_{}".format(initial_solution_num)],
                # "random_map": all_step_data["random_map_{}".format(initial_solution_num)]
            }
            print_distribution_step_curve(curve_data=step_curve_data, save_folder=save_folder,
                                          file_name="{}_{}_step".format(target_instance_path,
                                                                        initial_solution_num),
                                          title="{}, and the map solution num is {},\n"
                                                "with max_step={},\n"
                                                "while sample_num={} in decoder mapping".format(
                                              target_instance_path, initial_solution_num,
                                              max_eval, eval_sample_num
                                          ), x_name="Step")


def output_multi_result(eval_sample_num: int = 64, max_eval: int = 6400,
                        problem_root_dir: str = "../data/problem_instance"):
    print("\n\n\n------------------\n\n\n\n\n")
    print("eval_sample_num={}, max_eval={}".format(eval_sample_num, max_eval))
    all_result = pickle.load(
        open(Path("../HillClimbing_logs", "mapSamplNum-{}maxEval-{}.pkl".format(eval_sample_num, max_eval)), "rb"))
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        for task_index in range(30):
            # print("{}\t{}\t{}".format(target_problem_type, target_dimension, task_index), end="\t")
            result = all_result["{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index)]
            for initial_type in ["max_map"]:
                for initial_solution_num in [2, 4]:
                    print(result["{}_{}".format(initial_type, initial_solution_num)]["y"], end='\t')
            print(result["random_initial"]["y"])


if __name__ == '__main__':
    os.chdir(Path(os.path.dirname(os.path.abspath(__file__)), "../"))
    logging.disable(logging.INFO)
    start = time.time()
    run_hill_climbing_multi(eval_sample_num=64, max_eval=800, problem_root_dir="../data/problem_instance")
    output_multi_result(eval_sample_num=64, max_eval=800, problem_root_dir="../data/problem_instance")
    output_multi_statistic(eval_sample_num=64, max_eval=800, problem_root_dir="../data/problem_instance")

    print(start - time.time())
