import datetime
import gc
import logging
import os
import pickle
import time
from copy import deepcopy
from multiprocessing import Manager, Process
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from decoder_mapping import DecoderMapping
from types_ import *


def get_initial_source_index_by_surrogate_mapping(problem_root_dir: str, target_instance_path: str, eval_sample_num=128,
                                                  initial_solution_num: int = 16, gpu_index: int = 0) -> List[int]:
    known_x: List[NpArray] = []
    task_args = {
        "model_name": "SurrogateVAE",
        "config_name": "surrogate_vae_mapping",
        "sample_num": eval_sample_num,
        "gpu_index": gpu_index,
        "target_problem_dir": Path(problem_root_dir, "valid", target_instance_path),
        "target_index": int(target_instance_path.split("_")[-2])
    }
    know_x_quality, initial_solution, solution_strings, source_indices = [], [], set(), []
    instance_list = os.listdir(Path(problem_root_dir, "train"))
    instance_list.sort()
    for index, source_instance_path in enumerate(instance_list):
        task_args["source_index"] = int(source_instance_path.split("_")[-2])
        task_args["source_problem_dir"] = Path(problem_root_dir, "train", source_instance_path)
        dp = DecoderMapping(**deepcopy(task_args))
        dp.load_mapping_model(sample_num=eval_sample_num)
        solutions = dp.get_topk_target_solution(k=initial_solution_num)
        for solution in solutions:
            solution_str = "".join([str(bit) for bit in solution])
            if solution_str not in solution_strings:
                solution_strings.add(solution_str)
                known_x.append(solution)
                know_x_quality.append(dp.target_instance.evaluate(solution))
                source_indices.append(index)
        del dp
    sorted_index = np.argsort(know_x_quality)[::-1]
    source_problem_indices = [source_indices[index] for index in sorted_index]
    gc.collect()
    return source_problem_indices[:initial_solution_num]


def initial_solution_task(result_dict, problem_root_dir: str, target_instance_path: str, gpu_index: int = 0,
                          eval_sample_num: int = 128, initial_solution_num=4, task_index: int = 0):
    a = get_initial_source_index_by_surrogate_mapping(problem_root_dir=problem_root_dir,
                                                      target_instance_path=target_instance_path,
                                                      initial_solution_num=initial_solution_num,
                                                      gpu_index=gpu_index, eval_sample_num=eval_sample_num)
    result_dict["{}_{}_{}".format(target_instance_path, eval_sample_num, task_index)] = deepcopy(a)
    print("Finish", target_instance_path, task_index, datetime.datetime.now())


def async_save_result(result_dict, log_path):
    while True:
        time.sleep(600)
        store_value = dict(deepcopy(result_dict))
        pickle.dump(store_value, open(log_path, "wb"))


def run_initial_solution_multi(eval_sample_num: int = 64, problem_root_dir: str = "../data/problem_instance",
                               initial_solution_num=4):
    if not os.path.exists(Path("../initial_logs")):
        os.mkdir(Path("../initial_logs"))
    log_path = Path("../initial_logs", "sourceProblemRecord-{}-{}.pkl".format(eval_sample_num, initial_solution_num))
    last_result = pickle.load(open(log_path, "rb")) if os.path.exists(log_path) else {}
    manager = Manager()
    all_result = manager.dict(last_result)
    max_parallel_num = 120
    useful_gpu = [0, 1, 2, 3, 4, 5, 6]
    task_args = {
        "result_dict": all_result,
        "problem_root_dir": "../data/problem_instance",
        "eval_sample_num": eval_sample_num,
        "initial_solution_num": initial_solution_num,
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


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x


def analysis_source_indices(eval_sample_num: int = 64, problem_root_dir: str = "../data/problem_instance",
                            initial_solution_num=4):
    log_path = Path("../initial_logs", "sourceProblemRecord-{}-{}.pkl".format(eval_sample_num, initial_solution_num))
    all_result = pickle.load(open(log_path, "rb")) if os.path.exists(log_path) else {}
    all_features = {}
    target_instances_paths = os.listdir(Path(problem_root_dir, "valid"))
    target_instances_paths.sort()
    for target_instance_path in target_instances_paths:
        feature = [0 for _ in os.listdir(Path(problem_root_dir, "train"))]
        elements = target_instance_path.split("_")
        dimension, instance_index, problem = int(elements[-1]), int(elements[-2]), "_".join(elements[:-2])
        for task_index in range(30):
            for index in all_result["{}_{}_{}".format(target_instance_path, eval_sample_num, task_index)]:
                feature[index] += 1
        # feature = sigmoid(np.array(feature))
        # feature = softmax(np.array(feature)/np.sum(feature))
        feature = np.array(feature)/np.sum(feature)
        if problem not in all_features.keys():
            all_features[problem] = {}
        if dimension not in all_features[problem].keys():
            all_features[problem][dimension] = {}
        all_features[problem][dimension][instance_index] = feature
    pickle.dump(all_features, open(
        Path("../initial_logs", "instanceFeatures-{}-{}.pkl".format(eval_sample_num, initial_solution_num)), "wb"))


def pca_and_clustering(eval_sample_num: int = 64, problem_root_dir: str = "../data/problem_instance",
                       initial_solution_num=4):
    features_dict = pickle.load(open(
        Path("../initial_logs", "instanceFeatures-{}-{}.pkl".format(eval_sample_num, initial_solution_num)), "rb"))
    all_features = []
    problem_class_list = []
    pca = PCA(n_components=2)
    for problem in features_dict.keys():
        for dimension in features_dict[problem].keys():
            for instance_index in features_dict[problem][dimension].keys():
                all_features.append(features_dict[problem][dimension][instance_index])
                problem_class_list.append(problem)
    pca.fit(all_features)
    problem_marker = {
        "compiler_args_selection_problem": "o",
        "com_influence_max_problem": "v",
        "zero_one_knapsack_problem": "2",
        "match_max_problem": "s",
        "anchor_selection_problem": "p",
        "max_cut_problem": "x"
    }
    problem_color = {
        "compiler_args_selection_problem": "red",
        "com_influence_max_problem": "blue",
        "zero_one_knapsack_problem": "green",
        "match_max_problem": "orange",
        "anchor_selection_problem": "black",
        "max_cut_problem": "pink"
    }
    for problem in features_dict.keys():
        for dimension in features_dict[problem].keys():
            for instance_index in features_dict[problem][dimension].keys():
                new_x = pca.transform([features_dict[problem][dimension][instance_index]])[0]
                plt.scatter(new_x[0], new_x[1], marker=problem_marker[problem], label=problem,
                            color=problem_color[problem])
    plt.show()
    plt.clf()
    plt.figure(figsize=[8, 8], dpi=500)
    label_colors = ["#3D3B40", "#DC84F3", "#7D0A0A", "#52D3D8", "#FB8B24"]
    problem_list = ["Compiler Arguments Selection Problem",
                    "Competitive / Complementary \nInfluence Maximization Problem", "Zero/one Knapsack Problem",
                    "Match Max Problem", "Anchor Selection Problem", "Max Cut Problem"]
    marker_list = ["o", "v", "2", "s", "p", "x"]
    kmeans = KMeans(n_clusters=5, random_state=1088, n_init="auto")
    kmeans.fit(all_features)
    for problem in features_dict.keys():
        for dimension in features_dict[problem].keys():
            for instance_index in features_dict[problem][dimension].keys():
                new_x = pca.transform([features_dict[problem][dimension][instance_index]])[0]
                label = kmeans.predict([features_dict[problem][dimension][instance_index]])[0]
                plt.scatter(new_x[0], new_x[1], marker=problem_marker[problem], color=label_colors[label])

    custom_legend1 = [
        mpatches.Patch(color=label_colors[0], label='Class 1'),
        mpatches.Patch(color=label_colors[1], label='Class 2'),
        mpatches.Patch(color=label_colors[2], label='Class 3'),
        mpatches.Patch(color=label_colors[3], label='Class 4'),
        mpatches.Patch(color=label_colors[4], label='Class 5'),
    ]

    custom_legend2 = [
        plt.scatter([], [], marker=marker_list[temp_index], label=problem_list[temp_index], color="#000")
        for temp_index in range(len(problem_list))
    ]
    # plt.yticks(np.arange(-0.6, 1.4, 0.2))
    # plt.xticks(np.arange(-0.8, 1.0, 0.2))
    first_legend = plt.legend(handles=custom_legend1, loc='upper left', title='W-D-L')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=custom_legend2, loc='upper right', title='Problem Class')
    plt.show()


if __name__ == '__main__':
    logging.disable(logging.INFO)
    start = time.time()
    run_initial_solution_multi(eval_sample_num=64, problem_root_dir="../data/problem_instance", initial_solution_num=4)
    analysis_source_indices(eval_sample_num=64, problem_root_dir="../data/problem_instance", initial_solution_num=4)
    pca_and_clustering(eval_sample_num=64, problem_root_dir="../data/problem_instance", initial_solution_num=4)

    print(start - time.time())
