import logging
import os
import pickle
import random
import time
from copy import deepcopy
from multiprocessing import Process, Manager
from pathlib import Path

import yaml
from pytorch_lightning.utilities.seed import seed_everything

from src.decoder_mapping import DecoderMapping
from src.experiment_problem import train_problem_types, load_problem_data, load_problem_instance, \
    generate_data_from_problem_instance
from src.types_ import *


# cpu_num = 240  # 这里设置成你想运行的CPU个数
# torch.set_num_threads(cpu_num)
def decoder_mapping_training_task(task_args):
    with open("../configs/{}.yaml".format(task_args["config_name"]), 'r') as file:
        config = yaml.safe_load(file)
    source_instance = load_problem_instance(problem_dir=task_args["source_problem_dir"])
    target_instance = load_problem_instance(problem_dir=task_args["target_problem_dir"])
    config["logging_params"]["name"] = "Mapping-{}_{}_{}2{}_{}_{}-{}-{}".format(
        source_instance.__class__.__name__, source_instance.dimension, task_args["source_index"],
        target_instance.__class__.__name__, target_instance.dimension, task_args["target_index"],
        task_args["model_name"], task_args["sample_num"])
    log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
    if not os.path.exists(Path(log_path, "best_model.pt")):
        dp = DecoderMapping(**deepcopy(task_args))
        dp.fine_tuning_mapping_decoder()


def start_training(eval_sample_num=64):
    logging.disable(logging.INFO)
    # torch.multiprocessing.set_start_method('spawn')
    process_list: List[Process] = []
    problem_root_dir = "../data/problem_instance"
    useful_gpu = [0, 1, 2, 3, 4, 5, 6]
    max_parallel_num = 80
    task_index = 0
    task_args = {
        "model_name": "SurrogateVAE",
        "config_name": "surrogate_vae_mapping",
        "sample_num": eval_sample_num
    }

    for source_instance_path in os.listdir(Path(problem_root_dir, "train")):
        task_args["source_index"] = int(source_instance_path.split("_")[-2])
        task_args["source_problem_dir"] = Path(problem_root_dir, "train", source_instance_path)
        for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
            task_args["target_index"] = int(target_instance_path.split("_")[-2])
            task_args["target_problem_dir"] = Path(problem_root_dir, "valid", target_instance_path)
            task_args["gpu_index"] = useful_gpu[task_index % len(useful_gpu)]
            task_index += 1
            alive_num = max_parallel_num + 100
            while alive_num >= max_parallel_num:
                alive_num = sum([1 if process.is_alive() else 0 for process in process_list])
                time.sleep(0.01)
            for process in process_list[:]:
                if not process.is_alive():
                    process.join()
                    process.close()
                    process_list.remove(process)

            print(task_index, alive_num, task_args["gpu_index"])
            p = Process(target=decoder_mapping_training_task, args=(task_args,))
            p.start()
            process_list.append(p)

    for p in process_list:
        p.join()


def start_eval_task(return_dict, task_args, target_instance_path, eval_sample_num, problem_root_dir, top_k):
    pearsons, spearmans, results = [], [], []
    for source_instance_path in os.listdir(Path(problem_root_dir, "train")):
        task_args["source_index"] = int(source_instance_path.split("_")[-2])
        task_args["source_problem_dir"] = Path(problem_root_dir, "train", source_instance_path)
        seed_everything(2333, workers=True)
        dp = DecoderMapping(**deepcopy(task_args))
        dp.load_mapping_model(sample_num=eval_sample_num)
        pearson, spearman = dp.evaluate_correlation()
        topk_x = dp.get_topk_target_solution(k=top_k)
        result = [("".join([str(bit) for bit in x]), dp.target_instance.evaluate(x)) for x in topk_x]
        pearsons.append(pearson), spearmans.append(spearman), results.append(result)
    return_dict["{}_{}".format(target_instance_path, top_k)] = (pearsons, spearmans, results)


def eval_and_statistic(eval_sample_num=64, top_ks=None):
    if top_ks is None:
        top_ks = [4, 8, 16]
    hit_ks = [1, 3, 5, 10]
    logging.disable(logging.INFO)
    problem_root_dir = "../data/problem_instance"
    task_args = {
        "model_name": "SurrogateVAE",
        "config_name": "surrogate_vae_mapping",
        "sample_num": eval_sample_num,
    }
    process_list = []
    useful_gpu = [0, 1, 2, 3, 4, 5, 6]
    log_path = "all_data_corr_{}.pkl".format(eval_sample_num)
    last_result = pickle.load(open(log_path, "rb")) if os.path.exists(log_path) else {}
    manager = Manager()
    return_dict = manager.dict(last_result)
    for top_k in top_ks:
        for task_index, target_instance_path in enumerate(os.listdir(Path(problem_root_dir, "valid"))):
            if "{}_{}".format(target_instance_path, top_k) in return_dict:
                continue
            task_args["target_index"] = int(target_instance_path.split("_")[-2])
            task_args["target_problem_dir"] = Path(problem_root_dir, "valid", target_instance_path)
            task_args["gpu_index"] = useful_gpu[task_index % len(useful_gpu)]
            p = Process(target=start_eval_task, args=(
                return_dict, deepcopy(task_args), target_instance_path, eval_sample_num, problem_root_dir, top_k))
            p.start()
            process_list.append(p)
    [p.join() for p in process_list]
    pickle.dump(dict(return_dict), open(log_path, "wb"))
    problem_list = ["compiler_args_selection_problem", "com_influence_max_problem", "zero_one_knapsack_problem",
                    "match_max_problem", "anchor_selection_problem", "max_cut_problem"]
    dimension_list = [40, 60, 80, 100]
    dimension_indices = {dim: set() for dim in dimension_list}
    [dimension_indices[int(item.split("_")[-2])].add(int(item.split("_")[-3])) for item in return_dict.keys()]
    top_k = 4
    for problem in problem_list:
        for dimension in dimension_list:
            print(problem, dimension, end=" ")
            for hit_k in hit_ks:
                corr_hits = []
                rand_hits = []
                for index in dimension_indices[dimension]:
                    target_instance_path = "{}_{}_{}".format(problem, index, dimension)
                    pearsons, spearmans, results = return_dict["{}_{}".format(target_instance_path, top_k)]
                    k_indices = np.argsort([np.mean([e[1] for e in result]) for result in results])[::-1][:hit_k]
                    pos_indices = set()
                    for i in range(len(spearmans)):
                        if pearsons[i] > 0 and spearmans[i] > 0:
                            pos_indices.add(i)
                    pos_num = len(pos_indices)
                    corr_hits.append(
                        min(1, sum([1 if i in pos_indices else 0 for i in k_indices])))

                    random_rate, random_index = [], list(range(len(pearsons)))
                    for _ in range(100):
                        random.shuffle(random_index)
                        random_indices = set(random_index[:pos_num])
                        random_rate.append(
                            min(1, sum([1 if i in random_indices else 0 for i in k_indices])))
                    rand_hits.append(np.mean(random_rate))
                print("{:.2f} {:.2f}".format(np.mean(corr_hits), np.mean(rand_hits)), end=" ")

            print()


def random_instance_quality(eval_sample_num=128):
    logging.disable(logging.INFO)
    problem_root_dir = "../data/problem_instance"
    seed_everything(2333, workers=True)
    for target_problem_type in train_problem_types.keys():
        for target_dimension in [30, 40]:
            for target_index in range(1):
                target_instance = load_problem_instance(problem_dir=problem_root_dir, problem_type=target_problem_type,
                                                        instance_type="valid", index=target_index,
                                                        dimension=target_dimension)
                target_x, _ = generate_data_from_problem_instance(problem=target_instance, sample_num=128)
                all_result = []
                for x in target_x:
                    result = target_instance.evaluate(x)
                    _, target_y = load_problem_data(problem_dir=problem_root_dir,
                                                    problem_type=target_problem_type, instance_type="valid",
                                                    dimension=target_dimension, index=target_index)
                    result = (result - np.min(target_y)) / (np.max(target_y) - np.min(target_y))
                    all_result.append(result)
                print(np.mean(all_result), "\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


if __name__ == '__main__':
    # start_training(eval_sample_num=64)
    eval_and_statistic(eval_sample_num=64, top_ks=[4])
    # random_instance_quality(eval_sample_num=128)
