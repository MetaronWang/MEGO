import os
import pickle
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

from types_ import *


def hold_best(sequence: List[Union[int, float]]) -> List[Union[int, float]]:
    new_sequence = [sequence[0]]
    for i in range(1, len(sequence)):
        new_sequence.append(max(new_sequence[i - 1], sequence[i]))
    return new_sequence


def GA_statistic(eval_sample_num: int = 64, pop_size: int = 128, max_eval_rate: int = 10, cut_eval=800,
                 initial_solution_num_list=None, elite_k: int = 1,
                 problem_root_dir: str = "../data/problem_instance"):
    if initial_solution_num_list is None:
        initial_solution_num_list = [2, 4]
    print("\n\n\n------------------\n\n\n\n\n")
    print("eval_sample_num={}, pop_size={}, max_eval_rate={}".format(eval_sample_num, pop_size, max_eval_rate))
    all_result = pickle.load(open(Path("../GA_logs", "mapSamplNum-{}_popSize-{}_popEvalRate-{}_eliteK-{}"
                                                     "_adjustInitial.pkl".format(eval_sample_num, pop_size,
                                                                                 max_eval_rate, elite_k)), "rb"))
    result_dict = {}
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        max_eval = max_eval_rate * pop_size
        print("{}\t{}".format(target_instance_path, ""), end="\t")
        elements = target_instance_path.split("_")
        dimension, index, problem = int(elements[-1]), int(elements[-2]), "_".join(elements[:-2])
        if problem not in result_dict:
            result_dict[problem] = {}
        if dimension not in result_dict[problem]:
            result_dict[problem][dimension] = {}
        result_dict[problem][dimension][index] = {key: [0, 0, 0, 0, 0] for key in initial_solution_num_list}
        results = [
            all_result["{}_{}_{}_{}_{}".format(target_instance_path, eval_sample_num, pop_size, max_eval, task_index)]
            for task_index in range(30)]
        all_step_data = {}
        all_step_data["random_initial"] = [hold_best(result["random_initial"]["step_history"][:cut_eval]) for result in
                                           results]
        random_initial_data = [result[-1] for result in all_step_data["random_initial"]]
        initial_type = "max_map"
        for initial_solution_num in initial_solution_num_list:
            all_step_data["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)] = [
                hold_best(
                    result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]["step_history"][:cut_eval])
                for result in results]
            map_data = [result[-1] for result in
                        all_step_data["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]]
            # print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
            result_dict[problem][dimension][index][initial_solution_num][3] = (np.mean(map_data), np.std(map_data))
            if ranksums(map_data, random_initial_data, alternative="greater")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][0] += 1
                # print("↑", end="\t")
            elif ranksums(map_data, random_initial_data, alternative="less")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][2] += 1
                # print("↓", end="\t")
            else:
                result_dict[problem][dimension][index][initial_solution_num][1] += 1
            #     print("→", end="\t")
            # print("{:.2f}±{:.2f}".format(np.mean(random_initial_data), np.std(random_initial_data)), end="\t")
            result_dict[problem][dimension][index][initial_solution_num][4] = (
                np.mean(random_initial_data), np.std(random_initial_data))
            result_dict[problem][dimension][index][initial_solution_num].append([{
                "map_steps":
                    result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]["step_history"][
                    :result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]["len_known"]],
                "random_steps": result["random_initial"]["step_history"],
                "step_history": result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]["step_history"]
            } for result in results])
            # for result in results:
            #     print("{:.2f}".format(max(result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]["step_history"][
            #               64:result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)][
            #                   "len_known"]]) -
            #           max(result["{}_{}_{}".format(initial_type, pop_size, initial_solution_num)]["step_history"][:64]))
            #           ,end=" ")
            # print("\n")
            result_dict[problem][dimension][index][initial_solution_num] += [0, 0, 0]
            step_data = [max(result["map_steps"]) for result in
                         result_dict[problem][dimension][index][initial_solution_num][5]]
            random_step_data = [max(result["random_steps"][:len(result["map_steps"])]) for result in
                                result_dict[problem][dimension][index][initial_solution_num][5]]
            # print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
            if ranksums(step_data, random_step_data, alternative="greater")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][6] += 1
                # print("↑", end="\t")
            elif ranksums(step_data, random_step_data, alternative="less")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][8] += 1
                # print("↓", end="\t")
            else:
                result_dict[problem][dimension][index][initial_solution_num][7] += 1

        print()
    return result_dict


def HillClimbing_statistic(eval_sample_num: int = 64, max_eval: int = 6400, cut_eval=800,
                           initial_solution_num_list=None, problem_root_dir: str = "../data/problem_instance"):
    if initial_solution_num_list is None:
        initial_solution_num_list = [2, 4]
    print("\n\n\n------------------\n\n\n\n\n")
    print("eval_sample_num={}, max_eval={}".format(eval_sample_num, max_eval))
    all_result = pickle.load(
        open(Path("../HillClimbing_logs", "mapSamplNum-{}maxEval-{}.pkl".format(eval_sample_num, max_eval)), "rb"))
    result_dict = {}
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        elements = target_instance_path.split("_")
        dimension, index, problem = int(elements[-1]), int(elements[-2]), "_".join(elements[:-2])
        if problem not in result_dict:
            result_dict[problem] = {}
        if dimension not in result_dict[problem]:
            result_dict[problem][dimension] = {}
        result_dict[problem][dimension][index] = {key: [0, 0, 0, 0, 0] for key in initial_solution_num_list}
        print("{}\t{}".format(target_instance_path, ""), end="\t")
        results = [all_result["{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index)]
                   for task_index in range(30)]
        all_step_data = {}
        all_step_data["random_initial"] = [hold_best(result["random_initial"]["step_history"][:cut_eval]) for result in
                                           results]
        random_initial_data = [result[-1] for result in all_step_data["random_initial"]]
        initial_type = "max_map"
        for initial_solution_num in initial_solution_num_list:
            all_step_data["{}_{}".format(initial_type, initial_solution_num)] = [
                hold_best(result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][:cut_eval])
                for result in results]
            map_data = [result[-1] for result in all_step_data["{}_{}".format(initial_type, initial_solution_num)]]
            print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
            result_dict[problem][dimension][index][initial_solution_num][3] = (np.mean(map_data), np.std(map_data))
            if ranksums(map_data, random_initial_data, alternative="greater")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][0] += 1
                print("↑", end="\t")
            elif ranksums(map_data, random_initial_data, alternative="less")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][2] += 1
                print("↓", end="\t")
            else:
                result_dict[problem][dimension][index][initial_solution_num][1] += 1
                print("→", end="\t")
            print("{:.2f}±{:.2f}".format(np.mean(random_initial_data), np.std(random_initial_data)), end="\t")
            result_dict[problem][dimension][index][initial_solution_num][4] = (
                np.mean(random_initial_data), np.std(random_initial_data))
            result_dict[problem][dimension][index][initial_solution_num].append([{
                "map_steps":
                    result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][
                    :result["{}_{}".format(initial_type, initial_solution_num)]["len_known"]],
                "random_steps": result["random_initial"]["step_history"],
                "step_history": result["{}_{}".format(initial_type, initial_solution_num)]["step_history"]
            } for result in results])
            result_dict[problem][dimension][index][initial_solution_num] += [0, 0, 0]
            step_data = [max(result["map_steps"]) for result in
                         result_dict[problem][dimension][index][initial_solution_num][5]]
            random_step_data = [max(result["random_steps"][:len(result["map_steps"])]) for result in
                                result_dict[problem][dimension][index][initial_solution_num][5]]
            # print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
            if ranksums(step_data, random_step_data, alternative="greater")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][6] += 1
                # print("↑", end="\t")
            elif ranksums(step_data, random_step_data, alternative="less")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][8] += 1
                # print("↓", end="\t")
            else:
                result_dict[problem][dimension][index][initial_solution_num][7] += 1
        print()
    return result_dict


def SMAC_statistic(eval_sample_num: int = 64, max_eval: int = 6400, cut_eval=800,
                   initial_solution_num_list=None, problem_root_dir: str = "../data/problem_instance"):
    if initial_solution_num_list is None:
        initial_solution_num_list = [2, 4]
    print("\n\n\n------------------\n\n\n\n\n")
    print("eval_sample_num={}, max_eval={}".format(eval_sample_num, max_eval))
    all_result = pickle.load(
        open(Path("../SMAC_logs", "mapSamplNum-{}maxEval-{}.pkl".format(eval_sample_num, max_eval)), "rb"))
    result_dict = {}
    for target_instance_path in os.listdir(Path(problem_root_dir, "valid")):
        elements = target_instance_path.split("_")
        dimension, index, problem = int(elements[-1]), int(elements[-2]), "_".join(elements[:-2])
        if problem not in result_dict:
            result_dict[problem] = {}
        if dimension not in result_dict[problem]:
            result_dict[problem][dimension] = {}
        result_dict[problem][dimension][index] = {key: [0, 0, 0, 0, 0] for key in initial_solution_num_list}
        print("{}\t{}".format(target_instance_path, ""), end="\t")
        results = [all_result["{}_{}_{}_{}".format(target_instance_path, eval_sample_num, max_eval, task_index)]
                   for task_index in range(30)]
        all_step_data = {}
        all_step_data["random_initial"] = [hold_best(result["random_initial"]["step_history"][:cut_eval]) for result in
                                           results]
        random_initial_data = [result[-1] for result in all_step_data["random_initial"]]
        initial_type = "max_map"
        for initial_solution_num in initial_solution_num_list:
            all_step_data["{}_{}".format(initial_type, initial_solution_num)] = [
                hold_best(result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][:cut_eval])
                for result in results]
            map_data = [result[-1] for result in all_step_data["{}_{}".format(initial_type, initial_solution_num)]]
            print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
            result_dict[problem][dimension][index][initial_solution_num][3] = (np.mean(map_data), np.std(map_data))
            if ranksums(map_data, random_initial_data, alternative="greater")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][0] += 1
                print("↑", end="\t")
            elif ranksums(map_data, random_initial_data, alternative="less")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][2] += 1
                print("↓", end="\t")
            else:
                result_dict[problem][dimension][index][initial_solution_num][1] += 1
                print("→", end="\t")
            print("{:.2f}±{:.2f}".format(np.mean(random_initial_data), np.std(random_initial_data)), end="\t")
            result_dict[problem][dimension][index][initial_solution_num][4] = (
                np.mean(random_initial_data), np.std(random_initial_data))
            temp_result = []
            for result in results:
                initial_qualities = result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][
                                    64:result["{}_{}".format(initial_type, initial_solution_num)]["len_known"]]
                random.shuffle(initial_qualities)
                result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][
                64:result["{}_{}".format(initial_type, initial_solution_num)]["len_known"]] = initial_qualities
                temp_dict = {
                    "map_steps": result["{}_{}".format(initial_type, initial_solution_num)]["step_history"][
                                 :result["{}_{}".format(initial_type, initial_solution_num)]["len_known"]],
                    "random_steps": result["random_initial"]["step_history"],
                    "step_history": result["{}_{}".format(initial_type, initial_solution_num)]["step_history"]
                }
                temp_result.append(temp_dict)
            result_dict[problem][dimension][index][initial_solution_num].append(temp_result)
            result_dict[problem][dimension][index][initial_solution_num] += [0, 0, 0]
            step_data = [max(result["map_steps"]) for result in
                         result_dict[problem][dimension][index][initial_solution_num][5]]
            random_step_data = [max(result["random_steps"][:len(result["map_steps"])]) for result in
                                result_dict[problem][dimension][index][initial_solution_num][5]]
            # print("{:.2f}±{:.2f}".format(np.mean(map_data), np.std(map_data)), end="")
            if ranksums(step_data, random_step_data, alternative="greater")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][6] += 1
                # print("↑", end="\t")
            elif ranksums(step_data, random_step_data, alternative="less")[1] < 0.05:
                result_dict[problem][dimension][index][initial_solution_num][8] += 1
                # print("↓", end="\t")
            else:
                result_dict[problem][dimension][index][initial_solution_num][7] += 1
        print()
    return result_dict


def get_exceed_step(sequence, threshold) -> int:
    for index, element in enumerate(sequence):
        if element >= threshold:
            return index + 1
    return len(sequence)


def main():
    initial_solution_num_list = [4]
    all_data = {}
    all_data["GA"] = GA_statistic(eval_sample_num=64, pop_size=32, max_eval_rate=25, cut_eval=800, elite_k=1,
                                  problem_root_dir="../data/problem_instance",
                                  initial_solution_num_list=initial_solution_num_list)
    all_data["HillClimbing"] = HillClimbing_statistic(eval_sample_num=64, max_eval=800, cut_eval=800,
                                                      problem_root_dir="../data/problem_instance",
                                                      initial_solution_num_list=initial_solution_num_list)
    all_data["SMAC"] = SMAC_statistic(eval_sample_num=64, max_eval=800, cut_eval=800,
                                      problem_root_dir="../data/problem_instance",
                                      initial_solution_num_list=initial_solution_num_list)
    pickle.dump(all_data, open("all_data.pkl", "wb"))
    # return
    all_data = pickle.load(open("all_data.pkl", "rb"))
    problem_list = ["compiler_args_selection_problem", "com_influence_max_problem", "zero_one_knapsack_problem",
                    "match_max_problem", "anchor_selection_problem", "max_cut_problem"]
    dimension_list = [40, 60, 80, 100]
    algorithm_list = ["GA", "HillClimbing", "SMAC"]
    print("\n\n\n\n")
    print("\t\t" + "\t".join(
        ["{}{}".format(algorithm, "\t".join(["" for _ in range(2 * len(initial_solution_num_list))])) for algorithm
         in algorithm_list]))
    print("\t\t" + "\t".join(
        ["a={}\t".format(initial_solution_num) for _ in algorithm_list for initial_solution_num in
         initial_solution_num_list]))
    print("Problem\tDim\t" + "\t".join(["Map\tRand" for _ in initial_solution_num_list for _ in algorithm_list]))
    for problem in problem_list:
        for dimension in dimension_list:
            indices = list(all_data["GA"][problem][dimension].keys())
            all_ranks = {}
            for index in indices:
                mean_value = {}
                print("{}\t{}".format(problem, dimension), end="\t")
                for algorithm in algorithm_list:
                    for initial_solution_num in initial_solution_num_list:
                        algorithm_data = all_data[algorithm][problem][dimension][index][initial_solution_num]
                        mean_value[algorithm + "_map-a={}".format(initial_solution_num)] = algorithm_data[3][0]
                        mean_value[algorithm + "_origin"] = algorithm_data[4][0]
                        print("{:.2f}±{:.2f}{}\t{:.2f}±{:.2f}".format(
                            algorithm_data[3][0], algorithm_data[3][1],
                            "↑" if algorithm_data[0] == 1 else "→" if algorithm_data[1] == 1 else "↓",
                            algorithm_data[4][0], algorithm_data[4][1],
                        ), end="\t")
                all_tag = list(mean_value.keys())
                all_tag.sort(key=lambda item: mean_value[item], reverse=True)
                all_ranks[index] = {tag: rank + 1 for rank, tag in enumerate(all_tag)}
                print()

            print("{}\t{}".format(problem, dimension), end="\t")
            for algorithm in algorithm_list:
                for initial_solution_num in initial_solution_num_list:
                    data = all_data[algorithm][problem][dimension]
                    print("{}-{}-{}".format(sum([data[index][initial_solution_num][0] for index in indices]),
                                            sum([data[index][initial_solution_num][1] for index in indices]),
                                            sum([data[index][initial_solution_num][2] for index in indices])
                                            ), end="\t\t")
            print()

            print("{}\t{}".format(problem, dimension), end="\t")
            for algorithm in algorithm_list:
                for initial_solution_num in initial_solution_num_list:
                    print("{:.2f}\t{:.2f}".format(np.mean(
                        [all_ranks[index][algorithm + "_map-a={}".format(initial_solution_num)] for index in indices]),
                        np.mean(
                            [all_ranks[index][algorithm + "_origin"] for index in indices])),
                        end="\t")
            print()

    print("\n\n\n\n")
    print("\t\t" + "\t".join(
        ["{}{}".format(algorithm, "\t".join(["" for _ in range(2 * len(initial_solution_num_list))])) for algorithm
         in algorithm_list]))
    print("\t\t" + "\t".join(
        ["a={}\t".format(initial_solution_num) for _ in algorithm_list for initial_solution_num in
         initial_solution_num_list]))
    print("Problem\tDim\t" + "\t".join(["Map\tRand" for _ in initial_solution_num_list for _ in algorithm_list]))
    for problem in problem_list:
        for dimension in dimension_list:
            indices = list(all_data["GA"][problem][dimension].keys())
            all_ranks = {}
            wdls = {initial_solution_num: {algorithm: {index: [0, 0, 0] for index in indices} for algorithm in
                                           algorithm_list} for initial_solution_num in initial_solution_num_list}
            for index in indices:
                mean_value = {}
                print("{}\t{}".format(problem, dimension), end="\t")
                for algorithm in algorithm_list:
                    if algorithm not in mean_value:
                        for initial_solution_num in initial_solution_num_list:
                            algorithm_data = all_data[algorithm][problem][dimension][index][initial_solution_num][5]
                            map_value = [max(algorithm_data[task_index]['map_steps']) for task_index in range(30)]
                            random_value = [
                                max(algorithm_data[task_index]['random_steps'][:len(algorithm_data[0]['map_steps'])])
                                for task_index in range(30)]
                            map_mean, map_std = np.mean(map_value), np.std(map_value)
                            random_mean, random_std = np.mean(random_value), np.std(random_value)
                            if ranksums(map_value, random_value, alternative="greater")[1] < 0.05:
                                wdls[initial_solution_num][algorithm][index][0] += 1
                            elif ranksums(map_value, random_value, alternative="less")[1] < 0.05:
                                wdls[initial_solution_num][algorithm][index][2] += 1
                            else:
                                wdls[initial_solution_num][algorithm][index][1] += 1
                            mean_value[algorithm + "_map-a={}".format(initial_solution_num)] = map_mean
                            mean_value[algorithm + "_origin"] = random_mean
                            print("{:.2f}±{:.2f}{}\t{:.2f}±{:.2f}".format(
                                map_mean, map_std,
                                "↑" if wdls[initial_solution_num][algorithm][index][0] == 1 else \
                                    ("→" if wdls[initial_solution_num][algorithm][index][1] == 1 else "↓"),
                                random_mean, random_std
                            ), end="\t")
                all_tag = list(mean_value.keys())
                all_tag.sort(key=lambda item: mean_value[item], reverse=True)
                all_ranks[index] = {tag: rank + 1 for rank, tag in enumerate(all_tag)}
                print()

            print("{}\t{}".format(problem, dimension), end="\t")
            for algorithm in algorithm_list:
                for initial_solution_num in initial_solution_num_list:
                    data = wdls[initial_solution_num][algorithm]
                    print("{}-{}-{}".format(sum([data[index][0] for index in indices]),
                                            sum([data[index][1] for index in indices]),
                                            sum([data[index][2] for index in indices])
                                            ), end="\t\t")
            print()

            print("{}\t{}".format(problem, dimension), end="\t")
            for algorithm in algorithm_list:
                for initial_solution_num in initial_solution_num_list:
                    print("{:.2f}\t{:.2f}".format(np.mean(
                        [all_ranks[index][algorithm + "_map-a={}".format(initial_solution_num)] for index in indices]),
                        np.mean(
                            [all_ranks[index][algorithm + "_origin"] for index in indices])),
                        end="\t")
            print()

    for initial_solution_num in initial_solution_num_list:
        print("---------initial solution num={}---------".format(initial_solution_num))
        print("\n\n\n\n")
        print("\t\t" + "\t".join(["{}\t\t".format(algorithm) for algorithm in algorithm_list]))
        print("Problem\tDim\t" + "\t".join(["Initial Step\tExceed Step\tRatio" for _ in algorithm_list]))
        for problem in problem_list:
            for dimension in dimension_list:
                indices = list(all_data["GA"][problem][dimension].keys())
                print("{}\t{}".format(problem, dimension), end="\t")
                for algorithm in algorithm_list:
                    initial_steps, exceed_steps = [], []
                    for index in indices:
                        step_datas = all_data[algorithm][problem][dimension][index][initial_solution_num][5]
                        for step_data in step_datas:
                            initial_steps.append(len(step_data["map_steps"]))
                            exceed_steps.append(get_exceed_step(step_data["random_steps"], max(step_data["map_steps"])))
                    print("{:.2f}\t{:.2f}\t{:.2f}".format(np.mean(initial_steps), np.mean(exceed_steps),
                                                          np.mean(exceed_steps) / np.mean(initial_steps)), end="\t")
                print()


def draw_bar():
    initial_solution_num_list = [4]
    all_data = pickle.load(open("all_data.pkl", "rb"))
    # problem_list = ["compiler_args_selection_problem", "com_influence_max_problem", "zero_one_knapsack_problem",
    #                 "match_max_problem", "anchor_selection_problem", "max_cut_problem"]
    # problem_list = ["zero_one_knapsack_problem", "match_max_problem", "max_cut_problem"]
    # problem_legend_list = ["Zero/one Knapsack Problem", "Match Max Problem", "Max Cut Problem"]
    problem_list = ["compiler_args_selection_problem", "com_influence_max_problem", "anchor_selection_problem"]
    problem_legend_list = ["Compiler Arguments Selection Problem",
                           "Competitive / Complementary Influence \nMaximization Problem", "Anchor Selection Problem"]
    dimension_list = [40, 60, 80, 100]
    algorithm_list = ["GA", "HillClimbing", "SMAC"]
    algorithm_legend_list = ["GA", "HC", "BO"]
    for initial_solution_num in initial_solution_num_list:
        for algorithm_index, algorithm in enumerate(algorithm_list):
            plt.clf()
            plt.figure(figsize=[8, 6], dpi=500)
            x = np.arange(0, len(dimension_list) * 3, 3)
            wdl_data = {}
            for dimension in dimension_list:
                for problem in problem_list:
                    indices = list(all_data["GA"][problem][dimension].keys())
                    data = all_data[algorithm][problem][dimension]
                    wdl_data["{}_{}".format(problem, dimension)] = (
                        sum([data[index][initial_solution_num][0] for index in indices]),
                        sum([data[index][initial_solution_num][1] for index in indices]),
                        sum([data[index][initial_solution_num][2] for index in indices])
                    )
            color_list = ["#A8DF8E", "#FB8B24", "#9A031E"]
            hatch_list = ["/", ".", "\\"]
            width = 0.5
            for hatch_index, problem in enumerate(problem_list):
                plt.bar(x - width * (1 - hatch_index),
                        [wdl_data["{}_{}".format(problem, dimension)][0] for dimension in dimension_list],
                        width=width, edgecolor='k',
                        hatch=hatch_list[hatch_index], color=color_list[0])
                plt.bar(x - width * (1 - hatch_index),
                        [wdl_data["{}_{}".format(problem, dimension)][1] for dimension in dimension_list],
                        width=width, edgecolor='k',
                        bottom=[wdl_data["{}_{}".format(problem, dimension)][0] for dimension in dimension_list],
                        hatch=hatch_list[hatch_index], color=color_list[1])
                plt.bar(x - width * (1 - hatch_index),
                        [wdl_data["{}_{}".format(problem, dimension)][2] for dimension in dimension_list],
                        width=width, edgecolor='k',
                        bottom=[wdl_data["{}_{}".format(problem, dimension)][0] +
                                wdl_data["{}_{}".format(problem, dimension)][1] for dimension in dimension_list],
                        hatch=hatch_list[hatch_index], color=color_list[2])
            custom_legend1 = [
                mpatches.Patch(color=color_list[0], label='W'),
                mpatches.Patch(color=color_list[1], label='D'),
                mpatches.Patch(color=color_list[2], label='L'),
            ]
            custom_legend2 = [
                mpatches.Patch(facecolor="#FFFF", edgecolor="#000", hatch=hatch_list[0], label=problem_legend_list[0]),
                mpatches.Patch(facecolor="#FFFF", edgecolor="#000", hatch=hatch_list[1], label=problem_legend_list[1]),
                mpatches.Patch(facecolor="#FFFF", edgecolor="#000", hatch=hatch_list[2], label=problem_legend_list[2]),
            ]
            first_legend = plt.legend(handles=custom_legend1, loc='upper left', title='W-D-L')
            plt.gca().add_artist(first_legend)
            plt.legend(handles=custom_legend2, loc='upper right', title='Problem Class', fontsize=9)

            plt.xlabel('Dimension', fontsize=11)
            plt.ylabel('W-D-L', fontsize=11)
            plt.xticks(x, dimension_list)
            plt.yticks([0, 1, 2, 3, 4, 5])
            plt.title(algorithm_legend_list[algorithm_index])
            plt.show()


def statistic_compiler_args():
    initial_solution_num = 4
    bo_data = pickle.load(open("all_data.pkl", "rb"))["SMAC"]["compiler_args_selection_problem"]
    smartest_data = pickle.load(open("../SMARTEST_logs/result_1600.pkl", "rb"))
    pass
    problem_type = "compiler_args_selection_problem"
    problem = "Compiler Arguments Selection Problem"
    dimension_list = [40, 60, 80, 100]
    wdls = [[0, 0, 0] for _ in range(5)]
    mean_ranks = []
    for dimension in dimension_list:
        indices = list(bo_data[dimension].keys())
        for index in indices:
            print("{}\t{}".format(problem, dimension), end="\t")
            bo_value = bo_data[dimension][index][initial_solution_num]
            init_map_value = [max(bo_value[5][task_index]['map_steps']) for task_index in range(30)]
            bo_random_800_value = [max(bo_value[5][task_index]['random_steps']) for task_index in range(30)]
            bo_map_800_value = [max(bo_value[5][task_index]['step_history']) for task_index in range(30)]

            smartest_value = [smartest_data["{}_{}_{}_{}".format(problem_type, index, dimension, task_index)] for
                              task_index in range(30)]
            smartest_800_100_value, smartest_1600_100_value = \
                [max(value["smartest_100"]["step_history"][:800]) for value in smartest_value], \
                    [max(value["smartest_100"]["step_history"][:1600]) for value in smartest_value]
            compare_data = [init_map_value, bo_random_800_value, bo_map_800_value, smartest_800_100_value,
                            smartest_1600_100_value]
            mean_values = [np.mean(data) for data in compare_data]
            mean_ranks.append(np.argsort(np.argsort((mean_values))[::-1]) + 1)
            for data_index, data in enumerate(compare_data):
                print("{:.2f}±{:.2f}".format(np.mean(data), np.std(data)), end="")
                if data_index != 2:
                    if ranksums(data, compare_data[2], alternative="greater")[1] < 0.05:
                        wdls[data_index][0] += 1
                        print("↑", end="")
                    elif ranksums(data, compare_data[2], alternative="less")[1] < 0.05:
                        wdls[data_index][2] += 1
                        print("↓", end="")
                    else:
                        wdls[data_index][1] += 1
                        print("→", end="")
                print(end="\t")
            print()
    print("wdl\t\t", end="")
    for wdl in wdls:
        print("{}-{}-{}".format(wdl[0], wdl[1], wdl[2]), end="\t")
    print("\nrank\t\t", end="")
    mean_ranks = np.array(mean_ranks).transpose()
    ranks_mean = np.mean(mean_ranks, axis=1)
    for rank_mean in ranks_mean:
        print("{:.2f}".format(rank_mean), end="\t")


if __name__ == '__main__':
    main()
    # draw_bar()
    # statistic_compiler_args()
    # compare_initail_solution()
    # get_acculate_rate()
