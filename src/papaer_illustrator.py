import os
import pickle
from pathlib import Path

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import mpl_toolkits.axisartist as axisartist
import numpy as np
import pandas as pds
import seaborn as sns
from matplotlib.scale import register_scale, ScaleBase
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from types_ import *

dimensions = [40, 60, 80, 100]


class MyScale(ScaleBase):
    name = 'myscale'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0., vmin), vmax

    class SquareRootTransform(transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 3

        def inverted(self):
            return MyScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a) ** (1 / 3)

        def inverted(self):
            return MyScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


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
        feature = sigmoid(np.array(feature))
        # feature = softmax(np.array(feature)/np.sum(feature))
        # feature = np.array(feature) / np.sum(feature)
        if problem not in all_features.keys():
            all_features[problem] = {}
        if dimension not in all_features[problem].keys():
            all_features[problem][dimension] = {}
        all_features[problem][dimension][instance_index] = feature
    pickle.dump(all_features, open(
        Path("../initial_logs", "instanceFeatures-{}-{}.pkl".format(eval_sample_num, initial_solution_num)), "wb"))


def get_instance_min_max(all_data: Tuple[List[List[Dict[str, List]]]]) -> List[Tuple[float, float]]:
    instance_num = len(all_data[0])
    min_max: List[Tuple[float, float]] = []
    for instance_index in range(instance_num):
        known_data = []
        for data in all_data:
            for sub_data in data[instance_index]:
                for value in sub_data["random_steps"] + sub_data["step_history"]:
                    if value > -1000000000000000:
                        known_data.append(value)
        min_max.append((min(known_data), max(known_data)))
    return min_max


def normalize_instance_data(data: List[Dict[str, List]],
                            min_max: Tuple[float, float]) -> List[Dict[str, List]]:
    min_value, max_value = min_max
    return [{
        "map_steps": [(value - min_value) / (max_value - min_value) for value in sub_data["map_steps"]],
        "random_steps": [(value - min_value) / (max_value - min_value) for value in sub_data["random_steps"]],
        "step_history": [(value - min_value) / (max_value - min_value) for value in sub_data["step_history"]],
    } for sub_data in data]


def get_exceed_step(data: Dict[str, List]):
    map_max_value = max(data["map_steps"])
    baseline_index = 800
    for i in range(800):
        if data["random_steps"][i] >= map_max_value:
            baseline_index = i
            break
    return baseline_index


def line_plot(all_data, problems, filename):
    plt.clf()
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    no_mapping_data_raw = pickle.load(open(
        "../initial_logs/withoutMapping-SamplNum-64-padding0.pkl", "rb"
    ))
    ga_data, hc_data, bo_data = all_data["GA"], all_data['HillClimbing'], all_data['SMAC']
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(24, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    register_scale(MyScale)
    no_mapping_data = {}
    for key in no_mapping_data_raw.keys():
        elements = key.split("_")
        ins_id, dim = int(elements[-4]), int(elements[-3])
        domain = "_".join(elements[:-4])
        if domain not in no_mapping_data:
            no_mapping_data[domain] = {}
        if dim not in no_mapping_data[domain]:
            no_mapping_data[domain][dim] = {}
        if ins_id not in no_mapping_data[domain][dim]:
            no_mapping_data[domain][dim][ins_id] = []
        no_mapping_data[domain][dim][ins_id].append(no_mapping_data_raw[key])
    for i in range(len(problems)):
        for j in range(len(dimensions)):
            ax: matplotlib.axes.Axes = axes[i][j]
            ga_run_data_temp = [data[4][5] for data in ga_data[problems[i]][dimensions[j]].values()]
            hc_run_data_temp = [data[4][5] for data in hc_data[problems[i]][dimensions[j]].values()]
            bo_run_data_temp = [data[4][5] for data in bo_data[problems[i]][dimensions[j]].values()]
            ins_ids = list(ga_data[problems[i]][dimensions[j]].keys())
            instance_min_max = get_instance_min_max((ga_run_data_temp, hc_run_data_temp, bo_run_data_temp))
            ga_run_data = [run_data for ins_data in [normalize_instance_data(instance_data, instance_min_max[index])
                                                     for index, instance_data in enumerate(ga_run_data_temp)] for
                           run_data in ins_data]
            hc_run_data = [run_data for ins_data in [normalize_instance_data(instance_data, instance_min_max[index])
                                                     for index, instance_data in enumerate(hc_run_data_temp)] for
                           run_data in ins_data]
            bo_run_data = [run_data for ins_data in [normalize_instance_data(instance_data, instance_min_max[index])
                                                     for index, instance_data in enumerate(bo_run_data_temp)] for
                           run_data in ins_data]
            no_mapping_run_data = [[item[4] for item in no_mapping_data[problems[i]][dimensions[j]][ins_id]] for ins_id
                                   in ins_ids]
            no_mapping_norm_mean = np.mean([
                (np.array([value[0] for value in item]) - instance_min_max[i_id][0]) /
                (instance_min_max[i_id][1] - instance_min_max[i_id][0])
                for i_id, item in enumerate(no_mapping_run_data)
            ])
            no_mapping_step_mean = np.mean([
                np.array([value[1] + 64 for value in item]) for i_id, item in enumerate(no_mapping_run_data)
            ])
            # sample_values = []
            # for i_id, ins_id in enumerate(ins_ids):
            #     problem_dir = Path("../data/problem_instance/valid", f"{problems[i]}_{ins_id}_{dimensions[j]}")
            #     y = np.load(str(Path(problem_dir, "y.npy")))
            #     indices = np.load(str(Path(problem_dir, "indices_64.npy")))
            #     sample_values.append(
            #         (y[indices] - instance_min_max[i_id][0]) / (instance_min_max[i_id][1] - instance_min_max[i_id][0])
            #     )
            # sample_value_mean = np.mean(sample_values)
            data = {
                r"\#FEs ×100": [],
                "Initial Type": [],
                "Objective Value": [],
                "Map": []
            }
            map_step_mean = np.mean([np.mean([len(run_data["map_steps"]) for run_data in run_datas]) for run_datas in
                                     (ga_run_data, bo_run_data, hc_run_data)])
            map_best_mean = np.mean([np.mean([max(run_data["map_steps"]) for run_data in run_datas]) for run_datas in
                                     (ga_run_data, bo_run_data, hc_run_data)])
            sample_value_mean = np.mean([np.mean([max(run_data["map_steps"][:64]) for run_data in run_datas]) for run_datas in
                                     (ga_run_data, bo_run_data, hc_run_data)])
            no_mapping_norm_mean = max(sample_value_mean, no_mapping_norm_mean)
            print((map_step_mean, map_best_mean), (no_mapping_step_mean, no_mapping_norm_mean))
            for run_datas, name in [(ga_run_data, "GA"), (bo_run_data, "BO"), (hc_run_data, "HC")]:
                for run_data in run_datas:
                    for length, key, out_name, mapping in [(800, "step_history", name, "map"),
                                                           (800, "random_steps", name, "random")]:
                        last_value = -1e8
                        for index in range(length):
                            last_value = max(last_value, run_data[key][index] if index < len(run_data[key]) else -1e8)
                            data[r"\#FEs ×100"].append(index / 100)
                            data["Objective Value"].append(last_value)
                            data["Initial Type"].append(out_name)
                            data["Map"].append(mapping)
            df = pds.DataFrame(data)
            hue_order = ["GA", "HC", "BO"]
            style_order = ["map", "random"]
            sns.lineplot(df, x=r"\#FEs ×100", y="Objective Value", hue="Initial Type",
                         style="Map", markers=False, errorbar=None, linewidth='1', ax=ax, hue_order=hue_order,
                         style_order=style_order)
            ax.axvline(x=map_step_mean / 100, color='red', linewidth='1', linestyle='-',
                       ymax=(map_best_mean ** 3) / (1.02 ** 3))
            ax.axhline(y=map_best_mean, color='red', linewidth='1', linestyle='-',
                       xmin=(map_step_mean / 100 + 0.30) / 8.30)
            ax.plot(map_step_mean / 100, map_best_mean, color='red', marker='*', markersize=14)
            ax.plot(no_mapping_step_mean / 100, no_mapping_norm_mean, color='blue', marker='s', markersize=8)
            ax.set_yscale("myscale")
            ax.set_yticks([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_ylim((0, 1.02))
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_xlim((-0.30, 8))
            ax.set_title("")
            # if i != 0 or j != 0:
            #     ax.legend_.remove()
            ax.legend_.remove()
    plt.savefig("{}.pdf".format(filename), backend='pgf')
    plt.show(dpi=400)


def acc_rate_plot_errbar(all_data, problems, filename):
    plt.clf()
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    ga_data, hc_data, bo_data = all_data["GA"], all_data['HillClimbing'], all_data['SMAC']
    data = {
        "Problem": [],
        "Value": [],
        "Method": []
    }
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    angles = np.concatenate((angles, [angles[0] + 2 * np.pi]))
    for i in range(len(problems)):
        for j in range(len(dimensions)):
            ga_run_data = [get_exceed_step(run_data) / len(run_data["map_steps"]) for ins_data in
                           [data[4][5] for data in ga_data[problems[i]][dimensions[j]].values()] for run_data
                           in ins_data]
            hc_run_data = [get_exceed_step(run_data) / len(run_data["map_steps"]) for ins_data in
                           [data[4][5] for data in hc_data[problems[i]][dimensions[j]].values()] for run_data
                           in ins_data]
            bo_run_data = [get_exceed_step(run_data) / len(run_data["map_steps"]) for ins_data in
                           [data[4][5] for data in bo_data[problems[i]][dimensions[j]].values()] for run_data
                           in ins_data]
            data["Problem"] += [angles[j * 3 + i] for _ in ga_run_data]
            data["Value"] += ga_run_data
            data["Method"] += ["GA" for _ in ga_run_data]

            data["Problem"] += [angles[j * 3 + i] for _ in ga_run_data]
            data["Value"] += hc_run_data
            data["Method"] += ["HC" for _ in ga_run_data]

            data["Problem"] += [angles[j * 3 + i] for _ in ga_run_data]
            data["Value"] += bo_run_data
            data["Method"] += ["BO" for _ in ga_run_data]
            if i == 0 and j == 0:
                data["Problem"] += [angles[12] for _ in ga_run_data]
                data["Value"] += ga_run_data
                data["Method"] += ["GA" for _ in ga_run_data]

                data["Problem"] += [angles[12] for _ in ga_run_data]
                data["Value"] += hc_run_data
                data["Method"] += ["HC" for _ in ga_run_data]

                data["Problem"] += [angles[12] for _ in ga_run_data]
                data["Value"] += bo_run_data
                data["Method"] += ["BO" for _ in ga_run_data]

    fig = plt.figure(facecolor="white", figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    plt.grid(color='gray', linestyle='-.', linewidth=0.5)
    df = pds.DataFrame(data)
    sns.lineplot(df, x="Problem", y="Value", hue="Method", markers=False, errorbar=None, linewidth='1')
    # plt.legend(loc="best")
    radar_labels = [r"$f_{}$".format(chr(i + 97)) for i in range(12)]
    plt.thetagrids(angles[:-1] * 180 / np.pi, radar_labels)
    plt.yticks([1, 2, 4, 6])
    plt.ylim((0, 8.5))
    ax.legend_.remove()
    plt.xlabel("")
    plt.ylabel("")
    # plt.show()
    plt.savefig("{}.pdf".format(filename), backend='pgf')


def wdl_800_steps_plot(all_data, problems, filename):
    plt.clf()
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    ga_data, hc_data, bo_data = all_data["GA"], all_data['HillClimbing'], all_data['SMAC']

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    problem_labels = [r"$f_{}$".format(chr(i + 97)) for i in range(12)]
    hatch_list = [".", "*", "x", "/"]
    for method_index, method_data in enumerate([ga_data, hc_data, bo_data]):
        ax = axes[method_index]
        for i in range(len(problems)):
            for j in range(len(dimensions)):
                run_value = [data[4][:3] for data in method_data[problems[i]][dimensions[j]].values()]
                problem = i * 4 + j + i * 0.5
                win, draw, loss = sum([i[0] for i in run_value]), sum([i[1] for i in run_value]), sum(
                    [i[2] for i in run_value]),
                ax.bar(problem, win, label="Win", color="#32CD32", edgecolor='k', hatch=hatch_list[j], width=0.9,
                       linewidth=0.5)
                ax.bar(problem, draw, bottom=win, label="Draw", color="#1E90FF", edgecolor='k', hatch=hatch_list[j],
                       width=0.9, linewidth=0.5)
                ax.bar(problem, loss, bottom=win + draw, label="Loss", color="#F6BD60", edgecolor='k',
                       hatch=hatch_list[j], width=0.9, linewidth=0.5)
        ax.set_ylabel(r"\#W-D-L")
        # ax.set_xlabel("Problem")
        ax.set_xticks([i * 4 + j + i * 0.5 for i in range(len(problems)) for j in range(len(dimensions))],
                      problem_labels)
        ax.set_yticks([0, 1, 2, 3])
    # plt.legend(title="W-D-L", loc="upper left")
    # plt.show()
    plt.savefig("{}.pdf".format(filename), backend='pgf')


def wdl_map_steps_plot(all_data, problems, filename):
    plt.clf()
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    ga_data, hc_data, bo_data = all_data["GA"], all_data['HillClimbing'], all_data['SMAC']

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    problem_labels = [r"$f_{}$".format(chr(i + 97)) for i in range(12)]
    hatch_list = [".", "*", "x", "/"]
    for method_index, method_data in enumerate([ga_data, hc_data, bo_data]):
        ax = axes[method_index]
        for i in range(len(problems)):
            for j in range(len(dimensions)):
                run_value = [data[4][-3:] for data in method_data[problems[i]][dimensions[j]].values()]
                problem = i * 4 + j + i * 0.5
                win, draw, loss = sum([i[0] for i in run_value]), sum([i[1] for i in run_value]), sum(
                    [i[2] for i in run_value]),
                ax.bar(problem, win, label="Win", color="#32CD32", edgecolor='k', hatch=hatch_list[j], width=0.9,
                       linewidth=0.5)
                ax.bar(problem, draw, bottom=win, label="Draw", color="#1E90FF", edgecolor='k', hatch=hatch_list[j],
                       width=0.9, linewidth=0.5)
                ax.bar(problem, loss, bottom=win + draw, label="Loss", color="#F6BD60", edgecolor='k',
                       hatch=hatch_list[j], width=0.9, linewidth=0.5)
        ax.set_ylabel(r"\#W-D-L")
        # ax.set_xlabel("Problem")
        ax.set_xticks([i * 4 + j + i * 0.5 for i in range(len(problems)) for j in range(len(dimensions))],
                      problem_labels)
        ax.set_yticks([0, 1, 2, 3])
    # plt.legend(title="W-D-L", loc="upper left")
    # plt.show()
    plt.savefig("{}.pdf".format(filename), backend='pgf')


def corr_heat(problems, filename):
    # 每组8个数， 分别是k=1，3，5，10时corr和rand的hit@K
    all_data = {
        "compiler_args_selection_problem": {40: [0.67, 0.44, 1, 0.77, 1, 0.9, 1, 0.98],
                                            60: [0.33, 0.57, 0.67, 0.95, 1, 0.99, 1, 1],
                                            80: [0.33, 0.41, 0.67, 0.76, 0.67, 0.88, 1, 1],
                                            100: [0.33, 0.33, 0.67, 0.69, 1, 0.77, 1, 0.94]},
        "com_influence_max_problem": {40: [0.33, 0.7, 1, 0.98, 1, 1, 1, 1],
                                      60: [0.33, 0.58, 1, 0.91, 1, 0.97, 1, 1],
                                      80: [0.33, 0.43, 0.67, 0.78, 1, 0.92, 1, 0.98],
                                      100: [0.67, 0.7, 1, 0.97, 1, 0.99, 1, 1]},
        "zero_one_knapsack_problem": {40: [0.33, 0.67, 1, 0.97, 1, 1, 1, 1],
                                      60: [0.67, 0.48, 0.67, 0.83, 1, 0.89, 1, 0.98],
                                      80: [1, 0.74, 1, 0.99, 1, 1, 1, 1],
                                      100: [1, 0.67, 1, 0.96, 1, 1, 1, 1]},
        "match_max_problem": {40: [0.33, 0.27, 1, 0.65, 1, 0.81, 1, 0.97],
                              60: [0.67, 0.54, 1, 0.94, 1, 0.99, 1, 1],
                              80: [0.33, 0.45, 0.67, 0.84, 1, 0.95, 1, 1],
                              100: [0.67, 0.35, 0.67, 0.68, 1, 0.87, 1, 0.99]},
        "anchor_selection_problem": {40: [0.33, 0.45, 0.67, 0.86, 0.67, 0.96, 1, 1],
                                     60: [1, 0.59, 1, 0.82, 1, 0.91, 1, 0.97],
                                     80: [1, 0.54, 1, 0.9, 1, 0.97, 1, 1],
                                     100: [0.33, 0.5, 1, 0.86, 1, 0.87, 1, 0.97]},
        "max_cut_problem": {40: [1, 0.68, 1, 0.96, 1, 0.99, 1, 1],
                            60: [0, 0.49, 0.67, 0.86, 0.67, 0.96, 1, 1],
                            80: [0.33, 0.64, 1, 0.96, 1, 0.99, 1, 1],
                            100: [0.33, 0.51, 1, 0.89, 1, 0.96, 1, 1]}
    }
    ks = [1, 3, 5, 10]
    plt.clf()
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    fig, axes = plt.subplots(nrows=len(ks), ncols=len(problems), sharex=False, sharey=False, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    map_color = [("", ""), ("", ""), ("", ""), ("", "")]
    typo = ["-", "+", "="]
    data_tensor = np.array([[[all_data[problem][dimension]] for dimension in dimensions] for problem in problems],
                           dtype=np.float32)
    self_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('self-cmap',
                                                                    [(0, '#018383'),
                                                                     (1, '#4EF037')], N=256)
    for k_index, k in enumerate(ks):
        k_data = np.array([[[all_data[problem][dimension][k_index * 2], all_data[problem][dimension][k_index * 2 + 1]]
                            for dimension in dimensions] for problem in problems], dtype=np.float32)
        all_elements = np.unique(k_data.reshape((-1)))
        all_elements.sort()
        key_map = {element: index / (len(all_elements) - 1) for index, element in enumerate(all_elements)}
        # k_norm = (k_data - np.min(k_data)) / (np.max(k_data) - np.min(k_data))
        k_norm = np.array([[[key_map[data] for data in dimension] for dimension in problem] for problem in k_data])
        for pro_index, problem in enumerate(problems):
            ax = axes[k_index][pro_index]
            data = k_norm[pro_index]
            data_origin = k_data[pro_index]
            im = ax.imshow(data.transpose(), vmin=np.min(k_norm), vmax=np.max(k_norm),
                           cmap=self_cmap)
            # if k_index == 0 and pro_index == 0:
            #     cbar = ax.figure.colorbar(im, ax=ax)
            #     cbar.ax.set_ylabel("Color bar", rotation=-90)
            ax.set_xticks(np.arange(len(dimensions)), dimensions)
            ax.set_yticks([], [])
            for i in range(len(dimensions)):
                for j in range(2):
                    text = ax.text(i, j, data_origin[i, j], ha="center", va="center", color='r', fontsize="small")
    # plt.legend()
    # plt.show()
    plt.savefig("{}.pdf".format(filename), backend='pgf')


def pca_and_clustering(problem_list, eval_sample_num: int = 64, initial_solution_num=4, filename="figure5"):
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    features_dict = pickle.load(open(
        Path("../initial_logs", "instanceFeatures-{}-{}.pkl".format(eval_sample_num, initial_solution_num)), "rb"))
    all_features = []
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30, n_iter=3000, random_state=1088)
    transformed_index = {}
    for problem in problem_list:
        if problem not in transformed_index:
            transformed_index[problem] = {}
        for dimension in features_dict[problem].keys():
            if dimension not in transformed_index[problem]:
                transformed_index[problem][dimension] = {}
            for instance_index in features_dict[problem][dimension].keys():
                transformed_index[problem][dimension][instance_index] = len(all_features)
                all_features.append(features_dict[problem][dimension][instance_index])
    transformed = tsne.fit_transform(np.array(all_features))
    problem_marker = {
        "compiler_args_selection_problem": "o",
        "com_influence_max_problem": "v",
        "zero_one_knapsack_problem": "2",
        "match_max_problem": "s",
        "anchor_selection_problem": "p",
        "max_cut_problem": "x"
    }
    problem_name = {
        "compiler_args_selection_problem": "CA",
        "com_influence_max_problem": "CIM",
        "zero_one_knapsack_problem": "KP",
        "match_max_problem": "OM",
        "anchor_selection_problem": "AS",
        "max_cut_problem": "MC"
    }
    fig = plt.figure(figsize=(5, 5), dpi=500)
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    label_colors = ["#3D3B40", "#DC84F3", "#7D0A0A", "#52D3D8", "#FB8B24"]
    kmeans = KMeans(n_clusters=5, random_state=1088, n_init="auto", init="random")
    labels = kmeans.fit_predict(all_features)
    for problem in features_dict.keys():
        for dimension in features_dict[problem].keys():
            for instance_index in features_dict[problem][dimension].keys():
                new_x = transformed[transformed_index[problem][dimension][instance_index]]
                label = labels[transformed_index[problem][dimension][instance_index]]
                ax.scatter(new_x[0], new_x[1], marker=problem_marker[problem], color=label_colors[label])
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    plt.savefig("{}_a.pdf".format(filename), backend='pgf')
    plt.clf()
    custom_legend1 = [
        mpatches.Patch(color=label_colors[0], label='Class 1'),
        mpatches.Patch(color=label_colors[1], label='Class 2'),
        mpatches.Patch(color=label_colors[2], label='Class 3'),
        mpatches.Patch(color=label_colors[3], label='Class 4'),
        mpatches.Patch(color=label_colors[4], label='Class 5'),
    ]

    custom_legend2 = [
        ax.scatter([], [], marker=problem_marker[problem], label=problem_name[problem], color="#000")
        for problem in problem_list
    ]

    first_legend = plt.legend(handles=custom_legend1, loc='upper left', title='New Class')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=custom_legend2, loc='upper right', title='Problem Class')
    plt.savefig("{}_ref.pdf".format(filename), backend='pgf')
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=False, sharey=False, figsize=(22, 5))
    # plt.subplots_adjust(wspace=0.1, hspace=0.4)
    label_problem_num = [{problem: 0 for problem in problem_list} for label in range(kmeans.n_clusters)]
    for problem in problem_list:
        for dimension in features_dict[problem].keys():
            for instance_index in features_dict[problem][dimension].keys():
                label = labels[transformed_index[problem][dimension][instance_index]]
                label_problem_num[label][problem] += 1
    for index, ax in enumerate(axes):
        ax.bar([problem_name[problem] for problem in problem_list],
               [label_problem_num[index][problem] for problem in problem_list], color=label_colors[index])
        ax.set_ylabel("\#Problem")
        ax.set_ylim((0, 8))
    plt.savefig("{}_b-f.pdf".format(filename), backend='pgf')
    matrix = []
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=500)
    ax = fig.add_subplot()
    self_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('self-cmap', [(0, '#018383'),
                                                                                  (1, '#4EF037')], N=256)
    data = np.array(all_features)
    im = ax.imshow(data.transpose(), vmax=np.max(data), vmin=np.min(data), cmap='plasma')
    ax.set_yticks([9 * i - 0.5 for i in range(3)], [9 * i for i in range(3)])
    ax.set_xticks([12 * i - 0.5 for i in range(6)], [9 * i for i in range(6)])
    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         text = ax.text(i, j, data[i, j], ha="center", va="center", color='r', fontsize="small")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Color bar", rotation=-90)
    plt.savefig("{}_g.pdf".format(filename), backend='pgf')


def example_figure(k=10, filename="fig2"):
    # records = pickle.load(open("example_figure_com_influence_max_problem_3_100.pkl", "rb"))
    # records = pickle.load(open("example_figure_com_influence_max_problem_6_80.pkl", "rb"))
    # records = pickle.load(open("example_figure_compiler_args_selection_problem_6_80.pkl", "rb"))
    # records = pickle.load(open("example_figure_train_com_influence_max_problem_3_100.pkl", "rb"))
    # records = pickle.load(open("example_figure_train_com_influence_max_problem_6_80.pkl", "rb"))
    records = pickle.load(open("example_figure_train_compiler_args_selection_problem_6_80.pkl", "rb"))
    for source_instance_NAME in records.keys():
        record = records["max_cut_problem_5_40"]
        # record = records[source_instance_NAME]
        y, origin_map, finetuning = record['y'], record['origin_map_y_prime'], record['finetuning_y_prime']
        y_argsort, origin_map_argsort, finetuning_argsort = np.argsort(y), np.argsort(origin_map), np.argsort(
            finetuning)
        y_c, origin_map_c, finetuning_c = [1 for _ in range(len(y))], [1 for _ in range(len(y))], [1 for _ in
                                                                                                   range(len(y))]
        y_c[y_argsort[0]], origin_map_c[origin_map_argsort[0]], finetuning_c[finetuning_argsort[0]] = 1, 1, 1
        for index in range(1, len(y)):
            c = 1 + index // (len(y) // k + 1)
            y_c[y_argsort[index]] = c if y[y_argsort[index]] != y[y_argsort[index - 1]] else y_c[y_argsort[index - 1]]
            origin_map_c[origin_map_argsort[index]] = c if origin_map[origin_map_argsort[index]] != origin_map[
                origin_map_argsort[index - 1]] else origin_map_c[origin_map_argsort[index - 1]]
            finetuning_c[finetuning_argsort[index]] = c if finetuning[finetuning_argsort[index]] != finetuning[
                finetuning_argsort[index - 1]] else finetuning_c[finetuning_argsort[index - 1]]
        pass
        y_c, origin_map_c, finetuning_c = np.array(y_c), np.array(origin_map_c), np.array(finetuning_c)
        y_c_argsort = np.argsort(y_c)

        plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 'large'
        fig = plt.figure(figsize=(5, 5))
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        ax.plot(list(range(len(y_c))), y_c[y_c_argsort], label="Train Instance", color="tab:blue")
        ax.plot(list(range(len(y_c))), origin_map_c[y_c_argsort], label="Original Output on Test Instance",
                color="tab:orange")
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)
        ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["left"].set_axisline_style("-|>", size=1.5)
        # ax.set_xticks([], [])
        ax.set_ylim((0, k + 1))
        ax.set_xlabel("Index of Solutions")
        ax.set_ylabel("Level of Fitness")
        plt.legend()
        # plt.show()
        plt.savefig("{}_c.pdf".format(filename), backend='pgf')
        plt.clf()

        plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 'large'
        fig = plt.figure(figsize=(5, 5))
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        ax.plot(list(range(len(y_c))), y_c[y_c_argsort], label="Input Solution on Train Instance", color="tab:blue")
        ax.plot(list(range(len(y_c))), finetuning_c[y_c_argsort], label="Finetuning Output on Test Instance",
                color="tab:green")
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)
        ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["left"].set_axisline_style("-|>", size=1.5)
        # ax.set_xticks([], [])
        ax.set_ylim((0, k + 1))
        ax.set_xlabel("Index of Solutions")
        ax.set_ylabel("Level of Fitness")
        # ax.set_title(source_instance_NAME)
        plt.legend()
        plt.savefig("{}_d.pdf".format(filename), backend='pgf')
        # plt.show()
        break


if __name__ == '__main__':
    # example_figure()
    # analysis_source_indices(eval_sample_num=64, problem_root_dir="../data/problem_instance", initial_solution_num=4)
    train_problems = ["match_max_problem", "max_cut_problem", "zero_one_knapsack_problem"]
    test_problems = ["anchor_selection_problem", "com_influence_max_problem", "compiler_args_selection_problem"]
    # pca_and_clustering(train_problems + test_problems, eval_sample_num=64, initial_solution_num=4, filename="figure5")
    # corr_heat(train_problems + test_problems, "heatmap")
    load_data = pickle.load(open("./all_data.pkl", "rb"))
    line_plot(load_data, train_problems, "fig3_al_train")
    line_plot(load_data, test_problems, "fig3_al_test")
    # acc_rate_plot_errbar(load_data, train_problems, "fig3_m_train")
    # acc_rate_plot_errbar(load_data, test_problems, "fig3_m_test")
    # wdl_map_steps_plot(load_data, train_problems, "fig3_n_train")
    # wdl_map_steps_plot(load_data, test_problems, "fig3_n_test")
    # wdl_800_steps_plot(load_data, train_problems, "fig3_o_train")
    # wdl_800_steps_plot(load_data, test_problems, "fig3_o_test")
