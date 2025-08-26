import os
import sys
import time

sys.path.append(os.getenv('IMP_MODULE_PATH'))
# 'IMP_CPP.cpython-310-x86_64-linux-gnu.so'
import ComIMP

root_dir = "/data/01-Optimization-Problem-Lib/data/IMP/"

dataset_dir_dic = {
    "Test1": ("../../../../data/IMP/", "testcase1"),
    "Wiki": (root_dir + "Wiki/", "Wiki_WC"),
    "Epinions": (root_dir + "Epinions/", "Epinions_WC"),
    "LiveJournal": (root_dir + "LiveJournal/", "LiveJournal_WC"),
    "Twitter": (root_dir + "Twitter/", "Twitter_WC"),
    "Facebook": (root_dir + "Facebook/", "Facebook_WC"),
    "Flixster": (root_dir + "Flixster/", "Flixster_WC"),
}

dataset_directed_dic = {
    "Test1": True,
    "Wiki": True,
    "Epinions": True,
    "LiveJournal": True,
    "Twitter": True,
    "Facebook": False,
    "Flixster": False,
}


def get_seed_path(dataset_name: str, seed_name: str) -> str:
    graph_dir = dataset_dir_dic[dataset_name][0]
    seed_path = os.path.join(graph_dir, "seed")
    seed_path = os.path.join(seed_path, seed_name)
    # seed_path = dataset_dir_dic[dataset_name][0] + 'seed/' + seed_name
    return seed_path


def load_seeds(seed_path: str) -> list:
    seeds = []
    with open(seed_path, 'r') as f:
        for line in f:
            seeds.append(int(line.strip()))
    return seeds


def test_compatible():
    dataset = 'Wiki'
    graph_dir = dataset_dir_dic[dataset][0]
    graph_name = dataset_dir_dic[dataset][1]
    directed = dataset_directed_dic[dataset]
    sample_num = 10000
    thread_num = 16
    is_load_graph = False

    gap = [0.5, 0.7, 0.5, 0.7]
    # seeds_b = [2]
    # seeds_a = [1]
    # seeds_a_path = '../../../../data/IMP/seeds_a'
    # seeds_b_path = '../../../../data/IMP/seeds_b'
    seeds_a_path = get_seed_path(dataset, graph_name + '_50_seed')
    seeds_b_path = get_seed_path(dataset, graph_name + '_100_seed')

    print('seeds_a_path', seeds_a_path)
    print('seeds_b_path', seeds_b_path)
    seeds_a = load_seeds(seeds_a_path)
    seeds_b = load_seeds(seeds_b_path)

    print(graph_dir, graph_name)
    st = time.time()
    strategy = ComIMP.Strategy(graph_dir, graph_name, directed, gap, seeds_b, sample_num, thread_num, is_load_graph)
    et = time.time()
    elapsed_time = et - st
    print('Load Graph Time: ', elapsed_time, 'seconds')

    # st = time.time()
    # strategy.serializeGraph()
    # et = time.time()
    # elapsed_time = et - st
    # print('Save Graph Time: ', elapsed_time, 'seconds')

    com_n = strategy.getVerNum()
    print(com_n)
    # is_seed_a = [0 for i in range(com_n)]
    # for s in seeds_a:
    #     is_seed_a[s] = 1

    st = time.time()
    com_exp = strategy.evaluate(seeds_a)
    et = time.time()
    elapsed_time = et - st
    print('Evaluate Seed Time: ', elapsed_time, 'seconds')
    print(com_exp)


if __name__ == '__main__':
    test_compatible()
