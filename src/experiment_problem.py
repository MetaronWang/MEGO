import os
import pickle
import random
from multiprocessing import Process
from pathlib import Path

from src.problem.base_problem import BaseProblem
from src.problem.match_max_problem import MatchMaxProblem
from src.problem.max_cut_problem import MaxCutProblem
from src.problem.zero_one_knapsack_problem import ZeroOneKnapsackProblem
from src.problem.anchor_selection_problem import AnchorSelectionProblem
from src.problem.com_influence_max_problem import ComInfluenceMaxProblem
from src.problem.compiler_args_selectionProblem import CompilerArgsSelectionProblem
from src.types_ import *

train_problem_types: Dict[str, Type[BaseProblem]] = {
    "match_max_problem": MatchMaxProblem,
    "max_cut_problem": MaxCutProblem,
    "zero_one_knapsack_problem": ZeroOneKnapsackProblem,
}

valid_problem_types: Dict[str, Type[BaseProblem]] = {
    "match_max_problem": MatchMaxProblem,
    "max_cut_problem": MaxCutProblem,
    "zero_one_knapsack_problem": ZeroOneKnapsackProblem,
    "anchor_selection_problem": AnchorSelectionProblem,
    "com_influence_max_problem": ComInfluenceMaxProblem,
    "compiler_args_selection_problem": CompilerArgsSelectionProblem,
}


def generate_problem_instance(ins_dir: str = '../data/problem_instance', train_num: int = 5, valid_num: int = 1,
                              train_dimensions=None, valid_dimensions=None):
    if train_dimensions is None:
        train_dimensions = [30, 35, 40]
    if valid_dimensions is None:
        valid_dimensions = [40, 60, 80, 100]
    if not os.path.exists(ins_dir):
        os.mkdir(ins_dir)
    if not os.path.exists(Path(ins_dir, "train")):
        os.mkdir(Path(ins_dir, "train"))
    if not os.path.exists(Path(ins_dir, "valid")):
        os.mkdir(Path(ins_dir, "valid"))
    for problem_type in train_problem_types.keys():
        for index in range(train_num):
            train_dimension: int = train_dimensions[index % len(train_dimensions)]
            if os.path.exists(Path(ins_dir, "train", "{}_{}_{}".format(problem_type, index, train_dimension))):
                continue
            print("mkdir", str(Path(ins_dir, "train", "{}_{}_{}".format(problem_type, index, train_dimension))))
            os.mkdir(Path(ins_dir, "train", "{}_{}_{}".format(problem_type, index, train_dimension)))
            if problem_type != "anchor_selection_problem":
                problem_instance = train_problem_types[problem_type](dimension=train_dimension)
            else:
                problem_instance = AnchorSelectionProblem(dimension=train_dimension,
                                                          model_type="5p2v" if index % 2 == 1 else "4p3v")
            pickle.dump(
                problem_instance,
                open(Path(ins_dir, "train", "{}_{}_{}".format(problem_type, index, train_dimension), "problem.pkl"),
                     "wb")
            )
    for problem_type in valid_problem_types.keys():
        for index in range(valid_num):
            valid_dimension: int = valid_dimensions[index % len(valid_dimensions)]
            if os.path.exists(Path(ins_dir, "valid", "{}_{}_{}".format(problem_type, index, valid_dimension))):
                continue
            print("mkdir", str(Path(ins_dir, "valid", "{}_{}_{}".format(problem_type, index, valid_dimension))))
            os.mkdir(Path(ins_dir, "valid", "{}_{}_{}".format(problem_type, index, valid_dimension)))
            problem_instance = valid_problem_types[problem_type](dimension=valid_dimension)
            pickle.dump(
                problem_instance,
                open(Path(ins_dir, "valid", "{}_{}_{}".format(problem_type, index, valid_dimension), "problem.pkl"),
                     "wb")
            )


def load_problem_instance(problem_dir: Path) -> BaseProblem:
    return pickle.load(open(Path(problem_dir, "problem.pkl"), "rb"))


def generate_data_from_problem_instance(problem: BaseProblem, sample_num=20000):
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(problem.dimension))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(problem.dimension))
        solutions.add(temp)
    x: NpArray = np.array(list(solutions), dtype=np.float32)
    y: NpArray = np.zeros([sample_num], dtype=np.float32)
    for i in range(sample_num):
        y[i] = problem.evaluate(solution=np.array(x[i], dtype=np.int32))
    return x, y


def generate_problem_data(problem_dir: Path, sample_num=20000):
    if os.path.exists(Path(problem_dir, "x.npy")) and os.path.exists(Path(problem_dir, "y.npy")):
        return
    problem = load_problem_instance(problem_dir=problem_dir)
    x, y = generate_data_from_problem_instance(problem, sample_num=sample_num)
    np.save(str(Path(problem_dir, "x.npy")), x)
    np.save(str(Path(problem_dir, "y.npy")), y)
    print("Generate New DATA for", str(problem_dir))


def load_problem_data(problem_dir):
    x = np.load(str(Path(problem_dir, "x.npy")))
    y = np.load(str(Path(problem_dir, "y.npy")))
    return x, y


def generate_only_solution(dimension: int = 30, sample_num: int = 10000):
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(dimension))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(dimension))
        solutions.add(temp)
    return np.array(list(solutions), dtype=np.float32)


def load_sample_indices(problem_dir: Path, sample_num: int = 1000) -> NpArray:
    indices_path = Path(problem_dir, "indices_{}.npy".format(sample_num))
    if os.path.exists(indices_path):
        return np.load(open(indices_path, 'rb'))
    else:
        x, y = load_problem_data(problem_dir=problem_dir)
        length = len(x)
        indices = np.random.choice(length, sample_num, replace=False)
        np.save(open(indices_path, "wb"), indices)
        return indices


if __name__ == '__main__':
    root_dir = "../data/problem_instance"
    process_list = []
    generate_problem_instance(ins_dir=root_dir, train_num=9, valid_num=12)
    for instance_path in os.listdir(Path(root_dir, "train")):
        p = Process(target=generate_problem_data, args=(Path(root_dir, "train", instance_path), 20000))
        p.start()
        process_list.append(p)
    for instance_path in os.listdir(Path(root_dir, "valid")):
        p = Process(target=generate_problem_data, args=(Path(root_dir, "valid", instance_path), 1024))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
