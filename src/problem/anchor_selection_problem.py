import random
import sys
import time
from os import path
from pathlib import Path

import numpy as np

from src.problem.base_problem import BaseProblem
from src.types_ import *

sys.path.append(str(Path(path.dirname(path.abspath(__file__)), "../../lib/anchor_selection")))

_anchor_selection_dataset_names = [
    "botanical_garden",
    "delivery_area",
    "facade",
    "lounge",
    "old_computer",
    "relief_2",
    "terrains",
    "boulders",
    "door",
    "kicker",
    "meadow",
    "pipes",
    "statue",
    "bridge",
    "electro",
    "lecture_room",
    "observatory",
    "playground",
    "terrace",
    "courtyard",
    "exhibition_hall",
    "living_room",
    "office",
    "relief",
    "terrace_2"
]


class AnchorSelectionProblem(BaseProblem):
    def __init__(self, dimension: int, instance_num: int = 100000, model_type: str = "4p3v", **kwargs):
        super().__init__(dimension, **kwargs)
        self.dimension = dimension
        self.max_k: int = random.randint(int(self.dimension * 0.1), int(self.dimension * 0.6))
        assert model_type in ["4p3v", "5p2v"]
        self.model_type: str = model_type
        if self.model_type == "4p3v":
            import LM4P3V as LM
        else:
            import LM5P2V as LM
        self.dataset: str = np.random.choice(_anchor_selection_dataset_names)
        self.instance_num: int = instance_num
        self.set_file_path = str(Path(path.dirname(path.abspath(__file__)), "../../data/dataset/anchor_selection",
                                      "{}_trainParam.txt/".format(self.model_type)))
        dataset_dir: str = str(Path(path.dirname(path.abspath(__file__)), "../../data/dataset/anchor_selection",
                                    "multi_view_dslr_calibration_undistorted/{}".format(self.dataset)))
        result = LM.sample_instances(dataset_dir, 10)
        sample_num = (self.instance_num // len(result) + 2) * 10
        result = LM.sample_instances(dataset_dir, sample_num)
        self.instances: List[List[float]] = result[:self.instance_num]
        self.candidate_anchors = np.random.choice(range(len(self.instances)), size=dimension, replace=False)
        self.candidate_anchors_dominates: List[List[int]] = self.get_dominate_list(self.candidate_anchors)

    def load_samples(self, filename):
        print("Load Data from:", filename)
        self.datasets = ["LOAD_DATA"]
        file = open(filename, "r", encoding="utf8")
        line = file.readline()
        self.instance_num = int(line.strip())
        self.instances: List[List[float]] = []
        for _ in range(self.instance_num):
            line = file.readline()
            self.instances.append([float(element) for element in line.strip().split(" ")])
        print("Loading finished, the new instance num is", self.instance_num)

    def get_dominate_list(self, solution: List[int]):
        if self.model_type == "4p3v":
            import LM4P3V as LM
        else:
            import LM5P2V as LM
        dominate_list = LM.set_cover_list(self.set_file_path, self.instances, solution)
        return dominate_list

    def get_dominate_list_from_instance_file(self, filename, solution: List[int]):
        if self.model_type == "4p3v":
            import LM4P3V as LM
        else:
            import LM5P2V as LM
        dominate_list = LM.get_set_cover_list_from_file(self.set_file_path, filename, solution)
        return dominate_list

    def evaluate(self, solution: Union[NpArray, List[int]]):
        anchors = []
        dominate = set()
        for index in range(self.dimension):
            if len(anchors) >= self.max_k:
                break
            if solution[index] == 1:
                anchors.append(self.candidate_anchors[index])
                for instance in self.candidate_anchors_dominates[index]:
                    dominate.add(instance)
        return len(dominate)

    def old_evaluate(self, solution: Union[NpArray, List[int]]):
        anchors = []
        for index in range(self.dimension):
            if len(anchors) >= self.max_k:
                break
            if solution[index] == 1:
                anchors.append(self.candidate_anchors[index])
        if self.model_type == "4p3v":
            import LM4P3V as LM
        else:
            import LM5P2V as LM
        dominate = LM.set_cover_parallel(self.set_file_path, self.instances, anchors)
        return len(dominate)


if __name__ == '__main__':
    problem = AnchorSelectionProblem(50, model_type="4p3v")
    for _ in range(100):
        st = time.time()
        solution = np.random.choice(2, size=50)
        a = problem.evaluate(solution)
        b = problem.old_evaluate(solution)
        print(a-b)
