import os
import time
from copy import deepcopy
from pathlib import Path

from src.decoder_mapping import DecoderMapping


def eval_time(problem_root_dir: str, target_instance_path: str, source_instance_path: str, eval_sample_num=64,
              initial_solution_num: int = 16, gpu_index: int = 0):
    task_args = {
        "model_name": "SurrogateVAE",
        "config_name": "surrogate_vae_mapping",
        "sample_num": eval_sample_num,
        "gpu_index": gpu_index,
        "target_problem_dir": Path(problem_root_dir, "valid", target_instance_path),
        "target_index": int(target_instance_path.split("_")[-2])
    }
    task_args["source_index"] = int(source_instance_path.split("_")[-2])
    task_args["source_problem_dir"] = Path(problem_root_dir, "train", source_instance_path)
    dp = DecoderMapping(**deepcopy(task_args))
    start = time.time()
    dp.fine_tuning_mapping_decoder(save_model=False)
    fine_tuning_time = time.time()
    print(fine_tuning_time-start)
    solutions = dp.get_topk_target_solution(k=initial_solution_num)
    sample_time = time.time()
    print(sample_time-fine_tuning_time)

if __name__ == '__main__':
    eval_time(eval_sample_num=64, problem_root_dir="../data/problem_instance",
              source_instance_path="match_max_problem_8_40",
              target_instance_path="match_max_problem_3_100",
              initial_solution_num=2
              )
    eval_time(eval_sample_num=64, problem_root_dir="../data/problem_instance",
              source_instance_path="match_max_problem_8_40",
              target_instance_path="anchor_selection_problem_10_80",
              initial_solution_num=2
              )