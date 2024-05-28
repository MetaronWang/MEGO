import logging
import os
from itertools import chain
from pathlib import Path

import torch
import yaml
from pytorch_lightning.utilities.seed import seed_everything
from scipy.stats import spearmanr, pearsonr
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.experiment_build_surrogate import all_model, Surrogate, BaseProblem
from src.experiment_problem import generate_data_from_problem_instance, load_problem_data, load_problem_instance, \
    load_sample_indices
from src.regression.dataset import SolutionMappingData, ZeroOneProblemData
from src.types_ import *


class DecoderMapping():
    def __init__(self, source_problem_dir: Path, target_problem_dir: Path, source_index: int = 0, target_index: int = 0,
                 model_name: str = "SurrogateVAE", config_name: str = "surrogate_vae_mapping",
                 sample_num: int = 128, source_coefficient: int = 4, gpu_index: int = 0,
                 source_surrogate_date: str = None):
        self.source_index: int = source_index
        self.target_index: int = target_index
        self.source_problem_dir: Path = source_problem_dir
        self.target_problem_dir: Path = target_problem_dir
        self.model_name: str = model_name
        self.config_name: str = config_name
        self.sample_num: int = sample_num
        self.source_coefficient = source_coefficient
        self.gpu_index: int = gpu_index
        self.source_surrogate_date = source_surrogate_date
        self.source_instance: BaseProblem = load_problem_instance(problem_dir=self.source_problem_dir)
        self.target_instance: BaseProblem = load_problem_instance(problem_dir=self.target_problem_dir)
        self.source_dimension: int = self.source_instance.dimension
        self.target_dimension: int = self.target_instance.dimension
        self.source_model: Surrogate = self.load_source_surrogate_model()
        self.mapping_model: Surrogate = self.source_model

    def load_source_surrogate_model(self) -> Surrogate:
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        with open("../configs/{}.yaml".format(self.config_name), 'r') as file:
            config = yaml.safe_load(file)
        config["logging_params"]["name"] = "{}-{}-{}".format(self.source_instance.__class__.__name__, self.source_index,
                                                             self.model_name, )
        log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
        with open(Path(log_path, "HyperParam.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        config["trainer_params"]["gpus"] = [self.gpu_index]
        config["model_params"]["out_dim"] = self.target_instance.dimension
        model = all_model[self.model_name](**config['model_params']).to(device)
        weight_param = torch.load(str(Path(log_path, "best_model.pt")), map_location=device)
        if self.target_instance.dimension == self.source_instance.dimension:
            model.load_state_dict(weight_param)
        else:
            weight_param = {k: v.to(device) for k, v in weight_param.items() if "final_layer" not in k}
            model_dict = model.state_dict()
            model_dict.update(weight_param)
            model.load_state_dict(model_dict)
        return model.to(device)

    def load_mapping_model(self, sample_num: int = 128) -> bool:
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        with open("../configs/{}.yaml".format(self.config_name), 'r') as file:
            config = yaml.safe_load(file)
        config["logging_params"]["name"] = "Mapping-{}_{}_{}2{}_{}_{}-{}-{}".format(
            self.source_instance.__class__.__name__, self.source_instance.dimension, self.source_index,
            self.target_instance.__class__.__name__, self.target_instance.dimension, self.target_index, self.model_name,
            sample_num)
        log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
        if not os.path.exists(Path(log_path, "best_model.pt")):
            return False
        with open(Path(log_path, "HyperParam.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        config["trainer_params"]["gpus"] = [self.gpu_index]
        config["model_params"]["out_dim"] = self.target_instance.dimension
        model = all_model[self.model_name](**config['model_params']).to(device)
        model.load_state_dict(torch.load(str(Path(log_path, "best_model.pt")), map_location=device))
        self.mapping_model = model.to(device)
        return True

    def sample_source_data(self, sample_num=512) -> Tuple[NpArray, NpArray]:
        source_x, source_y = load_problem_data(problem_dir=self.source_problem_dir)
        source_indices = load_sample_indices(problem_dir=self.source_problem_dir, sample_num=sample_num)
        return source_x[source_indices], source_y[source_indices]

    def sample_target_data(self, sample_num=128) -> Tuple[NpArray, NpArray]:
        target_x, target_y = load_problem_data(problem_dir=self.target_problem_dir)

        target_indices = load_sample_indices(problem_dir=self.target_problem_dir, sample_num=sample_num)
        return target_x[target_indices], target_y[target_indices]

    def generate_cartesian_mapping_data(self, source_coefficient=4) -> Tuple[NpArray, NpArray]:
        source_x, source_y = self.sample_source_data(sample_num=source_coefficient * self.sample_num)
        target_x, target_y = self.sample_target_data(sample_num=self.sample_num)
        target_rank_indices = np.argsort(target_y)[::-1]
        source_rank_indices = np.argsort(source_y)[::-1]
        source_quality_solutions, target_quality_solutions = {}, {}
        for index in range(len(source_y)):
            if source_y[source_rank_indices[index]] not in source_quality_solutions:
                source_quality_solutions[source_y[source_rank_indices[index]]] = []
            source_quality_solutions[source_y[source_rank_indices[index]]].append(source_x[source_rank_indices[index]])
        for index in range(len(target_y)):
            if target_y[target_rank_indices[index]] not in target_quality_solutions:
                target_quality_solutions[target_y[target_rank_indices[index]]] = []
            target_quality_solutions[target_y[target_rank_indices[index]]].append(target_x[target_rank_indices[index]])
        rank_level_num = min(len(source_quality_solutions), len(target_quality_solutions))
        ref_quality_levels, eval_quality_levels = list(source_quality_solutions.keys()), list(
            target_quality_solutions.keys())
        ref_quality_levels.sort(reverse=True)
        eval_quality_levels.sort(reverse=True)
        ref_input, eval_output = [], []
        for index in range(rank_level_num):
            for rel_solution in source_quality_solutions[ref_quality_levels[index]]:
                for eval_solution in target_quality_solutions[eval_quality_levels[index]]:
                    ref_input.append(rel_solution)
                    eval_output.append(eval_solution)
        ref_input, eval_output = np.array(ref_input, dtype=np.float32), np.array(eval_output, dtype=np.float32)
        # print(len(ref_input), len(eval_output))
        indices = np.arange(len(ref_input))
        np.random.shuffle(indices)
        return ref_input[indices], eval_output[indices]

    def fine_tuning_mapping_decoder(self):
        logging.disable(logging.INFO)
        source_input, target_output = self.generate_cartesian_mapping_data(source_coefficient=self.source_coefficient)
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        with open("../configs/{}.yaml".format(self.config_name), 'r') as file:
            config = yaml.safe_load(file)
        config["model_params"]["in_dim"] = self.source_instance.dimension
        config["model_params"]["latent_dim"] = self.source_instance.dimension * config["model_params"][
            "latent_dim_coefficient"]
        config["trainer_params"]["gpus"] = [self.gpu_index]
        config["logging_params"]["name"] = "Mapping-{}_{}_{}2{}_{}_{}-{}-{}".format(
            self.source_instance.__class__.__name__, self.source_instance.dimension, self.source_index,
            self.target_instance.__class__.__name__, self.target_instance.dimension, self.target_index, self.model_name,
            self.sample_num)
        seed_everything(config['exp_params']['manual_seed'], True)
        train_dataloader = DataLoader(SolutionMappingData(source_input, target_output, "train"),
                                      batch_size=config['data_params']['train_batch_size'], shuffle=True,
                                      num_workers=config['data_params']['num_workers'])
        valid_dataloader = DataLoader(SolutionMappingData(source_input, target_output, "valid"),
                                      batch_size=config['data_params']['val_batch_size'], shuffle=True,
                                      num_workers=config['data_params']['num_workers'])
        log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
        if not os.path.exists(Path(config['logging_params']['save_dir'])):
            os.mkdir(Path(config['logging_params']['save_dir']))
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        np.save(str(Path(log_path, "source_input.npy")), source_input)
        np.save(str(Path(log_path, "target_output.npy")), target_output)
        writer = SummaryWriter(str(log_path))
        yaml.dump(config, open(Path(log_path, "HyperParam.yaml"), "w"))
        for (name, param) in self.mapping_model.named_parameters():
            param.requires_grad = False
        for (name, param) in self.mapping_model.decoder.named_parameters():
            param.requires_grad = True
        for (name, param) in self.mapping_model.final_layer.named_parameters():
            param.requires_grad = True
        for (name, param) in self.mapping_model.decoder_input.named_parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            chain(self.mapping_model.decoder.parameters(), self.mapping_model.final_layer.parameters(),
                  self.mapping_model.decoder_input.parameters()), lr=config['exp_params']['LR'],
            weight_decay=config['exp_params']['weight_decay'])

        # epoch_bar = tqdm(range(int(config['trainer_params']['max_epochs'])))
        best_val_loss = np.inf
        best_state_dict = self.mapping_model.state_dict()
        for epoch in range(int(config['trainer_params']['max_epochs'])):
            loss_records = {}
            for ref_solution, eval_solution in train_dataloader:
                optimizer.zero_grad()
                train_loss = self.mapping_model.mapping_loss_function(ref_solution.to(device), eval_solution.to(device))
                train_loss['loss'].backward()
                optimizer.step()
                for key in train_loss.keys():
                    if key not in loss_records:
                        loss_records[key] = []
                    loss_records[key].append(
                        train_loss[key] if key != "loss" else train_loss[key].cpu().detach().numpy())
            for ref_solution, eval_solution in valid_dataloader:
                valid_loss = self.mapping_model.mapping_loss_function(ref_solution.to(device), eval_solution.to(device))
                for key in valid_loss.keys():
                    if "val_{}".format(key) not in loss_records:
                        loss_records["val_{}".format(key)] = []
                    loss_records["val_{}".format(key)].append(
                        valid_loss[key] if key != "loss" else valid_loss[key].cpu().detach().numpy())
            if np.mean(loss_records['val_loss']) < best_val_loss:
                best_val_loss = np.mean(loss_records['val_loss'])
                best_state_dict = self.mapping_model.state_dict()

            for key in loss_records.keys():
                writer.add_scalar(key, np.mean(loss_records[key]), epoch)

            # epoch_bar.set_description("Epoch {}".format(epoch))
            # epoch_bar.set_postfix_str("MSE {:.5f}".format(np.mean(loss_records['loss'])))
        torch.save(best_state_dict, Path(log_path, "best_model.pt"))
        print("Finish the surrogate Task of {}_{} to {}_{}".format(self.source_instance.__class__.__name__,
                                                                   self.source_index,
                                                                   self.target_instance.__class__.__name__,
                                                                   self.target_index))

    def get_topk_target_solution(self, k=10, source_sample_num=2000000) -> NpArray:
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        # source_x, source_y = generate_data_from_problem_instance(self.source_instance, sample_num=source_sample_num)
        # source_data = ZeroOneProblemData(source_x, source_y, 'all')
        self.mapping_model.eval()
        topk_x, topk_y = [], []
        # source_dataloader = DataLoader(source_data, batch_size=1024, shuffle=False, num_workers=2)
        # for solution, _ in source_dataloader:
        #     solution = solution.to(device)
        for _ in range(source_sample_num // 1024):
            solution = torch.randint(0, 2, (1024, self.source_instance.dimension), dtype=torch.float32, device=device)
            for _ in range(1):
                target_output, mu, log_var, performance = self.mapping_model(solution)
                _, indices = torch.topk(performance, k=min(k * 10, 512))
                target_output = target_output[indices].cpu().detach().numpy()
                performance = performance[indices].cpu().detach().numpy()
                target_output = (target_output > 0.5).astype(np.int_)
                target_output, performance = list(target_output) + topk_x, list(performance) + topk_y
                sort_index, solution_str_set = np.argsort(performance)[::-1], set()
                topk_x, topk_y = [], []
                for index in sort_index:
                    solution_str = "".join([str(bit) for bit in target_output[index]])
                    if len(solution_str_set) >= k:
                        break
                    if solution_str not in solution_str_set:
                        solution_str_set.add(solution_str)
                        topk_x.append(target_output[index])
                        topk_y.append(performance[index])
        return np.array(topk_x)

    def get_random_k_target_solution(self, k=10) -> NpArray:
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        source_x, _ = generate_data_from_problem_instance(self.source_instance, sample_num=k)
        all_output, solution_str_set = [], set()
        for _ in range(10):
            if len(all_output) >= k:
                break
            target_output, _, _, _ = self.mapping_model(torch.from_numpy(source_x).to(device))
            target_output = target_output.cpu().detach().numpy()
            target_output = (target_output > 0.5).astype(np.int_)
            for solution in target_output:
                if len(all_output) >= k:
                    break
                solution_str = "".join([str(bit) for bit in solution])
                if solution_str not in solution_str_set:
                    solution_str_set.add(solution_str)
                    all_output.append(solution)
        return np.array(all_output)

    def eval_mapping_quality(self, topk: int = 10, source_sample_num=100000) -> NpArray:
        topk_x = self.get_topk_target_solution(k=topk, source_sample_num=source_sample_num)
        return np.array([self.target_instance.evaluate(x) for x in topk_x])

    def eval_random_quality(self, k=10):
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        exact_mapping_quality: List[List[float]] = [[] for _ in range(k)]
        source_x, _ = generate_data_from_problem_instance(self.source_instance, sample_num=k)
        for _ in range(50):
            target_output, _, _, _ = self.mapping_model(torch.from_numpy(source_x).to(device))
            target_output = target_output.cpu().detach().numpy()
            target_output = (target_output > 0.5).astype(np.int_)
            for i in range(k):
                exact_mapping_quality[i].append(self.target_instance.evaluate(target_output[i]))
        return np.array([np.mean(exact_mapping_quality[i]) for i in range(k)])

    def evaluate_correlation(self) -> Tuple[float, float]:
        device = torch.device("cuda:{}".format(self.gpu_index)) if self.gpu_index >= 0 else torch.device("cpu")
        self.source_model.eval()
        target_x, target_y = self.sample_target_data()
        if self.target_dimension > self.source_dimension:
            target_x = np.array([x[:self.source_dimension] for x in target_x])
        elif self.target_dimension < self.source_dimension:
            target_x = np.pad(target_x, pad_width=((0, 0), (0, self.source_dimension - self.target_dimension)),
                              mode="constant", constant_values=0)
        _, _, _, predict_y = self.source_model(torch.from_numpy(target_x).to(device))
        predict_y = predict_y.cpu().cpu().detach().numpy()
        rank_y = np.argsort(np.argsort(target_y)[::-1])
        predict_rank = np.argsort(np.argsort(predict_y)[::-1])
        pearson = pearsonr(target_y, predict_y)
        spearman = spearmanr(np.array(rank_y, dtype=np.int32), np.array(predict_rank, dtype=np.int32))
        return pearson.statistic, spearman.statistic


if __name__ == '__main__':
    problem_root_dir = "../data/problem_instance"
    source_problem_dir = Path(problem_root_dir, "train", "max_cut_problem_0_30")
    target_problem_dir = Path(problem_root_dir, "valid", "anchor_selection_problem_7_100")
    dp = DecoderMapping(source_problem_dir=source_problem_dir, target_problem_dir=target_problem_dir)
    # dp.load_mapping_model(sample_num=64)
    dp.get_topk_target_solution(source_sample_num=10000)