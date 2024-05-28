import os
from multiprocessing import Process
from pathlib import Path

import torch
import yaml
from pytorch_lightning.utilities.seed import seed_everything
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.regression.dataset import ZeroOneProblemData
from src.regression.surrogate import Surrogate
from src.regression.surrogate_ae import SurrogateAE
from src.regression.surrogate_mlp import SurrogateMLP
from src.regression.surrogate_vae import SurrogateVAE
from src.problem.base_problem import BaseProblem
from src.experiment_problem import load_problem_instance, load_problem_data
from src.types_ import *

all_model: Dict[str, Type[Surrogate]] = {
    "SurrogateMLP": SurrogateMLP,
    "SurrogateAE": SurrogateAE,
    "SurrogateVAE": SurrogateVAE
}


def train_task(x: NpArray, y: NpArray, problem_instance: BaseProblem, problem_index: int = 0,
               model_name: str = "SurrogateMLP", config_name: str = "surrogate_mlp", gpu_index: int = 0):
    device = torch.device("cuda:{}".format(gpu_index)) if gpu_index >= 0 else torch.device("cpu")
    with open("../configs/{}.yaml".format(config_name), 'r') as file:
        config = yaml.safe_load(file)
    config["model_params"]["in_dim"] = problem_instance.dimension
    config["model_params"]["latent_dim"] = problem_instance.dimension * config["model_params"]["latent_dim_coefficient"]
    config["trainer_params"]["gpus"] = [gpu_index]
    config["logging_params"]["name"] = "{}-{}-{}".format(
        problem_instance.__class__.__name__,
        problem_index,
        model_name,
    )
    seed_everything(config['exp_params']['manual_seed'], True)
    model = all_model[model_name](**config['model_params']).to(device)
    train_data = ZeroOneProblemData(x, y, 'train')
    valid_data = ZeroOneProblemData(x, y, 'valid')
    train_dataloader = DataLoader(train_data, batch_size=config['data_params']['train_batch_size'], shuffle=True,
                                  num_workers=config['data_params']['num_workers'])
    valid_dataloader = DataLoader(valid_data, batch_size=config['data_params']['val_batch_size'], shuffle=True,
                                  num_workers=config['data_params']['num_workers'])
    log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
    if not os.path.exists(Path(config['logging_params']['save_dir'])):
        os.mkdir(Path(config['logging_params']['save_dir']))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(str(log_path))
    yaml.dump(config, open(Path(log_path, "HyperParam.yaml"), "w"))
    optimizer = optim.Adam(model.parameters(),
                           lr=config['exp_params']['LR'],
                           weight_decay=config['exp_params']['weight_decay'])
    # writer.add_graph(model)
    # epoch_bar = tqdm(range(int(config['trainer_params']['max_epochs'])))
    best_val_loss = np.inf
    for epoch in range(int(config['trainer_params']['max_epochs'])):
        loss_records = {}
        for solution, quality in train_dataloader:
            optimizer.zero_grad()
            train_loss = model.loss_function(solution.to(device), quality.to(device))
            train_loss['loss'].backward()
            optimizer.step()
            for key in train_loss.keys():
                if key not in loss_records:
                    loss_records[key] = []
                loss_records[key].append(train_loss[key] if key != "loss" else train_loss[key].cpu().detach().numpy())
        for solution, quality in valid_dataloader:
            valid_loss = model.loss_function(solution.to(device), quality.to(device))
            for key in valid_loss.keys():
                if "val_{}".format(key) not in loss_records:
                    loss_records["val_{}".format(key)] = []
                loss_records["val_{}".format(key)].append(
                    valid_loss[key] if key != "loss" else valid_loss[key].cpu().detach().numpy())
        if np.mean(loss_records['val_loss']) < best_val_loss:
            best_val_loss = np.mean(loss_records['val_loss'])
            torch.save(model.state_dict(), Path(log_path, "best_model.pt"))

        for key in loss_records.keys():
            writer.add_scalar(key, np.mean(loss_records[key]), epoch)

        # epoch_bar.set_description("Epoch {}".format(epoch))
        # epoch_bar.set_postfix_str("MSE {:.5f}".format(np.mean(loss_records['loss'])))
    print("Finish the surrogate Task of {}_{}".format(problem_instance.__class__.__name__, problem_index))


def main():
    problem_root_dir = "../data/problem_instance"
    process_list = []
    useful_gpu = [0, 1, 2, 3, 4, 5, 6]
    for task_index, instance_path in enumerate(os.listdir(Path(problem_root_dir, "train"))):
        index = int(instance_path.split("_")[-2])
        problem_instance = load_problem_instance(problem_dir=Path(problem_root_dir, "train", instance_path))
        x, y = load_problem_data(problem_dir=Path(problem_root_dir, "train", instance_path))
        p = Process(target=train_task,
                    args=(x, y, problem_instance, index, "SurrogateVAE", "surrogate_vae",
                          useful_gpu[task_index % len(useful_gpu)]))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()


if __name__ == '__main__':
    main()
    # problem_root_dir = "../data/problem_instance"
    # problem_type = "max_cut_problem"
    # index = 0
    # problem_instance = load_problem_instance(problem_dir=problem_root_dir, problem_type=problem_type,
    #                                          instance_type="train", dimension=30, index=index)
    #
    # x, y = load_problem_data(problem_dir=problem_root_dir, problem_type=problem_type, instance_type="train",
    #                          dimension=30, index=index)
    # train_task(x, y, problem_instance, index, "SurrogateVAE", "surrogate_vae", 0)
