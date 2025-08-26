import sklearn.metrics
import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from sklearn.metrics import mean_absolute_error


class Surrogate(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, **kwargs) -> None:
        super(Surrogate, self).__init__()

    def predict(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        pass

    def loss_function(self, X: Tensor, y: Tensor) -> dict:
        pass

    def mapping_loss_function(self, X: Tensor, y: Tensor) -> dict:
        pass