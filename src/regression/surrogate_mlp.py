import sklearn.metrics
import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from sklearn.metrics import mean_absolute_error
from src.regression.surrogate import Surrogate


class SurrogateMLP(Surrogate):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_dim: int,
                 mlp_dims: List = None,
                 lamBDa: float = 1,
                 **kwargs) -> None:
        super(SurrogateMLP, self).__init__()

        self.input_dim = in_dim
        self.lamBDa = lamBDa

        modules = []

        if mlp_dims is None:
            mlp_dims = [100]

        # Build Decoder
        modules = []

        for mlp_dim in mlp_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, mlp_dim),
                    nn.BatchNorm1d(mlp_dim),
                )
            )
            in_dim = mlp_dim
        modules.append(
            nn.Sequential(
                nn.Linear(mlp_dims[-1], 1),
                nn.ReLU(),
                nn.Flatten(start_dim=0)
            )
        )
        self.performance_predictor = nn.Sequential(*modules)

    def predict(self, input: Tensor) -> Tensor:
        result = self.performance_predictor(input)
        return result

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return [self.performance_predictor(input)]

    def loss_function(self, X: Tensor, y: Tensor) -> dict:
        self.num_iter += 1
        y_prime = self.forward(X)[0]

        performance_loss = F.mse_loss(y_prime, y)

        mae_loss = mean_absolute_error(y_prime.cpu().detach().numpy(), y.cpu().detach().numpy())

        loss = performance_loss

        return {'loss': loss, "MSE_Loss": performance_loss.cpu().detach().numpy(), "MAE_Loss": mae_loss}
