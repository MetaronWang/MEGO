import sklearn.metrics
import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from sklearn.metrics import mean_absolute_error
from src.regression.surrogate import Surrogate


class SurrogateAE(Surrogate):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 mlp_dims: List = None,
                 lamBDa: float = 1,
                 margin: float = 2,
                 **kwargs) -> None:
        super(SurrogateAE, self).__init__()
        self.input_dim = in_dim
        self.latent_dim = latent_dim
        self.lamBDa = lamBDa

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 128]

        if mlp_dims is None:
            mlp_dims = [64, 128, 128]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.Hardtanh(min_val=-margin / 2, max_val=margin / 2)
        )
        modules = []
        in_dim = latent_dim
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

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        z = self.fc(result)

        return z

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def predict(self, input: Tensor) -> Tensor:
        result = self.performance_predictor(input)
        return result

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        z = self.encode(input)
        return [self.decode(z), self.performance_predictor(z)]

    def loss_function(self, X: Tensor, y: Tensor) -> dict:
        self.num_iter += 1
        forward_result = self.forward(X)
        recons = forward_result[0]
        y_prime = forward_result[1]

        recons_loss = F.mse_loss(recons, X)
        performance_loss = F.mse_loss(y_prime, y)

        mae_loss = mean_absolute_error(y_prime.cpu().detach().numpy(), y.cpu().detach().numpy())

        loss = recons_loss + self.lamBDa * performance_loss

        return {'loss': loss, "MSE_Loss": performance_loss.cpu().detach().numpy(), "MAE_Loss": mae_loss,
                "Reconstruction_Loss": recons_loss.cpu().detach().numpy()}
