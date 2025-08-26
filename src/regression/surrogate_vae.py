import sklearn.metrics
import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from sklearn.metrics import mean_absolute_error
from src.regression.surrogate import Surrogate


class SurrogateVAE(Surrogate):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 mlp_dims: List = None,
                 lamBDa: float = 1,
                 gamma: float = 0.0025,
                 margin: float = 2,
                 out_dim: int = None,
                 **kwargs) -> None:
        super(SurrogateVAE, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim if out_dim else in_dim
        self.latent_dim = latent_dim
        self.lamBDa = lamBDa
        self.gamma = gamma

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
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

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
            nn.Linear(hidden_dims[-1], self.output_dim),
            nn.BatchNorm1d(self.output_dim),
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
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Log variance of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var, self.performance_predictor(z)]

    def loss_function(self, X: Tensor, y: Tensor) -> dict:
        self.num_iter += 1
        forward_result = self.forward(X)
        recons = forward_result[0]
        mu = forward_result[1]
        log_var = forward_result[2]
        y_prime = forward_result[3]

        recons_loss = F.mse_loss(recons, X)
        performance_loss = F.mse_loss(y_prime, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        mae_loss = mean_absolute_error(y_prime.cpu().detach().numpy(), y.cpu().detach().numpy())

        loss = recons_loss + self.lamBDa * performance_loss + self.gamma * kld_loss

        return {'loss': loss, "MSE_Loss": performance_loss.cpu().detach().numpy(), "MAE_Loss": mae_loss,
                "Reconstruction_Loss": recons_loss.cpu().detach().numpy(), "KLD_Loss": kld_loss.cpu().detach().numpy()}

    def mapping_loss_function(self, X: Tensor, y: Tensor) -> dict:
        self.num_iter += 1
        forward_result = self.forward(X)
        recons = forward_result[0]

        recons_loss = F.mse_loss(recons, y)

        loss = recons_loss

        return {'loss': loss, "Reconstruction_Loss": recons_loss.cpu().detach().numpy()}
