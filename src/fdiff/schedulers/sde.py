"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs. Adapted from https://github.com/yang-song/score_sde."""
import abc
import torch
import numpy as np
from torch import device
import math


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, fourier_noise_scaling=False):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.noise_scaling = fourier_noise_scaling

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    def initialize(self, max_len: int, device: str | device) -> None:
        """Finish the initialization of the scheduler by setting G (scaling diagonal) and the device.

        Args:
            max_len (_type_): _description_
            device (_type_): _description_
        """
        if not self.noise_scaling:
            # We will get the identity by putting G in the diagonal
            G = torch.ones(max_len, device=device)
        else:
            G = 1 / (math.sqrt(2 * max_len)) * torch.ones(max_len, device=device)
            # Double the variance for the first component
            G[0] *= math.sqrt(2)

        self.G = G  # Tensor of size (max_len)
        self.G_matrix = torch.diag(G)  # Tensor of size (max_len, max_len)
        assert G.shape[0] == max_len

        # Set the device
        self.device = device
