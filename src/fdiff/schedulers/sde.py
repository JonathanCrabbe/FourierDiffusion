"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs. Adapted from https://github.com/yang-song/score_sde."""
import abc
import math
from collections import namedtuple

import numpy as np
import torch
from torch import device

SamplingOutput = namedtuple("SamplingOutput", ["prev_sample"])


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, fourier_noise_scaling: bool = False):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.noise_scaling = fourier_noise_scaling

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""

    @abc.abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""

    @abc.abstractmethod
    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$."""

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


class VEScheduler(SDE):
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
    ):
        """Construct a Variance Exploding SDE.
        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(fourier_noise_scaling)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

        self.device = None
        self.G = None

    @property
    def T(self) -> float:
        return 1.0

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # perturbation kernel P(X(t)|X(0)) parameters
        if self.G is None:
            self.initialize(x.shape[1], x.device)

        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = (sigma_min * (sigma_max / sigma_min) ** t).view(-1, 1) * self.G
        mean = x
        return mean, std

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x0 = original_samples
        mean, _ = self.marginal_prob(x0, timesteps)

        # Note that the std is not used here because the noise has been scaled prior to calling the function
        sample = mean + noise
        return sample

    def get_sigma(
        self, timestep: torch.Tensor | float | np.ndarray
    ) -> torch.Tensor | float | np.ndarray:
        return torch.tensor(
            self.sigma_min * (self.sigma_max / self.sigma_min) ** timestep,
            device=self.device,
        )

    def set_timesteps(self, num_diffusion_steps: int) -> None:
        self.timesteps = torch.linspace(
            1.0, self.eps, num_diffusion_steps, device=self.device
        )
        self.step_size = self.timesteps[0] - self.timesteps[1]

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        # Reshape the G matrix to be (1, max_len, max_len)
        scaling_matrix = self.G_matrix.view(
            -1, self.G_matrix.shape[0], self.G_matrix.shape[1]
        )
        scaling_matrix = self.sigma_max * scaling_matrix

        z = torch.randn(*shape, device=self.device)
        # Return G@z where z \sim N(0,I)
        return torch.matmul(scaling_matrix, z)

    def step(
        self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """

        sqrt_derivative = (
            self.sigma_min
            * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
            * (self.sigma_max / self.sigma_min) ** (timestep)
        )

        diffusion = torch.diag_embed(sqrt_derivative * self.G)

        # Compute drift for the reverse
        drift = -(
            torch.matmul(diffusion * diffusion, model_output)
        )  # Notice that the drift of the forward is 0

        # Sample noise
        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output


class VPScheduler(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
    ):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
          G: tensor of size max_len
        """
        super().__init__(fourier_noise_scaling)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.eps = eps

        # To be initialized later
        self.device = None
        self.G = None

    @property
    def T(self) -> float:
        return 1.0

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # first check if G has been init.
        if self.G is None:
            self.initialize(x.shape[1], x.device)

        # Compute -1/2*\int_0^t \beta(s) ds
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )

        mean = (
            torch.exp(log_mean_coeff[(...,) + (None,) * len(x.shape[1:])]) * x
        )  # mean: (batch_size, max_len, n_channels)

        std = (
            torch.sqrt((1.0 - torch.exp(2.0 * log_mean_coeff.view(-1, 1)))) * self.G
        )  # std: (batch_size, max_len)

        return mean, std

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x0 = original_samples
        mean, _ = self.marginal_prob(x0, timesteps)

        # Note that the std is not used here because the noise has been scaled prior to calling the function
        sample = mean + noise
        return sample

    def get_beta(self, timestep: torch.Tensor | float | np.ndarray) -> torch.Tensor:
        return torch.tensor(
            self.beta_0 + timestep * (self.beta_1 - self.beta_0), device=self.device
        )

    def set_timesteps(self, num_diffusion_steps: int) -> None:
        self.timesteps = torch.linspace(
            1.0, self.eps, num_diffusion_steps, device=self.device
        )
        self.step_size = self.timesteps[0] - self.timesteps[1]

    def prior_sampling(self, shape: tuple | list | torch.Size) -> torch.Tensor:
        # Reshape the G matrix to be (1, max_len, max_len)
        scaling_matrix = self.G_matrix.view(
            -1, self.G_matrix.shape[0], self.G_matrix.shape[1]
        )
        z = torch.randn(*shape, device=self.device)

        # Return G@z where z \sim N(0,I)
        return torch.matmul(scaling_matrix, z)

    def step(
        self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """
        beta = self.get_beta(timestep)
        diffusion = torch.diag_embed(torch.sqrt(beta).view(-1, 1) * self.G)

        # Compute drift
        drift = -0.5 * beta.view(-1, 1, 1) * sample - (
            torch.matmul(diffusion * diffusion, model_output)
        )

        # Sample noise
        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output
