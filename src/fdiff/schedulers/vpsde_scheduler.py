from fdiff.schedulers.sde import SDE
import torch
import numpy as np
import math
from torch import device
from collections import namedtuple

SamplingOutput = namedtuple("SamplingOutput", ["prev_sample"])


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
    def T(self):
        return 1

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
    ):
        x0 = original_samples
        mean, _ = self.marginal_prob(x0, timesteps)

        # Note that the std is not used here because the noise has been scaled prior to calling the function
        sample = mean + noise
        return sample

    def get_beta(
        self, timestep: torch.Tensor | float | np.ndarray
    ) -> torch.Tensor | float | np.ndarray:
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
