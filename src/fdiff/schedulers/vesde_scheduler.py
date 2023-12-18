from fdiff.schedulers.sde import SDE
import torch
import numpy as np
from collections import namedtuple
import math

SamplingOutput = namedtuple("SamplingOutput", ["prev_sample"])


class VEScheduler(SDE):
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        fourier_noise_scaling=False,
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
    def T(self):
        return 1

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
    ):
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

    def prior_sampling(self, shape):
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
        )  # Notive that the drift of the forward is 0

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
