import math
from typing import Optional

import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from fdiff.models.score_models import ScoreModule
from fdiff.schedulers.custom_ddpm_scheduler import CustomDDPMScheduler
from fdiff.schedulers.vesde_scheduler import VEScheduler
from fdiff.schedulers.vpsde_scheduler import VPScheduler
from fdiff.utils.dataclasses import DiffusableBatch


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModule,
        sample_batch_size: int,
    ) -> None:
        self.score_model = score_model
        self.noise_scheduler = score_model.noise_scheduler

        # Disable clipping for the noise scheduler
        if isinstance(self.noise_scheduler, (DDPMScheduler, CustomDDPMScheduler)):
            self.noise_scheduler.config.clip_sample = False

        self.sample_batch_size = sample_batch_size
        self.n_channels = score_model.n_channels
        self.max_len = score_model.max_len

    def reverse_diffusion_step(self, batch: DiffusableBatch) -> torch.Tensor:
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # Check the validity of the timestep (current implementation assumes same time for all samples)
        assert timesteps is not None and timesteps.size(0) == len(batch)
        assert torch.min(timesteps) == torch.max(timesteps)

        # Predict score for the current batch
        score = self.score_model(batch)
        # Apply a step of reverse diffusion
        output = self.noise_scheduler.step(
            model_output=score, timestep=timesteps[0].item(), sample=X
        )

        X_prev = output.prev_sample
        assert isinstance(X_prev, torch.Tensor)

        return X_prev

    def sample(
        self, num_samples: int, num_diffusion_steps: Optional[int] = None
    ) -> torch.Tensor:
        # Set the score model in eval mode and move it to GPU
        self.score_model.eval()

        # If the number of diffusion steps is not provided, use the number of training steps
        num_diffusion_steps = (
            self.score_model.num_training_steps
            if num_diffusion_steps is None
            else num_diffusion_steps
        )
        self.noise_scheduler.set_timesteps(num_diffusion_steps)

        # Create the list that will store the samples
        all_samples = []

        # Compute the required amount of batches
        num_batches = max(1, num_samples // self.sample_batch_size)

        # No need to track gradients when sampling
        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc="Sampling",
                unit="batch",
                leave=False,
                colour="blue",
            ):
                # Compute the batch size
                batch_size = min(
                    num_samples - batch_idx * self.sample_batch_size,
                    self.sample_batch_size,
                )
                # Sample from noise distribution
                X = self.sample_prior(batch_size)

                # Perform the diffusion step by step
                for t in tqdm(
                    self.noise_scheduler.timesteps,
                    desc="Diffusion",
                    unit="step",
                    leave=False,
                    colour="green",
                ):
                    # Define timesteps for the batch
                    timesteps = torch.full(
                        (batch_size,),
                        t,
                        dtype=torch.long
                        if isinstance(t.item(), int)
                        else torch.float32,
                        device=self.score_model.device,
                        requires_grad=False,
                    )
                    # Create diffusable batch
                    batch = DiffusableBatch(X=X, y=None, timesteps=timesteps)
                    # Return denoised X

                    X = self.reverse_diffusion_step(batch)

                # Add the samples to the list
                all_samples.append(X.cpu())

        return torch.cat(all_samples, dim=0)

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        # depending on the scheduler, the prior distribution might be different
        if isinstance(self.noise_scheduler, DDPMScheduler):
            X = torch.randn(
                (batch_size, self.max_len, self.n_channels),
                device=self.score_model.device,
                requires_grad=False,
            )

        elif isinstance(self.noise_scheduler, VPScheduler) or isinstance(
            self.noise_scheduler, VEScheduler
        ):
            X = self.noise_scheduler.prior_sampling(
                (batch_size, self.max_len, self.n_channels)
            )

        elif isinstance(self.noise_scheduler, CustomDDPMScheduler):
            X = torch.randn(
                (batch_size, self.max_len, self.n_channels),
                device=self.score_model.device,
                requires_grad=False,
            )
            G = (
                1
                / (math.sqrt(2 * self.max_len))
                * torch.eye(self.max_len, device=self.score_model.device)
            )
            G[0, 0] *= math.sqrt(2)

            # Scale by the covariance matrix
            X = torch.matmul(G, X)

        else:
            raise NotImplementedError("Scheduler not recognized.")

        return X
