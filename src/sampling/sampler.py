from typing import Optional

import torch

from models.score_models import ScoreModule
from utils.dataclasses import DiffusableBatch


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModule,
        sample_batch_size: int,
    ) -> None:
        self.score_model = score_model
        self.noise_scheduler = score_model.noise_scheduler
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
        num_batches = num_samples // self.sample_batch_size

        for batch_idx in range(num_batches):
            # Compute the batch size
            batch_size = min(
                num_samples - batch_idx * self.sample_batch_size, self.sample_batch_size
            )
            # Sample from noise distribution
            X = torch.randn(
                (batch_size, self.max_len, self.n_channels),
                device=self.score_model.device,
            )

            # Perform the diffusion step by step
            for t in self.noise_scheduler.timesteps:
                # Define timesteps for the batch
                print(t)

                timesteps = torch.full(
                    (batch_size,), t, dtype=torch.long, device=self.score_model.device
                )

                # Create diffusable batch
                batch = DiffusableBatch(X=X, y=None, timesteps=timesteps)

                # Return denoised X
                X = self.reverse_diffusion_step(batch)

            # Add the samples to the list
            all_samples.append(X.cpu())

        return torch.cat(all_samples, dim=0)
