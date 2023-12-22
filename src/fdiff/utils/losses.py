from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from fdiff.schedulers.sde import SDE
from fdiff.utils.dataclasses import DiffusableBatch


# Courtesy of https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py
def get_sde_loss_fn(
    scheduler: SDE,
    train: bool,
    reduce_mean: bool = True,
    likelihood_weighting: bool = False,
) -> Callable[[nn.Module, DiffusableBatch], torch.Tensor]:
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model: nn.Module, batch: DiffusableBatch) -> torch.Tensor:
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        if train:
            model.train()
        else:
            model.eval()

        X = batch.X
        y = batch.y
        timesteps = batch.timesteps

        # Sample a time step uniformly from [eps, T]
        if timesteps is None:
            timesteps = (
                torch.rand(X.shape[0], device=X.device) * (scheduler.T - scheduler.eps)
                + scheduler.eps
            )

        # Sample the gaussian noise
        z = torch.randn_like(X)  # (batch_size, max_len, n_channels)

        _, std = scheduler.marginal_prob(X, timesteps)  # (batch_size, max_len)
        var = std**2  # (batch_size, max_len)

        std_matrix = torch.diag_embed(std)  # (batch_size, max_len, max_len)
        inverse_std_matrix = torch.diag_embed(1 / std)  # (batch_size, max_len, max_len)

        # compute Sigma^{1/2}z to be used for forward sampling: noise is x(t)
        noise = torch.matmul(std_matrix, z)  # (batch_size, max_len, n_channels)

        # compute Sigma^{-1/2}z to be used for the loss: target_noise is grad log p(x(t)|x(0))
        target_noise = torch.matmul(
            inverse_std_matrix, z
        )  # (batch_size, max_len, n_channels)

        # Do the perturbation
        X_noisy = scheduler.add_noise(
            original_samples=X, noise=noise, timesteps=timesteps
        )

        noisy_batch = DiffusableBatch(X=X_noisy, y=y, timesteps=timesteps)

        # Compute the score function
        score = model(noisy_batch)

        if not likelihood_weighting:
            # lambda(t) = E[||\grad log p(x(t)|x(0))||^2]

            # Compute 1/tr(\Sigma^{-1})
            weighting_factor = 1.0 / torch.sum(1.0 / var, dim=1)  # (batch_size,)
            assert weighting_factor.shape == (X.shape[0],)

            # 1/tr(\Sigma^{-1}) * ||s + \Sigma^{-1/2}z||^2
            losses = weighting_factor.view(-1, 1, 1) * torch.square(
                score + target_noise
            )

            # No relative minus size because:
            # log(p(x(t)|x(0))) = -1/2 * (x(t) -mean)^{T} Cov^{-1} (x(t) - mean) + C
            # grad log(p(x(t)|x(0))) = (-1) * Cov^{-1} (x(t) - mean)

            # Reduction
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        else:
            # Compute the Mahalanobis distance, cf. https://arxiv.org/pdf/2111.13606.pdf + https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf

            # 1) s - \grad log p(x)
            difference = score + target_noise  # (batch_size, max_len, n_channels)

            # 2) Sigma(s - \grad log p(x))
            scaled_difference = torch.matmul(std_matrix, difference)

            # 3) Compute the loss
            losses = torch.square(scaled_difference)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss(
    scheduler: DDPMScheduler, train: bool, max_time: int
) -> Callable[[nn.Module, DiffusableBatch], torch.Tensor]:
    def loss_fn(model: nn.Module, batch: DiffusableBatch) -> torch.Tensor:
        if train:
            model.train()
        else:
            model.eval()

        X = batch.X
        timesteps = batch.timesteps

        # If no timesteps are provided, sample them randomly
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=max_time,
                size=(len(batch),),
                dtype=torch.long,
                device=batch.device,
            )

        noise = torch.randn_like(X, device=batch.device)

        assert hasattr(scheduler, "add_noise")

        # Add the noise to obtain x(t) given x(0)
        X_noisy = scheduler.add_noise(
            original_samples=X, noise=noise, timesteps=timesteps
        )
        noisy_batch = DiffusableBatch(X=X_noisy, y=batch.y, timesteps=timesteps)

        # Predict noise from score model
        noise_pred = model(noisy_batch)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    return loss_fn
