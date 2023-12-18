import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Callable, Optional
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from fdiff.utils.losses import get_sde_loss_fn, get_ddpm_loss
from fdiff.models.transformer import (
    PositionalEncoding,
    TimeEncoding,
    GaussianFourierProjection,
)
from fdiff.utils.dataclasses import DiffusableBatch
from fdiff.schedulers.custom_ddpm_scheduler import CustomDDPMScheduler
from fdiff.schedulers.vpsde_scheduler import VPScheduler
from fdiff.schedulers.vesde_scheduler import VEScheduler


class ScoreModule(pl.LightningModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        noise_scheduler: DDPMScheduler
        | CustomDDPMScheduler
        | VPScheduler
        | VEScheduler,
        fourier_noise_scaling: bool = True,
        d_model: int = 60,
        num_layers: int = 3,
        n_head: int = 12,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
        likelihood_weighting: bool = False,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.max_len = max_len
        self.n_channels = n_channels

        self.noise_scheduler = noise_scheduler
        self.num_warmup_steps = num_training_steps // 10
        self.num_training_steps = num_training_steps
        self.lr_max = lr_max
        self.d_model = d_model
        self.scale_noise = fourier_noise_scaling

        # Loss function
        self.likelihood_weighting = likelihood_weighting
        self.training_loss_fn, self.validation_loss_fn = self.set_loss_fn()

        # Model components
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.max_len)
        self.time_encoder = self.set_time_encoder()
        self.embedder = nn.Linear(in_features=n_channels, out_features=d_model)
        self.unembedder = nn.Linear(in_features=d_model, out_features=n_channels)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add positional encoding
        X = self.pos_encoder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone
        X = self.backbone(X)

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)

        return X

    def training_step(
        self, batch: DiffusableBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)

        self.log_dict(
            {"train/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=True,
        )
        return loss

    def validation_step(
        self, batch: DiffusableBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def set_loss_fn(self) -> tuple[Callable, Callable]:
        # depending on the scheduler, get the right loss function
        if isinstance(self.noise_scheduler, DDPMScheduler):
            self.max_time = self.noise_scheduler.config.num_train_timesteps

            training_loss_fn = get_ddpm_loss(
                scheduler=self.noise_scheduler, train=True, max_time=self.max_time
            )
            validation_loss_fn = get_ddpm_loss(
                scheduler=self.noise_scheduler, train=False, max_time=self.max_time
            )
            return training_loss_fn, validation_loss_fn

        elif isinstance(self.noise_scheduler, VPScheduler) or isinstance(
            self.noise_scheduler, VEScheduler
        ):
            training_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=True,
                likelihood_weighting=self.likelihood_weighting,
            )
            validation_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=False,
                likelihood_weighting=self.likelihood_weighting,
            )

            return training_loss_fn, validation_loss_fn

        else:
            raise NotImplementedError(
                "Scheduler not implemented yet, cannot set loss function"
            )

    def set_time_encoder(self) -> nn.Module:
        if isinstance(self.noise_scheduler, DDPMScheduler):
            return TimeEncoding(d_model=self.d_model, max_time=self.max_time)

        elif isinstance(self.noise_scheduler, VPScheduler) or isinstance(
            self.noise_scheduler, VEScheduler
        ):
            return GaussianFourierProjection(d_model=self.d_model)

        else:
            raise NotImplementedError(
                "Scheduler not implemented yet, cannot set loss function"
            )
