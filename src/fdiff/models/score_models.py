import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from fdiff.models.transformer import PositionalEncoding, TimeEncoding
from fdiff.utils.dataclasses import DiffusableBatch


class ScoreModule(pl.LightningModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        noise_scheduler: DDPMScheduler,
        d_model: int = 60,
        num_layers: int = 3,
        n_head: int = 12,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.max_len = max_len
        self.n_channels = n_channels
        assert hasattr(noise_scheduler, "config")
        self.max_time = noise_scheduler.config.num_train_timesteps
        assert isinstance(self.max_time, int)
        self.noise_scheduler = noise_scheduler
        self.num_warmup_steps = num_training_steps // 10
        self.num_training_steps = num_training_steps
        self.lr_max = lr_max

        # Model components
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.max_len)
        self.time_encoder = TimeEncoding(d_model=d_model, max_time=self.max_time)
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
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # If no timesteps are provided, sample them randomly
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.max_time,
                size=(len(batch),),
                dtype=torch.long,
                device=batch.device,
            )

        # Sample noise from distribution and add it to X
        noise = torch.randn_like(X, device=batch.device)
        assert hasattr(self.noise_scheduler, "add_noise")
        X_noisy = self.noise_scheduler.add_noise(
            original_samples=X, noise=noise, timesteps=timesteps
        )
        noisy_batch = DiffusableBatch(X=X_noisy, y=batch.y, timesteps=timesteps)

        # Predict noise from score model
        noise_pred = self.forward(noisy_batch)
        loss = F.mse_loss(noise_pred, noise)
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
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # If no timesteps are provided, sample them randomly
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.max_time,
                size=(len(batch),),
                dtype=torch.long,
                device=batch.device,
            )

        # Sample noise from distribution and add it to X
        noise = torch.randn_like(X, device=batch.device)
        assert hasattr(self.noise_scheduler, "add_noise")
        X_noisy = self.noise_scheduler.add_noise(
            original_samples=X, noise=noise, timesteps=timesteps
        )
        noisy_batch = DiffusableBatch(X=X_noisy, y=batch.y, timesteps=timesteps)

        # Predict noise from score model
        noise_pred = self.forward(noisy_batch)
        loss = F.mse_loss(noise_pred, noise)
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
