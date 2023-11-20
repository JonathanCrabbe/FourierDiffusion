from copy import deepcopy

import pytorch_lightning as pl
import torch
from diffusers import DDPMScheduler

from src.models.score_models import ScoreModule
from src.utils.dataclasses import DiffusableBatch

from .test_datamodules import DummyDatamodule

n_head = 4
d_model = 8
n_chanels = 3
max_len = 20
num_layers = 2
batch_size = 5
n_diffusion_steps = 10


def instantiate_score_model() -> ScoreModule:
    noise_scheduler = DDPMScheduler(num_train_timesteps=n_diffusion_steps)
    score_model = ScoreModule(
        n_channels=n_chanels,
        max_len=max_len,
        noise_scheduler=noise_scheduler,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        num_training_steps=10,
    )
    return score_model


def instantiate_trainer() -> pl.Trainer:
    return pl.Trainer(max_epochs=1, accelerator="cpu")


def test_score_module():
    torch.manual_seed(42)
    score_model = instantiate_score_model()

    # Check that the forward call produces tensor of the right shape
    X = torch.randn((batch_size, max_len, n_chanels))
    timesteps = torch.randint(low=0, high=n_diffusion_steps, size=(batch_size,))
    batch = DiffusableBatch(X=X, timesteps=timesteps)
    score = score_model(batch)
    assert isinstance(score, torch.Tensor)
    assert score.size() == X.size()

    # Check that the training  updates the parameters
    trainer = instantiate_trainer()
    datamodule = DummyDatamodule(
        n_channels=n_chanels, max_len=max_len, batch_size=batch_size
    )
    params_before = deepcopy(score_model.state_dict())
    trainer.fit(model=score_model, datamodule=datamodule)
    params_after = deepcopy(score_model.state_dict())

    for param_name in params_before:
        assert not torch.allclose(
            params_before[param_name], params_after[param_name]
        ), f"Parameter {param_name} did not change during training"
