from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.schedulers.sde import SDE, VEScheduler, VPScheduler
from fdiff.utils.dataclasses import DiffusableBatch

from .test_datamodules import DummyDatamodule

n_head = 4
d_model = 8
n_channels = 3
max_len = 20
num_layers = 2
num_diffusion_steps = 10
low = 0
high = 10
num_samples = 48
beta_min = 0.01
beta_max = 20
batch_size = 50


@pytest.mark.parametrize(
    "scheduler_type",
    [
        VEScheduler,
        VPScheduler,
    ],
)
def test_forward(scheduler_type: SDE) -> None:
    # Create the SDE
    scheduler: SDE = scheduler_type()

    # Create a dummy time series
    x = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    noise = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    timesteps = torch.rand(size=(batch_size,), device="cpu")
    x_noisy = scheduler.add_noise(original_samples=x, noise=noise, timesteps=timesteps)

    assert x_noisy.shape == x.shape


@pytest.mark.parametrize(
    "scheduler_type",
    [
        VEScheduler,
        VPScheduler,
    ],
)
def test_backward(scheduler_type: SDE) -> None:
    t = 0.5

    scheduler: SDE = scheduler_type()
    scheduler.set_noise_scaling(max_len=max_len)
    scheduler.set_timesteps(num_diffusion_steps=1000)

    noise = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    model_output = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")

    scheduler_output = scheduler.step(model_output, timestep=t, sample=noise)
    assert scheduler_output.prev_sample.shape == noise.shape


@pytest.mark.parametrize(
    "scheduler_type",
    [
        VEScheduler,
        VPScheduler,
    ],
)
def test_training(scheduler_type: SDE) -> None:
    torch.manual_seed(42)
    noise_scheduler = scheduler_type()
    score_model = instantiate_score_model(noise_scheduler)

    # Check that the forward call produces tensor of the right shape
    X = torch.randn((batch_size, max_len, n_channels))
    timesteps = torch.rand(size=(batch_size,))
    batch = DiffusableBatch(X=X, timesteps=timesteps)
    score = score_model(batch)
    assert isinstance(score, torch.Tensor)
    assert score.size() == X.size()

    # Check that the training  updates the parameters
    trainer = instantiate_trainer()
    datamodule = DummyDatamodule(
        n_channels=n_channels, max_len=max_len, batch_size=batch_size
    )
    params_before = deepcopy(score_model.state_dict())
    params_before = {k: v for k, v in params_before.items() if v.requires_grad}

    trainer.fit(model=score_model, datamodule=datamodule)
    params_after = deepcopy(score_model.state_dict())
    params_after = {k: v for k, v in params_after.items() if v.requires_grad}

    # only look at the params which require grad

    for param_name in params_before:
        assert not torch.allclose(
            params_before[param_name], params_after[param_name]
        ), f"Parameter {param_name} did not change during training"

    # Create a sampler
    sampler = DiffusionSampler(score_model=score_model, sample_batch_size=batch_size)

    # Sample from the sampler
    samples = sampler.sample(
        num_samples=num_samples, num_diffusion_steps=num_diffusion_steps
    )

    # Check the shape of the samples
    assert samples.shape == (num_samples, max_len, n_channels)


def instantiate_score_model(scheduler: SDE) -> ScoreModule:
    score_model = ScoreModule(
        n_channels=n_channels,
        max_len=max_len,
        noise_scheduler=scheduler,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        num_training_steps=10,
    )
    return score_model


def instantiate_trainer() -> pl.Trainer:
    return pl.Trainer(max_epochs=1, accelerator="cpu")
