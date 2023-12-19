from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
import torch

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.schedulers.vpsde_scheduler import VPScheduler
from fdiff.utils.dataclasses import DiffusableBatch

n_head = 4
d_model = 8
n_channels = 3
max_len = 20
num_layers = 2
num_diffusion_steps = 10
low = 0
high = 10
num_samples = 48
beta_0 = 0.01
beta_1 = 20
batch_size = 50


def test_noise_adder():
    """Test the noise adder."""
    # Set the parameters
    beta_min = 0.01
    beta_max = 1
    max_len = 20
    G = torch.ones(max_len)
    G[0] *= 2

    # Create the SDE
    scheduler = VPScheduler(beta_min=beta_min, beta_max=beta_max)

    # Create a dummy time series
    x = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    noise = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    timesteps = torch.rand(size=(batch_size,), device="cpu")
    x_noisy = scheduler.add_noise(original_samples=x, noise=noise, timesteps=timesteps)

    assert x_noisy.shape == x.shape

    beta = scheduler.get_beta(timestep=timesteps)
    # Check that each element of beta is between beta_min and beta_max
    assert torch.all(beta >= beta_min)
    assert torch.all(beta <= beta_max)

    scheduler.set_timesteps(num_diffusion_steps=1000)

    model_output = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    timesteps = torch.ones(size=(batch_size,), device="cpu") * 0.5
    scheduler_output = scheduler.step(model_output, timestep=timesteps, sample=x_noisy)
    assert scheduler_output.prev_sample.shape == x_noisy.shape


def instantiate_score_model() -> ScoreModule:
    noise_scheduler = VPScheduler(
        beta_min=beta_0, beta_max=beta_1, fourier_noise_scaling=False
    )
    score_model = ScoreModule(
        n_channels=n_channels,
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


def test_score_module_with_vpsde():
    torch.manual_seed(42)
    score_model = instantiate_score_model()

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


class DummyDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = batch_size,
        max_len: int = max_len,
        n_channels: int = n_channels,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
        )
        self.max_len = max_len
        self.n_channels = n_channels
        self.batch_size = batch_size

    def setup(self, stage: str = "fit") -> None:
        torch.manual_seed(self.random_seed)
        self.X_train = torch.randn(
            (10 * self.batch_size, self.max_len, self.n_channels),
            dtype=torch.float32,
        )
        self.y_train = torch.randint(
            low=low, high=high, size=(10 * self.batch_size,), dtype=torch.long
        )
        self.X_test = torch.randn_like(self.X_train)
        self.y_test = torch.randint_like(self.y_train, low=low, high=high)
        self.compute_feature_statistics()

    def download_data(self) -> None:
        ...

    def compute_feature_statistics(self) -> None:
        """Compute the mean and standard deviation
        of the features, along the batch dimension."""
        self.feature_mean = self.X_train.mean(dim=0)
        self.feature_std = self.X_train.std(dim=0)

    @property
    def dataset_name(self) -> str:
        return "dummy"


if __name__ == "__main__":
    # test_noise_adder()
    test_score_module_with_vpsde()
