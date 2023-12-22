import pytest
from diffusers import DDPMScheduler

from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.schedulers.sde import SDE, VEScheduler, VPScheduler

n_channels = 3
max_len = 50
num_diffusion_steps = 10
batch_size = 12
num_samples = 48


@pytest.mark.parametrize(
    "noise_scheduler",
    [
        DDPMScheduler(
            num_train_timesteps=10,
        ),
        VPScheduler(),
        VEScheduler(),
    ],
)
def test_sampler(noise_scheduler: DDPMScheduler | SDE) -> None:
    # Create a score model
    score_model = ScoreModule(
        n_channels=n_channels, max_len=max_len, noise_scheduler=noise_scheduler
    )

    # In case of SDEs, set the noise scaling
    if isinstance(noise_scheduler, SDE):
        noise_scheduler.set_noise_scaling(max_len=max_len)

    # Create a sampler
    sampler = DiffusionSampler(score_model=score_model, sample_batch_size=batch_size)

    # Sample from the sampler
    samples = sampler.sample(
        num_samples=num_samples, num_diffusion_steps=num_diffusion_steps
    )

    # Check the shape of the samples
    assert samples.shape == (num_samples, max_len, n_channels)
