import numpy as np
import pytest
from ot.sliced import sliced_wasserstein_distance

from fdiff.sampling.metrics import MarginalWasserstein, SlicedWasserstein
from fdiff.utils.tensors import check_flat_array

random_seed = 42
n_channels = 1
max_len = 2
n_samples = 1000
num_directions = 1000
EPS = 1e-5

test_data_wasserstein = [0.0, 0.1, 1.0]


@pytest.mark.parametrize("shift", test_data_wasserstein)
def test_sliced_waserstein(shift: float) -> None:
    # Set random seed
    np.random.seed(random_seed)

    # Initialize two datasets
    dataset1 = np.random.rand(n_samples, max_len, n_channels)
    dataset2 = np.random.rand(n_samples, max_len, n_channels) + shift

    # Evaluate the sliced distance with pot
    pot_estimate = sliced_wasserstein_distance(
        X_t=check_flat_array(dataset1),
        X_s=check_flat_array(dataset2),
        n_projections=num_directions,
        seed=random_seed,
    )

    # Compute sliced wasserstein distance
    sw = SlicedWasserstein(
        original_samples=dataset1,
        random_seed=random_seed,
        num_directions=num_directions,
        save_all_distances=True,
    )
    metrics = sw(dataset2)

    assert (
        np.abs(
            metrics["sliced_wasserstein_mean"]
            - np.mean(metrics["sliced_wasserstein_all"])
        )
        <= EPS
    )
    assert metrics["sliced_wasserstein_mean"] <= metrics["sliced_wasserstein_max"]
    assert np.abs(metrics["sliced_wasserstein_mean"] - pot_estimate) <= 0.1


@pytest.mark.parametrize("shift", test_data_wasserstein)
def test_marginal_waserstein(shift: float) -> None:
    # Set random seed
    np.random.seed(random_seed)

    # Initialize two datasets
    dataset1 = np.random.rand(n_samples, max_len, n_channels)
    dataset2 = np.random.rand(n_samples, max_len, n_channels) + shift

    # Ground-truth distance
    ground_truth = shift

    # Compute sliced wasserstein distance
    mw = MarginalWasserstein(
        original_samples=dataset1, random_seed=random_seed, save_all_distances=True
    )
    metrics = mw(dataset2)

    assert (
        np.abs(
            metrics["marginal_wasserstein_mean"]
            - np.mean(metrics["marginal_wasserstein_all"])
        )
        <= EPS
    )
    assert metrics["marginal_wasserstein_mean"] <= metrics["marginal_wasserstein_max"]
    assert np.abs(metrics["marginal_wasserstein_mean"] - ground_truth) <= 0.1
    assert np.abs(metrics["marginal_wasserstein_max"] - ground_truth) <= 0.1
