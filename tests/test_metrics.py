import numpy as np
import pytest

from fdiff.sampling.metrics import MarginalWasserstein, SlicedWasserstein

random_seed = 42
n_dim = 2
n_samples = 10000

test_data_wasserstein = [0.0, 0.1, 1.0, 5.0]


@pytest.mark.parametrize("shift", test_data_wasserstein)
def test_sliced_waserstein(shift: float):
    # Set random seed
    np.random.seed(random_seed)

    # Initialize two datasets
    dataset1 = np.random.rand(n_samples, n_dim)
    dataset2 = np.random.rand(n_samples, n_dim) + shift

    # Ground-truth distance
    ground_truth = shift

    # Compute sliced wasserstein distance
    sw = SlicedWasserstein(
        original_samples=dataset1, random_seed=random_seed, num_directions=100
    )
    metrics = sw(dataset2)

    assert metrics["sliced_wasserstein_mean"] <= metrics["sliced_wasserstein_max"]
    assert np.abs(metrics["sliced_wasserstein_mean"] - ground_truth) <= 0.1


@pytest.mark.parametrize("shift", test_data_wasserstein)
def test_marginal_waserstein(shift: float):
    # Set random seed
    np.random.seed(random_seed)

    # Initialize two datasets
    dataset1 = np.random.rand(n_samples, n_dim)
    dataset2 = np.random.rand(n_samples, n_dim) + shift

    # Ground-truth distance
    ground_truth = shift

    # Compute sliced wasserstein distance
    mw = MarginalWasserstein(original_samples=dataset1, random_seed=random_seed)
    metrics = mw(dataset2)

    assert metrics["marginal_wasserstein_mean"] <= metrics["marginal_wasserstein_max"]
    assert np.abs(metrics["marginal_wasserstein_mean"] - ground_truth) <= 0.1
    assert np.abs(metrics["marginal_wasserstein_max"] - ground_truth) <= 0.1
