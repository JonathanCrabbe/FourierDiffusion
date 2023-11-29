from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from typing import Optional

import numpy as np
import torch

from fdiff.utils.tensors import check_flat_array
from fdiff.utils.wasserstein import WassersteinDistances


class Metric(ABC):
    def __init__(self, original_samples: np.ndarray | torch.Tensor) -> None:
        self.original_samples = check_flat_array(original_samples)

    @abstractmethod
    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, float]:
        ...

    @abstractproperty
    def name(self) -> str:
        ...

    @property
    def baseline_metrics(self) -> dict[str, float]:
        return {}


class MetricCollection:
    def __init__(
        self,
        metrics: list[Metric],
        original_samples: Optional[np.ndarray | torch.Tensor] = None,
        include_baselines: bool = True,
    ) -> None:
        for i, metric in enumerate(metrics):
            # If metric is partially instantiated, instantiate it with original samples
            if isinstance(metric, partial):
                assert (
                    original_samples is not None
                ), f"Original samples must be provided for metric {metric.name} to be instantiated."
                metrics[i] = metric(original_samples=original_samples)  # type: ignore
        self.metrics = metrics
        self.include_baselines = include_baselines

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, float]:
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric(other_samples))
        if self.include_baselines:
            metric_dict.update(self.baseline_metrics)
        return dict(sorted(metric_dict.items(), key=lambda item: item[0]))

    @property
    def baseline_metrics(self) -> dict[str, float]:
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.baseline_metrics)
        return metric_dict


class SlicedWasserstein(Metric):
    def __init__(
        self,
        original_samples: np.ndarray | torch.Tensor,
        random_seed: int,
        num_directions: int,
    ) -> None:
        super().__init__(original_samples=original_samples)
        self.random_seed = random_seed
        self.num_directions = num_directions

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, float]:
        wd = WassersteinDistances(
            original_data=self.original_samples,
            other_data=check_flat_array(other_samples),
            seed=self.random_seed,
        )
        distances = wd.sliced_distances(self.num_directions)
        return {
            "sliced_wasserstein_mean": float(np.mean(distances)),
            "sliced_wasserstein_max": float(np.max(distances)),
        }

    @property
    def baseline_metrics(self) -> dict[str, float]:
        # Compute the Wasserstein distance between 2 folds of the original samples
        n_samples = self.original_samples.shape[0]
        wd_self = WassersteinDistances(
            original_data=self.original_samples[: n_samples // 2],
            other_data=self.original_samples[n_samples // 2 :],
            seed=self.random_seed,
        )
        distances_self = wd_self.sliced_distances(self.num_directions)

        # Compute the Wasserstein distance with a generator that only outputs the average sample
        avg_sample = np.mean(self.original_samples, axis=0, keepdims=True)
        wd_dummy = WassersteinDistances(
            original_data=self.original_samples,
            other_data=avg_sample,
            seed=self.random_seed,
        )
        distances_dummy = wd_dummy.sliced_distances(self.num_directions)

        # Return the baselines as a dict
        return {
            "sliced_wasserstein_mean_self": float(np.mean(distances_self)),
            "sliced_wasserstein_max_self": float(np.max(distances_self)),
            "sliced_wasserstein_mean_dummy": float(np.mean(distances_dummy).item()),
            "sliced_wasserstein_max_dummy": float(np.max(distances_dummy)),
        }

    @property
    def name(self) -> str:
        return "sliced_wasserstein"


class MarginalWasserstein(Metric):
    def __init__(
        self,
        original_samples: np.ndarray | torch.Tensor,
        random_seed: int,
    ) -> None:
        super().__init__(original_samples=original_samples)
        self.random_seed = random_seed

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, float]:
        wd = WassersteinDistances(
            original_data=self.original_samples,
            other_data=check_flat_array(other_samples),
            seed=self.random_seed,
        )
        distances = wd.marginal_distances()
        return {
            "marginal_wasserstein_mean": float(np.mean(distances)),
            "marginal_wasserstein_max": float(np.max(distances)),
        }

    @property
    def baseline_metrics(self) -> dict[str, float]:
        # Compute the Wasserstein distance between 2 folds of the original samples
        n_samples = self.original_samples.shape[0]
        wd_self = WassersteinDistances(
            original_data=self.original_samples[: n_samples // 2],
            other_data=self.original_samples[n_samples // 2 :],
            seed=self.random_seed,
        )
        distances_self = wd_self.marginal_distances()

        # Compute the Wasserstein distance with a generator that only outputs the average sample
        avg_sample = np.mean(self.original_samples, axis=0, keepdims=True)
        wd_dummy = WassersteinDistances(
            original_data=self.original_samples,
            other_data=avg_sample,
            seed=self.random_seed,
        )
        distances_dummy = wd_dummy.marginal_distances()

        # Return the baselines as a dict
        return {
            "marginal_wasserstein_mean_self": float(np.mean(distances_self)),
            "marginal_wasserstein_max_self": float(np.max(distances_self)),
            "marginal_wasserstein_mean_dummy": float(np.mean(distances_dummy)),
            "marginal_wasserstein_max_dummy": float(np.max(distances_dummy)),
        }

    @property
    def name(self) -> str:
        return "marginal_wasserstein"
