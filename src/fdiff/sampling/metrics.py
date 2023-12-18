from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fdiff.utils.fourier import dft
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
        metrics_time: list[Metric] = []
        metrics_freq: list[Metric] = []

        original_samples_freq = (
            dft(original_samples) if original_samples is not None else None
        )

        for metric in metrics:
            # If metric is partially instantiated, instantiate it with original samples
            if isinstance(metric, partial):
                assert (
                    original_samples is not None
                ), f"Original samples must be provided for metric {metric.name} to be instantiated."
                metrics_time.append(metric(original_samples=original_samples))  # type: ignore
                metrics_freq.append(metric(original_samples=original_samples_freq))  # type: ignore
        self.metrics_time = metrics_time
        self.metrics_freq = metrics_freq
        self.include_baselines = include_baselines

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, float]:
        metric_dict = {}
        other_samples_freq = dft(other_samples)
        for metric_time, metric_freq in zip(self.metrics_time, self.metrics_freq):
            metric_dict.update(
                {f"time_{k}": v for k, v in metric_time(other_samples).items()}
            )
            metric_dict.update(
                {f"freq_{k}": v for k, v in metric_freq(other_samples_freq).items()}
            )
        if self.include_baselines:
            metric_dict.update(self.baseline_metrics)
        return dict(sorted(metric_dict.items(), key=lambda item: item[0]))

    @property
    def baseline_metrics(self) -> dict[str, float]:
        metric_dict = {}
        for metric_time, metric_freq in zip(self.metrics_time, self.metrics_freq):
            metric_dict.update(
                {f"time_{k}": v for k, v in metric_time.baseline_metrics.items()}
            )
            metric_dict.update(
                {f"freq_{k}": v for k, v in metric_freq.baseline_metrics.items()}
            )
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

    def save(self, other_samples: np.ndarray | torch.Tensor, path: str | Path) -> None:
        # Save the distances array for post-processing
        wd = WassersteinDistances(
            original_data=self.original_samples,
            other_data=check_flat_array(other_samples),
            seed=self.random_seed,
        )
        distances = wd.marginal_distances()
        np.save(path, distances)

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
