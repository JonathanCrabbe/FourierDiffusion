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


class MetricCollection:
    def __init__(
        self,
        metrics: list[Metric],
        original_samples: Optional[np.ndarray | torch.Tensor] = None,
    ) -> None:
        for i, metric in enumerate(metrics):
            # If metric is partially instantiated, instantiate it with original samples
            if isinstance(metric, partial):
                assert (
                    original_samples is not None
                ), f"Original samples must be provided for metric {metric.name} to be instantiated."
                metrics[i] = metric(original_samples=original_samples)
        self.metrics = metrics

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> dict[str, float]:
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric(other_samples))
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
            "sliced_wasserstein_mean": np.mean(distances).item(),
            "sliced_wasserstein_max": np.max(distances),
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
            "marginal_wasserstein_mean": np.mean(distances).item(),
            "marginal_wasserstein_max": np.max(distances),
        }

    @property
    def name(self) -> str:
        return "marginal_wasserstein"
