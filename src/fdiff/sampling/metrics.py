from abc import ABC, abstractmethod, abstractproperty

import numpy as np

from fdiff.utils.wasserstein import WassersteinDistances


class Metric(ABC):
    def __init__(self, original_samples: np.ndarray) -> None:
        self.original_samples = original_samples

    @abstractmethod
    def __call__(self, other_samples: np.ndarray) -> dict[str, float]:
        ...

    @abstractproperty
    def name(self) -> str:
        ...


class MetricCollection:
    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    def __call__(self, other_samples: np.ndarray) -> dict[str, float]:
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric(other_samples))
        return metric_dict


class SlicedWasserstein(Metric):
    def __init__(
        self, original_samples: np.ndarray, random_seed: int, num_directions: int
    ) -> None:
        super().__init__(original_samples=original_samples)
        self.random_seed = random_seed
        self.num_directions = num_directions

    def __call__(self, other_samples: np.ndarray) -> dict[str, float]:
        wd = WassersteinDistances(
            original_data=self.original_samples,
            other_data=other_samples,
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
        original_samples: np.ndarray,
        random_seed: int,
    ) -> None:
        super().__init__(original_samples=original_samples)
        self.random_seed = random_seed

    def __call__(self, other_samples: np.ndarray) -> dict[str, float]:
        wd = WassersteinDistances(
            original_data=self.original_samples,
            other_data=other_samples,
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
