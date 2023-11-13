from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class DiffusableBatch:
    X: torch.Tensor
    y: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def device(self) -> torch.device:
        return self.X.device


def collate_batch(data: list[dict[str, torch.Tensor]]) -> DiffusableBatch:
    assert "X" in data[0], "The construction of a batch requires a 'X' key."

    X = torch.stack([example["X"] for example in data])
    y = torch.stack([example["y"] for example in data]) if "y" in data[0] else None
    timesteps = (
        torch.stack([example["timestep"] for example in data])
        if "timestep" in data[0]
        else None
    )
    return DiffusableBatch(X=X, y=y, timesteps=timesteps)
