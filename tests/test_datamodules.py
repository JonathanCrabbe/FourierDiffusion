from pathlib import Path

import torch

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.utils.dataclasses import DiffusableBatch

N = 30
M = 3
batch_size = 32
low = 0
high = 10


class DummyDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            data_dir=data_dir, random_seed=random_seed, batch_size=batch_size
        )

    def prepare_data(self) -> None:
        super().prepare_data()

        self.X_train = torch.randn((10 * batch_size, M, N), dtype=torch.float32)
        self.y_train = torch.randint(
            low=low, high=high, size=(10 * batch_size,), dtype=torch.long
        )
        self.X_test = torch.randn_like(self.X_train)
        self.y_test = torch.randint_like(self.y_train, low=low, high=high)

    def download_data(self) -> None:
        ...

    @property
    def dataset_name(self) -> str:
        return "dummy"


def test_dataloader():
    datamodule = DummyDatamodule()
    datamodule.prepare_data()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        assert isinstance(batch, DiffusableBatch)
        assert batch.X.shape == (batch_size, M, N)
        assert batch.y.shape == (batch_size,)
