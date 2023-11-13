from pathlib import Path

import torch

from src.dataloaders.datamodules import Datamodule
from src.utils.dataclasses import DiffusableBatch

max_len = 30
n_channels = 3
batch_size = 32
low = 0
high = 10


class DummyDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = batch_size,
        max_len: int = max_len,
        n_channels: int = n_channels,
    ) -> None:
        super().__init__(
            data_dir=data_dir, random_seed=random_seed, batch_size=batch_size
        )
        self.max_len = max_len
        self.n_channels = n_channels
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.X_train = torch.randn(
            (10 * self.batch_size, self.max_len, self.n_channels), dtype=torch.float32
        )
        self.y_train = torch.randint(
            low=low, high=high, size=(10 * self.batch_size,), dtype=torch.long
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
        assert batch.X.shape == (batch_size, max_len, n_channels)
        assert batch.y.shape == (batch_size,)
