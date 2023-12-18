from pathlib import Path

import torch

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.utils.dataclasses import DiffusableBatch
from fdiff.utils.fourier import idft

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
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.max_len = max_len
        self.n_channels = n_channels
        self.batch_size = batch_size

    def setup(self, stage: str = "fit") -> None:
        torch.manual_seed(self.random_seed)
        self.X_train = torch.randn(
            (10 * self.batch_size, self.max_len, self.n_channels), dtype=torch.float32
        )
        self.y_train = torch.randint(
            low=low, high=high, size=(10 * self.batch_size,), dtype=torch.long
        )
        self.X_test = torch.randn_like(self.X_train)
        self.y_test = torch.randint_like(self.y_train, low=low, high=high)
        self.compute_feature_statistics()

    def download_data(self) -> None:
        ...

    def compute_feature_statistics(self) -> None:
        """Compute the mean and standard deviation of the features, along the batch dimension."""
        self.feature_mean = self.X_train.mean(dim=0)
        self.feature_std = self.X_train.std(dim=0)

    @property
    def dataset_name(self) -> str:
        return "dummy"


def test_dataloader() -> None:
    datamodule = DummyDatamodule()
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        assert isinstance(batch, DiffusableBatch)
        assert batch.X.shape == (batch_size, max_len, n_channels)
        assert batch.y.shape == (batch_size,)


def test_fourier_transform() -> None:
    # Default datamodule
    datamodule = DummyDatamodule()
    datamodule.prepare_data()
    datamodule.setup()

    # Fourier datamodule
    datamodule_fourier = DummyDatamodule(fourier_transform=True)
    datamodule_fourier.prepare_data()
    datamodule_fourier.setup()

    X = datamodule.train_dataloader().dataset.X
    X_tilde = datamodule_fourier.train_dataloader().dataset.X

    assert torch.allclose(X, idft(X_tilde), atol=1e-5)


def test_standardization() -> None:
    # Default datamodule
    datamodule = DummyDatamodule(standardize=True)
    datamodule.prepare_data()
    datamodule.setup()

    train_dataset = datamodule.train_dataloader().dataset

    X_0 = train_dataset.X[0]
    X_0_standardized = train_dataset[0]["X"]
    X_0_unscaled = (
        X_0_standardized * train_dataset.feature_std + train_dataset.feature_mean
    )

    # Assert that X_train and X_unscaled are close
    assert X_0.shape == X_0_unscaled.shape

    assert torch.allclose(X_0, X_0_unscaled, atol=1e-5)

    val_dataset = datamodule.val_dataloader().dataset

    X_0 = val_dataset.X[0]
    X_0_standardized = val_dataset[0]["X"]
    X_0_unscaled = X_0_standardized * val_dataset.feature_std + val_dataset.feature_mean

    # Assert that X_train and X_unscaled are close
    assert X_0.shape == X_0_unscaled.shape

    assert torch.allclose(X_0, X_0_unscaled, atol=1e-5)

    assert torch.allclose(
        val_dataset.feature_mean, train_dataset.feature_mean, atol=1e-5
    )
    assert torch.allclose(val_dataset.feature_std, train_dataset.feature_std, atol=1e-5)
