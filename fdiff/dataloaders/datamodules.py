import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from fdiff.utils.dataclasses import collate_batch


class DiffusionDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        data = {}
        data["X"] = self.X[index]
        if self.y is not None:
            data["y"] = self.y[index]
        return data


class Datamodule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_dir: Path = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
    ) -> None:
        self.data_dir = data_dir / self.dataset_name
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.X_train = torch.Tensor()
        self.y_train: Optional[torch.Tensor] = None
        self.X_test = torch.Tensor()
        self.y_test: Optional[torch.Tensor] = None

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}")
            os.makedirs(self.data_dir)
            self.download_data()
            logging.info(f"Dataset {self.dataset_name} downloaded in {self.data_dir}")

    @abstractmethod
    def download_data(self) -> None:
        """Download the data."""
        ...

    def train_dataloader(self) -> DataLoader:
        train_set = DiffusionDataset(X=self.X_train, y=self.y_train)
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        test_set = DiffusionDataset(X=self.X_test, y=self.y_test)
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )

    @abstractproperty
    def dataset_name(self) -> str:
        ...


class ECGDatamodule(Datamodule):
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

        # Read CSV; extract features and labels
        path_train = self.data_dir / "mitbih_train.csv"
        path_test = self.data_dir / "mitbih_test.csv"

        # Read data
        df_train = pd.read_csv(path_train)
        X_train = df_train.iloc[:, :187].values
        y_train = df_train.iloc[:, 187].values
        df_test = pd.read_csv(path_test)
        X_test = df_test.iloc[:, :187].values
        y_test = df_test.iloc[:, 187].values

        # Convert to tensor
        self.X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

    def download_data(self) -> None:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "shayanfazeli/heartbeat", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "ecg"
