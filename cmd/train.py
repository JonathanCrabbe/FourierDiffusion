import logging
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.dataloaders.datamodules import Datamodule
from src.models.score_models import ScoreModule


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Instatiate all the components
        self.score_model: ScoreModule = instantiate(cfg.score_model)
        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.datamodule: Datamodule = instantiate(cfg.datamodule)

        # Set-up dataset
        self.datamodule.prepare_data()
        self.datamodule.setup()

        # Finish instantiation of the model if necessary
        if isinstance(self.score_model, partial):
            self.score_model = self.score_model(**self.datamodule.dataset_parameters)

    def train(self) -> None:
        self.trainer.fit(model=self.score_model, datamodule=self.datamodule)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
