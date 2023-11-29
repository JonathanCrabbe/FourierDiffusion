import logging
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.utils.extraction import dict_to_str, get_training_params
from fdiff.utils.wandb import maybe_initialize_wandb


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the training script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Maybe initialize wandb
        maybe_initialize_wandb(cfg)

        # Instatiate all the components
        self.score_model: ScoreModule = instantiate(cfg.score_model)
        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.datamodule: Datamodule = instantiate(cfg.datamodule)

        # Set-up dataset
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        # Finish instantiation of the model if necessary
        if isinstance(self.score_model, partial):
            training_params = get_training_params(self.datamodule, self.trainer)
            self.score_model = self.score_model(**training_params)

    def train(self) -> None:
        self.trainer.fit(model=self.score_model, datamodule=self.datamodule)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
