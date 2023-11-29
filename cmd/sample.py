import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.extraction import dict_to_str, get_best_checkpoint


class SamplingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        self.random_seed: int = cfg.random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the sampling script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Instantiate datamodule and random seed
        self.datamodule: Datamodule = instantiate(cfg.datamodule)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        # Get model path and id
        self.model_path = Path(cfg.model_path)
        self.model_id = cfg.model_id

        # Get number of steps and samples
        self.num_samples: int = cfg.num_samples
        self.num_diffusion_steps: int = cfg.num_diffusion_steps

        # Load score model from checkpoint
        chekcpoint_dir = self.model_path / self.model_id / "checkpoints"
        best_checkpoint_path = get_best_checkpoint(chekcpoint_dir)

        self.score_model = ScoreModule.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path
        )
        if torch.cuda.is_available():
            self.score_model.to(device=torch.device("cuda"))

        # Instantiate sampler
        sampler_partial = instantiate(cfg.sampler)
        self.sampler: DiffusionSampler = sampler_partial(score_model=self.score_model)

        # Instantiate metrics
        metrics_partial = instantiate(cfg.metrics)
        self.metrics = metrics_partial(original_samples=self.datamodule.X_train)

    def sample(self) -> None:
        X = self.sampler.sample(
            num_samples=self.num_samples, num_diffusion_steps=self.num_diffusion_steps
        )
        results = self.metrics(X)
        logging.info(f"Metrics:\n{dict_to_str(results)}")
        logging.info(
            f"Saving samples to {self.model_path / self.model_id / 'samples.pt'}"
        )
        torch.save(X, self.model_path / self.model_id / "samples.pt")


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    runner = SamplingRunner(cfg)
    runner.sample()


if __name__ == "__main__":
    main()
