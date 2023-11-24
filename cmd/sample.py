import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.score_models import ScoreModule
from sampling.sampler import DiffusionSampler
from utils.extraction import flatten_config, get_best_checkpoint


class SamplingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the sampling script! You are using the following config:\n{flatten_config(cfg)}"
        )

        # Model path and id
        self.model_path = Path(cfg.model_path)
        self.model_id = cfg.model_id

        # Number of steps and samples
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

    def sample(self) -> None:
        X = self.sampler.sample(
            num_samples=self.num_samples, num_diffusion_steps=self.num_diffusion_steps
        )
        torch.save(X, self.model_path / self.model_id / "samples.pt")


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    runner = SamplingRunner(cfg)
    runner.sample()


if __name__ == "__main__":
    main()
