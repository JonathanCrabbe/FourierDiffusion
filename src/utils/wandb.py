from omegaconf import DictConfig

import wandb
from utils.extraction import flatten_config


def maybe_initialize_wandb(cfg: DictConfig) -> str | None:
    """Initialize wandb if necessary."""
    cfg_flat = flatten_config(cfg)
    if "pytorch_lightning.loggers.WandbLogger" in cfg_flat.values():
        wandb.init(
            project="FourierDiffusion",
            config=cfg_flat,
        )
        return wandb.run.name
    else:
        return None
