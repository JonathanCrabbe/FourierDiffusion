from omegaconf import DictConfig

import wandb
from fdiff.utils.extraction import flatten_config


def maybe_initialize_wandb(cfg: DictConfig) -> str | None:
    """Initialize wandb if necessary."""
    cfg_flat = flatten_config(cfg)
    if "pytorch_lightning.loggers.WandbLogger" in cfg_flat.values():
        wandb.init(
            project="FourierDiffusion",
            config=cfg_flat,
        )
        assert wandb.run is not None
        run_id = wandb.run.id
        assert isinstance(run_id, str)
        return run_id
    else:
        return None
