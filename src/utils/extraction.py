import re
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from dataloaders.datamodules import Datamodule


def get_training_params(datamodule: Datamodule, trainer: pl.Trainer) -> dict[str, Any]:
    params = datamodule.dataset_parameters
    params["num_training_steps"] *= trainer.max_epochs
    return params


def flatten_config(cfg: DictConfig | dict) -> dict[str, Any]:
    """Flatten a Hydra/dict config into a dict. This is useful for logging.

    Args:
        cfg (DictConfig | dict): Config to flatten.

    Returns:
        dict[str, Any]: Flatenned config.
    """
    cfg_dict = (
        OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, DictConfig)
        else cfg
    )

    cfg_flat: dict[str, Any] = {}
    for k, v in cfg_dict.items():
        # If the value is a dict, make a recursive call
        if isinstance(v, dict):
            if "_target_" in v:
                cfg_flat[k] = v["_target_"]
            cfg_flat.update(**flatten_config(v))
        # If the value is a list, make a recursive call for each element
        elif isinstance(v, list):
            v_ls = []
            for v_i in v:
                if isinstance(v_i, dict):
                    if "_target_" in v_i:
                        v_ls.append(v_i["_target_"])
                    cfg_flat.update(**flatten_config(v_i))
            cfg_flat[k] = v_ls
        # Exclude uninformative keys
        elif k not in {"_target_", "_partial_"}:
            cfg_flat[k] = v
    return cfg_flat


def get_best_checkpoint(checkpoint_path: Path) -> Path:
    """Get the path to the best checkpoint of a model.

    Args:
        checkpoint_path (Path): Path to the model.

    Returns:
        Path: Path to the best checkpoint.
    """
    # Get the path to the best checkpoint
    pattern = r"(.+?)epoch=(\d+)-val_loss=(\d+\.\d+).ckpt"
    best_loss = float("inf")
    for checkpoint in checkpoint_path.glob("*.ckpt"):
        loss = float(re.match(pattern, str(checkpoint)).group(3))
        if loss < best_loss:
            best_loss = loss
            best_checkpoint_path = checkpoint
    return best_checkpoint_path