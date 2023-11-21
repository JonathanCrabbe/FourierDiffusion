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
        if isinstance(v, dict):
            if "_target_" in v:
                cfg_flat[k] = v["_target_"]
            cfg_flat.update(**flatten_config(v))
        elif isinstance(v, list):
            v_ls = []
            for v_i in v:
                if isinstance(v_i, dict):
                    if "_target_" in v_i:
                        v_ls.append(v_i["_target_"])
                    cfg_flat.update(**flatten_config(v_i))
            cfg_flat[k] = v_ls
        elif k not in {"_target_", "_partial_"}:
            cfg_flat[k] = v
    return cfg_flat
