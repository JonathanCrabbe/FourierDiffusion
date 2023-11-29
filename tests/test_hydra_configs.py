import os
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig


# Function to find all YAML files in a directory
def find_yaml_files(directory: Path):
    yaml_files = []
    for path in directory.glob("*.yaml"):
        rel_path = os.path.relpath(str(path), directory)
        yaml_files.append(rel_path)
    return yaml_files


# Pytest fixture to load and instantiate Hydra configurations
@pytest.fixture(params=find_yaml_files(Path.cwd() / "cmd/conf"))
def hydra_config(request):
    # Initialize hydra
    with initialize(config_path="../cmd/conf"):
        # Load YAML configuration file
        config_file = request.param
        # Check that the YAML file corresponds to a dictionnary
        with open(Path.cwd() / "cmd/conf" / config_file, "r") as f:
            config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                pytest.skip(
                    f"Config file {config_file} does not correspond to a dictionnary"
                )
        # Try to compose and instantiate the configuration
        config = compose(
            config_file,
            overrides=[
                "++datamodule.data_dir='./data'",
                "++trainer.logger=False",
                "++model_path=./lightning_logs",
            ],
        )
        instantiate(config)
    return config


# Test function to ensure successful instantiation of Hydra configurations
def test_hydra_config_instantiation(hydra_config):
    assert isinstance(
        hydra_config, DictConfig
    ), f"Hydra config is not an DictConfig object but a {type(hydra_config)}"
