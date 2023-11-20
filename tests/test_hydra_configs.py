import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig


# Function to find all YAML files in a directory
def find_yaml_files(directory: Path):
    yaml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                rel_dir = os.path.relpath(root, directory)
                rel_path = os.path.join(rel_dir, file) if rel_dir != "." else file
                yaml_files.append(rel_path)
    return yaml_files


# Pytest fixture to load and instantiate Hydra configurations
@pytest.fixture(params=find_yaml_files(Path.cwd() / "cmd/conf"))
def hydra_config(request):
    # Initialize hydra
    with initialize(config_path="../cmd/conf"):
        # Load YAML configuration file
        config_file = request.param
        config = compose(
            config_file,
            overrides=[f"++datamodule.data_dir='./data'"],
        )
        instantiate(config)
    return config


# Test function to ensure successful instantiation of Hydra configurations
def test_hydra_config_instantiation(hydra_config):
    assert isinstance(
        hydra_config, DictConfig
    ), f"Hydra config is not an DictConfig object but a {type(hydra_config)}"
