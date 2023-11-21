from omegaconf import DictConfig

from utils.extraction import flatten_config


def test_flatten_config():
    cfg_dict = {
        "Option1": "Value1",
        "Option2": {
            "_target_": "Value2",
            "Option3": "Value3",
            "Option4": {"_partial_": True},
        },
    }
    cfg = DictConfig(cfg_dict)
    cfg_flat = flatten_config(cfg)
    assert cfg_flat == {
        "Option1": "Value1",
        "Option2": "Value2",
        "Option3": "Value3",
    }
