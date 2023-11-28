from omegaconf import DictConfig

from fdiff.utils.extraction import flatten_config


def test_flatten_config():
    cfg_dict = {
        "Option1": "Value1",
        "Option2": {
            "_target_": "Value2",
            "Option3": "Value3",
            "Option4": {"_partial_": True},
        },
        "Option5": [
            {"_target_": "Value5_0", "Option6": "Value6"},
            {"_target_": "Value5_1"},
        ],
    }
    cfg = DictConfig(cfg_dict)
    cfg_flat = flatten_config(cfg)
    assert cfg_flat == {
        "Option1": "Value1",
        "Option2": "Value2",
        "Option3": "Value3",
        "Option5": ["Value5_0", "Value5_1"],
        "Option6": "Value6",
    }
