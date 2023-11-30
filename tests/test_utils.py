import torch
from omegaconf import DictConfig

from fdiff.utils.extraction import flatten_config
from fdiff.utils.fourier import dft, idft

max_len = 100
n_channels = 3
batch_size = 100


def test_flatten_config() -> None:
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


def test_dft() -> None:
    # Create a random real time series
    x = torch.randn(batch_size, max_len, n_channels)

    # Compute the DFT
    x_tilde = dft(x)

    # Compute the inverse DFT
    x_hat = idft(x_tilde)

    # Check that the inverse DFT is the original time series
    assert torch.allclose(x, x_hat, atol=1e-5)
