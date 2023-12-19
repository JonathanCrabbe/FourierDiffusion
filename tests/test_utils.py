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
    x_even = torch.randn(batch_size, max_len, n_channels)
    x_odd = torch.randn(batch_size, max_len + 1, n_channels)

    # Check that IDFT of DFT is identity
    x_even_hat = idft(dft(x_even))
    x_odd_hat = idft(dft(x_odd))
    assert torch.allclose(x_even, x_even_hat, atol=1e-5)
    assert torch.allclose(x_odd, x_odd_hat, atol=1e-5)

    # Check that DFT of IDFT is identity
    x_even_hat = dft(idft(x_even))
    x_odd_hat = dft(idft(x_odd))
    assert torch.allclose(x_even, x_even_hat, atol=1e-5)
    assert torch.allclose(x_odd, x_odd_hat, atol=1e-5)
