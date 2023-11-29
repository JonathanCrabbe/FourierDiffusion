import numpy as np
import torch


def check_flat_array(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """If necessary, the tensor is converted to a 2d numpy array.

    Args:
        x (torch.Tensor | np.ndarray): Tensor to convert.

    Returns:
        np.ndarray: 2d array
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    assert isinstance(
        x, np.ndarray
    ), f"x must be a numpy array or a torch tensor. Got {type(x)}"
    assert x.ndim == 2, f"x must be a 2d array. Got {x.ndim}d array."
    return x
