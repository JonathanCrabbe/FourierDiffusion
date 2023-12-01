import math

import torch
from torch.fft import irfft, rfft


def dft(x: torch.Tensor) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)

    # Compute the FFT until the Nyquist frequency
    dft_full = rfft(x, dim=1, norm="ortho")
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    # The first harmonic corresponds to the mean, which is always real
    zero_padding = torch.zeros_like(dft_im[:, 0, :], device=x.device)
    assert torch.allclose(
        dft_im[:, 0, :], zero_padding
    ), f"The first harmonic of a real time series should be real, yet got imaginary part {dft_im[:, 0, :]}."
    dft_im = dft_im[:, 1:]

    # If max_len is even, the last component is always zero
    if max_len % 2 == 0:
        assert torch.allclose(
            dft_im[:, -1, :], zero_padding
        ), f"Got an even {max_len=}, which should be real at the Nyquist frequency, yet got imaginary part {dft_im[:, -1, :]}."
        dft_im = dft_im[:, :-1]

    # Concatenate real and imaginary parts
    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    assert (
        x_tilde.size() == x.size()
    ), f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."

    return x_tilde.detach()


def idft(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse DFT of the input DFT that only contains non-redundant components.

    Args:
        x (torch.Tensor): DFT of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: Inverse DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)
    n_real = math.ceil((max_len + 1) / 2)

    # Extract real and imaginary parts
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)))
    x_im = torch.cat((zero_padding, x_im), dim=1)

    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    assert (
        x_im.size() == x_re.size()
    ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

    x_freq = torch.complex(x_re, x_im)

    # Apply IFFT
    x_time = irfft(x_freq, n=max_len, dim=1, norm="ortho")

    assert isinstance(x_time, torch.Tensor)
    assert (
        x_time.size() == x.size()
    ), f"The inverse DFT and the input should have the same size. Got {x_time.size()} and {x.size()} instead."

    return x_time.detach()
