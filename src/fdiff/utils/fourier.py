import math

import torch
from einops import rearrange
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


def spectral_density(x: torch.Tensor, apply_dft: bool = True) -> torch.Tensor:
    """Compute the spectral density of the input time series.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).
        apply_dft (bool, optional): Whether to apply the DFT to the input. Defaults to True.

    Returns:
        torch.Tensor: Spectral density of x with the size (batch_size, n_frequencies, n_channels).
    """

    max_len = x.size(1)
    x = dft(x) if apply_dft else x

    # Extract real and imaginary parts
    n_real = math.ceil((max_len + 1) / 2)
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

    # Compute the spectral density
    x_dens = x_re**2 + x_im**2
    assert isinstance(x_dens, torch.Tensor)
    return x_dens


def localization_metrics(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the localization metrics for the input time series.

    Args:
        X (torch.Tensor): Input time series of shape (batch_size, max_len, n_channels).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Delocalization in the time domain and in the frequency domain for each sample
    """

    max_len = X.shape[1]

    # Compute the energy distribution over time for the time series
    X_energy = torch.sum(X**2, dim=2, keepdim=True) / torch.sum(
        X**2, dim=(1, 2), keepdim=True
    )
    X_energy = rearrange(X_energy, "batch time 1 -> batch time")

    # Compute the energy distribution over frequency for the time series
    X_spec = spectral_density(X)
    X_spec_mirror = (
        torch.flip(X_spec[:, 1:, :], dims=(1,))
        if max_len % 2 != 0
        else torch.flip(X_spec[:, 1:-1, :], dims=(1,))
    )  # Add the mirrored frequencies beyond the Nyquist frequency
    X_spec = torch.cat((X_spec, X_spec_mirror), dim=1)
    X_spec = torch.sum(X_spec, dim=2, keepdim=True) / torch.sum(
        X_spec, dim=(1, 2), keepdim=True
    )
    X_spec = rearrange(X_spec, "batch freq 1 -> batch freq")
    assert (
        X_spec.shape[1] == max_len
    ), f"Spectral density has incorrect shape at dimension 1, expected {max_len}, got {X_spec.shape[1]} instead."

    # Compute the cyclic distance between each time steps
    t = torch.arange(max_len, dtype=torch.float)
    t1 = rearrange(t, "time -> time 1 ")
    t2 = rearrange(t, "time -> 1 time ")
    cyclic_distance = torch.min(torch.abs(t1 - t2), max_len - torch.abs(t1 - t2))

    # Compute the delocalization of the signal in time domain
    X_loc = torch.einsum("bt, ts -> bs", X_energy, cyclic_distance**2)
    X_loc = torch.min(X_loc, dim=1)[0]

    # Compute the delocalization of the signal in frequency domain
    X_spec_loc = torch.einsum("bt, ts -> bs", X_spec, cyclic_distance**2)
    X_spec_loc = torch.min(X_spec_loc, dim=1)[0]

    return X_loc, X_spec_loc


def smooth_frequency(X: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooths the signal in the frequency domain by convolving it with a Gaussian kernel.

    Args:
        X (torch.Tensor): Time series to smooth of shape (batch_size, max_len, n_channels).
        sigma (float): Gaussian kernel width.

    Returns:
        torch.Tensor: Smoothed signal in the frequency domain of shape (batch_size, max_len, n_channels).
    """

    # Compute Nyquist frequency
    max_len = X.shape[1]
    nyquist_freq = max_len / 2

    # Define Gaussian kernel for each frequency pair
    k = torch.cat(
        (
            torch.arange(0, nyquist_freq, dtype=torch.float32),
            torch.arange(1, nyquist_freq, dtype=torch.float32),
        )
    )
    k1 = rearrange(k, "time -> time 1 ")
    k2 = rearrange(k, "time -> 1 time ")
    gaussian_kernel = torch.exp(-(((k1 - k2) / (sigma)) ** 2) / 2)
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=0, keepdim=True)

    # Convolve X with the Gaussian kernel in the frequency domain
    X = dft(X)
    X = torch.einsum("btc, ts -> bsc", X, gaussian_kernel)
    X = idft(X)
    return X
