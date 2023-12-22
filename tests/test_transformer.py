import numpy as np
import pytest
import torch

from fdiff.models.transformer import (
    GaussianFourierProjection,
    PositionalEncoding,
    TimeEncoding,
)

max_len = 20  # maximum time series length
max_time = 100  # maximum diffusion time step
batch_size = 16  # number of time series in a batch
d_model = 5  # embedding dimension
EPS = 1e-5  # tolerance for floating point errors


def test_positional_encoding() -> None:
    torch.manual_seed(42)

    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
    X = torch.randn((batch_size, max_len, d_model))
    enc_out = pos_encoder(X)

    # Check that the output has consistent shape
    assert enc_out.shape == X.shape

    # Check that the max_norm constrtaint for positional encoding is satisfied
    assert torch.max(torch.sum((enc_out - X) ** 2, dim=-1)) <= d_model + EPS

    # Check that each time step is assigned the right encoding vector
    for b in range(batch_size):
        for l in range(max_len):
            assert torch.allclose(
                (enc_out - X)[b, l, :], pos_encoder.embedding.weight[l, :], atol=EPS
            )


@pytest.mark.parametrize(
    "time_encoder",
    [
        TimeEncoding(d_model=d_model, max_time=max_time),
        GaussianFourierProjection(d_model=d_model),
    ],
)
def test_time_encoding(time_encoder: TimeEncoding | GaussianFourierProjection) -> None:
    torch.manual_seed(42)

    X = torch.randn((batch_size, max_len, d_model))
    timesteps = torch.randint(low=0, high=max_len, size=(batch_size,))

    enc_out = time_encoder(X, timesteps)

    # Check that the output has consistent shape
    assert enc_out.shape == X.shape

    # Check that the max_norm constrtaint for time encoding is satisfied
    assert torch.max(torch.sum((enc_out - X) ** 2, dim=-1)) <= d_model + EPS

    # Check that each batch element is assigned the right encoding vector
    for b in range(batch_size):
        for l in range(max_len):
            if isinstance(time_encoder, TimeEncoding):
                ground_truth = time_encoder.embedding(timesteps[b])
            else:

                def fourier_embedding(t: torch.Tensor) -> torch.Tensor:
                    time_proj = t[:, None] * time_encoder.W[None, :] * 2 * np.pi
                    embeddings = torch.cat(
                        [torch.sin(time_proj), torch.cos(time_proj)], dim=-1
                    )

                    # Slice to get exactly d_model
                    t_emb = embeddings[:, : time_encoder.d_model]

                    projected_emb: torch.Tensor = time_encoder.dense(t_emb)

                    return projected_emb

                ground_truth = fourier_embedding(timesteps)[b]  # (d_model)

            assert torch.allclose((enc_out - X)[b, l, :], ground_truth, atol=EPS)
