import torch

from fdiff.models.transformer import PositionalEncoding, TimeEncoding

max_len = 20  # maximum time series length
max_time = 100  # maximum diffusion time step
batch_size = 16  # number of time series in a batch
d_model = 5  # embedding dimension
EPS = 1e-5  # tolerance for floating point errors


def test_positional_encoding():
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


def test_time_encoding():
    torch.manual_seed(42)

    time_encoder = TimeEncoding(d_model=d_model, max_time=max_time)
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
            assert torch.allclose(
                (enc_out - X)[b, l, :], time_encoder.embedding(timesteps[b]), atol=EPS
            )
