import torch

from fdiff.models.transformer import PositionalEncoding

max_len = 20
batch_size = 16
d_model = 5
EPS = 1e-5


def test_positional_encoding():
    torch.manual_seed(42)

    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
    X = torch.randn((batch_size, d_model, max_len))
    enc_out = pos_encoder(X)

    # Check that the output has consistent shape
    assert enc_out.shape == X.shape

    # Check that the max_norm constrtaint for positional encoding is satisfied
    assert torch.max(torch.sum((enc_out - X) ** 2, dim=1)) <= d_model + EPS

    # Check that time step is assigned the right emconding vector
    for b in range(batch_size):
        for l in range(max_len):
            assert torch.allclose(
                (enc_out - X)[b, :, l], pos_encoder.pe[0, :, l], atol=EPS
            )
