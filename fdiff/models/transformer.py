import math

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        # Position vector broadcasted accross batch dimension
        position = torch.arange(max_len).unsqueeze(0)  # (1, max_len)

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_len, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_len, d_emb)

        # Positional encodings broadcasted accross batch dimension
        pe = self.embedding(position)  # (1, max_len, d_emb)
        self.pe = rearrange(pe, "b l d -> b d l")  # (1, d_emb, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, d_embed, max_len]``

        Returns:
            x: Tensor, shape ``[batch_size, d_embed, max_len]``
        """
        x = x + self.pe
        return x
