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
        self.pe = self.embedding(position)  # (1, max_len, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a positional encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the positional encoding should be added

        Returns:
            torch.Tensor: Tensor with an additional positional encoding
        """
        x = x + self.pe
        return x


class TimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_time: int):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_time, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_time, d_emb)

    def forward(self, x: torch.Tensor, timesteps: torch.LongTensor) -> torch.Tensor:
        """Adds a time encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the time encoding should be added
            timesteps (torch.LongTensor): Tensor of shape (batch_size,) containing the current timestep for each sample in the batch

        Returns:
            torch.Tensor: Tensor with an additional time encoding
        """
        t_emb = self.embedding(timesteps)  # (batch_size, d_emb)
        t_emb = t_emb.unsqueeze(1)  # (batch_size, 1, d_emb)
        assert isinstance(t_emb, torch.Tensor)
        return x + t_emb
