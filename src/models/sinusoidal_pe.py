"""
Sinusoidal Positional Encodings for Temporal and Diffusion Timesteps.

Provides standard sinusoidal PE for sequence positions and diffusion
timestep embeddings following the formulation in "Attention Is All You Need"
(Vaswani et al., 2017) and "Denoising Diffusion Probabilistic Models"
(Ho et al., 2020).
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            Positionally encoded tensor of same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DiffusionTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion models.
    
    Maps scalar timestep t → R^d_model following Ho et al. (2020).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch,) integer diffusion timesteps
        Returns:
            (batch, d_model) timestep embeddings
        """
        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device).float() * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.d_model % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return self.mlp(emb)
