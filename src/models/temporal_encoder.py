"""
Temporal Transformer Encoder.

Encodes an agent's observed trajectory using multi-head self-attention
over time steps, capturing velocity profiles, acceleration patterns,
and higher-order motion dynamics.

Architecture:
    Input Embedding → Positional Encoding → N × [Self-Attention + FFN] → Output
"""

import torch
import torch.nn as nn
from .sinusoidal_pe import SinusoidalPositionalEncoding


class TemporalTransformerEncoder(nn.Module):
    """
    Encodes temporal dynamics of a single agent's trajectory.

    Given observed positions (x, y) over T_obs timesteps, produces a
    temporally-aware feature sequence and a summary context vector.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input coordinates (default: 2 for x, y).
    embed_dim : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    ff_dim : int
        Feed-forward inner dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Input projection: (x, y) → embed_dim
        # We also include velocity features: (dx, dy)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, embed_dim),  # pos + velocity
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Context aggregation: learnable [CLS]-style token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _compute_velocity(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity (finite differences) from position sequence.

        Args:
            positions: (batch, T, 2) observed positions
        Returns:
            (batch, T, 2) velocity features (first step is zero-padded)
        """
        velocity = torch.zeros_like(positions)
        velocity[:, 1:, :] = positions[:, 1:, :] - positions[:, :-1, :]
        return velocity

    def forward(
        self, obs_traj: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_traj: (batch, T_obs, 2) observed trajectory positions
            mask: (batch, T_obs) boolean mask, True = valid timestep

        Returns:
            temporal_features: (batch, T_obs, embed_dim) per-step features
            temporal_context: (batch, embed_dim) summary context vector
        """
        B, T, _ = obs_traj.shape

        # Compute velocity features
        velocity = self._compute_velocity(obs_traj)

        # Concatenate position + velocity → (B, T, 4)
        features = torch.cat([obs_traj, velocity], dim=-1)

        # Project to embedding space
        x = self.input_proj(features)  # (B, T, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, embed_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Build attention mask if provided
        attn_mask = None
        if mask is not None:
            # Extend mask for CLS token (always valid)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)
            # TransformerEncoder expects: True = IGNORE position
            attn_mask = ~extended_mask

        # Encode
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract CLS token as context vector
        temporal_context = self.output_proj(encoded[:, 0, :])  # (B, embed_dim)

        # Per-step features (excluding CLS)
        temporal_features = encoded[:, 1:, :]  # (B, T, embed_dim)

        return temporal_features, temporal_context
