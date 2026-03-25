"""
Gated Fusion Module.

Combines temporal and social context vectors through a learned gating
mechanism. The gate adaptively weights the contribution of each stream
based on the current context, allowing the model to dynamically
balance individual motion patterns vs social interaction signals.

Inspired by GRU-style gating and highway network mechanisms.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated fusion of temporal and social context vectors.

    Uses a sigmoid gate to blend two context vectors:
        fused = gate * temporal + (1 - gate) * social

    The gate is computed from the concatenation of both vectors,
    allowing context-dependent weighting.

    Parameters
    ----------
    embed_dim : int
        Dimension of input context vectors.
    gate_type : str
        Type of gating: "sigmoid" (element-wise) or "softmax" (vector-level).
    """

    def __init__(self, embed_dim: int, gate_type: str = "sigmoid"):
        super().__init__()
        self.gate_type = gate_type

        if gate_type == "sigmoid":
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
        elif gate_type == "softmax":
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 2, 2),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

        # Residual projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        temporal_context: torch.Tensor,
        social_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal_context: (B, D) temporal stream output
            social_context: (B, D) social stream output

        Returns:
            fused_context: (B, D) fused context vector
        """
        combined = torch.cat([temporal_context, social_context], dim=-1)

        if self.gate_type == "sigmoid":
            gate = self.gate_net(combined)  # (B, D)
            fused = gate * temporal_context + (1 - gate) * social_context
        else:
            weights = self.gate_net(combined)  # (B, 2)
            w_t = weights[:, 0:1]  # (B, 1)
            w_s = weights[:, 1:2]  # (B, 1)
            fused = w_t * temporal_context + w_s * social_context

        # Residual output
        return self.output_proj(fused) + fused
