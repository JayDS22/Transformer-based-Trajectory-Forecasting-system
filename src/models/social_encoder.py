"""
Social Transformer Encoder.

Models interactions between agents using cross-attention. Each agent
attends to its spatial neighbors to capture social forces such as
collision avoidance, group behavior, and leader-follower dynamics.

Architecture:
    Per-agent embedding → Cross-Attention (query=ego, key/value=neighbors) 
    → N layers → Social context
"""

import torch
import torch.nn as nn
from .sinusoidal_pe import SinusoidalPositionalEncoding


class SocialCrossAttentionLayer(nn.Module):
    """
    Single layer of social cross-attention.
    
    The ego agent's features serve as queries, while neighbor features
    serve as keys and values. This allows selective attention to
    relevant neighbors based on relative position and motion.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        
        # Pre-norm cross-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        ego_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            ego_features: (B, 1, D) ego agent features
            neighbor_features: (B, N_max, D) neighbor features
            neighbor_mask: (B, N_max) True = valid neighbor
            
        Returns:
            (B, 1, D) updated ego features with social context
        """
        # Cross-attention: ego attends to neighbors
        q = self.norm1(ego_features)
        kv = self.norm_kv(neighbor_features)
        
        # Mask: True = IGNORE for MultiheadAttention
        key_padding_mask = None
        if neighbor_mask is not None:
            key_padding_mask = ~neighbor_mask
            # For samples with ALL neighbors masked, ensure at least one is unmasked
            # to prevent NaN from fully-masked attention
            all_masked = key_padding_mask.all(dim=-1)  # (B,)
            if all_masked.any():
                key_padding_mask[all_masked, 0] = False  # unmask first slot
        
        attn_out, _ = self.cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
        )
        ego_features = ego_features + attn_out
        
        # FFN
        ego_features = ego_features + self.ffn(self.norm2(ego_features))
        
        return ego_features


class SocialTransformerEncoder(nn.Module):
    """
    Encodes social interactions between agents via cross-attention.
    
    For each agent, we compute relative position/velocity features
    for all neighbors within a spatial radius, then apply cross-attention
    layers where the ego agent attends to neighbor embeddings.

    Parameters
    ----------
    input_dim : int
        Coordinate dimension (2 for x, y).
    embed_dim : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of cross-attention layers.
    ff_dim : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate.
    neighbor_radius : float
        Maximum distance (meters) to consider neighbors.
    """

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        neighbor_radius: float = 3.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.neighbor_radius = neighbor_radius

        # Relative feature projection
        # Input: relative_pos(2) + relative_vel(2) + distance(1) + bearing(1) = 6
        self.neighbor_proj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Ego summary projection (from temporal context)
        self.ego_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Stacked cross-attention layers
        self.layers = nn.ModuleList([
            SocialCrossAttentionLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def compute_relative_features(
        self,
        ego_pos: torch.Tensor,
        ego_vel: torch.Tensor,
        neighbor_pos: torch.Tensor,
        neighbor_vel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relative features between ego and each neighbor.

        Args:
            ego_pos: (B, 2) ego position at current timestep
            ego_vel: (B, 2) ego velocity at current timestep
            neighbor_pos: (B, N_max, 2) neighbor positions
            neighbor_vel: (B, N_max, 2) neighbor velocities

        Returns:
            (B, N_max, 6) relative features
        """
        # Relative position
        rel_pos = neighbor_pos - ego_pos.unsqueeze(1)  # (B, N, 2)
        
        # Relative velocity
        rel_vel = neighbor_vel - ego_vel.unsqueeze(1)  # (B, N, 2)
        
        # Euclidean distance (with epsilon for stability)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
        
        # Bearing angle (safe atan2)
        bearing = torch.atan2(
            rel_pos[..., 1:2] + 1e-8, rel_pos[..., 0:1] + 1e-8
        )  # (B, N, 1)
        
        return torch.cat([rel_pos, rel_vel, dist, bearing], dim=-1)  # (B, N, 6)

    def forward(
        self,
        ego_context: torch.Tensor,
        ego_pos: torch.Tensor,
        ego_vel: torch.Tensor,
        neighbor_pos: torch.Tensor,
        neighbor_vel: torch.Tensor,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            ego_context: (B, embed_dim) temporal context from ego agent
            ego_pos: (B, 2) current position
            ego_vel: (B, 2) current velocity
            neighbor_pos: (B, N_max, 2) neighbor last-observed positions
            neighbor_vel: (B, N_max, 2) neighbor last-observed velocities
            neighbor_mask: (B, N_max) True = valid neighbor

        Returns:
            social_context: (B, embed_dim) social interaction context
        """
        B = ego_context.shape[0]
        
        # Compute relative features
        rel_features = self.compute_relative_features(
            ego_pos, ego_vel, neighbor_pos, neighbor_vel
        )  # (B, N_max, 6)

        # Apply distance-based masking
        if neighbor_mask is not None:
            dist = torch.norm(neighbor_pos - ego_pos.unsqueeze(1), dim=-1)
            radius_mask = dist < self.neighbor_radius
            neighbor_mask = neighbor_mask & radius_mask

        # Project neighbor features
        neighbor_embeddings = self.neighbor_proj(rel_features)  # (B, N_max, D)

        # Project ego context as query
        ego_query = self.ego_proj(ego_context).unsqueeze(1)  # (B, 1, D)

        # Apply cross-attention layers
        x = ego_query
        for layer in self.layers:
            x = layer(x, neighbor_embeddings, neighbor_mask)

        # Squeeze and project
        social_context = self.output_proj(x.squeeze(1))  # (B, D)

        return social_context
