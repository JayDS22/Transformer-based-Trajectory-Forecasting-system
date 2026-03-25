"""
MotionTransformer: Full Model Assembly.

Combines the Temporal Encoder, Social Encoder, Gated Fusion, and
Diffusion Decoder into the complete trajectory forecasting pipeline.

    Observed Trajectories → [Temporal Encoder] ─┐
                                                  ├→ [Gated Fusion] → [Diffusion Decoder] → Predictions
    Neighbor Trajectories → [Social Encoder]  ───┘
"""

import torch
import torch.nn as nn

from .temporal_encoder import TemporalTransformerEncoder
from .social_encoder import SocialTransformerEncoder
from .gated_fusion import GatedFusion
from .diffusion_decoder import DiffusionTrajectoryDecoder


class MotionTransformer(nn.Module):
    """
    MotionTransformer: Attention-Based Multi-Agent Trajectory Forecasting
    with Diffusion Refinement.

    Parameters
    ----------
    obs_len : int
        Number of observed timesteps.
    pred_len : int
        Number of future timesteps to predict.
    input_dim : int
        Coordinate dimension (2 for x, y).
    embed_dim : int
        Hidden dimension for all sub-modules.
    config : dict
        Full configuration dictionary (from YAML).
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        input_dim: int = 2,
        embed_dim: int = 128,
        config: dict = None,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Default config
        cfg = config or {}
        t_cfg = cfg.get("temporal", {})
        s_cfg = cfg.get("social", {})
        f_cfg = cfg.get("fusion", {})
        d_cfg = cfg.get("diffusion", {})

        # ---- Temporal Encoder ----
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=t_cfg.get("num_heads", 8),
            num_layers=t_cfg.get("num_layers", 4),
            ff_dim=t_cfg.get("ff_dim", 512),
            dropout=t_cfg.get("dropout", 0.1),
        )

        # ---- Social Encoder ----
        self.social_encoder = SocialTransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=s_cfg.get("num_heads", 8),
            num_layers=s_cfg.get("num_layers", 3),
            ff_dim=s_cfg.get("ff_dim", 512),
            dropout=s_cfg.get("dropout", 0.1),
            neighbor_radius=s_cfg.get("neighbor_radius", 3.0),
        )

        # ---- Gated Fusion ----
        self.fusion = GatedFusion(
            embed_dim=embed_dim,
            gate_type=f_cfg.get("gate_type", "sigmoid"),
        )

        # ---- Diffusion Decoder ----
        self.diffusion_decoder = DiffusionTrajectoryDecoder(
            pred_len=pred_len,
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_steps=d_cfg.get("num_steps", 100),
            beta_start=d_cfg.get("beta_start", 0.0001),
            beta_end=d_cfg.get("beta_end", 0.05),
            beta_schedule=d_cfg.get("beta_schedule", "cosine"),
            num_layers=d_cfg.get("decoder_layers", 6),
            num_heads=d_cfg.get("decoder_heads", 8),
            ff_dim=d_cfg.get("decoder_ff_dim", 512),
        )

    def encode(
        self,
        obs_traj: torch.Tensor,
        neighbor_obs: torch.Tensor = None,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode observed trajectories into conditioning context.

        Args:
            obs_traj: (B, T_obs, 2) observed trajectory of ego agent
            neighbor_obs: (B, N_max, T_obs, 2) observed trajectories of neighbors
            neighbor_mask: (B, N_max) True = valid neighbor

        Returns:
            context: (B, embed_dim) fused context vector
        """
        # Temporal encoding
        temporal_features, temporal_context = self.temporal_encoder(obs_traj)

        # Social encoding
        has_neighbors = (
            neighbor_obs is not None
            and neighbor_obs.shape[1] > 0
            and neighbor_mask is not None
            and neighbor_mask.any()
        )

        if has_neighbors:
            # Extract last-observed position and velocity for ego
            ego_pos = obs_traj[:, -1, :]  # (B, 2)
            ego_vel = obs_traj[:, -1, :] - obs_traj[:, -2, :]  # (B, 2)

            # Extract last-observed position and velocity for neighbors
            n_pos = neighbor_obs[:, :, -1, :]  # (B, N, 2)
            n_vel = neighbor_obs[:, :, -1, :] - neighbor_obs[:, :, -2, :]

            social_context = self.social_encoder(
                ego_context=temporal_context,
                ego_pos=ego_pos,
                ego_vel=ego_vel,
                neighbor_pos=n_pos,
                neighbor_vel=n_vel,
                neighbor_mask=neighbor_mask,
            )

            # Safety: replace any NaN with zeros
            social_context = torch.nan_to_num(social_context, nan=0.0)
        else:
            # No neighbors → use zeros
            social_context = torch.zeros_like(temporal_context)

        # Fuse temporal + social
        context = self.fusion(temporal_context, social_context)

        return context

    def forward(
        self,
        obs_traj: torch.Tensor,
        pred_traj_gt: torch.Tensor,
        neighbor_obs: torch.Tensor = None,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Training forward pass: compute diffusion loss.

        Args:
            obs_traj: (B, T_obs, 2) observed trajectories
            pred_traj_gt: (B, T_pred, 2) ground truth future trajectories
            neighbor_obs: (B, N_max, T_obs, 2) neighbor trajectories
            neighbor_mask: (B, N_max) neighbor validity mask

        Returns:
            loss: scalar diffusion training loss
        """
        # Normalize trajectories relative to last observed position
        last_pos = obs_traj[:, -1:, :]  # (B, 1, 2)
        obs_normalized = obs_traj - last_pos
        pred_relative = pred_traj_gt - last_pos

        # Normalize neighbors too
        if neighbor_obs is not None:
            neighbor_normalized = neighbor_obs - last_pos.unsqueeze(1)
        else:
            neighbor_normalized = neighbor_obs

        # Encode normalized trajectories
        context = self.encode(obs_normalized, neighbor_normalized, neighbor_mask)

        # Compute diffusion loss on relative predictions
        loss = self.diffusion_decoder.compute_loss(pred_relative, context)

        return loss

    @torch.no_grad()
    def predict(
        self,
        obs_traj: torch.Tensor,
        neighbor_obs: torch.Tensor = None,
        neighbor_mask: torch.Tensor = None,
        num_samples: int = 20,
        use_ddim: bool = False,
        ddim_steps: int = 20,
    ) -> torch.Tensor:
        """
        Inference: generate K diverse trajectory predictions.

        Args:
            obs_traj: (B, T_obs, 2) observed trajectories
            neighbor_obs: (B, N_max, T_obs, 2) neighbor trajectories
            neighbor_mask: (B, N_max) neighbor validity mask
            num_samples: K trajectory samples
            use_ddim: use DDIM for faster sampling
            ddim_steps: number of DDIM steps

        Returns:
            predictions: (B, K, T_pred, 2) absolute trajectory predictions
        """
        # Normalize relative to last observed position
        last_pos = obs_traj[:, -1:, :]
        obs_normalized = obs_traj - last_pos

        if neighbor_obs is not None:
            neighbor_normalized = neighbor_obs - last_pos.unsqueeze(1)
        else:
            neighbor_normalized = neighbor_obs

        # Encode normalized trajectories
        context = self.encode(obs_normalized, neighbor_normalized, neighbor_mask)

        # Sample from diffusion decoder
        if use_ddim:
            relative_preds = self.diffusion_decoder.sample_ddim(
                context, num_samples, ddim_steps
            )
        else:
            relative_preds = self.diffusion_decoder.sample(context, num_samples)

        # Convert from relative to absolute coordinates
        last_pos = obs_traj[:, -1:, :].unsqueeze(1)  # (B, 1, 1, 2)
        predictions = relative_preds + last_pos

        return predictions

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_component_params(self) -> dict:
        """Get parameter count for each component."""
        return {
            "temporal_encoder": sum(
                p.numel() for p in self.temporal_encoder.parameters()
            ),
            "social_encoder": sum(
                p.numel() for p in self.social_encoder.parameters()
            ),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "diffusion_decoder": sum(
                p.numel() for p in self.diffusion_decoder.parameters()
            ),
        }
