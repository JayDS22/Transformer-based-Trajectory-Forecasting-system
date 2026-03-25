"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) Trajectory Decoder.

Generates future trajectories by iteratively denoising random Gaussian noise,
conditioned on the fused temporal-social context. This allows sampling diverse,
multimodal trajectory predictions that capture the inherent uncertainty of
future motion.

Forward Process:  x_0 → x_1 → ... → x_T  (gradually add noise)
Reverse Process:  x_T → x_{T-1} → ... → x_0  (learned denoising)

The denoising network is a Transformer that takes:
  - Noisy trajectory tokens (the trajectory being denoised)
  - Diffusion timestep embedding
  - Conditioning context (from encoder)

References:
  - Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
  - Gu et al., "Stochastic Trajectory Prediction via Motion Indeterminacy
    Diffusion" (CVPR 2022)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sinusoidal_pe import DiffusionTimestepEmbedding, SinusoidalPositionalEncoding


class ConditionalDenoiserBlock(nn.Module):
    """
    Single Transformer block for the denoising network.
    
    Uses self-attention over noisy trajectory tokens with
    cross-attention to conditioning context, modulated by
    the diffusion timestep.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()

        # Self-attention over noisy trajectory
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to conditioning context
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Timestep-conditioned scale-shift (AdaLN)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 4),
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_pred, D) noisy trajectory tokens
            context: (B, 1, D) conditioning context
            t_emb: (B, D) timestep embedding

        Returns:
            (B, T_pred, D) refined trajectory tokens
        """
        # AdaLN modulation from timestep
        ada_params = self.adaLN(t_emb).unsqueeze(1)  # (B, 1, 4D)
        scale1, shift1, scale2, shift2 = ada_params.chunk(4, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + scale1) + shift1
        h_attn, _ = self.self_attn(h, h, h)
        x = x + h_attn

        # Cross-attention to context
        q = self.norm2(x)
        kv = self.norm_ctx(context)
        cross_out, _ = self.cross_attn(q, kv, kv)
        x = x + cross_out

        # FFN with AdaLN
        h = self.norm3(x) * (1 + scale2) + shift2
        x = x + self.ffn(h)

        return x


class DiffusionTrajectoryDecoder(nn.Module):
    """
    Conditional DDPM for trajectory generation.

    Parameters
    ----------
    pred_len : int
        Number of future timesteps to predict.
    input_dim : int
        Coordinate dimension (2 for x, y).
    embed_dim : int
        Hidden dimension.
    num_steps : int
        Number of diffusion steps T.
    beta_start : float
        Starting noise schedule value.
    beta_end : float
        Ending noise schedule value.
    beta_schedule : str
        "linear" or "cosine" schedule.
    num_layers : int
        Number of denoiser Transformer blocks.
    num_heads : int
        Attention heads in denoiser.
    ff_dim : int
        FFN dimension in denoiser.
    """

    def __init__(
        self,
        pred_len: int = 12,
        input_dim: int = 2,
        embed_dim: int = 128,
        num_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.05,
        beta_schedule: str = "cosine",
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_steps = num_steps

        # ---- Noise schedule ----
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        # ---- Denoising network ----
        # Trajectory input projection
        self.traj_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Positional encoding for trajectory steps
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=pred_len + 10)

        # Timestep embedding
        self.time_embed = DiffusionTimestepEmbedding(embed_dim)

        # Context projection (from encoder)
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Denoiser blocks
        self.blocks = nn.ModuleList([
            ConditionalDenoiserBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection: embed_dim → (x, y)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim),
        )

        # Zero-initialize final layer for stable training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    @staticmethod
    def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule from Nichol & Dhariwal (2021)."""
        steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
        alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clamp(betas.float(), 0.0001, 0.999)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t given x_0 and timestep t.

        Args:
            x_0: (B, T_pred, 2) clean trajectories
            t: (B,) timestep indices
            noise: optional pre-sampled noise

        Returns:
            x_t: (B, T_pred, 2) noisy trajectories
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise ε_θ(x_t, t, c) using the denoising network.

        Args:
            x_t: (B, T_pred, 2) noisy trajectory
            t: (B,) diffusion timestep
            context: (B, D) conditioning context from encoder

        Returns:
            (B, T_pred, 2) predicted noise
        """
        B = x_t.shape[0]

        # Project noisy trajectory to embedding space
        h = self.traj_proj(x_t)  # (B, T_pred, D)
        h = self.pos_enc(h)

        # Timestep embedding
        t_emb = self.time_embed(t)  # (B, D)

        # Context conditioning
        ctx = self.context_proj(context).unsqueeze(1)  # (B, 1, D)

        # Apply denoiser blocks
        for block in self.blocks:
            h = block(h, ctx, t_emb)

        # Project to coordinate space
        predicted_noise = self.output_proj(h)  # (B, T_pred, 2)

        return predicted_noise

    def compute_loss(
        self,
        x_0: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss (simplified ELBO).

        L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t, c)||² ]

        Args:
            x_0: (B, T_pred, 2) ground truth future trajectories
            context: (B, D) conditioning context

        Returns:
            Scalar loss
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.predict_noise(x_t, t, context)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        num_samples: int = 20,
    ) -> torch.Tensor:
        """
        Generate trajectory samples via reverse diffusion (DDPM sampling).

        Args:
            context: (B, D) conditioning context
            num_samples: number of trajectory samples per agent (K)

        Returns:
            (B, K, T_pred, 2) sampled trajectories
        """
        B = context.shape[0]
        device = context.device

        all_samples = []

        for _ in range(num_samples):
            # Start from pure noise
            x_t = torch.randn(B, self.pred_len, self.input_dim, device=device)

            # Reverse diffusion
            for t_idx in reversed(range(self.num_steps)):
                t = torch.full((B,), t_idx, device=device, dtype=torch.long)

                # Predict noise
                pred_noise = self.predict_noise(x_t, t, context)

                # Compute posterior mean
                x0_pred = self._predict_x0(x_t, t, pred_noise)
                x0_pred = torch.clamp(x0_pred, -10, 10)  # Stability clamp
                mean = (
                    self.posterior_mean_coef1[t_idx] * x0_pred
                    + self.posterior_mean_coef2[t_idx] * x_t
                )

                # Add noise (except at t=0)
                if t_idx > 0:
                    noise = torch.randn_like(x_t)
                    var = self.posterior_variance[t_idx]
                    x_t = mean + torch.sqrt(var) * noise
                else:
                    x_t = mean

            all_samples.append(x_t)

        # Stack: (B, K, T_pred, 2)
        return torch.stack(all_samples, dim=1)

    def _predict_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x_0 from x_t and predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    @torch.no_grad()
    def sample_ddim(
        self,
        context: torch.Tensor,
        num_samples: int = 20,
        ddim_steps: int = 20,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Accelerated sampling via DDIM (Song et al., 2020).

        Uses fewer denoising steps for faster inference.

        Args:
            context: (B, D) conditioning context
            num_samples: K samples per agent
            ddim_steps: number of DDIM steps (< num_steps)
            eta: stochasticity (0 = deterministic DDIM, 1 = DDPM)

        Returns:
            (B, K, T_pred, 2) sampled trajectories
        """
        B = context.shape[0]
        device = context.device

        # DDIM sub-sequence of timesteps
        step_size = self.num_steps // ddim_steps
        timesteps = list(range(0, self.num_steps, step_size))[::-1]

        all_samples = []

        for _ in range(num_samples):
            x_t = torch.randn(B, self.pred_len, self.input_dim, device=device)

            for i, t_idx in enumerate(timesteps):
                t = torch.full((B,), t_idx, device=device, dtype=torch.long)
                pred_noise = self.predict_noise(x_t, t, context)

                # Predict x_0
                x_0_pred = self._predict_x0(x_t, t, pred_noise)
                x_0_pred = torch.clamp(x_0_pred, -10, 10)  # Stability

                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_t = self.alphas_cumprod[t_idx]
                    alpha_prev = self.alphas_cumprod[t_prev]

                    sigma = eta * torch.sqrt(
                        (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                    )

                    # Direction pointing to x_t
                    dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise

                    x_t = torch.sqrt(alpha_prev) * x_0_pred + dir_xt
                    if sigma > 0:
                        x_t += sigma * torch.randn_like(x_t)
                else:
                    x_t = x_0_pred

            all_samples.append(x_t)

        return torch.stack(all_samples, dim=1)
