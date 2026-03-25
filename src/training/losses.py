"""
Loss functions for MotionTransformer training.

Includes diffusion loss, diversity loss, and optional auxiliary losses
for trajectory forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Simple diffusion training loss (noise prediction MSE).
    This is the core loss - delegated to the diffusion decoder.
    """

    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted_noise, target_noise)


class DiversityLoss(nn.Module):
    """
    Encourages diverse trajectory samples by penalizing similarity
    between different samples for the same agent.

    L_div = -min_{i≠j} ||sample_i - sample_j||²

    This pushes samples apart, preventing mode collapse.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: (B, K, T_pred, 2) trajectory samples

        Returns:
            Scalar diversity loss (negative = more diverse)
        """
        B, K, T, D = samples.shape

        if K < 2:
            return torch.tensor(0.0, device=samples.device)

        # Flatten spatial-temporal dims
        flat = samples.reshape(B, K, T * D)  # (B, K, T*D)

        # Pairwise L2 distances between samples
        # (B, K, 1, T*D) - (B, 1, K, T*D) → (B, K, K)
        dists = torch.cdist(flat, flat, p=2)  # (B, K, K)

        # Mask diagonal (self-distances)
        mask = torch.eye(K, device=samples.device).bool().unsqueeze(0)
        dists = dists.masked_fill(mask, float("inf"))

        # Minimum pairwise distance per batch
        min_dist = dists.min(dim=-1)[0].min(dim=-1)[0]  # (B,)

        # Negative mean min-distance as loss (minimize → maximize diversity)
        return -self.weight * min_dist.mean()


class BestOfKLoss(nn.Module):
    """
    Best-of-K loss: only backprop through the sample closest to GT.

    L = min_k ||sample_k - GT||²

    Used during fine-tuning to improve best-sample quality.
    """

    def forward(
        self, samples: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            samples: (B, K, T_pred, 2)
            ground_truth: (B, T_pred, 2)

        Returns:
            Scalar best-of-K ADE loss
        """
        gt_expanded = ground_truth.unsqueeze(1)  # (B, 1, T, 2)
        errors = torch.norm(samples - gt_expanded, dim=-1)  # (B, K, T)
        ade_per_sample = errors.mean(dim=-1)  # (B, K)
        best_ade = ade_per_sample.min(dim=-1)[0]  # (B,)
        return best_ade.mean()


class CombinedLoss(nn.Module):
    """Combined training loss with configurable weights."""

    def __init__(self, diffusion_weight: float = 1.0, diversity_weight: float = 0.1):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.diversity_loss = DiversityLoss(weight=diversity_weight)

    def forward(
        self,
        diffusion_loss: torch.Tensor,
        samples: torch.Tensor = None,
    ) -> dict:
        total = self.diffusion_weight * diffusion_loss

        losses = {"diffusion": diffusion_loss.item()}

        if samples is not None:
            div_loss = self.diversity_loss(samples)
            total = total + div_loss
            losses["diversity"] = div_loss.item()

        losses["total"] = total.item()
        return total, losses
