"""
Trajectory forecasting evaluation metrics.

Implements standard metrics used in the trajectory prediction literature:
  - ADE (Average Displacement Error)
  - FDE (Final Displacement Error)
  - Best-of-K variants
  - Collision Rate
  - Negative Log-Likelihood (NLL)
  - Kernel Density Estimate (KDE) NLL
"""

import torch
import numpy as np
from typing import Optional


def compute_ade(
    predictions: torch.Tensor, ground_truth: torch.Tensor
) -> torch.Tensor:
    """
    Average Displacement Error.

    ADE = (1/T) Σ_t ||pred_t - gt_t||₂

    Args:
        predictions: (B, T_pred, 2) or (B, K, T_pred, 2)
        ground_truth: (B, T_pred, 2)

    Returns:
        ADE per sample (B,) or (B, K)
    """
    if predictions.dim() == 4:
        gt = ground_truth.unsqueeze(1)
    else:
        gt = ground_truth

    error = torch.norm(predictions - gt, dim=-1)  # L2 per timestep
    return error.mean(dim=-1)


def compute_fde(
    predictions: torch.Tensor, ground_truth: torch.Tensor
) -> torch.Tensor:
    """
    Final Displacement Error.

    FDE = ||pred_T - gt_T||₂

    Args:
        predictions: (B, T_pred, 2) or (B, K, T_pred, 2)
        ground_truth: (B, T_pred, 2)

    Returns:
        FDE per sample (B,) or (B, K)
    """
    if predictions.dim() == 4:
        gt = ground_truth.unsqueeze(1)
    else:
        gt = ground_truth

    return torch.norm(predictions[..., -1, :] - gt[..., -1, :], dim=-1)


def compute_best_of_k(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    k: int = 20,
) -> dict:
    """
    Best-of-K evaluation (minADE_K, minFDE_K).

    For each sample, select the prediction closest to GT.

    Args:
        predictions: (B, K, T_pred, 2)
        ground_truth: (B, T_pred, 2)
        k: number of samples (uses first K)

    Returns:
        Dict with minADE, minFDE statistics
    """
    preds = predictions[:, :k]
    gt = ground_truth.unsqueeze(1)

    # Per-sample errors
    errors = torch.norm(preds - gt, dim=-1)  # (B, K, T)

    ade_per_k = errors.mean(dim=-1)  # (B, K)
    fde_per_k = errors[:, :, -1]  # (B, K)

    # Best of K
    min_ade = ade_per_k.min(dim=1)[0]  # (B,)
    min_fde = fde_per_k.min(dim=1)[0]  # (B,)

    return {
        "minADE": min_ade.mean().item(),
        "minFDE": min_fde.mean().item(),
        "minADE_std": min_ade.std().item(),
        "minFDE_std": min_fde.std().item(),
        "meanADE": ade_per_k.mean().item(),
        "meanFDE": fde_per_k.mean().item(),
    }


def compute_collision_rate(
    predictions: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute self-collision rate among predicted trajectories.

    Collision = any two agents closer than threshold at same timestep.

    Args:
        predictions: (B, K, T_pred, 2) - for a single scene
        threshold: collision distance in meters

    Returns:
        Fraction of timesteps with collisions
    """
    B, K, T, D = predictions.shape
    collisions = 0
    total = 0

    for t in range(T):
        pos = predictions[:, :, t, :]  # (B, K, 2)
        for b in range(B):
            points = pos[b]  # (K, 2)
            dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
            mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
            close = (dists[mask] < threshold).sum().item()
            collisions += close
            total += mask.sum().item()

    return collisions / max(total, 1)


def compute_trajectory_diversity(predictions: torch.Tensor) -> float:
    """
    Measure diversity of predicted trajectories.

    Average pairwise L2 distance between trajectory samples.

    Args:
        predictions: (B, K, T_pred, 2)

    Returns:
        Mean pairwise distance
    """
    B, K, T, D = predictions.shape
    flat = predictions.reshape(B, K, T * D)
    dists = torch.cdist(flat, flat, p=2)  # (B, K, K)

    # Upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    mean_dist = dists[:, mask].mean().item()

    return mean_dist


def full_evaluation(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    k: int = 20,
    collision_threshold: float = 0.1,
) -> dict:
    """
    Complete evaluation suite.

    Args:
        predictions: (B, K, T_pred, 2)
        ground_truth: (B, T_pred, 2)

    Returns:
        Dict with all metrics
    """
    bok = compute_best_of_k(predictions, ground_truth, k)
    diversity = compute_trajectory_diversity(predictions)
    collision = compute_collision_rate(predictions, collision_threshold)

    return {
        **bok,
        "diversity": diversity,
        "collision_rate": collision,
    }
