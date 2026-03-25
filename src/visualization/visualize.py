"""
Trajectory Visualization Utilities.

Generates publication-quality trajectory plots showing:
  - Observed trajectory
  - Ground truth future
  - Multiple predicted samples
  - Social neighborhood
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Optional


# Publication-quality style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
})

# Color palette
COLORS = {
    "observed": "#2c3e50",
    "ground_truth": "#27ae60",
    "prediction": "#3498db",
    "prediction_alpha": 0.15,
    "best_prediction": "#e74c3c",
    "neighbor": "#95a5a6",
    "neighbor_pred": "#bdc3c7",
}


def plot_trajectory_predictions(
    obs_traj: np.ndarray,
    pred_gt: np.ndarray,
    predictions: np.ndarray,
    neighbor_obs: Optional[np.ndarray] = None,
    neighbor_mask: Optional[np.ndarray] = None,
    title: str = "Trajectory Prediction",
    save_path: Optional[str] = None,
    show_best: bool = True,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Plot trajectory predictions with observed path and ground truth.

    Args:
        obs_traj: (T_obs, 2) observed trajectory
        pred_gt: (T_pred, 2) ground truth future
        predictions: (K, T_pred, 2) predicted trajectories
        neighbor_obs: (N, T_obs, 2) neighbor trajectories
        neighbor_mask: (N,) validity mask
        title: plot title
        save_path: path to save figure
        show_best: highlight best prediction
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot neighbors
    if neighbor_obs is not None and neighbor_mask is not None:
        for i in range(len(neighbor_obs)):
            if neighbor_mask[i]:
                n_traj = neighbor_obs[i]
                ax.plot(
                    n_traj[:, 0], n_traj[:, 1],
                    color=COLORS["neighbor"], linewidth=1.0,
                    alpha=0.5, zorder=1,
                )
                ax.scatter(
                    n_traj[-1, 0], n_traj[-1, 1],
                    color=COLORS["neighbor"], s=20, alpha=0.5, zorder=1,
                )

    # Plot all prediction samples
    K = predictions.shape[0]
    for k in range(K):
        full_pred = np.concatenate([obs_traj[-1:], predictions[k]], axis=0)
        ax.plot(
            full_pred[:, 0], full_pred[:, 1],
            color=COLORS["prediction"],
            alpha=COLORS["prediction_alpha"],
            linewidth=1.0,
            zorder=2,
        )

    # Find and highlight best prediction
    if show_best and pred_gt is not None:
        errors = np.linalg.norm(predictions - pred_gt[None], axis=-1).mean(axis=-1)
        best_idx = errors.argmin()
        best_pred = np.concatenate([obs_traj[-1:], predictions[best_idx]], axis=0)
        ax.plot(
            best_pred[:, 0], best_pred[:, 1],
            color=COLORS["best_prediction"],
            linewidth=2.0,
            label=f"Best prediction (ADE: {errors[best_idx]:.3f}m)",
            zorder=4,
        )
        ax.scatter(
            best_pred[-1, 0], best_pred[-1, 1],
            color=COLORS["best_prediction"],
            s=60, marker="*", zorder=5,
        )

    # Plot ground truth
    if pred_gt is not None:
        full_gt = np.concatenate([obs_traj[-1:], pred_gt], axis=0)
        ax.plot(
            full_gt[:, 0], full_gt[:, 1],
            color=COLORS["ground_truth"],
            linewidth=2.5,
            linestyle="--",
            label="Ground truth",
            zorder=3,
        )
        ax.scatter(
            full_gt[-1, 0], full_gt[-1, 1],
            color=COLORS["ground_truth"],
            s=80, marker="D", zorder=5,
        )

    # Plot observed trajectory
    ax.plot(
        obs_traj[:, 0], obs_traj[:, 1],
        color=COLORS["observed"],
        linewidth=2.5,
        label="Observed",
        zorder=4,
    )
    ax.scatter(
        obs_traj[0, 0], obs_traj[0, 1],
        color=COLORS["observed"],
        s=80, marker="o", zorder=5, label="Start",
    )
    ax.scatter(
        obs_traj[-1, 0], obs_traj[-1, 1],
        color=COLORS["observed"],
        s=80, marker="s", zorder=5, label="Current",
    )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: list[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training loss and evaluation metrics over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]

    # Loss curve
    axes[0].plot(epochs, losses, color="#e74c3c", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Diffusion Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")

    # ADE curve
    eval_epochs = [h["epoch"] for h in history if "ade" in h and h["ade"]]
    ades = [h["ade"] for h in history if "ade" in h and h["ade"]]
    if eval_epochs:
        axes[1].plot(eval_epochs, ades, color="#3498db", linewidth=2, marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("ADE (meters)")
        axes[1].set_title("Best-of-20 ADE")

    # FDE curve
    fdes = [h["fde"] for h in history if "fde" in h and h["fde"]]
    if eval_epochs:
        axes[2].plot(eval_epochs, fdes, color="#27ae60", linewidth=2, marker="s")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("FDE (meters)")
        axes[2].set_title("Best-of-20 FDE")

    fig.suptitle("MotionTransformer Training Progress", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    labels: list[str] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_multi_scene_comparison(
    scene_results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a comparison bar chart across scenes."""
    scenes = list(scene_results.keys())
    ades = [scene_results[s]["minADE"] for s in scenes]
    fdes = [scene_results[s]["minFDE"] for s in scenes]

    x = np.arange(len(scenes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, ades, width, label="ADE", color="#3498db")
    bars2 = ax.bar(x + width / 2, fdes, width, label="FDE", color="#e74c3c")

    ax.set_xlabel("Scene")
    ax.set_ylabel("Error (meters)")
    ax.set_title("Per-Scene Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenes])
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
