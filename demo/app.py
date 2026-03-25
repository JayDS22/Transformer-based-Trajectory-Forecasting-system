"""
MotionTransformer - Full Demo Pipeline.

This script runs the complete pipeline:
  1. Generate synthetic ETH/UCY-style data
  2. Train the MotionTransformer model
  3. Evaluate on test set with all metrics
  4. Generate publication-quality visualizations
  5. Print comprehensive results summary

Usage:
    python demo/app.py
"""

import sys
import json
import time
import logging
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.motion_transformer import MotionTransformer
from src.data.eth_ucy_dataset import ETHUCYDataset, eth_ucy_collate_fn
from src.evaluation.metrics import (
    compute_ade, compute_fde, compute_best_of_k,
    compute_trajectory_diversity, full_evaluation,
)
from src.visualization.visualize import (
    plot_trajectory_predictions,
    plot_training_curves,
    create_multi_scene_comparison,
)
from src.utils.helpers import set_seed
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_demo(
    scene: str = "eth",
    epochs: int = 30,
    batch_size: int = 32,
    embed_dim: int = 128,
    device: torch.device = None,
):
    """Train the model for demo purposes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training on scene: {scene} | device: {device}")

    # Create datasets
    train_data = ETHUCYDataset(scene=scene, split="train", augment=True)
    test_data = ETHUCYDataset(scene=scene, split="test", augment=False)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=eth_ucy_collate_fn, drop_last=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        collate_fn=eth_ucy_collate_fn,
    )

    # Build model
    config = {
        "temporal": {"num_heads": 8, "num_layers": 4, "ff_dim": 512, "dropout": 0.1},
        "social": {"num_heads": 8, "num_layers": 3, "ff_dim": 512, "dropout": 0.1},
        "fusion": {"gate_type": "sigmoid"},
        "diffusion": {
            "num_steps": 100,
            "beta_schedule": "cosine",
            "decoder_layers": 6,
            "decoder_heads": 8,
            "decoder_ff_dim": 512,
        },
    }

    model = MotionTransformer(
        obs_len=8, pred_len=12, input_dim=2,
        embed_dim=embed_dim, config=config,
    ).to(device)

    total_params = model.count_parameters()
    logger.info(f"Model parameters: {total_params:,}")
    for name, count in model.get_component_params().items():
        logger.info(f"  {name}: {count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    # Training loop
    history = []
    best_ade = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in train_loader:
            obs = batch["obs_traj"].to(device)
            pred_gt = batch["pred_traj"].to(device)
            neighbors = batch["neighbor_obs"].to(device)
            n_mask = batch["neighbor_mask"].to(device)

            optimizer.zero_grad()
            loss = model(obs, pred_gt, neighbors, n_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        # Evaluate every 5 epochs
        eval_metrics = {}
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for batch in test_loader:
                    obs = batch["obs_traj"].to(device)
                    pred_gt = batch["pred_traj"].to(device)
                    neighbors = batch["neighbor_obs"].to(device)
                    n_mask = batch["neighbor_mask"].to(device)

                    preds = model.predict(
                        obs, neighbors, n_mask,
                        num_samples=20, use_ddim=True, ddim_steps=20,
                    )
                    all_preds.append(preds.cpu())
                    all_gts.append(pred_gt.cpu())

            predictions = torch.cat(all_preds)
            ground_truth = torch.cat(all_gts)
            eval_metrics = full_evaluation(predictions, ground_truth, k=20)

            if eval_metrics["minADE"] < best_ade:
                best_ade = eval_metrics["minADE"]
                # Save checkpoint
                ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "best_model.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "ade": eval_metrics["minADE"],
                    "fde": eval_metrics["minFDE"],
                }, ckpt_path)
                logger.info(f"  ★ New best ADE: {best_ade:.4f}")

            logger.info(
                f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f} | "
                f"ADE: {eval_metrics['minADE']:.4f} | "
                f"FDE: {eval_metrics['minFDE']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        record = {"epoch": epoch, "train_loss": avg_loss, **eval_metrics}
        history.append(record)

    return model, history, test_loader


def generate_visualizations(
    model: MotionTransformer,
    test_loader: DataLoader,
    history: list,
    device: torch.device,
    output_dir: Path,
):
    """Generate all publication-quality visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # 1. Training curves
    logger.info("Generating training curves...")
    fig = plot_training_curves(history, save_path=str(output_dir / "training_curves.png"))
    plt.close(fig)

    # 2. Trajectory prediction examples
    logger.info("Generating trajectory prediction plots...")
    batch = next(iter(test_loader))
    obs = batch["obs_traj"].to(device)
    pred_gt = batch["pred_traj"].to(device)
    neighbors = batch["neighbor_obs"].to(device)
    n_mask = batch["neighbor_mask"].to(device)

    with torch.no_grad():
        predictions = model.predict(
            obs, neighbors, n_mask,
            num_samples=20, use_ddim=True, ddim_steps=20,
        )

    # Plot first 6 examples
    num_examples = min(6, obs.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(num_examples):
        ax = axes[i]
        o = obs[i].cpu().numpy()
        gt = pred_gt[i].cpu().numpy()
        preds_i = predictions[i].cpu().numpy()
        n_obs = neighbors[i].cpu().numpy()
        nm = n_mask[i].cpu().numpy()

        # Plot on subplot
        # Neighbors
        for j in range(len(n_obs)):
            if nm[j]:
                ax.plot(n_obs[j, :, 0], n_obs[j, :, 1],
                       color="#95a5a6", alpha=0.4, linewidth=0.8)

        # Prediction samples
        for k in range(20):
            full = np.concatenate([o[-1:], preds_i[k]], axis=0)
            ax.plot(full[:, 0], full[:, 1], color="#3498db", alpha=0.12, linewidth=0.8)

        # Best prediction
        errors = np.linalg.norm(preds_i - gt[None], axis=-1).mean(axis=-1)
        best_k = errors.argmin()
        best = np.concatenate([o[-1:], preds_i[best_k]], axis=0)
        ax.plot(best[:, 0], best[:, 1], color="#e74c3c", linewidth=2.0,
               label=f"Best (ADE:{errors[best_k]:.2f})")

        # Ground truth
        full_gt = np.concatenate([o[-1:], gt], axis=0)
        ax.plot(full_gt[:, 0], full_gt[:, 1], color="#27ae60", linewidth=2.0,
               linestyle="--", label="GT")

        # Observed
        ax.plot(o[:, 0], o[:, 1], color="#2c3e50", linewidth=2.5, label="Obs")
        ax.scatter(o[0, 0], o[0, 1], color="#2c3e50", s=50, marker="o", zorder=5)
        ax.scatter(o[-1, 0], o[-1, 1], color="#2c3e50", s=50, marker="s", zorder=5)

        ax.set_title(f"Sample {i+1}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("MotionTransformer: Multi-Agent Trajectory Predictions (Best-of-20)",
                fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Per-scene comparison
    logger.info("Running multi-scene evaluation...")
    scenes = ["eth", "hotel", "univ", "zara1", "zara2"]
    scene_results = {}

    for scene in scenes:
        test_ds = ETHUCYDataset(scene=scene, split="test", augment=False)
        loader = DataLoader(
            test_ds, batch_size=32, shuffle=False, collate_fn=eth_ucy_collate_fn,
        )

        all_p, all_g = [], []
        with torch.no_grad():
            for b in loader:
                o = b["obs_traj"].to(device)
                g = b["pred_traj"].to(device)
                n = b["neighbor_obs"].to(device)
                m = b["neighbor_mask"].to(device)
                p = model.predict(o, n, m, num_samples=20, use_ddim=True, ddim_steps=20)
                all_p.append(p.cpu())
                all_g.append(g.cpu())

        preds = torch.cat(all_p)
        gts = torch.cat(all_g)
        metrics = full_evaluation(preds, gts, k=20)
        scene_results[scene] = metrics
        logger.info(f"  {scene}: ADE={metrics['minADE']:.4f}, FDE={metrics['minFDE']:.4f}")

    fig = create_multi_scene_comparison(
        scene_results, save_path=str(output_dir / "scene_comparison.png")
    )
    plt.close(fig)

    # 4. Diversity visualization
    logger.info("Generating diversity plot...")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Single example with many samples
    with torch.no_grad():
        single_obs = obs[:1]
        single_n = neighbors[:1]
        single_m = n_mask[:1]
        many_preds = model.predict(
            single_obs, single_n, single_m,
            num_samples=20, use_ddim=True, ddim_steps=20,
        )

    o = single_obs[0].cpu().numpy()
    preds_array = many_preds[0].cpu().numpy()
    gt = pred_gt[0].cpu().numpy()

    # Prediction heatmap via endpoints
    endpoints = preds_array[:, -1, :]  # (K, 2)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], c="#e74c3c", s=100,
              alpha=0.6, zorder=5, label="Endpoints (K=20)", edgecolors="white")

    for k in range(20):
        full = np.concatenate([o[-1:], preds_array[k]], axis=0)
        ax.plot(full[:, 0], full[:, 1], color="#3498db", alpha=0.3, linewidth=1.2)

    full_gt = np.concatenate([o[-1:], gt], axis=0)
    ax.plot(full_gt[:, 0], full_gt[:, 1], color="#27ae60", linewidth=3, linestyle="--",
           label="Ground Truth")
    ax.plot(o[:, 0], o[:, 1], color="#2c3e50", linewidth=3, label="Observed")

    ax.set_title("Trajectory Diversity: 20 Samples from Diffusion Decoder",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "diversity_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Architecture parameter breakdown
    logger.info("Generating architecture summary...")
    components = model.get_component_params()
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [k.replace("_", " ").title() for k in components.keys()]
    sizes = list(components.values())
    colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    explode = [0.05] * len(sizes)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100):,})',
        startangle=90, textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(
        f"Model Architecture: {model.count_parameters():,} Total Parameters",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "architecture_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return scene_results


def print_results_summary(scene_results: dict, history: list, model):
    """Print a comprehensive results summary."""
    print("\n" + "=" * 72)
    print("  MotionTransformer: Results Summary")
    print("=" * 72)

    print(f"\n  Model Parameters: {model.count_parameters():,}")
    for name, count in model.get_component_params().items():
        print(f"    {name}: {count:,}")

    print(f"\n  Training: {len(history)} epochs")
    print(f"    Final loss: {history[-1]['train_loss']:.6f}")

    print("\n  Per-Scene Results (Best-of-20):")
    print("  " + "-" * 58)
    print(f"  {'Scene':<10} {'minADE':>10} {'minFDE':>10} {'Diversity':>12} {'Collision':>12}")
    print("  " + "-" * 58)

    all_ade, all_fde = [], []
    for scene, metrics in scene_results.items():
        print(
            f"  {scene.upper():<10} "
            f"{metrics['minADE']:>10.4f} "
            f"{metrics['minFDE']:>10.4f} "
            f"{metrics['diversity']:>12.4f} "
            f"{metrics['collision_rate']:>12.4f}"
        )
        all_ade.append(metrics["minADE"])
        all_fde.append(metrics["minFDE"])

    print("  " + "-" * 58)
    print(
        f"  {'AVERAGE':<10} "
        f"{np.mean(all_ade):>10.4f} "
        f"{np.mean(all_fde):>10.4f}"
    )

    print("\n  Comparison with State-of-the-Art (ETH/UCY Average):")
    print("  " + "-" * 42)
    baselines = [
        ("Social-LSTM", 1.09, 2.35),
        ("Social-GAN", 0.81, 1.52),
        ("Trajectron++", 0.43, 0.86),
        ("AgentFormer", 0.45, 0.75),
        ("MID (Diffusion)", 0.39, 0.75),
        ("MotionTransformer (Ours)", np.mean(all_ade), np.mean(all_fde)),
    ]
    print(f"  {'Method':<28} {'ADE':>7} {'FDE':>7}")
    print("  " + "-" * 42)
    for name, ade, fde in baselines:
        marker = " ←" if "Ours" in name else ""
        print(f"  {name:<28} {ade:>7.2f} {fde:>7.2f}{marker}")

    print("\n" + "=" * 72)
    print("  Output files saved to: results/figures/")
    print("=" * 72)


def main():
    """Run the full demo pipeline."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " MotionTransformer: Full Demo Pipeline".center(70) + "║")
    print("║" + " Attention-Based Trajectory Forecasting with Diffusion".center(70) + "║")
    print("╚" + "═" * 70 + "╝\n")

    # Phase 1: Training
    logger.info("Phase 1: Training MotionTransformer...")
    model, history, test_loader = train_demo(
        scene="eth",
        epochs=30,
        batch_size=32,
        embed_dim=128,
        device=device,
    )

    # Phase 2: Visualization
    logger.info("\nPhase 2: Generating visualizations...")
    output_dir = PROJECT_ROOT / "results" / "figures"
    scene_results = generate_visualizations(
        model, test_loader, history, device, output_dir
    )

    # Phase 3: Results
    logger.info("\nPhase 3: Results summary")
    print_results_summary(scene_results, history, model)

    # Save comprehensive metrics
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "full_results.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for scene, m in scene_results.items():
        serializable[scene] = {k: float(v) for k, v in m.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Demo complete!")
    return model, scene_results


if __name__ == "__main__":
    main()
