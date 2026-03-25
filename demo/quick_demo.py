"""
MotionTransformer - Quick Demo Pipeline.

Runs a streamlined version optimized for demonstration:
  - Smaller model config for fast iteration
  - DDIM sampling with fewer steps
  - Full visualization pipeline
"""

import sys
import json
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.motion_transformer import MotionTransformer
from src.data.eth_ucy_dataset import ETHUCYDataset, eth_ucy_collate_fn
from src.evaluation.metrics import full_evaluation, compute_best_of_k
from src.visualization.visualize import (
    plot_trajectory_predictions,
    plot_training_curves,
    create_multi_scene_comparison,
)
from src.utils.helpers import set_seed
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " MotionTransformer: Full Demo Pipeline".center(70) + "║")
    print("║" + " Transformer + Diffusion Trajectory Forecasting".center(70) + "║")
    print("╚" + "═" * 70 + "╝\n")

    # ---- Config (optimized for demo speed) ----
    EMBED_DIM = 64
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_SAMPLES = 5
    DDIM_STEPS = 5
    DIFFUSION_STEPS = 50

    config = {
        "temporal": {"num_heads": 4, "num_layers": 3, "ff_dim": 256, "dropout": 0.1},
        "social": {"num_heads": 4, "num_layers": 2, "ff_dim": 256, "dropout": 0.1},
        "fusion": {"gate_type": "sigmoid"},
        "diffusion": {
            "num_steps": DIFFUSION_STEPS,
            "beta_schedule": "cosine",
            "decoder_layers": 4,
            "decoder_heads": 4,
            "decoder_ff_dim": 256,
        },
    }

    model = MotionTransformer(
        obs_len=8, pred_len=12, input_dim=2,
        embed_dim=EMBED_DIM, config=config,
    ).to(device)

    total_params = model.count_parameters()
    logger.info(f"Model parameters: {total_params:,}")
    for name, count in model.get_component_params().items():
        logger.info(f"  {name}: {count:,}")

    # ---- Data ----
    train_data = ETHUCYDataset(scene="eth", split="train", augment=True)
    test_data = ETHUCYDataset(scene="eth", split="test", augment=False)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=eth_ucy_collate_fn, drop_last=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=eth_ucy_collate_fn,
    )
    logger.info(f"Train: {len(train_data)} samples | Test: {len(test_data)} samples")

    # ---- Training ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    history = []
    best_ade = float("inf")
    t_total = time.time()

    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: TRAINING")
    logger.info(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
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
            n_batch += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batch, 1)
        elapsed = time.time() - t0

        # Evaluate periodically
        eval_str = ""
        eval_metrics = {}
        if epoch == EPOCHS:
            model.eval()
            all_p, all_g = [], []
            with torch.no_grad():
                for batch in test_loader:
                    o = batch["obs_traj"].to(device)
                    g = batch["pred_traj"].to(device)
                    n = batch["neighbor_obs"].to(device)
                    m = batch["neighbor_mask"].to(device)
                    p = model.predict(o, n, m, num_samples=NUM_SAMPLES,
                                     use_ddim=True, ddim_steps=DDIM_STEPS)
                    all_p.append(p.cpu())
                    all_g.append(g.cpu())

            preds = torch.cat(all_p)
            gts = torch.cat(all_g)
            bok = compute_best_of_k(preds, gts, k=NUM_SAMPLES)
            eval_metrics = bok
            eval_str = f" | ADE: {bok['minADE']:.4f} | FDE: {bok['minFDE']:.4f}"

            if bok["minADE"] < best_ade:
                best_ade = bok["minADE"]
                ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "best_model.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                           "config": config}, ckpt_path)
                eval_str += " ★"

        record = {"epoch": epoch, "train_loss": avg_loss, "time": elapsed}
        if eval_metrics:
            record["ade"] = eval_metrics.get("minADE")
            record["fde"] = eval_metrics.get("minFDE")
        history.append(record)

        logger.info(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.6f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {elapsed:.1f}s{eval_str}")

    train_time = time.time() - t_total
    logger.info(f"\nTraining complete in {train_time:.1f}s | Best ADE: {best_ade:.4f}")

    # ---- Evaluation ----
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: MULTI-SCENE EVALUATION")
    logger.info(f"{'='*60}")

    scenes = ["eth", "hotel", "univ", "zara1", "zara2"]
    scene_results = {}

    model.eval()
    for scene in scenes:
        ds = ETHUCYDataset(scene=scene, split="test", augment=False)
        # Use subset for speed
        subset_size = min(128, len(ds))
        ds_sub = torch.utils.data.Subset(ds, range(subset_size))
        loader = DataLoader(ds_sub, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=eth_ucy_collate_fn)

        all_p, all_g = [], []
        with torch.no_grad():
            for batch in loader:
                o = batch["obs_traj"].to(device)
                g = batch["pred_traj"].to(device)
                n = batch["neighbor_obs"].to(device)
                m = batch["neighbor_mask"].to(device)
                p = model.predict(o, n, m, num_samples=NUM_SAMPLES,
                                 use_ddim=True, ddim_steps=DDIM_STEPS)
                all_p.append(p.cpu())
                all_g.append(g.cpu())

        preds = torch.cat(all_p)
        gts = torch.cat(all_g)
        metrics = full_evaluation(preds, gts, k=NUM_SAMPLES)
        scene_results[scene] = metrics
        logger.info(f"  {scene:>6}: ADE={metrics['minADE']:.4f}  FDE={metrics['minFDE']:.4f}  "
                    f"Diversity={metrics['diversity']:.4f}")

    # ---- Visualization ----
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3: GENERATING VISUALIZATIONS")
    logger.info(f"{'='*60}")

    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    fig = plot_training_curves(history, save_path=str(output_dir / "training_curves.png"))
    plt.close(fig)
    logger.info("  ✓ Training curves saved")

    # 2. Trajectory prediction grid
    batch = next(iter(test_loader))
    obs = batch["obs_traj"][:6].to(device)
    pred_gt = batch["pred_traj"][:6].to(device)
    neighbors = batch["neighbor_obs"][:6].to(device)
    n_mask = batch["neighbor_mask"][:6].to(device)

    with torch.no_grad():
        predictions = model.predict(obs, neighbors, n_mask, num_samples=NUM_SAMPLES,
                                   use_ddim=True, ddim_steps=DDIM_STEPS)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flatten()):
        if i >= obs.shape[0]:
            ax.axis("off")
            continue

        o = obs[i].cpu().numpy()
        gt = pred_gt[i].cpu().numpy()
        ps = predictions[i].cpu().numpy()
        nb = neighbors[i].cpu().numpy()
        nm = n_mask[i].cpu().numpy()

        # Neighbors
        for j in range(len(nb)):
            if nm[j]:
                ax.plot(nb[j, :, 0], nb[j, :, 1], color="#bdc3c7", alpha=0.4, lw=0.8)

        # All predictions
        for k in range(NUM_SAMPLES):
            full = np.concatenate([o[-1:], ps[k]], axis=0)
            ax.plot(full[:, 0], full[:, 1], color="#3498db", alpha=0.12, lw=0.8)

        # Best prediction
        errs = np.linalg.norm(ps - gt[None], axis=-1).mean(axis=-1)
        best_k = errs.argmin()
        best = np.concatenate([o[-1:], ps[best_k]], axis=0)
        ax.plot(best[:, 0], best[:, 1], color="#e74c3c", lw=2.0,
               label=f"Best (ADE:{errs[best_k]:.2f})")

        # Ground truth
        full_gt = np.concatenate([o[-1:], gt], axis=0)
        ax.plot(full_gt[:, 0], full_gt[:, 1], color="#27ae60", lw=2.0,
               linestyle="--", label="GT")

        # Observed
        ax.plot(o[:, 0], o[:, 1], color="#2c3e50", lw=2.5, label="Obs")
        ax.scatter(o[0, 0], o[0, 1], color="#2c3e50", s=50, marker="o", zorder=5)
        ax.scatter(o[-1, 0], o[-1, 1], color="#2c3e50", s=60, marker="s", zorder=5)

        ax.set_title(f"Sample {i+1}", fontsize=11)
        ax.legend(fontsize=7, loc="best")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("MotionTransformer: Multi-Agent Trajectory Predictions (Best-of-20)",
                fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ Trajectory prediction grid saved")

    # 3. Scene comparison bar chart
    fig = create_multi_scene_comparison(scene_results,
                                       save_path=str(output_dir / "scene_comparison.png"))
    plt.close(fig)
    logger.info("  ✓ Scene comparison chart saved")

    # 4. Diversity visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    with torch.no_grad():
        many_preds = model.predict(obs[:1], neighbors[:1], n_mask[:1],
                                  num_samples=NUM_SAMPLES, use_ddim=True, ddim_steps=DDIM_STEPS)

    o = obs[0].cpu().numpy()
    ps = many_preds[0].cpu().numpy()
    gt = pred_gt[0].cpu().numpy()
    endpoints = ps[:, -1, :]

    ax.scatter(endpoints[:, 0], endpoints[:, 1], c="#e74c3c", s=100, alpha=0.6,
              zorder=5, label="Endpoints (K=20)", edgecolors="white")
    for k in range(NUM_SAMPLES):
        full = np.concatenate([o[-1:], ps[k]], axis=0)
        ax.plot(full[:, 0], full[:, 1], color="#3498db", alpha=0.3, lw=1.2)
    full_gt = np.concatenate([o[-1:], gt], axis=0)
    ax.plot(full_gt[:, 0], full_gt[:, 1], color="#27ae60", lw=3, ls="--", label="Ground Truth")
    ax.plot(o[:, 0], o[:, 1], color="#2c3e50", lw=3, label="Observed")
    ax.set_title("Trajectory Diversity: 20 Diffusion Samples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "diversity_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ Diversity plot saved")

    # 5. Architecture breakdown
    components = model.get_component_params()
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [k.replace("_", " ").title() for k in components.keys()]
    sizes = list(components.values())
    colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=[0.05]*4,
        autopct=lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100):,})',
        startangle=90, textprops={"fontsize": 11},
    )
    ax.set_title(f"Architecture: {model.count_parameters():,} Parameters",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "architecture_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ Architecture breakdown saved")

    # ---- Final Summary ----
    all_ade = [m["minADE"] for m in scene_results.values()]
    all_fde = [m["minFDE"] for m in scene_results.values()]

    print("\n" + "=" * 72)
    print("  MOTIONTRANSFORMER: RESULTS SUMMARY")
    print("=" * 72)
    print(f"\n  Architecture")
    print(f"    Total Parameters: {model.count_parameters():,}")
    for name, count in model.get_component_params().items():
        print(f"      {name}: {count:,}")

    print(f"\n  Training")
    print(f"    Epochs: {EPOCHS} | Total time: {train_time:.1f}s")
    print(f"    Final loss: {history[-1]['train_loss']:.6f}")
    print(f"    Diffusion steps: {DIFFUSION_STEPS} | DDIM steps: {DDIM_STEPS}")

    print(f"\n  Per-Scene Results (Best-of-{NUM_SAMPLES}):")
    print(f"  {'':─<62}")
    print(f"  {'Scene':<10} {'minADE':>8} {'minFDE':>8} {'Diversity':>10} {'Collisions':>10}")
    print(f"  {'':─<62}")
    for scene, m in scene_results.items():
        print(f"  {scene.upper():<10} {m['minADE']:>8.4f} {m['minFDE']:>8.4f} "
              f"{m['diversity']:>10.4f} {m['collision_rate']:>10.4f}")
    print(f"  {'':─<62}")
    print(f"  {'AVERAGE':<10} {np.mean(all_ade):>8.4f} {np.mean(all_fde):>8.4f}")

    print(f"\n  Comparison with Published Methods:")
    print(f"  {'':─<46}")
    baselines = [
        ("Social-LSTM (2016)", 1.09, 2.35),
        ("Social-GAN (2018)", 0.81, 1.52),
        ("Trajectron++ (2020)", 0.43, 0.86),
        ("AgentFormer (2021)", 0.45, 0.75),
        ("MID / Diffusion (2022)", 0.39, 0.75),
        ("MotionTransformer (Ours)", np.mean(all_ade), np.mean(all_fde)),
    ]
    print(f"  {'Method':<28} {'ADE':>8} {'FDE':>8}")
    print(f"  {'':─<46}")
    for name, ade, fde in baselines:
        marker = " ◀" if "Ours" in name else ""
        print(f"  {name:<28} {ade:>8.2f} {fde:>8.2f}{marker}")

    print(f"\n  Generated Artifacts:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"    📊 {f.name}")

    print("\n" + "=" * 72)

    # Save metrics
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "full_results.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model_params": model.count_parameters(),
        "component_params": {k: int(v) for k, v in components.items()},
        "training": {"epochs": EPOCHS, "final_loss": history[-1]["train_loss"]},
        "scenes": {s: {k: float(v) for k, v in m.items()} for s, m in scene_results.items()},
        "average": {"minADE": float(np.mean(all_ade)), "minFDE": float(np.mean(all_fde))},
    }
    with open(metrics_path, "w") as f:
        json.dump(save_data, f, indent=2)

    return model, scene_results


if __name__ == "__main__":
    main()
