"""
MotionTransformer Training Pipeline.

Implements the full training loop with:
  - Gradient clipping
  - Learning rate warmup + cosine annealing
  - Periodic evaluation
  - Checkpoint saving
  - TensorBoard logging
"""

import os
import sys
import time
import yaml
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.motion_transformer import MotionTransformer
from src.data.eth_ucy_dataset import ETHUCYDataset, eth_ucy_collate_fn
from src.training.losses import CombinedLoss
from src.training.scheduler import get_scheduler
from src.evaluation.metrics import compute_ade, compute_fde

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: MotionTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    epoch: int = 0,
    log_interval: int = 10,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        obs = batch["obs_traj"].to(device)
        pred_gt = batch["pred_traj"].to(device)
        neighbors = batch["neighbor_obs"].to(device)
        n_mask = batch["neighbor_mask"].to(device)

        optimizer.zero_grad()

        # Forward pass → diffusion loss
        loss = model(obs, pred_gt, neighbors, n_mask)

        # Backward
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / num_batches
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.6f} | Avg: {avg:.6f}"
            )

    return {"train_loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate(
    model: MotionTransformer,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 20,
    use_ddim: bool = True,
) -> dict:
    """Evaluate model on dataset."""
    model.eval()
    all_ade = []
    all_fde = []

    for batch in loader:
        obs = batch["obs_traj"].to(device)
        pred_gt = batch["pred_traj"].to(device)
        neighbors = batch["neighbor_obs"].to(device)
        n_mask = batch["neighbor_mask"].to(device)

        # Generate predictions
        predictions = model.predict(
            obs, neighbors, n_mask,
            num_samples=num_samples,
            use_ddim=use_ddim,
            ddim_steps=20,
        )  # (B, K, T_pred, 2)

        # Best-of-K metrics
        gt_expanded = pred_gt.unsqueeze(1)  # (B, 1, T, 2)
        errors = torch.norm(predictions - gt_expanded, dim=-1)  # (B, K, T)

        ade_per_sample = errors.mean(dim=-1)  # (B, K)
        fde_per_sample = errors[:, :, -1]  # (B, K)

        best_ade = ade_per_sample.min(dim=1)[0]  # (B,)
        best_fde = fde_per_sample.min(dim=1)[0]  # (B,)

        all_ade.append(best_ade.cpu())
        all_fde.append(best_fde.cpu())

    all_ade = torch.cat(all_ade)
    all_fde = torch.cat(all_fde)

    return {
        "ade": all_ade.mean().item(),
        "fde": all_fde.mean().item(),
        "ade_std": all_ade.std().item(),
        "fde_std": all_fde.std().item(),
    }


def train(config_path: str, dataset: str = "eth", epochs: int = None):
    """Main training function."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if epochs is not None:
        config["training"]["epochs"] = epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    data_cfg = config["data"]
    train_dataset = ETHUCYDataset(
        scene=dataset,
        obs_len=data_cfg["obs_len"],
        pred_len=data_cfg["pred_len"],
        max_neighbors=data_cfg.get("max_neighbors", 10),
        split="train",
        augment=data_cfg.get("augment", True),
    )
    test_dataset = ETHUCYDataset(
        scene=dataset,
        obs_len=data_cfg["obs_len"],
        pred_len=data_cfg["pred_len"],
        max_neighbors=data_cfg.get("max_neighbors", 10),
        split="test",
        augment=False,
    )

    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=eth_ucy_collate_fn,
        num_workers=0,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=eth_ucy_collate_fn,
        num_workers=0,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Model
    model_cfg = config["model"]
    model = MotionTransformer(
        obs_len=data_cfg["obs_len"],
        pred_len=data_cfg["pred_len"],
        input_dim=model_cfg["input_dim"],
        embed_dim=model_cfg["embed_dim"],
        config=model_cfg,
    ).to(device)

    total_params = model.count_parameters()
    component_params = model.get_component_params()
    logger.info(f"Total parameters: {total_params:,}")
    for name, count in component_params.items():
        logger.info(f"  {name}: {count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = get_scheduler(optimizer, train_cfg)

    # Training loop
    best_ade = float("inf")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)

    training_history = []

    num_epochs = train_cfg["epochs"]
    eval_interval = config["logging"].get("save_interval", 5)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=train_cfg.get("grad_clip", 1.0),
            epoch=epoch,
            log_interval=config["logging"].get("log_interval", 10),
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # Evaluate periodically
        eval_metrics = {}
        if epoch % eval_interval == 0 or epoch == num_epochs:
            logger.info(f"Evaluating at epoch {epoch}...")
            eval_metrics = evaluate(
                model, test_loader, device,
                num_samples=config["evaluation"].get("best_of_k", 20),
                use_ddim=True,
            )
            logger.info(
                f"  ADE: {eval_metrics['ade']:.4f} ± {eval_metrics['ade_std']:.4f} | "
                f"FDE: {eval_metrics['fde']:.4f} ± {eval_metrics['fde_std']:.4f}"
            )

            # Save best model
            if eval_metrics["ade"] < best_ade:
                best_ade = eval_metrics["ade"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ade": eval_metrics["ade"],
                    "fde": eval_metrics["fde"],
                    "config": config,
                }, results_dir / "checkpoints" / "best_model.pt")
                logger.info(f"  ★ New best ADE: {best_ade:.4f}")

        # Log
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "lr": current_lr,
            "time": elapsed,
            **eval_metrics,
        }
        training_history.append(record)

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Loss: {train_metrics['train_loss']:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

    # Save training history
    import json
    with open(results_dir / "metrics" / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"\nTraining complete. Best ADE: {best_ade:.4f}")
    return model, training_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MotionTransformer")
    parser.add_argument("--config", type=str, default="configs/eth_ucy.yaml")
    parser.add_argument("--dataset", type=str, default="eth")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train(args.config, args.dataset, args.epochs)
