"""
Evaluation pipeline for MotionTransformer.

Runs full evaluation on test set with all metrics and generates
visualizations and result tables.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.motion_transformer import MotionTransformer
from src.data.eth_ucy_dataset import ETHUCYDataset, eth_ucy_collate_fn
from src.evaluation.metrics import full_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset: str = "eth",
    num_samples: int = 20,
    use_ddim: bool = True,
):
    """Load checkpoint and run full evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model
    model_cfg = config["model"]
    data_cfg = config["data"]
    model = MotionTransformer(
        obs_len=data_cfg["obs_len"],
        pred_len=data_cfg["pred_len"],
        input_dim=model_cfg["input_dim"],
        embed_dim=model_cfg["embed_dim"],
        config=model_cfg,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from epoch {ckpt['epoch']}")

    # Data
    test_dataset = ETHUCYDataset(
        scene=dataset,
        obs_len=data_cfg["obs_len"],
        pred_len=data_cfg["pred_len"],
        split="test",
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=eth_ucy_collate_fn,
    )

    # Evaluate
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in test_loader:
            obs = batch["obs_traj"].to(device)
            pred_gt = batch["pred_traj"].to(device)
            neighbors = batch["neighbor_obs"].to(device)
            n_mask = batch["neighbor_mask"].to(device)

            predictions = model.predict(
                obs, neighbors, n_mask,
                num_samples=num_samples,
                use_ddim=use_ddim,
            )

            all_preds.append(predictions.cpu())
            all_gts.append(pred_gt.cpu())

    predictions = torch.cat(all_preds, dim=0)
    ground_truth = torch.cat(all_gts, dim=0)

    metrics = full_evaluation(predictions, ground_truth, k=num_samples)

    logger.info("\n" + "=" * 50)
    logger.info(f"EVALUATION RESULTS - {dataset.upper()}")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save results
    results_path = Path("results/metrics") / f"eval_{dataset}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="eth")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.dataset, args.num_samples)
