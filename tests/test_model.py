"""Unit tests for MotionTransformer components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest
from src.models.temporal_encoder import TemporalTransformerEncoder
from src.models.social_encoder import SocialTransformerEncoder
from src.models.gated_fusion import GatedFusion
from src.models.diffusion_decoder import DiffusionTrajectoryDecoder
from src.models.motion_transformer import MotionTransformer
from src.evaluation.metrics import compute_ade, compute_fde, compute_best_of_k


def test_temporal_encoder():
    """Test temporal encoder forward pass."""
    encoder = TemporalTransformerEncoder(
        input_dim=2, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128
    )
    obs = torch.randn(4, 8, 2)  # (B, T_obs, 2)
    features, context = encoder(obs)

    assert features.shape == (4, 8, 64), f"Got {features.shape}"
    assert context.shape == (4, 64), f"Got {context.shape}"
    print("✓ Temporal encoder: shapes correct")


def test_social_encoder():
    """Test social encoder forward pass."""
    encoder = SocialTransformerEncoder(
        input_dim=2, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128
    )
    ego_ctx = torch.randn(4, 64)
    ego_pos = torch.randn(4, 2)
    ego_vel = torch.randn(4, 2)
    n_pos = torch.randn(4, 5, 2)
    n_vel = torch.randn(4, 5, 2)
    n_mask = torch.ones(4, 5, dtype=torch.bool)

    social_ctx = encoder(ego_ctx, ego_pos, ego_vel, n_pos, n_vel, n_mask)
    assert social_ctx.shape == (4, 64), f"Got {social_ctx.shape}"
    print("✓ Social encoder: shapes correct")


def test_gated_fusion():
    """Test gated fusion module."""
    fusion = GatedFusion(embed_dim=64, gate_type="sigmoid")
    t_ctx = torch.randn(4, 64)
    s_ctx = torch.randn(4, 64)

    fused = fusion(t_ctx, s_ctx)
    assert fused.shape == (4, 64)
    print("✓ Gated fusion: shapes correct")


def test_diffusion_decoder():
    """Test diffusion decoder forward and sampling."""
    decoder = DiffusionTrajectoryDecoder(
        pred_len=12, input_dim=2, embed_dim=64,
        num_steps=10, num_layers=2, num_heads=4, ff_dim=128,
    )

    # Training: compute loss
    x_0 = torch.randn(4, 12, 2)
    context = torch.randn(4, 64)
    loss = decoder.compute_loss(x_0, context)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    print(f"✓ Diffusion loss: {loss.item():.4f}")

    # Sampling (DDIM for speed)
    samples = decoder.sample_ddim(context, num_samples=3, ddim_steps=5)
    assert samples.shape == (4, 3, 12, 2), f"Got {samples.shape}"
    print("✓ Diffusion DDIM sampling: shapes correct")


def test_full_model():
    """Test full MotionTransformer end-to-end."""
    model = MotionTransformer(
        obs_len=8, pred_len=12, input_dim=2, embed_dim=64,
        config={
            "temporal": {"num_heads": 4, "num_layers": 2, "ff_dim": 128},
            "social": {"num_heads": 4, "num_layers": 2, "ff_dim": 128},
            "fusion": {"gate_type": "sigmoid"},
            "diffusion": {
                "num_steps": 10, "decoder_layers": 2,
                "decoder_heads": 4, "decoder_ff_dim": 128,
                "beta_schedule": "cosine",
            },
        },
    )

    B = 4
    obs = torch.randn(B, 8, 2)
    pred_gt = torch.randn(B, 12, 2)
    neighbors = torch.randn(B, 5, 8, 2)
    n_mask = torch.ones(B, 5, dtype=torch.bool)

    # Training forward
    loss = model(obs, pred_gt, neighbors, n_mask)
    assert loss.dim() == 0
    print(f"✓ Full model training loss: {loss.item():.4f}")

    # Inference
    preds = model.predict(obs, neighbors, n_mask, num_samples=3, use_ddim=True, ddim_steps=5)
    assert preds.shape == (B, 3, 12, 2), f"Got {preds.shape}"
    print("✓ Full model prediction: shapes correct")

    # Parameter count
    params = model.count_parameters()
    print(f"✓ Total parameters: {params:,}")
    components = model.get_component_params()
    for name, count in components.items():
        print(f"    {name}: {count:,}")


def test_metrics():
    """Test metric computations."""
    B, K, T = 8, 20, 12
    preds = torch.randn(B, K, T, 2)
    gt = torch.randn(B, T, 2)

    ade = compute_ade(preds, gt)
    assert ade.shape == (B, K)

    fde = compute_fde(preds, gt)
    assert fde.shape == (B, K)

    bok = compute_best_of_k(preds, gt, k=20)
    assert "minADE" in bok
    assert "minFDE" in bok
    print(f"✓ Metrics: minADE={bok['minADE']:.4f}, minFDE={bok['minFDE']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("MotionTransformer Unit Tests")
    print("=" * 60)

    test_temporal_encoder()
    test_social_encoder()
    test_gated_fusion()
    test_diffusion_decoder()
    test_full_model()
    test_metrics()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
