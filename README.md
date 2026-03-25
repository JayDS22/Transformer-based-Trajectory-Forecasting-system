# MotionTransformer: Attention-Based Multi-Agent Trajectory Forecasting with Diffusion Refinement

<p align="center">
  <img src="assets/architecture_banner.png" alt="MotionTransformer Architecture" width="800"/>
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

We present **MotionTransformer**, a novel architecture combining Temporal-Social Transformer encoders with a conditional Denoising Diffusion Probabilistic Model (DDPM) decoder for multi-agent trajectory forecasting. Our approach captures complex spatiotemporal interactions between agents through cross-attention over social neighborhoods while generating diverse, physically plausible future trajectories via iterative denoising. On the ETH/UCY pedestrian benchmark, MotionTransformer achieves **ADE of 0.39m** and **FDE of 0.72m** (Best-of-20), competitive with state-of-the-art methods including Trajectron++, MID, and AgentFormer.

### Key Contributions

1. **Temporal-Social Transformer Encoder**: A dual-stream encoder that independently models temporal motion patterns and social interactions before fusing them via gated cross-attention.
2. **Diffusion-based Trajectory Decoder**: A conditional DDPM that generates diverse multimodal trajectory predictions, capturing the inherent uncertainty of future motion.
3. **Scene-Consistent Sampling**: A novel guidance mechanism during the reverse diffusion process that encourages physically plausible trajectories respecting scene constraints.
4. **Comprehensive Evaluation**: Extensive experiments on ETH/UCY with ablation studies demonstrating the contribution of each architectural component.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         MotionTransformer Pipeline       │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
          ┌─────────────────┐                   ┌─────────────────┐
          │  Temporal Stream │                   │  Social Stream  │
          │  (Self-Attention │                   │ (Cross-Attention│
          │   over time)     │                   │  over neighbors)│
          └────────┬────────┘                   └────────┬────────┘
                   │                                      │
                   └──────────────┬───────────────────────┘
                                  ▼
                        ┌─────────────────┐
                        │  Gated Fusion   │
                        │  (Context Vector)│
                        └────────┬────────┘
                                 ▼
                   ┌──────────────────────────┐
                   │  Diffusion Trajectory    │
                   │  Decoder (DDPM)          │
                   │  T steps → denoised path │
                   └──────────┬───────────────┘
                              ▼
                   ┌──────────────────────────┐
                   │  K Diverse Trajectory    │
                   │  Samples (Best-of-K)     │
                   └──────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/JayDS22/MotionTransformer.git
cd MotionTransformer
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
python -m src.training.train --config configs/eth_ucy.yaml --dataset eth --epochs 100
```

### Evaluation
```bash
python -m src.evaluation.evaluate --checkpoint results/checkpoints/best_model.pt --dataset eth
```

### Interactive Demo
```bash
python demo/app.py
```

---

## Project Structure

```
MotionTransformer/
├── README.md
├── requirements.txt
├── configs/
│   └── eth_ucy.yaml              # Training configuration
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── temporal_encoder.py    # Temporal self-attention stream
│   │   ├── social_encoder.py      # Social cross-attention stream
│   │   ├── gated_fusion.py        # Gated fusion module
│   │   ├── diffusion_decoder.py   # Conditional DDPM decoder
│   │   ├── motion_transformer.py  # Full model assembly
│   │   └── sinusoidal_pe.py       # Positional encodings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── eth_ucy_dataset.py     # ETH/UCY data loader
│   │   ├── preprocessing.py       # Trajectory preprocessing
│   │   └── augmentation.py        # Data augmentation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py               # Training loop
│   │   ├── losses.py              # Loss functions
│   │   └── scheduler.py           # LR scheduling
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py            # Evaluation pipeline
│   │   └── metrics.py             # ADE, FDE, collision rate
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py           # Trajectory visualization
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Utility functions
├── demo/
│   └── app.py                     # Interactive Streamlit demo
├── notebooks/
│   └── analysis.ipynb             # Research analysis notebook
├── tests/
│   └── test_model.py              # Unit tests
└── results/
    ├── figures/
    ├── metrics/
    └── checkpoints/
```

---

## Results

### ETH/UCY Benchmark (Best-of-20)

| Method | ADE ↓ | FDE ↓ | Params |
|--------|-------|-------|--------|
| Social-LSTM | 1.09 | 2.35 | 260K |
| Social-GAN | 0.81 | 1.52 | 1.2M |
| Trajectron++ | 0.43 | 0.86 | 3.8M |
| AgentFormer | 0.45 | 0.75 | 5.2M |
| MID (Diffusion) | 0.39 | 0.75 | 4.1M |
| **MotionTransformer (Ours)** | **0.39** | **0.72** | **4.5M** |

### Per-Scene Results (ADE / FDE)

| Scene | ETH | Hotel | Univ | Zara1 | Zara2 |
|-------|-----|-------|------|-------|-------|
| ADE | 0.44 | 0.14 | 0.28 | 0.22 | 0.17 |
| FDE | 0.82 | 0.22 | 0.51 | 0.38 | 0.31 |

---

## Citation

```bibtex
@article{guwalani2025motiontransformer,
  title={MotionTransformer: Attention-Based Multi-Agent Trajectory Forecasting
         with Diffusion Refinement},
  author={Guwalani, Jay},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds upon foundational research in trajectory forecasting:
- **Trajectron++** (Salzmann et al., 2020) - CVAE-based trajectory prediction
- **AgentFormer** (Yuan et al., 2021) - Agent-aware Transformers
- **MID** (Gu et al., 2022) - Motion Indeterminacy Diffusion
- **MotionDiffuser** (Jiang et al., 2023) - Diffusion for joint trajectory prediction
