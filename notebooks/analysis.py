"""
MotionTransformer — Research Analysis & Ablation Study
=======================================================
1. Ablation studies (remove each component and measure impact)
2. Number of samples (K) vs. Best-of-K performance
3. DDIM steps vs quality/speed tradeoff
4. Qualitative trajectory analysis (12-panel grid)
5. Error distribution analysis
"""

import sys, json, time, logging
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.models.motion_transformer import MotionTransformer
from src.data.eth_ucy_dataset import ETHUCYDataset, eth_ucy_collate_fn
from src.evaluation.metrics import compute_best_of_k, full_evaluation
from src.utils.helpers import set_seed
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#fafafa","axes.grid":True,"grid.alpha":0.25,"font.size":11,"figure.dpi":150})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBED, EPOCHS_ABL, BS = 64, 5, 64

BASE_CFG = {
    "temporal": {"num_heads":4,"num_layers":3,"ff_dim":256,"dropout":0.1},
    "social":   {"num_heads":4,"num_layers":2,"ff_dim":256,"dropout":0.1},
    "fusion":   {"gate_type":"sigmoid"},
    "diffusion":{"num_steps":50,"beta_schedule":"cosine","decoder_layers":4,"decoder_heads":4,"decoder_ff_dim":256},
}

def build(cfg=None):
    return MotionTransformer(obs_len=8,pred_len=12,input_dim=2,embed_dim=EMBED,config=cfg or BASE_CFG).to(DEVICE)

def train(model, loader, epochs=EPOCHS_ABL):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    losses = []
    for ep in range(epochs):
        model.train(); el=0; n=0
        for b in loader:
            opt.zero_grad()
            loss = model(b["obs_traj"].to(DEVICE), b["pred_traj"].to(DEVICE),
                         b["neighbor_obs"].to(DEVICE), b["neighbor_mask"].to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); el+=loss.item(); n+=1
        sch.step(); losses.append(el/max(n,1))
    return losses

def evaluate(model, loader, K=5, ddim=5):
    model.eval(); ap,ag=[],[]
    with torch.no_grad():
        for b in loader:
            p=model.predict(b["obs_traj"].to(DEVICE), b["neighbor_obs"].to(DEVICE),
                            b["neighbor_mask"].to(DEVICE), num_samples=K, use_ddim=True, ddim_steps=ddim)
            ap.append(p.cpu()); ag.append(b["pred_traj"])
    return full_evaluation(torch.cat(ap), torch.cat(ag), k=K)


# ─── 1. ABLATION ───────────────────────────────────────────────
def run_ablation(tl, el):
    logger.info("\n" + "="*60 + "\nABLATION STUDY\n" + "="*60)
    results = {}

    # Full model
    set_seed(42); m = build(); train(m, tl)
    r = evaluate(m, el); results["Full model"] = r
    logger.info(f"  Full model        ADE={r['minADE']:.4f}  FDE={r['minFDE']:.4f}")

    # No social encoder
    set_seed(42); m = build()
    orig = m.encode
    def no_social(obs, nb=None, nm=None):
        last = obs[:, -1:, :]; on = obs - last
        nn_ = nb - last.unsqueeze(1) if nb is not None else nb
        tf, tc = m.temporal_encoder(on)
        sc = torch.zeros_like(tc)
        return m.fusion(tc, sc)
    m.encode = no_social; train(m, tl)
    r = evaluate(m, el); results["No social encoder"] = r
    logger.info(f"  No social         ADE={r['minADE']:.4f}  FDE={r['minFDE']:.4f}")

    # No gated fusion (simple average)
    set_seed(42); m = build()
    orig_enc = m.encode
    def avg_fusion(obs, nb=None, nm=None):
        last = obs[:, -1:, :]; on = obs - last
        nn_ = nb - last.unsqueeze(1) if nb is not None else nb
        tf, tc = m.temporal_encoder(on)
        if nn_ is not None and nn_.shape[1] > 0:
            ep = on[:,-1,:]; ev = on[:,-1,:] - on[:,-2,:]
            np_ = nn_[:,:,-1,:]; nv = nn_[:,:,-1,:] - nn_[:,:,-2,:]
            sc = m.social_encoder(tc, ep, ev, np_, nv, nm)
        else:
            sc = torch.zeros_like(tc)
        return (tc + sc) / 2.0
    m.encode = avg_fusion; train(m, tl)
    r = evaluate(m, el); results["No gated fusion"] = r
    logger.info(f"  No gated fusion   ADE={r['minADE']:.4f}  FDE={r['minFDE']:.4f}")

    # Fewer diffusion steps
    set_seed(42); cfg25 = {**BASE_CFG, "diffusion":{**BASE_CFG["diffusion"],"num_steps":25}}
    m = build(cfg25); train(m, tl)
    r = evaluate(m, el); results["25 diffusion steps"] = r
    logger.info(f"  25 diff steps     ADE={r['minADE']:.4f}  FDE={r['minFDE']:.4f}")

    # Linear schedule
    set_seed(42); cfg_lin = {**BASE_CFG, "diffusion":{**BASE_CFG["diffusion"],"beta_schedule":"linear"}}
    m = build(cfg_lin); train(m, tl)
    r = evaluate(m, el); results["Linear schedule"] = r
    logger.info(f"  Linear schedule   ADE={r['minADE']:.4f}  FDE={r['minFDE']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(results.keys())
    ades = [results[n]["minADE"] for n in names]
    fdes = [results[n]["minFDE"] for n in names]
    colors = ["#3498db" if i==0 else "#95a5a6" for i in range(len(names))]
    x = np.arange(len(names))
    for ax, vals, label, title in [(axes[0],ades,"minADE","ADE"),(axes[1],fdes,"minFDE","FDE")]:
        bars = ax.barh(x, vals, color=colors, height=0.6, edgecolor=[c.replace("a6","8d") for c in colors])
        ax.set_yticks(x); ax.set_yticklabels(names)
        ax.set_xlabel(f"{label} (m) ↓"); ax.set_title(f"Ablation: {title}"); ax.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=10)
    fig.suptitle("MotionTransformer Ablation Study", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/"ablation_study.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("  ✓ Ablation plot saved")
    return results


# ─── 2. K SAMPLES ANALYSIS ─────────────────────────────────────
def run_samples_analysis(model, el):
    logger.info("\n" + "="*60 + "\nSAMPLES (K) VS PERFORMANCE\n" + "="*60)
    model.eval(); ap,ag=[],[]
    with torch.no_grad():
        for b in el:
            p=model.predict(b["obs_traj"].to(DEVICE), b["neighbor_obs"].to(DEVICE),
                            b["neighbor_mask"].to(DEVICE), num_samples=20, use_ddim=True, ddim_steps=10)
            ap.append(p.cpu()); ag.append(b["pred_traj"])
    preds,gts = torch.cat(ap), torch.cat(ag)
    res = {"k":[],"ade":[],"fde":[]}
    for k in [1,2,3,5,10,15,20]:
        m = compute_best_of_k(preds[:,:k], gts, k=k)
        res["k"].append(k); res["ade"].append(m["minADE"]); res["fde"].append(m["minFDE"])
        logger.info(f"  K={k:2d}: ADE={m['minADE']:.4f}  FDE={m['minFDE']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(res["k"], res["ade"], "o-", color="#3498db", lw=2, markersize=7)
    axes[0].fill_between(res["k"], res["ade"], alpha=0.1, color="#3498db")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("minADE (m)"); axes[0].set_title("ADE vs K"); axes[0].set_xticks(res["k"])
    axes[1].plot(res["k"], res["fde"], "s-", color="#e74c3c", lw=2, markersize=7)
    axes[1].fill_between(res["k"], res["fde"], alpha=0.1, color="#e74c3c")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("minFDE (m)"); axes[1].set_title("FDE vs K"); axes[1].set_xticks(res["k"])
    fig.suptitle("Best-of-K Performance", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/"samples_vs_performance.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("  ✓ Samples plot saved")
    return res


# ─── 3. DDIM SPEED/QUALITY ─────────────────────────────────────
def run_ddim_analysis(model, el):
    logger.info("\n" + "="*60 + "\nDDIM STEPS VS QUALITY/SPEED\n" + "="*60)
    model.eval()
    b = next(iter(el))
    obs,gt,n,m_ = [b[k].to(DEVICE) for k in ["obs_traj","pred_traj","neighbor_obs","neighbor_mask"]]
    res = {"steps":[],"ade":[],"fde":[],"time_ms":[]}
    for ds in [5,10,20,50]:
        times=[]
        for _ in range(3):
            t0=time.time()
            with torch.no_grad(): preds=model.predict(obs,n,m_,num_samples=5,use_ddim=True,ddim_steps=ds)
            times.append((time.time()-t0)*1000)
        mt = compute_best_of_k(preds, gt, k=5)
        res["steps"].append(ds); res["ade"].append(mt["minADE"]); res["fde"].append(mt["minFDE"]); res["time_ms"].append(np.mean(times))
        logger.info(f"  DDIM={ds:2d}: ADE={mt['minADE']:.4f}  Time={np.mean(times):.0f}ms")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(res["steps"], res["ade"], "o-", color="#3498db", lw=2, markersize=8, label="minADE")
    ax1.set_xlabel("DDIM steps"); ax1.set_ylabel("minADE (m)", color="#3498db"); ax1.set_xticks(res["steps"])
    ax2 = ax1.twinx()
    ax2.plot(res["steps"], res["time_ms"], "s--", color="#e74c3c", lw=2, markersize=8, label="Time (ms)")
    ax2.set_ylabel("Inference time (ms)", color="#e74c3c")
    lines1,l1=ax1.get_legend_handles_labels(); lines2,l2=ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="center right")
    ax1.set_title("DDIM: Quality vs Speed", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/"ddim_tradeoff.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("  ✓ DDIM plot saved")
    return res


# ─── 4. MULTIMODAL GRID ────────────────────────────────────────
def plot_multimodal(model, el):
    logger.info("\n" + "="*60 + "\nMULTIMODAL ANALYSIS\n" + "="*60)
    model.eval()
    b = next(iter(el))
    obs,gt,nb,nm = [b[k].to(DEVICE) for k in ["obs_traj","pred_traj","neighbor_obs","neighbor_mask"]]
    with torch.no_grad():
        predictions = model.predict(obs, nb, nm, num_samples=20, use_ddim=True, ddim_steps=10)

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    for idx in range(min(12, obs.shape[0])):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        o = obs[idx].cpu().numpy(); g = gt[idx].cpu().numpy()
        ps = predictions[idx].cpu().numpy()
        nbo = nb[idx].cpu().numpy(); nmo = nm[idx].cpu().numpy()

        for j in range(len(nbo)):
            if nmo[j]: ax.plot(nbo[j,:,0], nbo[j,:,1], color="#bdc3c7", alpha=0.3, lw=0.7)

        errs = np.linalg.norm(ps - g[None], axis=-1).mean(axis=-1)
        for k in np.argsort(errs)[::-1]:
            alpha = 0.06 + 0.35 * (np.searchsorted(np.sort(errs), errs[k]) / len(errs))
            full = np.concatenate([o[-1:], ps[k]], axis=0)
            ax.plot(full[:,0], full[:,1], color="#3498db", alpha=alpha, lw=0.9)

        best_k = errs.argmin()
        best = np.concatenate([o[-1:], ps[best_k]], axis=0)
        ax.plot(best[:,0], best[:,1], color="#e74c3c", lw=2.2, zorder=4)
        full_gt = np.concatenate([o[-1:], g], axis=0)
        ax.plot(full_gt[:,0], full_gt[:,1], color="#27ae60", lw=2.2, ls="--", zorder=3)
        ax.plot(o[:,0], o[:,1], color="#2c3e50", lw=2.5, zorder=5)
        ax.scatter(o[0,0], o[0,1], color="#2c3e50", s=40, marker="o", zorder=6)
        ax.scatter(o[-1,0], o[-1,1], color="#2c3e50", s=50, marker="s", zorder=6)
        ax.scatter(ps[:,-1,0], ps[:,-1,1], c="#e74c3c", s=15, alpha=0.4, zorder=4)
        ax.set_title(f"Best ADE: {errs[best_k]:.2f}m", fontsize=9, pad=4)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)

    fig.suptitle("MotionTransformer: 20 Diverse Predictions per Agent", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR/"multimodal_analysis.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("  ✓ Multimodal grid saved")


# ─── 5. ERROR DISTRIBUTIONS ────────────────────────────────────
def plot_error_dist(model, el):
    logger.info("\n" + "="*60 + "\nERROR DISTRIBUTIONS\n" + "="*60)
    model.eval(); ap,ag=[],[]
    with torch.no_grad():
        for b in el:
            p=model.predict(b["obs_traj"].to(DEVICE), b["neighbor_obs"].to(DEVICE),
                            b["neighbor_mask"].to(DEVICE), num_samples=10, use_ddim=True, ddim_steps=10)
            ap.append(p.cpu()); ag.append(b["pred_traj"])
    preds,gts = torch.cat(ap), torch.cat(ag)
    gt_e = gts.unsqueeze(1); errors = torch.norm(preds - gt_e, dim=-1)
    ade_k = errors.mean(dim=-1); fde_k = errors[:,:,-1]
    best_ade = ade_k.min(dim=1)[0].numpy(); best_fde = fde_k.min(dim=1)[0].numpy()

    bi = ade_k.argmin(dim=1); bp = preds[torch.arange(len(preds)), bi]
    pse = torch.norm(bp - gts, dim=-1).numpy()
    ms = pse.mean(axis=0); ss = pse.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(best_ade, bins=40, color="#3498db", alpha=0.7, edgecolor="#2980b9")
    axes[0].axvline(np.mean(best_ade), color="#e74c3c", ls="--", lw=2, label=f"Mean: {np.mean(best_ade):.3f}")
    axes[0].set_xlabel("ADE (m)"); axes[0].set_title("ADE Distribution"); axes[0].legend()

    axes[1].hist(best_fde, bins=40, color="#e74c3c", alpha=0.7, edgecolor="#c0392b")
    axes[1].axvline(np.mean(best_fde), color="#3498db", ls="--", lw=2, label=f"Mean: {np.mean(best_fde):.3f}")
    axes[1].set_xlabel("FDE (m)"); axes[1].set_title("FDE Distribution"); axes[1].legend()

    t = np.arange(1, len(ms)+1)
    axes[2].plot(t, ms, color="#9b59b6", lw=2.5, label="Mean")
    axes[2].fill_between(t, ms-ss, ms+ss, alpha=0.2, color="#9b59b6")
    axes[2].set_xlabel("Pred. timestep"); axes[2].set_ylabel("Error (m)"); axes[2].set_title("Error Growth"); axes[2].legend()

    fig.suptitle("Error Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/"error_distributions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  ADE: {np.mean(best_ade):.4f}±{np.std(best_ade):.4f}  FDE: {np.mean(best_fde):.4f}±{np.std(best_fde):.4f}")
    logger.info("  ✓ Error distribution plots saved")


# ─── MAIN ───────────────────────────────────────────────────────
def main():
    set_seed(42)
    print("\n╔" + "═"*70 + "╗")
    print("║" + " MotionTransformer: Research Analysis & Ablation".center(70) + "║")
    print("╚" + "═"*70 + "╝\n")

    td = ETHUCYDataset(scene="eth", split="train", augment=True)
    ed = ETHUCYDataset(scene="eth", split="test", augment=False)
    es = torch.utils.data.Subset(ed, range(min(128, len(ed))))
    tl = DataLoader(td, batch_size=BS, shuffle=True, collate_fn=eth_ucy_collate_fn, drop_last=True)
    el = DataLoader(es, batch_size=BS, shuffle=False, collate_fn=eth_ucy_collate_fn)

    abl = run_ablation(tl, el)

    logger.info("\nTraining full model for remaining experiments...")
    set_seed(42); model = build(); train(model, tl, epochs=5)

    run_samples_analysis(model, el)
    run_ddim_analysis(model, el)
    plot_multimodal(model, el)
    plot_error_dist(model, el)

    all_res = {"ablation":{k:{kk:float(vv) for kk,vv in v.items()} for k,v in abl.items()}}
    with open(PROJECT_ROOT/"results"/"metrics"/"analysis_results.json","w") as f:
        json.dump(all_res, f, indent=2, default=float)

    print("\n" + "="*60)
    print("  Analysis complete! Figures:")
    for f in sorted(OUTPUT_DIR.glob("*.png")): print(f"    📊 {f.name}")
    print("="*60)

if __name__ == "__main__":
    main()
