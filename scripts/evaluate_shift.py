#!/usr/bin/env python3
"""
Phase 3 — Evaluate RC-JEPA on clean and corrupted data.

Produces:
  - risk_coverage_curve.png
  - safe_degradation.png
  - per-corruption metrics printed to stdout

Usage
-----
python scripts/evaluate_shift.py \
    --config configs/eval_conformal.yaml \
    --ckpt_path outputs/best_predictor.pth \
    --threshold_file outputs/thresholds/tau_alpha0.1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import build_dataset, build_dataloader
from data.masking import MultiBlockMaskGenerator
from models.frozen_encoder import FrozenEncoder
from models.prob_predictor import ProbabilisticPredictor
from conformal.calibrator import ConformalCalibrator
from conformal.selector import RiskSelector
from utils import compute_uncertainty, compute_latent_error, set_seed


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@torch.no_grad()
def collect_scores(encoder, predictor, loader, mask_gen, device):
    predictor.eval()
    all_A, all_L = [], []
    for images, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        ctx_idx, tgt_idx = mask_gen(B)
        ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)

        all_reps = encoder.forward_full(images)
        D = all_reps.shape[-1]
        targets = torch.gather(
            all_reps, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, D))
        context = encoder.forward_context(images, ctx_idx)
        mu, log_sigma_sq = predictor(context, ctx_idx, tgt_idx)

        all_A.append(compute_uncertainty(log_sigma_sq).cpu().numpy())
        all_L.append(compute_latent_error(mu, targets).cpu().numpy())

    return np.concatenate(all_A), np.concatenate(all_L)


def plot_risk_coverage(coverages, risks, alpha, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, risks, linewidth=2, label="RC-JEPA")
    ax.axhline(y=alpha, color="red", linestyle="--", linewidth=1.5,
               label=f"Target α = {alpha}")
    ax.set_xlabel("Coverage", fontsize=13)
    ax.set_ylabel("Empirical Latent MSE (Risk)", fontsize=13)
    ax.set_title("Risk–Coverage Curve", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


def plot_safe_degradation(severities, risks, coverages, alpha, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_bar = sns.color_palette("muted")[0]
    color_line = sns.color_palette("muted")[3]

    x = np.arange(len(severities))
    bars = ax1.bar(x, risks, width=0.5, color=color_bar, alpha=0.8,
                   label="Empirical Risk")
    ax1.axhline(y=alpha, color="red", linestyle="--", linewidth=1.5,
                label=f"Target α = {alpha}")
    ax1.set_xlabel("Corruption Severity", fontsize=13)
    ax1.set_ylabel("Empirical Risk", fontsize=13, color=color_bar)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in severities])
    ax1.tick_params(axis="y", labelcolor=color_bar)

    ax2 = ax1.twinx()
    ax2.plot(x, coverages, "o-", color=color_line, linewidth=2, markersize=7,
             label="Coverage")
    ax2.set_ylabel("Coverage", fontsize=13, color=color_line)
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=10)

    ax1.set_title("Safe Degradation under Distribution Shift", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/eval_conformal.yaml")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--threshold_file", type=str, default=None)
    p.add_argument("--test_clean", type=str, default=None)
    p.add_argument("--test_corrupt", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- paths -----------------------------------------------------------
    ckpt_path = args.ckpt_path or cfg["calibration"]["ckpt_path"]
    threshold_file = args.threshold_file or \
        str(Path(cfg["evaluation"]["threshold_dir"]) / "tau_alpha0.1.json")
    test_clean = args.test_clean or cfg["evaluation"]["test_clean"]
    test_corrupt = args.test_corrupt or cfg["evaluation"]["test_corrupt"]
    output_dir = Path(args.output_dir or cfg["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- models ----------------------------------------------------------
    enc_cfg = cfg["model"]["encoder"]
    encoder = FrozenEncoder(
        model_name=enc_cfg["name"],
        img_size=enc_cfg["img_size"],
        patch_size=enc_cfg["patch_size"],
        checkpoint_path=enc_cfg.get("checkpoint_path"),
    ).to(device)

    pred_cfg = cfg["model"]["predictor"]
    predictor = ProbabilisticPredictor(
        encoder_dim=encoder.embed_dim,
        predictor_dim=pred_cfg["predictor_dim"],
        num_heads=pred_cfg["num_heads"],
        depth=pred_cfg["depth"],
        num_patches=encoder.num_patches,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    print(f"Loaded predictor from {ckpt_path}")

    mask_cfg = cfg["masking"]
    mask_gen = MultiBlockMaskGenerator(
        grid_h=encoder.grid_size[0],
        grid_w=encoder.grid_size[1],
        target_ratio=mask_cfg["target_ratio"],
        num_blocks=mask_cfg["num_blocks"],
    )

    selector = RiskSelector.from_file(threshold_file)
    alpha = selector.alpha or 0.10
    print(f"Threshold τ = {selector.tau_hat:.6f}  (α = {alpha})")

    # ---- 1. Clean evaluation ---------------------------------------------
    print("\n=== Clean Test Set ===")
    clean_ds = build_dataset(
        cfg["data"]["dataset"], test_clean,
        img_size=cfg["data"]["img_size"], is_train=False)
    clean_loader = build_dataloader(
        clean_ds, batch_size=128, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4), drop_last=False)

    A_clean, L_clean = collect_scores(
        encoder, predictor, clean_loader, mask_gen, device)
    res_clean = selector.evaluate(A_clean, L_clean)
    print(f"  Risk  = {res_clean['risk']:.6f}  (target ≤ {alpha})")
    print(f"  Coverage = {res_clean['coverage']:.2%}")

    # risk-coverage curve
    cal = ConformalCalibrator(alpha=alpha)
    covs, rsks = cal.risk_coverage_curve(A_clean, L_clean)
    plot_risk_coverage(covs, rsks, alpha, output_dir / "risk_coverage_curve.png")

    # ---- 2. Corruption evaluation ----------------------------------------
    severities = cfg["evaluation"].get("corruption_severities", [1, 2, 3, 4, 5])
    from data.corruptions import CORRUPTION_TYPES

    sev_risks, sev_covs = [], []

    for sev in severities:
        sev_A_list, sev_L_list = [], []
        found_any = False

        for corruption in CORRUPTION_TYPES:
            import os
            cdir = os.path.join(test_corrupt, corruption, str(sev))
            if not os.path.isdir(cdir):
                continue
            found_any = True
            try:
                from data.corruptions import CorruptedDataset
                from data.datasets import get_transform
                transform = get_transform(cfg["data"]["img_size"], is_train=False)
                cds = CorruptedDataset(test_corrupt, corruption, sev, transform)
                cloader = build_dataloader(
                    cds, batch_size=128, shuffle=False,
                    num_workers=cfg["data"].get("num_workers", 4),
                    drop_last=False)
                A_c, L_c = collect_scores(
                    encoder, predictor, cloader, mask_gen, device)
                sev_A_list.append(A_c)
                sev_L_list.append(L_c)
            except Exception as exc:
                print(f"  skip {corruption}/{sev}: {exc}")

        if not found_any:
            print(f"\nSeverity {sev}: no corruption data found, skipping.")
            sev_risks.append(float("nan"))
            sev_covs.append(float("nan"))
            continue

        A_sev = np.concatenate(sev_A_list)
        L_sev = np.concatenate(sev_L_list)
        res = selector.evaluate(A_sev, L_sev)
        sev_risks.append(res["risk"])
        sev_covs.append(res["coverage"])
        status = "Valid" if res["risk"] <= alpha else "VIOLATED"
        print(f"\nSeverity {sev}:  risk={res['risk']:.6f} ({status})  "
              f"coverage={res['coverage']:.2%}")

    # ---- safe degradation plot -------------------------------------------
    valid_idx = [i for i, r in enumerate(sev_risks) if not np.isnan(r)]
    if valid_idx:
        plot_safe_degradation(
            [severities[i] for i in valid_idx],
            [sev_risks[i] for i in valid_idx],
            [sev_covs[i] for i in valid_idx],
            alpha,
            output_dir / "safe_degradation.png",
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
