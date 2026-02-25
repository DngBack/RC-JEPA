#!/usr/bin/env python3
"""
Phase 2 — Calibrate the conformal risk threshold on a held-out set.

Usage
-----
python scripts/calibrate.py \
    --config configs/eval_conformal.yaml \
    --ckpt_path outputs/best_predictor.pth \
    --calib_dir /path/to/val_calib \
    --target_alpha 0.1
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
from utils import compute_uncertainty, compute_latent_error, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/eval_conformal.yaml")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--calib_dir", type=str, default=None)
    p.add_argument("--target_alpha", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--save_threshold", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def collect_scores(
    encoder: FrozenEncoder,
    predictor: ProbabilisticPredictor,
    loader,
    mask_gen: MultiBlockMaskGenerator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full pipeline on a dataset and return (A, L) arrays."""
    predictor.eval()
    all_A, all_L = [], []

    for images, _ in tqdm(loader, desc="Calibration"):
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        ctx_idx, tgt_idx = mask_gen(B)
        ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)

        all_reps = encoder.forward_full(images)
        D = all_reps.shape[-1]
        targets = torch.gather(
            all_reps, 1,
            tgt_idx.unsqueeze(-1).expand(-1, -1, D),
        )
        context = encoder.forward_context(images, ctx_idx)

        mu, log_sigma_sq = predictor(context, ctx_idx, tgt_idx)

        all_A.append(compute_uncertainty(log_sigma_sq).cpu().numpy())
        all_L.append(compute_latent_error(mu, targets).cpu().numpy())

    return np.concatenate(all_A), np.concatenate(all_L)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- overrides -------------------------------------------------------
    ckpt_path = args.ckpt_path or cfg["calibration"]["ckpt_path"]
    calib_dir = args.calib_dir or cfg["calibration"]["calib_dir"]
    alphas = [args.target_alpha] if args.target_alpha else cfg["calibration"]["alphas"]
    delta = args.delta or cfg["calibration"]["delta"]
    num_thresholds = cfg["calibration"].get("num_thresholds", 1000)

    # ---- encoder ---------------------------------------------------------
    enc_cfg = cfg["model"]["encoder"]
    encoder = FrozenEncoder(
        model_name=enc_cfg["name"],
        img_size=enc_cfg["img_size"],
        patch_size=enc_cfg["patch_size"],
        checkpoint_path=enc_cfg.get("checkpoint_path"),
    ).to(device)

    grid_h, grid_w = encoder.grid_size

    # ---- predictor -------------------------------------------------------
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
    print(f"Loaded predictor from {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # ---- mask generator --------------------------------------------------
    mask_cfg = cfg["masking"]
    mask_gen = MultiBlockMaskGenerator(
        grid_h=grid_h, grid_w=grid_w,
        target_ratio=mask_cfg["target_ratio"],
        num_blocks=mask_cfg["num_blocks"],
    )

    # ---- calibration data ------------------------------------------------
    dataset = build_dataset(
        dataset_name=cfg["data"]["dataset"],
        data_dir=calib_dir,
        img_size=cfg["data"]["img_size"],
        is_train=False,
    )
    loader = build_dataloader(
        dataset, batch_size=128, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        drop_last=False,
    )
    print(f"Calibration set: {len(dataset)} samples")

    # ---- collect scores --------------------------------------------------
    scores, errors = collect_scores(
        encoder, predictor, loader, mask_gen, device)

    from utils import spearman_correlation
    spear = spearman_correlation(scores, errors)
    print(f"Spearman(A, L) on cal set: {spear:.4f}")

    # ---- calibrate for each alpha ----------------------------------------
    save_dir = Path(args.save_threshold) if args.save_threshold else \
               Path(cfg["evaluation"].get("threshold_dir", "outputs/thresholds"))

    for alpha in alphas:
        calibrator = ConformalCalibrator(
            alpha=alpha, delta=delta, num_thresholds=num_thresholds)
        result = calibrator.calibrate(scores, errors)

        print(f"\nalpha={alpha:.2f}  →  tau_hat={result['tau_hat']:.6f}  "
              f"emp_risk={result['emp_risk']:.6f}  "
              f"coverage={result['coverage']:.2%}")

        out_path = save_dir if str(save_dir).endswith(".json") else \
                   save_dir / f"tau_alpha{alpha}.json"
        ConformalCalibrator.save(result, out_path)

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
