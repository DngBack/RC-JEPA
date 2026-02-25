#!/usr/bin/env python3
"""
Phase 1 — Train the probabilistic predictor on top of a frozen JEPA encoder.

Usage
-----
python scripts/train_predictor.py --config configs/train_rc_jepa.yaml

Only the lightweight predictor receives gradients; the encoder stays frozen.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
from models.losses import RCJEPALoss
from utils import (
    compute_uncertainty,
    compute_latent_error,
    cosine_schedule,
    set_seed,
    spearman_correlation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_rc_jepa.yaml")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--lambda_rank", type=float, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def encode_batch(
    encoder: FrozenEncoder,
    images: torch.Tensor,
    ctx_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen encoder to get context and target representations."""
    all_reps = encoder.forward_full(images)       # (B, N, D)
    B = images.shape[0]
    D = all_reps.shape[-1]

    tgt_expand = tgt_idx.unsqueeze(-1).expand(-1, -1, D)
    targets = torch.gather(all_reps, 1, tgt_expand)  # (B, N_tgt, D)

    context = encoder.forward_context(images, ctx_idx)  # (B, N_ctx, D)
    return context, targets


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.lambda_rank is not None:
        cfg["training"]["lambda_rank"] = args.lambda_rank

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- wandb -----------------------------------------------------------
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if args.wandb_project:
        use_wandb = True
        cfg.setdefault("wandb", {})["project"] = args.wandb_project

    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            config=cfg,
        )

    # ---- data ------------------------------------------------------------
    dataset = build_dataset(
        dataset_name=cfg["data"]["dataset"],
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        is_train=True,
    )
    loader = build_dataloader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"Dataset: {cfg['data']['dataset']}  |  {len(dataset)} samples")

    # ---- models ----------------------------------------------------------
    enc_cfg = cfg["model"]["encoder"]
    encoder = FrozenEncoder(
        model_name=enc_cfg["name"],
        img_size=enc_cfg["img_size"],
        patch_size=enc_cfg["patch_size"],
        checkpoint_path=enc_cfg.get("checkpoint_path"),
    ).to(device)

    grid_h, grid_w = encoder.grid_size
    num_patches = encoder.num_patches
    embed_dim = encoder.embed_dim

    pred_cfg = cfg["model"]["predictor"]
    predictor = ProbabilisticPredictor(
        encoder_dim=embed_dim,
        predictor_dim=pred_cfg["predictor_dim"],
        num_heads=pred_cfg["num_heads"],
        depth=pred_cfg["depth"],
        num_patches=num_patches,
    ).to(device)
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # ---- mask generator --------------------------------------------------
    mask_cfg = cfg["masking"]
    mask_gen = MultiBlockMaskGenerator(
        grid_h=grid_h,
        grid_w=grid_w,
        target_ratio=mask_cfg["target_ratio"],
        num_blocks=mask_cfg["num_blocks"],
        min_aspect=mask_cfg.get("min_block_aspect", 0.75),
        max_aspect=mask_cfg.get("max_block_aspect", 1.5),
    )

    # ---- optimiser & loss ------------------------------------------------
    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    criterion = RCJEPALoss(
        lambda_rank=t_cfg["lambda_rank"],
        gamma=t_cfg["gamma"],
    )

    save_dir = Path(t_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ---------------------------------------------------
    best_spearman = -1.0
    for epoch in range(1, t_cfg["epochs"] + 1):
        lr = cosine_schedule(
            optimizer, epoch - 1, t_cfg["epochs"],
            warmup_epochs=t_cfg["warmup_epochs"],
            base_lr=t_cfg["lr"],
        )
        predictor.train()

        epoch_nll, epoch_rank, epoch_total = 0.0, 0.0, 0.0
        all_A, all_L = [], []
        t0 = time.time()

        for step, (images, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}",
                                                 leave=False)):
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            ctx_idx, tgt_idx = mask_gen(B)
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            context, targets = encode_batch(encoder, images, ctx_idx, tgt_idx)

            mu, log_sigma_sq = predictor(context, ctx_idx, tgt_idx)
            total_loss, nll_loss, rank_loss = criterion(mu, log_sigma_sq, targets)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            epoch_nll += nll_loss.item()
            epoch_rank += rank_loss.item()
            epoch_total += total_loss.item()

            with torch.no_grad():
                all_A.append(compute_uncertainty(log_sigma_sq).cpu().numpy())
                all_L.append(compute_latent_error(mu, targets).cpu().numpy())

            if (step + 1) % t_cfg["log_interval"] == 0:
                tqdm.write(
                    f"  step {step+1:>5d}  loss={total_loss.item():.4f}  "
                    f"nll={nll_loss.item():.4f}  rank={rank_loss.item():.4f}"
                )

        n_steps = len(loader)
        A_all = np.concatenate(all_A)
        L_all = np.concatenate(all_L)
        spear = spearman_correlation(A_all, L_all)
        dt = time.time() - t0

        log_dict = {
            "epoch": epoch,
            "lr": lr,
            "loss/total": epoch_total / n_steps,
            "loss/nll": epoch_nll / n_steps,
            "loss/rank": epoch_rank / n_steps,
            "metrics/latent_spearman": spear,
        }
        print(
            f"Epoch {epoch:>3d}/{t_cfg['epochs']}  "
            f"loss={epoch_total/n_steps:.4f}  "
            f"nll={epoch_nll/n_steps:.4f}  "
            f"rank={epoch_rank/n_steps:.4f}  "
            f"spearman={spear:.4f}  "
            f"lr={lr:.2e}  "
            f"time={dt:.1f}s"
        )

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        # -- checkpointing --------------------------------------------------
        if spear > best_spearman:
            best_spearman = spear
            torch.save(
                {"predictor": predictor.state_dict(),
                 "epoch": epoch,
                 "spearman": spear,
                 "config": cfg},
                save_dir / "best_predictor.pth",
            )

        if epoch % t_cfg["save_interval"] == 0:
            torch.save(
                {"predictor": predictor.state_dict(),
                 "epoch": epoch,
                 "config": cfg},
                save_dir / f"predictor_epoch{epoch}.pth",
            )

    print(f"\nTraining complete.  Best Spearman = {best_spearman:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
