"""
Shared utilities for RC-JEPA.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from scipy.stats import spearmanr


# ------------------------------------------------------------------
# Score / error computation (work on raw model outputs)
# ------------------------------------------------------------------

def compute_uncertainty(log_sigma_sq: torch.Tensor) -> torch.Tensor:
    """
    A(x) = (1/K) sum_k ||log sigma_k^2||_1   (scalar per sample).

    Parameters
    ----------
    log_sigma_sq : (B, N_tgt, D)

    Returns
    -------
    (B,)
    """
    return log_sigma_sq.abs().sum(dim=-1).mean(dim=-1)


def compute_latent_error(
    mu: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """
    L(x) = (1/K) sum_k ||z_k - mu_k||_2^2   (scalar per sample).

    Parameters
    ----------
    mu      : (B, N_tgt, D)
    targets : (B, N_tgt, D)

    Returns
    -------
    (B,)
    """
    return ((targets - mu) ** 2).sum(dim=-1).mean(dim=-1)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays."""
    if len(a) < 3:
        return 0.0
    rho, _ = spearmanr(a, b)
    return float(rho)


def auroc_from_scores(errors: np.ndarray, scores: np.ndarray,
                      threshold: float | None = None) -> float:
    """
    Treat samples with L(x) above the median (or given threshold) as
    'high-error positives' and compute AUROC of the uncertainty score.
    """
    from sklearn.metrics import roc_auc_score
    if threshold is None:
        threshold = float(np.median(errors))
    labels = (errors > threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels, scores))


# ------------------------------------------------------------------
# Learning-rate schedule (cosine with linear warm-up)
# ------------------------------------------------------------------

def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 10,
    base_lr: float = 1.5e-4,
    min_lr: float = 1e-6,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
