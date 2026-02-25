"""
Inference-time risk selector.

Given a calibrated threshold tau_hat, decides whether to accept or abstain
on each sample based on its uncertainty score A(x).
"""

from __future__ import annotations

import torch
import numpy as np

from .calibrator import ConformalCalibrator


class RiskSelector:
    """
    s(x) = 1{A(x) <= tau_hat}

    Accepted samples have provably bounded expected latent error <= alpha.
    """

    def __init__(self, tau_hat: float, alpha: float | None = None):
        self.tau_hat = tau_hat
        self.alpha = alpha

    @classmethod
    def from_file(cls, path: str) -> "RiskSelector":
        result = ConformalCalibrator.load(path)
        return cls(
            tau_hat=result["tau_hat"],
            alpha=result.get("alpha"),
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_numpy(self, scores: np.ndarray) -> np.ndarray:
        """Boolean mask: True = accept, False = abstain."""
        return scores <= self.tau_hat

    def select_torch(self, scores: torch.Tensor) -> torch.Tensor:
        return scores <= self.tau_hat

    def __call__(
        self, scores: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(scores, torch.Tensor):
            return self.select_torch(scores)
        return self.select_numpy(scores)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scores: np.ndarray,
        errors: np.ndarray,
    ) -> dict:
        """Compute selective risk and coverage on a test set."""
        sel = self.select_numpy(scores)
        n_sel = int(sel.sum())
        n = len(scores)

        if n_sel == 0:
            return {"risk": float("nan"), "coverage": 0.0, "n_selected": 0, "n_total": n}

        return {
            "risk": float(errors[sel].mean()),
            "coverage": n_sel / n,
            "n_selected": n_sel,
            "n_total": n,
        }
