"""
Conformal Risk Control calibrator.

Given pre-computed uncertainty scores A(x) and latent errors L(x) on a
held-out calibration set, finds the optimal threshold tau_hat such that
the Hoeffding upper confidence bound on the selective risk stays below
the user-specified level alpha (with probability >= 1-delta).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ConformalCalibrator:

    def __init__(
        self,
        alpha: float = 0.10,
        delta: float = 0.10,
        num_thresholds: int = 1000,
    ):
        self.alpha = alpha
        self.delta = delta
        self.num_thresholds = num_thresholds

    # ------------------------------------------------------------------
    # Core calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        scores: np.ndarray,
        errors: np.ndarray,
    ) -> dict:
        """
        Parameters
        ----------
        scores : (n,) uncertainty scores  A(x_i)
        errors : (n,) latent errors        L(x_i)

        Returns
        -------
        dict with keys: tau_hat, alpha, delta, emp_risk, coverage,
                        num_cal, L_max
        """
        n = len(scores)
        assert len(errors) == n

        L_max = float(errors.max())
        taus = np.linspace(float(scores.min()), float(scores.max()),
                           self.num_thresholds)

        tau_hat = float(taus[0])
        best_risk = float("nan")
        best_coverage = 0.0
        found_valid = False

        for tau in taus:
            sel = scores <= tau
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue

            r_hat = float(errors[sel].mean())

            # Hoeffding UCB with Bonferroni correction over threshold grid
            correction = L_max * np.sqrt(
                np.log(self.num_thresholds / self.delta) / (2 * n_sel))
            ucb = r_hat + correction

            if ucb <= self.alpha:
                tau_hat = float(tau)
                best_risk = r_hat
                best_coverage = n_sel / n
                found_valid = True

        if not found_valid:
            # Fall back to the most conservative threshold (smallest tau
            # that selects at least one sample).
            for tau in taus:
                sel = scores <= tau
                if sel.sum() > 0:
                    tau_hat = float(tau)
                    best_risk = float(errors[sel].mean())
                    best_coverage = int(sel.sum()) / n
                    break

        return {
            "tau_hat": tau_hat,
            "alpha": self.alpha,
            "delta": self.delta,
            "emp_risk": best_risk,
            "coverage": best_coverage,
            "num_cal": n,
            "L_max": L_max,
            "valid_bound": found_valid,
        }

    # ------------------------------------------------------------------
    # Risk-coverage sweep (for plotting)
    # ------------------------------------------------------------------

    def risk_coverage_curve(
        self,
        scores: np.ndarray,
        errors: np.ndarray,
        num_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays (coverages, risks) swept over thresholds.
        """
        taus = np.linspace(float(scores.min()), float(scores.max()), num_points)
        coverages, risks = [], []
        for tau in taus:
            sel = scores <= tau
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            coverages.append(n_sel / len(scores))
            risks.append(float(errors[sel].mean()))
        return np.array(coverages), np.array(risks)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save(result: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Calibrator] saved → {path}")

    @staticmethod
    def load(path: str | Path) -> dict:
        with open(path) as f:
            return json.load(f)
