"""
RC-JEPA loss functions.

- LatentNLLLoss      : Gaussian negative log-likelihood in latent space
- LatentRankingLoss  : Pairwise margin ranking on (uncertainty, error)
- RCJEPALoss         : Weighted combination of the above
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentNLLLoss(nn.Module):
    """
    Per-dimension Gaussian NLL averaged over targets and batch.

    L = (1/K) sum_k (1/d) sum_j [ (z_kj - mu_kj)^2 / (2 sigma_kj^2)
                                   + 0.5 log sigma_kj^2 ]
    """

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma_sq: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        var = torch.exp(log_sigma_sq).clamp(min=1e-6)
        nll = 0.5 * ((targets - mu) ** 2 / var + log_sigma_sq)
        return nll.mean()


class LatentRankingLoss(nn.Module):
    """
    Pairwise margin ranking that encourages monotonic correlation between
    the learned uncertainty score A(x) and the true latent error L(x).

    L_rank = E_{i,j} [ max(0, gamma - (A_i - A_j) * sign(L_i - L_j)) ]
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    @staticmethod
    def uncertainty_score(log_sigma_sq: torch.Tensor) -> torch.Tensor:
        """A(x) = (1/K) sum_k ||log sigma_k^2||_1  → scalar per sample."""
        return log_sigma_sq.abs().sum(dim=-1).mean(dim=-1)  # (B,)

    @staticmethod
    def latent_error(mu: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """L(x) = (1/K) sum_k ||z_k - mu_k||_2^2  → scalar per sample."""
        return ((targets - mu) ** 2).sum(dim=-1).mean(dim=-1)  # (B,)

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma_sq: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        A = self.uncertainty_score(log_sigma_sq)   # (B,)
        L = self.latent_error(mu, targets)          # (B,)

        # all-pairs differences
        diff_A = A.unsqueeze(1) - A.unsqueeze(0)    # (B, B)
        sign_L = torch.sign(L.unsqueeze(1) - L.unsqueeze(0))  # (B, B)

        loss = torch.clamp(self.gamma - diff_A * sign_L, min=0.0)

        # upper-triangle only (skip diagonal + duplicates)
        B = A.shape[0]
        mask = torch.triu(torch.ones(B, B, device=A.device, dtype=torch.bool),
                          diagonal=1)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=A.device)

        return loss[mask].mean()


class RCJEPALoss(nn.Module):
    """Combined NLL + lambda * Ranking loss."""

    def __init__(self, lambda_rank: float = 0.5, gamma: float = 1.0):
        super().__init__()
        self.nll = LatentNLLLoss()
        self.rank = LatentRankingLoss(gamma=gamma)
        self.lambda_rank = lambda_rank

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma_sq: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_loss, nll_loss, rank_loss)."""
        l_nll = self.nll(mu, log_sigma_sq, targets)
        l_rank = self.rank(mu, log_sigma_sq, targets)
        total = l_nll + self.lambda_rank * l_rank
        return total, l_nll, l_rank
