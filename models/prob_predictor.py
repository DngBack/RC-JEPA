"""
Probabilistic latent predictor for RC-JEPA.

A lightweight transformer that receives context patch tokens from the frozen
encoder plus learnable mask tokens at target positions, then outputs a
Gaussian distribution (mu, log_sigma_sq) per target patch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .frozen_encoder import Block  # reuse the same transformer block


class ProbabilisticPredictor(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 192,
        num_heads: int = 6,
        depth: int = 4,
        num_patches: int = 196,
        mlp_ratio: float = 4.,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.encoder_dim = encoder_dim

        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(predictor_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        self.mu_head = nn.Linear(predictor_dim, encoder_dim)
        self.log_sigma_sq_head = nn.Linear(predictor_dim, encoder_dim)

        self._init_output_heads()

    def _init_output_heads(self) -> None:
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_sigma_sq_head.weight)
        # initialise log_sigma_sq bias so initial sigma ~ 1
        nn.init.zeros_(self.log_sigma_sq_head.bias)

    @staticmethod
    def _gather_pos(pos_embed: torch.Tensor, indices: torch.Tensor,
                    dim: int) -> torch.Tensor:
        """Gather positional embeddings for given patch indices."""
        B = indices.shape[0]
        idx = indices.unsqueeze(-1).expand(-1, -1, dim)
        return torch.gather(pos_embed.expand(B, -1, -1), 1, idx)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        context_tokens  : (B, N_ctx, D_enc)  – from frozen encoder
        context_indices : (B, N_ctx)          – patch grid positions
        target_indices  : (B, N_tgt)          – patch grid positions

        Returns
        -------
        mu            : (B, N_tgt, D_enc)
        log_sigma_sq  : (B, N_tgt, D_enc)
        """
        B, N_ctx, _ = context_tokens.shape
        N_tgt = target_indices.shape[1]

        ctx = self.input_proj(context_tokens)  # (B, N_ctx, D_pred)
        tgt = self.mask_token.expand(B, N_tgt, -1)  # (B, N_tgt, D_pred)

        ctx = ctx + self._gather_pos(
            self.pos_embed, context_indices, self.predictor_dim)
        tgt = tgt + self._gather_pos(
            self.pos_embed, target_indices, self.predictor_dim)

        tokens = torch.cat([ctx, tgt], dim=1)  # (B, N_ctx+N_tgt, D_pred)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        tgt_out = tokens[:, N_ctx:, :]  # (B, N_tgt, D_pred)

        mu = self.mu_head(tgt_out)
        log_sigma_sq = self.log_sigma_sq_head(tgt_out)
        return mu, log_sigma_sq
