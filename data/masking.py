"""
Multi-block masking strategy for JEPA-style training.

Target patches are selected as the union of K random rectangular blocks
on the patch grid. Context patches are the complement. The total number
of target/context patches is kept fixed across samples for batching.
"""

import random

import torch
import numpy as np


class MultiBlockMaskGenerator:

    def __init__(
        self,
        grid_h: int = 14,
        grid_w: int = 14,
        target_ratio: float = 0.6,
        num_blocks: int = 4,
        min_aspect: float = 0.75,
        max_aspect: float = 1.5,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.num_targets = int(self.num_patches * target_ratio)
        self.num_context = self.num_patches - self.num_targets
        self.num_blocks = num_blocks
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def _sample_block(self) -> set[int]:
        """Sample one random rectangular block, returning patch indices."""
        area_per_block = max(1, self.num_targets // self.num_blocks)
        aspect = random.uniform(self.min_aspect, self.max_aspect)

        block_h = max(1, min(int(round(np.sqrt(area_per_block * aspect))), self.grid_h))
        block_w = max(1, min(int(round(np.sqrt(area_per_block / aspect))), self.grid_w))

        top = random.randint(0, self.grid_h - block_h)
        left = random.randint(0, self.grid_w - block_w)

        indices: set[int] = set()
        for i in range(top, top + block_h):
            for j in range(left, left + block_w):
                indices.add(i * self.grid_w + j)
        return indices

    def _generate_single(self) -> tuple[list[int], list[int]]:
        target_set: set[int] = set()
        for _ in range(self.num_blocks):
            target_set |= self._sample_block()

        target_list = sorted(target_set)
        all_patches = set(range(self.num_patches))

        if len(target_list) > self.num_targets:
            target_list = sorted(random.sample(target_list, self.num_targets))
        elif len(target_list) < self.num_targets:
            remaining = sorted(all_patches - set(target_list))
            extra = random.sample(remaining, self.num_targets - len(target_list))
            target_list = sorted(target_list + extra)

        context_list = sorted(all_patches - set(target_list))
        return context_list, target_list

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context_indices : (B, N_ctx) long tensor of sorted context patch ids
        target_indices  : (B, N_tgt) long tensor of sorted target patch ids
        """
        ctx_all, tgt_all = [], []
        for _ in range(batch_size):
            c, t = self._generate_single()
            ctx_all.append(c)
            tgt_all.append(t)

        return (
            torch.tensor(ctx_all, dtype=torch.long),
            torch.tensor(tgt_all, dtype=torch.long),
        )
