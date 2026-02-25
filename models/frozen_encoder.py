"""
Frozen Vision Transformer encoder for RC-JEPA.

Provides a self-contained ViT implementation that can load weights from:
  1. I-JEPA checkpoints  (target_encoder / encoder key)
  2. Generic ViT checkpoints
  3. Random initialisation (for quick prototyping)

All parameters are frozen; the module is permanently in eval() mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ViT building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Minimal ViT (no CLS token) compatible with I-JEPA checkpoints."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)
        indices : (B, N_sel) optional – if given, only those patch positions
                  are retained before the transformer blocks.

        Returns
        -------
        (B, N | N_sel, D)
        """
        tokens = self.patch_embed(x)  # (B, N, D)

        if indices is not None:
            B, N_sel = indices.shape
            idx = indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            tokens = torch.gather(tokens, 1, idx)
            pos = self.pos_embed.expand(B, -1, -1)
            pos = torch.gather(pos, 1, idx)
            tokens = tokens + pos
        else:
            tokens = tokens + self.pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)

        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Predefined ViT configurations
# ---------------------------------------------------------------------------

VIT_CONFIGS: dict[str, dict] = {
    "vit_tiny":  dict(embed_dim=192,  depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384,  depth=12, num_heads=6),
    "vit_base":  dict(embed_dim=768,  depth=12, num_heads=12),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
    "vit_huge":  dict(embed_dim=1280, depth=32, num_heads=16),
}


# ---------------------------------------------------------------------------
# Frozen wrapper
# ---------------------------------------------------------------------------

class FrozenEncoder(nn.Module):
    """
    Wraps a VisionTransformer with all parameters frozen.

    Two forward modes
    -----------------
    forward_full(x)                 → target representations (full image)
    forward_context(x, ctx_indices) → context representations (subset)
    """

    def __init__(
        self,
        model_name: str = "vit_small",
        img_size: int = 224,
        patch_size: int = 16,
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        if model_name not in VIT_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from {list(VIT_CONFIGS)}")

        cfg = VIT_CONFIGS[model_name]
        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, **cfg)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self._freeze()

    # -- properties ----------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self.vit.embed_dim

    @property
    def num_patches(self) -> int:
        return self.vit.num_patches

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.vit.grid_size

    # -- checkpoint loading --------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        for key in ("target_encoder", "encoder", "model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

        cleaned = {}
        for k, v in ckpt.items():
            k = k.replace("module.", "").replace("backbone.", "")
            cleaned[k] = v

        msg = self.vit.load_state_dict(cleaned, strict=False)
        print(f"[FrozenEncoder] loaded checkpoint – {msg}")

    # -- freeze --------------------------------------------------------------

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):  # noqa: D102
        return super().train(False)

    # -- forward -------------------------------------------------------------

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Full-image pass → (B, N, D). Used for target representations."""
        return self.vit(x)

    def forward_context(
        self, x: torch.Tensor, context_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Context-only pass → (B, N_ctx, D)."""
        return self.vit(x, indices=context_indices)

    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_indices is not None:
            return self.forward_context(x, context_indices)
        return self.forward_full(x)
