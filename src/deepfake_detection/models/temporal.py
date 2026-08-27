"""Temporal transformer over per-frame spatial embeddings.

Given a clip's frame embeddings (T, D) from a frozen spatial detector,
this module projects them to a smaller dim, adds learnable positional
embeddings, runs a small transformer encoder, and pools (attention or
mean) to a single video-level fake logit. A padding mask handles
variable-length clips.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learned attention pooling over a masked sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class TemporalTransformer(nn.Module):
    """Video-level classifier over frame embeddings.

    Args:
        in_dim: spatial embedding dim (e.g. 1280 for EfficientNet-B0).
        dim: internal transformer width.
        depth: number of encoder layers.
        heads: attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: dropout prob.
        pool: 'attention' | 'mean' | 'max'.
        max_frames: upper bound on sequence length for positional embeds.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        dim: int = 256,
        depth: int = 3,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        pool: str = "attention",
        max_frames: int = 64,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pool = pool
        if pool == "attention":
            self.pooler = AttentionPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, D_in) frame embeddings; mask: (B, T) bool, True=valid.

        Returns logits (B, 1).
        """
        b, t, _ = x.shape
        x = self.proj(x)
        x = x + self.pos_embed[:, :t, :]

        # prepend CLS token
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, dim)

        key_padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, ~mask], dim=1)  # True = ignore

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.pool == "attention":
            pooled = self.pooler(x, key_padding_mask=key_padding_mask)
        elif self.pool == "mean":
            valid = (~key_padding_mask).unsqueeze(-1).float() if key_padding_mask is not None else torch.ones_like(x[:, :, :1])
            pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:  # max
            if key_padding_mask is not None:
                x = x.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
            pooled = x.max(dim=1).values

        return self.head(pooled)
