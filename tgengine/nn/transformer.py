"""Reusable transformer building blocks."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TransformerBlock(nn.Module):
    """Single pre-LN transformer layer (batch_first).

    Unlike TransformerSeqEncoder (which wraps nn.TransformerEncoder + mean pool),
    this is a standalone layer that returns all token outputs — useful when you
    need full sequence output (e.g., joint attention across concatenated sequences).

    Args:
        d_model: hidden dimension.
        n_heads: number of attention heads.
        dropout: dropout rate.
        ff_mult: feedforward hidden dim multiplier.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, d_model) input sequence.
        Returns:
            (B, S, d_model) output sequence.
        """
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
