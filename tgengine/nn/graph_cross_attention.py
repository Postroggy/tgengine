"""Graph Cross-Attention (GCA) for fusing temporal and structural streams.

Foundation-model component 3/4. The dynamic-graph foundation model has two
information streams:

  - Temporal stream: event sequence processed by the Mamba backbone.
    Produces a hidden state h (B, L, d_model) — the "what happened when"
    memory.
  - Structural stream: per-position graph-structure features (co-occurrence
    frequency, neighbor topology, recency rank). Produces struct tokens
    (B, L, d_struct) — the "who connects to whom" signal.

Simple concatenation buries the structural signal under the temporal
memory. GCA instead lets the structural stream *modulate* the temporal
state via cross-attention:

    Q = proj_q(h_temporal)          # temporal drives the query
    K = proj_k(struct)
    V = proj_v(struct)
    h' = h_temporal + dropout(softmax(QK^T/√d) · V)   # residual add-back

So the temporal SSM remains the main memory and structural info is an
auxiliary modulation — avoiding the DyG-Mamba failure mode (pure sequence
loses topology) while preventing structure from dominating.

The block is general-purpose: any (query, key_value) pair of token
sequences works. When key_value is None it is a no-op (returns query
unchanged), so a model can wire GCA conditionally on whether structure is
available.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphCrossAttention(nn.Module):
    """Cross-attention from a temporal query stream to a structural kv stream.

    Args:
        d_model: dimension of the query (temporal) stream.
        d_kv: dimension of the key/value (structural) stream. Defaults to
            d_model when omitted.
        n_heads: attention heads.
        dropout: attention + residual dropout.
        ff_mult: feedforward expansion after attention (0 = no FFN).
    """

    def __init__(
        self,
        d_model: int,
        d_kv: Optional[int] = None,
        n_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.d_kv = d_kv or d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(self.d_kv, d_model, bias=False)
        self.v_proj = nn.Linear(self.d_kv, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(self.d_kv)

        self.ff = None
        if ff_mult > 0:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ff_mult, d_model),
            )
            self.norm_ff = nn.LayerNorm(d_model)
            self.ff_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Optional[Tensor],
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-attend query to key_value, residual-add to query.

        Args:
            query: (B, Lq, d_model) temporal stream.
            key_value: (B, Lkv, d_kv) structural stream. If None, returns
                query unchanged (GCA disabled — no structure available).
            kv_mask: (B, Lkv) bool, True = valid structural token. Invalid
                tokens are masked out of the attention softmax.

        Returns:
            (B, Lq, d_model) modulated query.
        """
        if key_value is None:
            return query

        B, Lq, _ = query.shape
        Lkv = key_value.shape[1]

        q = self.q_proj(self.norm_q(query))  # (B, Lq, d_model)
        k = self.k_proj(self.norm_kv(key_value))  # (B, Lkv, d_model)
        v = self.v_proj(self.norm_kv(key_value))  # (B, Lkv, d_model)

        # Multi-head reshape: (B, n_heads, L, head_dim)
        q = q.view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lkv, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (use SDPA for fused kernel + masking).
        # SDPA bool attn_mask: True = attend (keep), False = mask out.
        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask[:, None, None, :].to(torch.bool)  # (B,1,1,Lkv)
            # A query row whose entire kv set is invalid would softmax to NaN.
            # For those (B,) rows, fall back to attending everywhere (degenerate
            # but finite). Cheaper than per-row Python: just OR in a "full" mask
            # for fully-invalid batches.
            all_invalid = (~kv_mask).all(dim=1)  # (B,)
            if all_invalid.any():
                full = torch.ones_like(attn_mask)
                # keep valid rows' original mask, replace fully-invalid rows with full
                row_keep = (~all_invalid)[:, None, None, None]  # (B,1,1,1)
                attn_mask = torch.where(row_keep, attn_mask, full)

        drop_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_p,
        )  # (B, n_heads, Lq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)
        query = query + self.resid_drop(out)

        if self.ff is not None:
            query = query + self.ff_drop(self.ff(self.norm_ff(query)))
        return query
