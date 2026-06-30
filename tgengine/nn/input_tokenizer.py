"""Foundation model input tokenizer.

Converts raw K-neighbor data (NeighborData + query time) into d_model-dim
tokens for the Mamba encoder. All features are domain-agnostic — no node IDs.

Feature hierarchy:
  - Event-level (per neighbor): edge_feat, Δt time encoding, pair features
  - Node-level (shared across K): recent_degree, activity_rate, Δt stats,
    recency — broadcast to each token position

Token assembly (additive, like transformer embeddings):
  token_i = edge_proj(e_i) + time_proj(time_enc(Δt_i))
            + pair_proj(pair_feat_i) + ctx_proj(node_ctx)

Structural token for GCA: struct_proj(node_ctx) → (B, 1, d_model)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from tgengine.core.batch import NeighborData


class TrainableSinusoidalTimeEncoding(nn.Module):
    """Sinusoidal time encoding with trainable frequencies.

    Maps scalar Δt to a d_time-dim vector using sin/cos at log-spaced
    trainable frequencies. Initialized to cover Δt ∈ [0, 1] (normalized
    timestamps from MixedDataset) across timescales from 1 to 10^4.

    TGPM (ICML 2026) uses the same approach: T_enc(Δt) = √(1/d_t)[cos(ωΔt), sin(ωΔt), ...]
    """

    def __init__(self, d_time: int, max_log_freq: float = 4.0):
        super().__init__()
        assert d_time % 2 == 0, "d_time must be even (sin/cos pairs)"
        self.d_time = d_time
        d_half = d_time // 2
        # Log-spaced frequencies: 2π × 10^0 to 2π × 10^max_log_freq
        log_freqs = torch.linspace(0, max_log_freq, d_half)
        freqs = 2 * math.pi * (10 ** log_freqs)
        self.freqs = nn.Parameter(freqs)
        self.scale = 1.0 / math.sqrt(d_time)

    def forward(self, dt: Tensor) -> Tensor:
        """
        Args:
            dt: (...) arbitrary shape of time deltas
        Returns:
            (..., d_time) time encoding
        """
        angles = dt.unsqueeze(-1) * self.freqs  # (..., d_half)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return enc * self.scale


def compute_node_context(
    timestamps: Tensor,
    mask: Tensor,
    query_time: Tensor,
) -> Tensor:
    """Compute node-level context features from K-neighbor buffer.

    All features are domain-agnostic scalars describing the temporal dynamics
    of a node's recent interaction history.

    Args:
        timestamps: (B, K) neighbor interaction timestamps (most recent first)
        mask: (B, K) bool, True for valid positions
        query_time: (B,) current query time

    Returns:
        (B, d_ctx) context features: [recent_degree, activity_rate,
        dt_mean, dt_var, recency_last]
    """
    B, K = timestamps.shape
    device = timestamps.device
    # Cast to float32: TemporalGraph stores timestamps as float64
    timestamps = timestamps.float()
    mask_f = mask.float()

    # 1. Recent degree: number of valid neighbors
    recent_degree = mask_f.sum(dim=1)  # (B,)

    # 2. Activity rate: interactions per unit time
    # Time span from oldest to most recent valid timestamp
    ts_masked_inf = timestamps.masked_fill(~mask, float("inf"))
    ts_masked_neg = timestamps.masked_fill(~mask, float("-inf"))
    ts_oldest = ts_masked_inf.min(dim=1).values  # (B,)
    ts_newest = ts_masked_neg.max(dim=1).values  # (B,)
    time_span = (ts_newest - ts_oldest).clamp(min=1e-8)
    # Handle all-padding: time_span=1e-8, recent_degree=0 → activity_rate=0
    activity_rate = recent_degree / time_span  # (B,)

    # 3-4. Inter-event gap statistics (mean, variance)
    # Gaps between consecutive valid timestamps (buffer is most-recent-first)
    # gap[i] = timestamps[i] - timestamps[i+1] ≥ 0
    if K > 1:
        gaps = timestamps[:, :-1] - timestamps[:, 1:]  # (B, K-1)
        gap_mask = mask[:, :-1] & mask[:, 1:]  # both endpoints valid
        gap_mask_f = gap_mask.float()
        gap_counts = gap_mask_f.sum(dim=1).clamp(min=1)  # (B,)
        valid_gaps = gaps * gap_mask_f  # zero out invalid
        dt_mean = valid_gaps.sum(dim=1) / gap_counts  # (B,)
        # Variance: E[(gap - mean)^2]
        dt_sq = (valid_gaps ** 2).sum(dim=1) / gap_counts
        dt_var = (dt_sq - dt_mean ** 2).clamp(min=0)  # (B,)
    else:
        dt_mean = torch.zeros(B, device=device)
        dt_var = torch.zeros(B, device=device)

    # 5. Recency: time since most recent interaction
    # For all-padding nodes, ts_newest = -inf → recency = inf → clamp
    recency_last = (query_time - ts_newest).clamp(min=0, max=1.0)  # (B,)

    # Stack: (B, 5)
    ctx = torch.stack([
        recent_degree,
        activity_rate,
        dt_mean,
        dt_var,
        recency_last,
    ], dim=1)

    # Zero out context for all-padding nodes
    has_neighbors = (recent_degree > 0).float().unsqueeze(1)
    ctx = ctx * has_neighbors

    return ctx


def compute_pair_features(
    neighbor_ids: Tensor,
    timestamps: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute per-position pair features from K-neighbor buffer.

    For each position i, compute features describing the relationship
    between the query node and this specific neighbor.

    Args:
        neighbor_ids: (B, K) neighbor node IDs
        timestamps: (B, K) interaction timestamps (most recent first)
        mask: (B, K) bool

    Returns:
        (B, K, d_pair) pair features: [pair_count, pair_recency]
        pair_count: how many times this neighbor appears in buffer
        pair_recency: time gap between first occurrence and this occurrence
            (0 if this IS the first/most-recent occurrence)
    """
    B, K = neighbor_ids.shape
    device = neighbor_ids.device
    # Cast to float32: TemporalGraph stores timestamps as float64
    timestamps = timestamps.float()
    mask_f = mask.float()

    # Pair count: for each position i, count positions j where
    # neighbor_ids[j] == neighbor_ids[i] and mask[j] is True.
    # Clamp padding IDs to 0 so they don't spuriously match each other;
    # the mask product below zeroes out padding positions anyway.
    safe_ids = neighbor_ids.clamp(min=0)
    # eq[b, j, i] = True if neighbor_ids[b,j] == neighbor_ids[b,i]
    eq = safe_ids.unsqueeze(1) == safe_ids.unsqueeze(2)  # (B, K_j, K_i)
    eq = eq & mask.unsqueeze(2)  # (B, K_j, K_i) — only count valid j
    pair_count = eq.sum(dim=1).float()  # (B, K_i) — count per position i
    pair_count = pair_count * mask_f  # zero out padding positions

    # Pair recency: time since the most recent (first) occurrence of the same
    # neighbor. For position i, find the smallest j where eq[:, j, i] is True.
    # Since buffer is most-recent-first, smallest j = most recent occurrence.
    positions = torch.arange(K, device=device).view(1, K, 1).expand(B, K, K)
    large_val = K + 1
    masked_pos = torch.where(eq, positions, torch.full_like(positions, large_val))
    first_j = masked_pos.min(dim=1).values  # (B, K_i) — index of first occurrence
    # Clamp to valid range in case all positions were masked (large_val > K-1)
    first_j = first_j.clamp(max=K - 1)
    first_ts = timestamps.gather(1, first_j)  # (B, K_i) — timestamp at first occurrence
    # pair_recency = first_ts - current_ts (≥ 0 since first is most recent)
    pair_recency = (first_ts - timestamps).clamp(min=0) * mask_f  # (B, K)

    # Stack: (B, K, 2)
    pair_feat = torch.stack([pair_count, pair_recency], dim=-1)
    return pair_feat


class InputTokenizer(nn.Module):
    """Convert K-neighbor raw data to d_model-dim tokens.

    Assembles each token additively from four feature groups:
      token_i = edge_proj(e_i) + time_proj(time_enc(Δt_i))
                + pair_proj(pair_feat_i) + ctx_proj(node_ctx)

    Also produces a structural token for GCA from node context features.

    All features are domain-agnostic (no node IDs used).

    Args:
        d_edge: input edge feature dimension (e.g. 172 for DyGLib LIAR)
        d_model: output token dimension
        d_time: time encoding dimension (must be even, default 64)
        d_ctx: number of node context features (default 5)
        d_pair: number of pair features (default 2)
    """

    def __init__(
        self,
        d_edge: int,
        d_model: int,
        d_time: int = 64,
        d_ctx: int = 5,
        d_pair: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ctx = d_ctx

        # Time encoding (trainable sinusoidal)
        self.time_enc = TrainableSinusoidalTimeEncoding(d_time)

        # Feature projections: each group → d_model
        self.edge_proj = nn.Linear(d_edge, d_model, bias=False)
        self.time_proj = nn.Linear(d_time, d_model, bias=False)
        self.pair_proj = nn.Linear(d_pair, d_model, bias=False)
        self.ctx_proj = nn.Linear(d_ctx, d_model, bias=False)

        # Structural token for GCA (from node context)
        self.struct_proj = nn.Linear(d_ctx, d_model, bias=False)

        # Learnable padding embedding
        self.padding_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.padding_emb, std=0.02)

        # Post-assembly layer norm
        self.token_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        nbr: NeighborData,
        query_time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Tokenize a node's K-neighbor buffer.

        Args:
            nbr: NeighborData with (neighbor_ids, timestamps, edge_feats, mask)
                All tensors on same device.
            query_time: (B,) query timestamps

        Returns:
            tokens: (B, K, d_model) input tokens for the encoder
            struct_token: (B, 1, d_model) structural token for GCA
        """
        B, K = nbr.neighbor_ids.shape
        mask = nbr.mask  # (B, K)

        # --- Node-level context features (shared across K positions) ---
        node_ctx = compute_node_context(nbr.timestamps, mask, query_time)  # (B, d_ctx)

        # --- Event-level features (per neighbor position) ---

        # 1. Edge feature projection: (B, K, d_edge) → (B, K, d_model)
        edge_tokens = self.edge_proj(nbr.edge_feats)

        # 2. Time encoding + projection: Δt → (B, K, d_time) → (B, K, d_model)
        # Cast to float32: TemporalGraph stores timestamps as float64, but
        # all linear layers and time encoding use float32.
        dt = (query_time.unsqueeze(1) - nbr.timestamps).clamp(min=0).float()  # (B, K)
        time_tokens = self.time_proj(self.time_enc(dt))

        # 3. Pair features: (B, K, d_pair) → (B, K, d_model)
        pair_feat = compute_pair_features(
            nbr.neighbor_ids, nbr.timestamps, mask,
        )  # (B, K, d_pair)
        pair_tokens = self.pair_proj(pair_feat)

        # 4. Node context broadcast: (B, d_ctx) → (B, 1, d_model) → (B, K, d_model)
        ctx_tokens = self.ctx_proj(node_ctx).unsqueeze(1)  # (B, 1, d_model)

        # --- Assemble tokens (additive) ---
        tokens = edge_tokens + time_tokens + pair_tokens + ctx_tokens  # (B, K, d_model)

        # Apply padding mask: replace padding positions with learned embedding
        mask_expanded = mask.unsqueeze(-1)  # (B, K, 1)
        tokens = torch.where(
            mask_expanded,
            tokens,
            self.padding_emb.view(1, 1, self.d_model).expand(B, K, self.d_model),
        )

        # Post-assembly normalization
        tokens = self.token_norm(tokens)

        # --- Structural token for GCA ---
        struct_token = self.struct_proj(node_ctx).unsqueeze(1)  # (B, 1, d_model)

        return tokens, struct_token
