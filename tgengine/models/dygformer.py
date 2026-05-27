"""DyGFormer: Multi-channel patch-based temporal graph model.

Faithful adaptation of the DyGFormer paper (https://arxiv.org/abs/2303.13047)
to TGEngine's PreparedBatch interface.

Architecture highlights:
  - 3 channels per token: edge_feat, time_enc, co-occurrence
  - patch_size groups P consecutive tokens into one patch vector
  - Joint src+dst transformer: src and dst patches attend to EACH OTHER
  - Two forward passes per batch: (src, dst) and (src, neg)
  - supports_independent_encode=False because co-occurrence is pair-dependent
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.core.batch import NeighborData
from tgengine.nn import FixedCosineTimeEncoder

PADDING_ID = -1


class _CoOccurrenceEncoder(nn.Module):
    """Per-neighbor co-occurrence encoder from DyGFormer Section 4.1.

    For each neighbor token of node a paired with node b, computes a 2-D feature:
      [self_count, cross_count]
    where self_count = times this neighbor appears in a's list,
          cross_count = times this neighbor appears in b's list.

    Both features are independently projected then summed.
    """

    def __init__(self, d_out: int):
        super().__init__()
        # Applied per-scalar → (B, K, 2, d_out) then summed over dim 2
        self.proj = nn.Sequential(
            nn.Linear(1, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, a_ids: Tensor, b_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            a_ids: (B, K) neighbor IDs for side A (int32).
            b_ids: (B, K) neighbor IDs for side B (int32).

        Returns:
            a_feat, b_feat — each (B, K, d_out) per-neighbor co-occurrence features.
        """
        a_pad = a_ids == PADDING_ID   # (B, K)
        b_pad = b_ids == PADDING_ID

        # self-co-occurrence: mask padding so -1 doesn't spuriously match -1
        a_self = (a_ids.unsqueeze(1) == a_ids.unsqueeze(2)).float()   # (B, K, K)
        b_self = (b_ids.unsqueeze(1) == b_ids.unsqueeze(2)).float()
        cross = (a_ids.unsqueeze(1) == b_ids.unsqueeze(2)).float()    # (B, K, K)

        a_self = a_self.masked_fill(a_pad.unsqueeze(2) | a_pad.unsqueeze(1), 0.0)
        b_self = b_self.masked_fill(b_pad.unsqueeze(2) | b_pad.unsqueeze(1), 0.0)
        cross = cross.masked_fill(a_pad.unsqueeze(2) | b_pad.unsqueeze(1), 0.0)

        # Per-neighbor 2D features matching DyGLib semantics:
        #   a_freq[b,j] = [count(a[b,j] in a[b]),  count(a[b,j] in b[b])]
        #   b_freq[b,j] = [count(b[b,j] in a[b]),  count(b[b,j] in b[b])]
        # cross[b,i,j] = (a[b,i]==b[b,j])
        #   cross.sum(2)[b,i] = count of a[b,i] in b[b]  → used for a_freq second column
        #   cross.sum(1)[b,j] = count of b[b,j] in a[b]  → used for b_freq first column
        a_freq = torch.stack([a_self.sum(1), cross.sum(2)], dim=2)    # (B, K, 2)
        b_freq = torch.stack([cross.sum(1), b_self.sum(1)], dim=2)    # (B, K, 2)

        # Project each scalar independently, sum; then zero padding positions (bias leak)
        a_feat = self.proj(a_freq.unsqueeze(-1)).sum(dim=2)            # (B, K, d_out)
        b_feat = self.proj(b_freq.unsqueeze(-1)).sum(dim=2)
        a_feat = a_feat.masked_fill(a_pad.unsqueeze(-1), 0.0)
        b_feat = b_feat.masked_fill(b_pad.unsqueeze(-1), 0.0)
        return a_feat, b_feat


class _TransformerLayer(nn.Module):
    """Single DyGFormer transformer layer (pre-LN, batch_first)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-LN attention
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(h)
        # Pre-LN FFN
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DyGFormer(TemporalModel):
    """DyGFormer model for link prediction in continuous-time dynamic graphs.

    Uses multi-channel patch encoding + joint (src, dst) transformer.
    Negative pairs are scored in a separate forward pass.

    Args:
        d_model: output embedding dimension.
        d_edge: edge feature dimension.
        d_time: time encoding dimension (Time2Vec output dim).
        d_channel: per-channel projection dimension.
        K: max neighbors to sample per node (= k in NeighborSpec).
        patch_size: group P consecutive tokens per patch (must divide K).
        n_layers: number of transformer layers.
        n_heads: number of attention heads.
        dropout: dropout rate.
        node_feat: optional static node feature matrix (num_nodes, d_node).
                   When provided, adds a 4th channel matching DyGLib's architecture.
                   Must be on CPU at init; moved to device automatically via register_buffer.
    """

    def __init__(
        self,
        d_model: int = 172,
        d_edge: int = 172,
        d_time: int = 100,
        d_channel: int = 50,
        K: int = 32,
        patch_size: int = 1,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
        node_feat: Optional[Tensor] = None,
    ):
        super().__init__()
        if K % patch_size != 0:
            raise ValueError(f"K={K} must be divisible by patch_size={patch_size}")

        self.K = K
        self.patch_size = patch_size
        self.n_patches = K // patch_size
        self.d_channel = d_channel

        # Instance-level gather_spec so K propagates to DataPipeline
        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, strategy="recency"),
            co_occurrence=False,  # co-occurrence computed internally from neighbor_ids
        )

        # Fixed cosine time encoding matches DyGLib exactly (1/10^linspace(0,9,d))
        self.time_enc = FixedCosineTimeEncoder(d_time)
        self.co_enc = _CoOccurrenceEncoder(d_channel)

        # Channel projections (patch_size tokens concatenated → d_channel)
        self.proj_edge = nn.Linear(patch_size * d_edge, d_channel)
        self.proj_time = nn.Linear(patch_size * d_time, d_channel)
        self.proj_co = nn.Linear(patch_size * d_channel, d_channel)

        # Optional 4th channel: static node features (matches DyGLib when available)
        if node_feat is not None:
            d_node = node_feat.shape[1]
            self.register_buffer("node_feat", node_feat.float())
            self.proj_node = nn.Linear(patch_size * d_node, d_channel)
            self.n_channels = 4
        else:
            self.node_feat = None
            self.proj_node = None
            self.n_channels = 3

        d_joint = self.n_channels * d_channel
        self.layers = nn.ModuleList([
            _TransformerLayer(d_joint, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_joint, d_model)

        # DyGLib-style MergeLayer decoder: cat(src, dst) → Linear → ReLU → Linear → 1
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        # Positive: encode (src, dst) jointly
        src_emb, dst_emb = self._encode_pair(
            batch.src_neighbors, batch.dst_neighbors, batch.time,
            batch.src, batch.dst,
        )
        # Negative: encode (src, neg) jointly — src_emb is recomputed with neg context
        src_emb_neg, neg_emb = self._encode_pair(
            batch.src_neighbors, batch.neg_neighbors, batch.time,
            batch.src, batch.neg,
        )

        pos_score = self.decoder(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)
        neg_score = self.decoder(torch.cat([src_emb_neg, neg_emb], dim=-1)).squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    # --- pair encoder ---

    def _encode_pair(
        self, a_nbrs: NeighborData, b_nbrs: NeighborData, time: Tensor,
        a_ids: Optional[Tensor] = None, b_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Encode a (node-a, node-b) pair jointly. Returns (a_emb, b_emb) (B, d_model)."""
        # Time deltas
        a_dt = time.unsqueeze(1).float() - a_nbrs.timestamps.float()  # (B, K)
        b_dt = time.unsqueeze(1).float() - b_nbrs.timestamps.float()

        # Per-neighbor time features
        a_time = self.time_enc(a_dt)   # (B, K, d_time)
        b_time = self.time_enc(b_dt)

        # Per-neighbor co-occurrence features
        a_co, b_co = self.co_enc(a_nbrs.neighbor_ids.long(), b_nbrs.neighbor_ids.long())  # (B,K,d_ch)

        # Optional node features: look up static node embeddings by neighbor ID
        a_node = b_node = None
        if self.node_feat is not None:
            safe_a = a_nbrs.neighbor_ids.long().clamp(0, self.node_feat.shape[0] - 1)
            safe_b = b_nbrs.neighbor_ids.long().clamp(0, self.node_feat.shape[0] - 1)
            a_node = self.node_feat[safe_a].masked_fill(~a_nbrs.mask.unsqueeze(-1), 0.0)
            b_node = self.node_feat[safe_b].masked_fill(~b_nbrs.mask.unsqueeze(-1), 0.0)

        # Build patch tokens for each channel, then interleave channels
        a_tok = self._build_token(a_nbrs.edge_feats, a_time, a_co, a_node)  # (B, n_patches, d_joint)
        b_tok = self._build_token(b_nbrs.edge_feats, b_time, b_co, b_node)

        # Neighbor padding mask: (B, n_patches) True = valid
        a_pmask = self._patch_padding_mask(a_nbrs.mask)
        b_pmask = self._patch_padding_mask(b_nbrs.mask)

        # Self-token: prepend query node at position 0 (DyGLib-style)
        # Provides node's own LIWC features even when history is empty (key for sparse Wikipedia nodes)
        if self.node_feat is not None and a_ids is not None and b_ids is not None:
            a_self = self._self_token(a_ids, time)   # (B, 1, d_joint)
            b_self = self._self_token(b_ids, time)
            a_tok = torch.cat([a_self, a_tok], dim=1)
            b_tok = torch.cat([b_self, b_tok], dim=1)
            ones = torch.ones(a_tok.shape[0], 1, dtype=torch.bool, device=a_tok.device)
            a_pmask = torch.cat([ones, a_pmask], dim=1)
            b_pmask = torch.cat([ones, b_pmask], dim=1)

        n_seq = a_tok.shape[1]
        # Joint sequence: (B, 2*n_seq, d_joint)
        joint = torch.cat([a_tok, b_tok], dim=1)
        for layer in self.layers:
            joint = layer(joint)

        a_out = joint[:, :n_seq, :]
        b_out = joint[:, n_seq:, :]

        # Simple mean over all patches (including padding), matching DyGLib
        a_emb = a_out.mean(dim=1)
        b_emb = b_out.mean(dim=1)

        return self.out_proj(a_emb), self.out_proj(b_emb)

    def _self_token(self, ids: Tensor, time: Tensor) -> Tensor:
        """Build (B, 1, d_joint) self-token representing the query node at t=now."""
        B, dev, P = ids.shape[0], ids.device, self.patch_size

        # Zero edge (no actual edge to self), zero co-occurrence
        e_emb = self.proj_edge(torch.zeros(B, 1, self.proj_edge.in_features, device=dev))
        c_emb = self.proj_co(torch.zeros(B, 1, self.proj_co.in_features, device=dev))

        # Time delta = 0 for self; repeat P times to fill patch
        t0 = self.time_enc(torch.zeros(B, 1, dtype=torch.float, device=dev))  # (B, 1, d_time)
        t_emb = self.proj_time(t0.repeat(1, 1, P))  # (B, 1, P*d_time)

        channels = [e_emb, t_emb, c_emb]
        if self.proj_node is not None:
            nf = self.node_feat[ids.long().clamp(0, self.node_feat.shape[0] - 1)]  # (B, d_node)
            n_in = nf.unsqueeze(1).repeat(1, 1, P)   # (B, 1, P*d_node)
            channels.append(self.proj_node(n_in))

        return torch.cat(channels, dim=-1)   # (B, 1, d_joint)

    def _build_token(
        self, edge_feats: Tensor, time_feats: Tensor, co_feats: Tensor,
        node_feats: Optional[Tensor] = None,
    ) -> Tensor:
        """Build (B, n_patches, n_channels*d_channel) patch tokens."""
        # Patchify each channel: (B, K, d) → (B, n_patches, P*d)
        e_p = self._patchify(edge_feats)   # (B, n_patches, P*d_edge)
        t_p = self._patchify(time_feats)   # (B, n_patches, P*d_time)
        c_p = self._patchify(co_feats)     # (B, n_patches, P*d_channel)

        # Project each channel to d_channel
        e_emb = self.proj_edge(e_p)    # (B, n_patches, d_channel)
        t_emb = self.proj_time(t_p)
        c_emb = self.proj_co(c_p)

        channels = [e_emb, t_emb, c_emb]
        if node_feats is not None and self.proj_node is not None:
            n_p = self._patchify(node_feats)   # (B, n_patches, P*d_node)
            channels.append(self.proj_node(n_p))

        # Concatenate channels: (B, n_patches, n_channels*d_channel)
        return torch.cat(channels, dim=-1)

    def _patchify(self, x: Tensor) -> Tensor:
        """(B, K, d) → (B, n_patches, patch_size*d)."""
        B, K, d = x.shape
        return x.reshape(B, self.n_patches, self.patch_size * d)

    def _patch_padding_mask(self, mask: Tensor) -> Tensor:
        """(B, K) bool → (B, n_patches) bool. Patch is valid if ANY token is valid."""
        B, K = mask.shape
        return mask.reshape(B, self.n_patches, self.patch_size).any(dim=2)  # (B, n_patches)

    def _masked_mean(self, x: Tensor, mask: Tensor) -> Tensor:
        """Mean pool x (B, T, d) over valid positions in mask (B, T)."""
        m = mask.float().unsqueeze(-1)           # (B, T, 1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    @property
    def supports_independent_encode(self) -> bool:
        return False
