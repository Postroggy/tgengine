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
        self.K = K
        self.patch_size = patch_size

        # Sequence = self + K neighbors. Pad to patch_size divisibility (matching DyGLib)
        seq_len = K + 1
        if seq_len % patch_size != 0:
            seq_len += patch_size - seq_len % patch_size
        self.n_patches = seq_len // patch_size
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
        # Negative: encode (neg_src, neg) jointly — uses random src when available
        neg_src_nbrs = batch.neg_src_neighbors if batch.neg_src_neighbors is not None else batch.src_neighbors
        neg_src_ids = batch.neg_src if batch.neg_src is not None else batch.src
        src_emb_neg, neg_emb = self._encode_pair(
            neg_src_nbrs, batch.neg_neighbors, batch.time,
            neg_src_ids, batch.neg,
        )

        pos_score = self.decoder(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)
        neg_score = self.decoder(torch.cat([src_emb_neg, neg_emb], dim=-1)).squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    # --- pair encoder ---

    def _encode_pair(
        self, a_nbrs: NeighborData, b_nbrs: NeighborData, time: Tensor,
        a_ids: Tensor, b_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode a (node-a, node-b) pair jointly.

        Self-token is integrated into the neighbor sequence at position 0
        BEFORE patching, matching DyGLib's architecture exactly.
        """
        B, K, dev = a_nbrs.edge_feats.shape[0], self.K, a_nbrs.edge_feats.device

        # ---- 1. Full neighbor-id sequences (self at position 0) ----
        a_nbr_ids = a_nbrs.neighbor_ids.long()   # (B, K), -1 for padding
        b_nbr_ids = b_nbrs.neighbor_ids.long()
        a_full_ids = torch.cat([a_ids.long().unsqueeze(1), a_nbr_ids], dim=1)  # (B, 1+K)
        b_full_ids = torch.cat([b_ids.long().unsqueeze(1), b_nbr_ids], dim=1)

        # Full mask: self at position 0 always valid
        ones = torch.ones(B, 1, dtype=torch.bool, device=dev)
        a_mask = torch.cat([ones, a_nbrs.mask], dim=1)    # (B, 1+K)
        b_mask = torch.cat([ones, b_nbrs.mask], dim=1)

        # ---- 2. Time channel (dt=0 for self) ----
        a_dt = time.unsqueeze(1).float() - a_nbrs.timestamps.float()   # (B, K)
        b_dt = time.unsqueeze(1).float() - b_nbrs.timestamps.float()
        a_time_nbr = self.time_enc(a_dt)   # (B, K, d_time)
        b_time_nbr = self.time_enc(b_dt)
        a_time_nbr = a_time_nbr.masked_fill(~a_nbrs.mask.unsqueeze(-1), 0.0)
        b_time_nbr = b_time_nbr.masked_fill(~b_nbrs.mask.unsqueeze(-1), 0.0)
        t0 = self.time_enc(torch.zeros(B, 1, device=dev))              # (B, 1, d_time)
        a_time = torch.cat([t0, a_time_nbr], dim=1)                    # (B, 1+K, d_time)
        b_time = torch.cat([t0, b_time_nbr], dim=1)

        # ---- 3. Edge channel (zeros for self, no edge to self) ----
        d_edge = a_nbrs.edge_feats.shape[-1]
        a_edge = torch.cat([
            torch.zeros(B, 1, d_edge, device=dev), a_nbrs.edge_feats,
        ], dim=1)   # (B, 1+K, d_edge)
        b_edge = torch.cat([
            torch.zeros(B, 1, d_edge, device=dev), b_nbrs.edge_feats,
        ], dim=1)

        # ---- 4. Co-occurrence (self participates in counting) ----
        a_co, b_co = self.co_enc(a_full_ids, b_full_ids)   # (B, 1+K, d_ch)

        # ---- 5. Node features channel (optional) ----
        a_node = b_node = None
        if self.node_feat is not None:
            a_node = self.node_feat[a_full_ids.clamp(min=0)]
            b_node = self.node_feat[b_full_ids.clamp(min=0)]

        # ---- 6. Pad to patch_size divisibility ----
        S = 1 + K   # sequence length before padding
        if S % self.patch_size != 0:
            pad = self.patch_size - S % self.patch_size
            a_edge = F.pad(a_edge, (0, 0, 0, pad))
            b_edge = F.pad(b_edge, (0, 0, 0, pad))
            a_time = F.pad(a_time, (0, 0, 0, pad))
            b_time = F.pad(b_time, (0, 0, 0, pad))
            a_co = F.pad(a_co, (0, 0, 0, pad))
            b_co = F.pad(b_co, (0, 0, 0, pad))
            a_mask = F.pad(a_mask, (0, pad), value=False)
            b_mask = F.pad(b_mask, (0, pad), value=False)
            if a_node is not None:
                a_node = F.pad(a_node, (0, 0, 0, pad))
                b_node = F.pad(b_node, (0, 0, 0, pad))

        # ---- 7. Patchify + project each channel ----
        a_tok = self._build_token(a_edge, a_time, a_co, a_node)  # (B, n_patches, d_joint)
        b_tok = self._build_token(b_edge, b_time, b_co, b_node)

        # ---- 8. Joint src+dst transformer ----
        n_patches = a_tok.shape[1]
        joint = torch.cat([a_tok, b_tok], dim=1)   # (B, 2*n_patches, d_joint)
        for layer in self.layers:
            joint = layer(joint)

        a_out = joint[:, :n_patches, :]
        b_out = joint[:, n_patches:, :]

        # Simple mean over all patches (matching DyGLib)
        a_emb = a_out.mean(dim=1)
        b_emb = b_out.mean(dim=1)

        return self.out_proj(a_emb), self.out_proj(b_emb)

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
        """(B, S, d) → (B, S//patch_size, patch_size*d)."""
        B, S, d = x.shape
        return x.reshape(B, S // self.patch_size, self.patch_size * d)

    @property
    def supports_independent_encode(self) -> bool:
        return False
