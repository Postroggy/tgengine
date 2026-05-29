"""FreeDyG: MLP-Mixer based temporal graph model with 4-channel encoding.

Faithful adaptation of the FreeDyG paper to TGEngine's PreparedBatch interface.

Architecture highlights:
  - 4 feature channels per token: edge, time, node (optional), NIF
  - NIF (Neighbor Interaction Frequency) = vectorized co-occurrence encoding,
    same mechanism as DyGFormer Section 4.1 but applied per-position
  - Each channel projected to d_channel, then concatenated → reduce to d_channel
  - FreeDyGMixerLayer: FFT FilterLayer + token-mixing + channel-mixing
  - Learned weighted aggregation (vs. mean pool in GraphMixer)
  - Two-pass forward like DyGFormer: (src,dst) and (src,neg) pairs
  - supports_independent_encode=False because NIF is pair-dependent
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import NeighborData, PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import EmbeddingBundle, ModelOutput, TemporalModel
from tgengine.nn import FixedCosineTimeEncoder, MergeDecoder
from tgengine.nn.mlp_mixer import FreeDyGMixerLayer

PADDING_ID = -1


class _NIFEncoder(nn.Module):
    """Per-neighbor co-occurrence (NIF) encoder for FreeDyG.

    Vectorized equivalent of FreeDyG's NIFEncoder (removes Python loop).
    For each token in a, computes [self_count, cross_count]:
      self_count  = times this token appears in a's neighbor list
      cross_count = times this token appears in b's neighbor list
    The 2-D feature is then projected to d_out via a shared MLP.
    """

    def __init__(self, d_out: int):
        super().__init__()
        # Each scalar encoded independently, then summed (same as DyGFormer)
        self.proj = nn.Sequential(
            nn.Linear(1, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, a_ids: Tensor, b_ids: Tensor, a_mask: Tensor, b_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            a_ids, b_ids: (B, K) neighbor IDs (int32).
            a_mask, b_mask: (B, K) valid positions.

        Returns:
            a_feat, b_feat — each (B, K, d_out).
        """
        a_pad, b_pad = a_ids == PADDING_ID, b_ids == PADDING_ID

        a_self = (a_ids.unsqueeze(1) == a_ids.unsqueeze(2)).float()   # (B, K, K)
        b_self = (b_ids.unsqueeze(1) == b_ids.unsqueeze(2)).float()
        cross  = (a_ids.unsqueeze(1) == b_ids.unsqueeze(2)).float()   # (B, K, K)

        a_self = a_self.masked_fill(a_pad.unsqueeze(2) | a_pad.unsqueeze(1), 0.0)
        b_self = b_self.masked_fill(b_pad.unsqueeze(2) | b_pad.unsqueeze(1), 0.0)
        cross  = cross.masked_fill(a_pad.unsqueeze(2)  | b_pad.unsqueeze(1), 0.0)

        # (B, K, 2): [self_count, cross_count]
        # a_freq[b,i] = [count of a[b,i] in a, count of a[b,i] in b]
        # b_freq[b,j] = [count of b[b,j] in a, count of b[b,j] in b]
        # cross.sum(2)[b,i] = count of b positions matching a[b,i]
        # cross.sum(1)[b,j] = count of a positions matching b[b,j]
        a_freq = torch.stack([a_self.sum(1), cross.sum(2)], dim=2)
        b_freq = torch.stack([cross.sum(1), b_self.sum(1)], dim=2)

        # project each scalar → d_out, sum over the 2 scalars
        a_feat = self.proj(a_freq.unsqueeze(-1)).sum(dim=2)            # (B, K, d_out)
        b_feat = self.proj(b_freq.unsqueeze(-1)).sum(dim=2)

        # zero out padding positions
        a_feat = a_feat.masked_fill(a_pad.unsqueeze(-1), 0.0)
        b_feat = b_feat.masked_fill(b_pad.unsqueeze(-1), 0.0)
        return a_feat, b_feat


class FreeDyG(TemporalModel):
    """FreeDyG: 4-channel MLP-Mixer temporal graph model.

    Args:
        d_model: Output embedding dimension.
        d_edge: Edge feature dimension.
        d_time: Time encoding dimension.
        d_nif: NIF feature dimension per position (projected from 2 scalars).
        K: Max neighbors per node.
        num_layers: Number of FreeDyGMixerLayer blocks.
        dropout: Dropout rate.
        token_expansion: Token FFN expansion factor (default 0.5).
        channel_expansion: Channel FFN expansion factor (default 4.0).
        node_raw_features: Optional (num_nodes, d_node) feature matrix. Non-trainable.
    """

    def __init__(
        self,
        d_model: int = 172,
        d_edge: int = 172,
        d_time: int = 100,
        d_nif: int = 172,
        K: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        token_expansion: float = 0.5,
        channel_expansion: float = 4.0,
        node_raw_features: Optional[Tensor] = None,
    ):
        super().__init__()
        self.K = K
        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, strategy="recency"),
            co_occurrence=False,  # NIF computed internally from neighbor_ids
        )

        self.time_enc = FixedCosineTimeEncoder(d_time, learnable=False)
        self.nif_enc = _NIFEncoder(d_nif)

        # Channel projections → d_edge each (as in reference)
        self.proj_edge = nn.Linear(d_edge, d_edge)
        self.proj_time = nn.Linear(d_time, d_edge)
        self.proj_nif  = nn.Linear(d_nif,  d_edge)

        if node_raw_features is not None:
            d_node = node_raw_features.shape[1]
            node_buf = torch.cat([
                torch.zeros(1, d_node, dtype=node_raw_features.dtype),
                node_raw_features,
            ], dim=0)
            self.register_buffer("node_raw_features", node_buf)
            self.proj_node = nn.Linear(d_node, d_edge)
            self._has_node_feat = True
        else:
            self.node_raw_features = None
            self._has_node_feat = False
            d_node = 0

        n_channels = 3 + (1 if self._has_node_feat else 0)
        self.reduce = nn.Linear(n_channels * d_edge, d_edge)

        self.mixers = nn.ModuleList([
            FreeDyGMixerLayer(K, d_edge, token_expansion, channel_expansion, dropout)
            for _ in range(num_layers)
        ])

        # Weighted aggregation (learned attention per position)
        self.weightagg = nn.Linear(d_edge, 1)

        self.out_proj = nn.Linear(d_edge, d_model)
        self.decoder = MergeDecoder(d_model)

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        src, dst = self._encode_pair(
            batch.src_neighbors, batch.dst_neighbors, batch.time
        )
        src_for_neg, neg = self._encode_pair(
            batch.src_neighbors, batch.neg_neighbors, batch.time
        )
        return EmbeddingBundle(src=src, dst=dst, neg=neg, src_for_neg=src_for_neg)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        # Positive pass: (src, dst) pair
        src_emb, dst_emb = self._encode_pair(
            batch.src_neighbors, batch.dst_neighbors, batch.time
        )
        # Negative pass: (src, neg) pair — src re-encoded in neg context
        src_emb_neg, neg_emb = self._encode_pair(
            batch.src_neighbors, batch.neg_neighbors, batch.time
        )

        pos_score = self.decoder(src_emb, dst_emb)
        neg_score = self.decoder(src_emb_neg, neg_emb)
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    def _encode_pair(
        self, a_nbrs: NeighborData, b_nbrs: NeighborData, time: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Encode a (a, b) pair. Returns (a_emb, b_emb) each (B, d_model)."""
        a_ids = a_nbrs.neighbor_ids.long()
        b_ids = b_nbrs.neighbor_ids.long()

        # Time features
        a_dt = time.unsqueeze(1).float() - a_nbrs.timestamps.float()
        b_dt = time.unsqueeze(1).float() - b_nbrs.timestamps.float()
        a_t = self.time_enc(a_dt)   # (B, K, d_time)
        b_t = self.time_enc(b_dt)

        # Zero out time features for padding positions (DyGLib semantics)
        a_t = a_t.masked_fill(~a_nbrs.mask.unsqueeze(-1), 0.0)
        b_t = b_t.masked_fill(~b_nbrs.mask.unsqueeze(-1), 0.0)

        # NIF features (vectorized co-occurrence)
        a_nif, b_nif = self.nif_enc(a_nbrs.neighbor_ids, b_nbrs.neighbor_ids,
                                    a_nbrs.mask, b_nbrs.mask)   # (B, K, d_nif)

        # Per-channel projections → (B, K, d_edge)
        a_feats = [self.proj_edge(a_nbrs.edge_feats),
                   self.proj_time(a_t),
                   self.proj_nif(a_nif)]
        b_feats = [self.proj_edge(b_nbrs.edge_feats),
                   self.proj_time(b_t),
                   self.proj_nif(b_nif)]

        if self._has_node_feat:
            a_node = self._lookup_node_feat(a_nbrs.neighbor_ids)  # (B, K, d_node)
            b_node = self._lookup_node_feat(b_nbrs.neighbor_ids)
            a_feats.append(self.proj_node(a_node))
            b_feats.append(self.proj_node(b_node))

        # Concatenate and reduce: (B, K, n*d_edge) → (B, K, d_edge)
        a_combined = self.reduce(torch.cat(a_feats, dim=-1))
        b_combined = self.reduce(torch.cat(b_feats, dim=-1))

        # MLP-Mixer layers
        for mixer in self.mixers:
            a_combined = mixer(a_combined)
            b_combined = mixer(b_combined)

        # Weighted aggregation
        a_emb = self._weighted_agg(a_combined)   # (B, d_edge)
        b_emb = self._weighted_agg(b_combined)

        return self.out_proj(a_emb), self.out_proj(b_emb)

    def _lookup_node_feat(self, neighbor_ids: Tensor) -> Tensor:
        """Lookup node features for neighbor IDs. Padding → zero row."""
        ids_shifted = (neighbor_ids.long() + 1).clamp(min=0)
        return self.node_raw_features[ids_shifted]

    def _weighted_agg(self, tokens: Tensor) -> Tensor:
        """Learned weighted aggregation: (B, K, C) → (B, C)."""
        w = self.weightagg(tokens)          # (B, K, 1)
        w = w.transpose(1, 2)               # (B, 1, K)
        out = w.matmul(tokens).squeeze(1)   # (B, C)
        return out

    @property
    def supports_independent_encode(self) -> bool:
        return False  # NIF requires both sides of the pair
