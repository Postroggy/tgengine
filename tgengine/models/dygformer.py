"""DyGFormer: Transformer-based temporal graph model with co-occurrence encoding.

Architecture:
  1. Encode src/dst neighbor sequences via TransformerSeqEncoder
  2. Encode co-neighbor counts (src ∩ dst shared neighbors)
  3. Score via MLP([src_emb; dst_emb; co_feat])

Co-occurrence makes src/dst scores interdependent, so supports_independent_encode=False.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgengine import GatherSpec, ModelOutput, NeighborSpec, PreparedBatch, TemporalModel
from tgengine.core.batch import NeighborData
from tgengine.nn import CoNeighborEncoder, Time2Vec, TransformerSeqEncoder


class DyGFormer(TemporalModel):
    """Transformer-based temporal graph model with co-occurrence features."""

    gather_spec = GatherSpec(
        neighbors=NeighborSpec(k=32, strategy="recency"),
        co_occurrence=True,
    )

    def __init__(
        self,
        d_model: int = 172,
        d_edge: int = 172,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_enc = Time2Vec(d_model)
        self.feat_proj = nn.Linear(d_edge + d_model, d_model)
        self.encoder = TransformerSeqEncoder(d_model, n_layers, n_heads, dropout)
        self.co_enc = CoNeighborEncoder(d_model)
        # Merge: [src; dst; co] → score
        self.merge = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self._encode(batch.src_neighbors, batch.time)   # (B, d)
        dst_emb = self._encode(batch.dst_neighbors, batch.time)   # (B, d)
        neg_emb = self._encode(batch.neg_neighbors, batch.time)   # (B, d)

        co_feat = self.co_enc(batch.co_occurrence)  # (B, d)

        pos_score = self._score(src_emb, dst_emb, co_feat)
        # For neg: reuse src_emb and co_feat (co-occurrence is (src,neg)-specific but
        # we use (src,dst) co-occurrence as an approximation during training)
        neg_score = self._score(src_emb, neg_emb, co_feat)

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    def _encode(self, nbrs: NeighborData, query_times: torch.Tensor) -> torch.Tensor:
        dt = query_times.unsqueeze(1) - nbrs.timestamps          # (B, K)
        time_feat = self.time_enc(dt)                             # (B, K, d)
        seq = self.feat_proj(torch.cat([nbrs.edge_feats, time_feat], dim=-1))  # (B, K, d)
        return self.encoder(seq, nbrs.mask)                       # (B, d)

    def _score(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        co_feat: torch.Tensor,
    ) -> torch.Tensor:
        return self.merge(torch.cat([src_emb, dst_emb, co_feat], dim=-1)).squeeze(-1)
