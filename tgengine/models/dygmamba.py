"""Example: DyGMamba — Mamba-based temporal graph model.

Demonstrates how to define a new model in TGEngine:
1. Set gather_spec to declare data needs
2. Implement forward() with neural network logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgengine.core.batch import PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.nn import BilinearDecoder, MambaSeqEncoder, Time2Vec


class DyGMamba(TemporalModel):
    """Mamba-based temporal graph model for link prediction."""

    gather_spec = GatherSpec(
        neighbors=NeighborSpec(k=32, strategy="recency"),
        co_occurrence=False,
    )

    def __init__(self, d_model: int = 172, d_edge: int = 172, n_layers: int = 2):
        super().__init__()
        self.time_enc = Time2Vec(d_model)
        self.feat_proj = nn.Linear(d_edge + d_model, d_model)
        self.encoder = MambaSeqEncoder(d_model, n_layers)
        self.decoder = BilinearDecoder(d_model)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self._encode_neighbors(batch.src_neighbors, batch.time)
        dst_emb = self._encode_neighbors(batch.dst_neighbors, batch.time)
        neg_emb = self._encode_neighbors(batch.neg_neighbors, batch.time)

        pos_score = self.decoder(src_emb, dst_emb)
        neg_score = self.decoder(src_emb, neg_emb)

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss += F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    def _encode_neighbors(self, nbrs, query_times):
        dt = query_times.unsqueeze(1) - nbrs.timestamps
        time_feat = self.time_enc(dt)
        seq = self.feat_proj(torch.cat([nbrs.edge_feats, time_feat], dim=-1))
        return self.encoder(seq, nbrs.mask)

    @property
    def supports_independent_encode(self) -> bool:
        return True

    def encode_nodes(self, neighbors, times):
        return self._encode_neighbors(neighbors, times)

    def score_pairs(self, src_emb, dst_emb):
        return self.decoder(src_emb, dst_emb)
