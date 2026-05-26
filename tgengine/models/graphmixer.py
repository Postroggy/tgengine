"""GraphMixer: MLP-Mixer based temporal graph model for link prediction.

Faithful adaptation of the GraphMixer paper (https://arxiv.org/abs/2302.11636)
to TGEngine's PreparedBatch interface.

Architecture:
  Link encoder:
    - Project (edge_feat || time_enc) → d_channel tokens
    - Apply N×MLPMixerLayer
    - Mean pool over valid positions → (B, d_channel)

  Node encoder (optional — requires node_raw_features):
    - Look up neighbor node features from stored buffer
    - Weighted softmax aggregation (invalid positions masked to -1e10)
    - Add the query node's own features → (B, d_node)

  Output:
    - Linear(d_channel + d_node, d_model) → node embedding
    - MergeDecoder for link scoring

Notes on K vs time_gap:
  Reference uses K=20 for link encoder and time_gap=2000 for node encoder
  (two separate neighbor queries). TGEngine uses the same K neighbors for both
  (buffer_size=32 constraint). The mechanism is identical; only K differs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.nn import HarmonicEncoder, MergeDecoder
from tgengine.nn.mlp_mixer import MLPMixerLayer


class GraphMixer(TemporalModel):
    """GraphMixer: MLP-Mixer link encoder + node feature aggregation.

    Args:
        d_model: Output embedding dimension.
        d_edge: Edge feature dimension.
        d_time: Time encoding output dimension.
        K: Max neighbors per node (= GatherSpec.neighbors.k).
        num_layers: Number of MLPMixerLayer blocks.
        token_expansion: Token FFN expansion factor (default 0.5, as in reference).
        channel_expansion: Channel FFN expansion factor (default 4.0).
        dropout: Dropout rate.
        node_raw_features: Optional (num_nodes, d_node) node feature matrix.
            If provided, enables node encoder. Non-trainable (frozen).
    """

    def __init__(
        self,
        d_model: int = 172,
        d_edge: int = 172,
        d_time: int = 100,
        K: int = 32,
        num_layers: int = 2,
        token_expansion: float = 0.5,
        channel_expansion: float = 4.0,
        dropout: float = 0.1,
        node_raw_features: Optional[Tensor] = None,
    ):
        super().__init__()

        self.K = K
        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, strategy="recency"),
            co_occurrence=False,
        )

        # time encoding: fixed harmonic (non-trainable, matching reference)
        self.time_enc = HarmonicEncoder(d_time)

        # num_channels = edge_feat_dim (as in reference)
        num_channels = d_edge
        self.projection = nn.Linear(d_edge + d_time, num_channels)

        self.mixers = nn.ModuleList([
            MLPMixerLayer(K, num_channels, token_expansion, channel_expansion, dropout)
            for _ in range(num_layers)
        ])

        # Node encoder (optional)
        if node_raw_features is not None:
            # Prepend a zero row (index 0 = padding sentinel in DyGLib)
            d_node = node_raw_features.shape[1]
            node_buf = torch.cat([
                torch.zeros(1, d_node, dtype=node_raw_features.dtype),
                node_raw_features,
            ], dim=0)
            self.register_buffer("node_raw_features", node_buf)
        else:
            self.node_raw_features = None
            d_node = 0
        self.d_node = d_node

        self.output_layer = nn.Linear(num_channels + d_node, d_model, bias=True)
        self.decoder = MergeDecoder(d_model)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self._encode(batch.src, batch.src_neighbors, batch.time)
        dst_emb = self._encode(batch.dst, batch.dst_neighbors, batch.time)
        neg_emb = self._encode(batch.neg, batch.neg_neighbors, batch.time)

        pos_score = self.decoder(src_emb, dst_emb)
        neg_score = self.decoder(src_emb, neg_emb)
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    def _encode(self, nodes: Tensor, nbrs, query_times: Tensor) -> Tensor:
        """Encode a node given its neighbor sequence."""
        # ---- link encoder ----
        dt = query_times.unsqueeze(1).float() - nbrs.timestamps.float()  # (B, K)
        t_feat = self.time_enc(dt)                                        # (B, K, d_time)

        combined = torch.cat([nbrs.edge_feats, t_feat], dim=-1)           # (B, K, d_edge+d_time)
        tokens = self.projection(combined)                                  # (B, K, num_channels)

        # zero-pad invalid positions before mixing
        tokens = tokens * nbrs.mask.unsqueeze(-1).float()

        for mixer in self.mixers:
            tokens = mixer(tokens)                                          # (B, K, num_channels)

        # masked mean pool
        m = nbrs.mask.float().unsqueeze(-1)                               # (B, K, 1)
        link_feat = (tokens * m).sum(1) / m.sum(1).clamp(min=1)          # (B, num_channels)

        # ---- node encoder ----
        if self.node_raw_features is not None:
            node_feat = self._node_encode(nodes, nbrs)                    # (B, d_node)
            combined_out = torch.cat([link_feat, node_feat], dim=-1)
        else:
            combined_out = link_feat

        return self.output_layer(combined_out)                             # (B, d_model)

    def _node_encode(self, nodes: Tensor, nbrs) -> Tensor:
        """Aggregate neighbor node features with softmax weighting."""
        # neighbor node feature lookup: (B, K, d_node)
        # shift neighbor_ids by +1 to align with node_raw_features (0 = padding)
        nbr_ids_shifted = (nbrs.neighbor_ids.long() + 1).clamp(min=0)
        nbr_feats = self.node_raw_features[nbr_ids_shifted]               # (B, K, d_node)

        # Mask: -1e10 for padding positions (as in reference)
        mask_float = nbrs.mask.float()                                     # (B, K)
        # If all neighbors padding, softmax still runs but on -1e10 — handle with clamp
        scores = mask_float.masked_fill(~nbrs.mask, -1e10)                # (B, K)
        attn = torch.softmax(scores, dim=1)                               # (B, K)
        agg = (nbr_feats * attn.unsqueeze(-1)).sum(dim=1)                 # (B, d_node)

        # Add query node's own features
        own_feat = self.node_raw_features[(nodes.long() + 1).clamp(min=0)]  # (B, d_node)
        return agg + own_feat

    @property
    def supports_independent_encode(self) -> bool:
        return True

    def encode_nodes(self, neighbors, times):
        # Dummy nodes tensor (not used for node encoder path in MRR)
        dummy_nodes = torch.zeros(neighbors.batch_size, dtype=torch.long,
                                  device=neighbors.device)
        return self._encode(dummy_nodes, neighbors, times)

    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        return self.decoder(src_emb, dst_emb)
