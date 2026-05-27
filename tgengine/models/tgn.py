"""TGN: Temporal Graph Network with node memory.

Stateful model that maintains a memory vector per node, updated after each
training batch via GRU. Memory captures long-range temporal context beyond
the fixed-size neighbor buffer.

Lifecycle (managed by Engine):
  evolve() → update memory after batch
  freeze()  → checkpoint memory before eval
  thaw()    → restore memory after eval
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.core.batch import NeighborData
from tgengine.nn import BilinearDecoder, GRUSeqEncoder, Time2Vec
from tgengine.nn.memory import NodeMemory


class TGN(TemporalModel):
    """Temporal Graph Network: GRU memory + neighbor aggregation."""

    gather_spec = GatherSpec(
        neighbors=NeighborSpec(k=10, strategy="recency"),
        memory=True,
    )

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 172,
        d_edge: int = 172,
        n_gru_layers: int = 1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.d_edge = d_edge

        self.memory = NodeMemory(num_nodes, d_model)
        self.time_enc = Time2Vec(d_model)
        # Project (edge_feat + time + mem) → d_model for sequence encoding
        self.feat_proj = nn.Linear(d_edge + d_model + d_model, d_model)
        self.encoder = GRUSeqEncoder(d_model, n_gru_layers)
        self.decoder = BilinearDecoder(d_model)

        # Message function: [src_mem | dst_mem | time_enc | edge_feat] → message
        self.msg_fn = nn.Linear(d_model * 2 + d_model + d_edge, d_model)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self._embed(batch.src, batch.src_neighbors, batch.time)
        dst_emb = self._embed(batch.dst, batch.dst_neighbors, batch.time)
        neg_emb = self._embed(batch.neg, batch.neg_neighbors, batch.time)

        pos_score = self.decoder(src_emb, dst_emb)
        neg_score = self.decoder(src_emb, neg_emb)
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    def _embed(self, nodes: Tensor, nbrs: NeighborData, query_times: Tensor) -> Tensor:
        """Build node embedding from memory + neighbor history."""
        mem = self.memory.read(nodes)                          # (B, d_model)
        dt = query_times.unsqueeze(1) - nbrs.timestamps       # (B, K)
        time_feat = self.time_enc(dt)                          # (B, K, d)
        # Tile memory to neighbor dim: (B, K, d)
        mem_tiled = mem.unsqueeze(1).expand(-1, nbrs.seq_len, -1)
        seq = self.feat_proj(
            torch.cat([nbrs.edge_feats, time_feat, mem_tiled], dim=-1)
        )  # (B, K, d)
        return self.encoder(seq, nbrs.mask)                    # (B, d)

    # --- Stateful lifecycle ---

    def evolve(self, src: Tensor, dst: Tensor, time: Tensor, edge_feat: Optional[Tensor] = None):
        """Update node memories after observing a batch of interactions.

        Message for src: cat([mem_src, mem_dst, delta_time_enc, edge])
        Message for dst: cat([mem_dst, mem_src, delta_time_enc, edge])  ← args swapped
        delta_time = current_time - node.last_updated_time  (matches DyGLib TGN)
        """
        if edge_feat is None:
            edge_feat = torch.zeros(src.shape[0], self.d_edge, device=src.device)

        src_mem = self.memory.read(src)   # (B, d)
        dst_mem = self.memory.read(dst)

        # Delta times since last memory update (recency signal)
        t = time.float()
        src_dt = (t - self.memory.last_times(src)).clamp(min=0)  # (B,)
        dst_dt = (t - self.memory.last_times(dst)).clamp(min=0)

        src_t_enc = self.time_enc(src_dt)  # (B, d_model)
        dst_t_enc = self.time_enc(dst_dt)

        # Src message: [src_mem | dst_mem | time | edge]
        src_msg = self.msg_fn(torch.cat([src_mem, dst_mem, src_t_enc, edge_feat], dim=-1))
        # Dst message: [dst_mem | src_mem | time | edge]  — swapped memory order
        dst_msg = self.msg_fn(torch.cat([dst_mem, src_mem, dst_t_enc, edge_feat], dim=-1))

        self.memory.update(src, src_msg, t)
        self.memory.update(dst, dst_msg, t)

    def freeze(self) -> Any:
        return self.memory.checkpoint()

    def thaw(self, state: Any):
        self.memory.restore(state)
