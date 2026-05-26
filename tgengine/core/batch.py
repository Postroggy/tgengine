from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class NeighborData:
    """Result of a temporal neighbor query. All tensors on same device."""

    neighbor_ids: Tensor  # (B, K) node IDs of neighbors
    timestamps: Tensor  # (B, K) interaction timestamps
    edge_feats: Tensor  # (B, K, d_edge) edge features
    mask: Tensor  # (B, K) bool — True for valid positions, False for padding

    @property
    def batch_size(self) -> int:
        return self.neighbor_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.neighbor_ids.shape[1]

    @property
    def device(self) -> torch.device:
        return self.neighbor_ids.device


@dataclass
class RawBatch:
    """A chronological batch of events before pipeline processing."""

    src: Tensor  # (B,) source node IDs
    dst: Tensor  # (B,) destination node IDs
    time: Tensor  # (B,) timestamps
    edge_feat: Optional[Tensor] = None  # (B, d_edge) edge features
    neg: Optional[Tensor] = None  # (B,) or (B, N_neg) negative node IDs
    edge_indices: Optional[Tensor] = None  # (B,) global edge indices (for TGB neg lookup)

    @property
    def batch_size(self) -> int:
        return self.src.shape[0]

    @property
    def device(self) -> torch.device:
        return self.src.device


@dataclass
class PreparedBatch:
    """Fully-assembled model input. All data fetched and on GPU.

    This is the ONLY input to TemporalModel.forward().
    Models should never need to access TemporalGraph directly.
    """

    src: Tensor  # (B,)
    dst: Tensor  # (B,)
    neg: Tensor  # (B,) or (B, N_neg)
    time: Tensor  # (B,)

    src_neighbors: NeighborData  # neighbors of src nodes
    dst_neighbors: NeighborData  # neighbors of dst nodes
    neg_neighbors: NeighborData  # neighbors of neg nodes

    co_occurrence: Optional[Tensor] = None  # (B, K) co-neighbor counts or None
    memory_emb: Optional[Tensor] = None  # (B, d_mem) for TGN-style models

    @property
    def batch_size(self) -> int:
        return self.src.shape[0]

    @property
    def device(self) -> torch.device:
        return self.src.device
