from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .batch import NeighborData


@dataclass
class Snapshot:
    """Lightweight graph snapshot for eval restore. O(1) to create."""

    write_pos: Tensor  # clone of per-node write positions
    num_edges: int  # total edges at snapshot time


class TemporalGraph:
    """GPU-resident temporal graph with circular buffer storage.

    Each node maintains a fixed-size ring buffer of its most recent interactions.
    All operations are batch-vectorized — no Python loops in the hot path.

    Args:
        num_nodes: Total number of nodes in the graph.
        buffer_size: Max neighbors stored per node (ring buffer capacity).
        edge_feat_dim: Dimensionality of edge features.
        device: Device for all tensors.
    """

    PADDING_ID: int = -1

    def __init__(
        self,
        num_nodes: int,
        buffer_size: int = 64,
        edge_feat_dim: int = 172,
        device: str | torch.device = "cuda",
    ):
        self.num_nodes = num_nodes
        self.buffer_size = buffer_size
        self.edge_feat_dim = edge_feat_dim
        self.device = torch.device(device)
        self._num_edges = 0

        # Ring buffers — all GPU resident
        self._neighbor_ids = torch.full(
            (num_nodes, buffer_size), self.PADDING_ID, dtype=torch.int32, device=self.device
        )
        self._neighbor_times = torch.zeros(
            (num_nodes, buffer_size), dtype=torch.float64, device=self.device
        )
        self._neighbor_feats = torch.zeros(
            (num_nodes, buffer_size, edge_feat_dim), dtype=torch.float32, device=self.device
        )
        self._write_pos = torch.zeros(num_nodes, dtype=torch.int64, device=self.device)

    def recent(self, nodes: Tensor, times: Tensor, k: int) -> NeighborData:
        """Get most recent k neighbors for each node before the query time.

        This is the primary data access operation. Batch-vectorized.

        Args:
            nodes: (N,) node IDs to query.
            times: (N,) query timestamps. Only neighbors with t < query_time are returned.
            k: number of recent neighbors to return.

        Returns:
            NeighborData with shape (N, k, ...).
        """
        N = nodes.shape[0]
        B = self.buffer_size

        # Read the ring buffers for queried nodes
        nbr_ids = self._neighbor_ids[nodes]  # (N, B)
        nbr_times = self._neighbor_times[nodes]  # (N, B)
        nbr_feats = self._neighbor_feats[nodes]  # (N, B, d)
        write_pos = self._write_pos[nodes]  # (N,)

        # Unroll ring buffer: map to chronological order (oldest → newest)
        offsets = torch.arange(B, device=self.device).unsqueeze(0)  # (1, B)
        chronological_idx = (write_pos.unsqueeze(1) - B + offsets) % B  # (N, B)

        # Gather in chronological order
        sorted_ids = torch.gather(nbr_ids, 1, chronological_idx)
        sorted_times = torch.gather(nbr_times, 1, chronological_idx)
        sorted_feats = torch.gather(
            nbr_feats, 1, chronological_idx.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim)
        )

        # Time mask: only keep neighbors strictly before query time
        time_mask = (sorted_times < times.unsqueeze(1)) & (sorted_ids != self.PADDING_ID)

        # For each node, take the last k valid entries (most recent)
        # We reverse so that most recent is first, then take top-k
        reversed_ids = sorted_ids.flip(1)
        reversed_times = sorted_times.flip(1)
        reversed_feats = sorted_feats.flip(1)
        reversed_mask = time_mask.flip(1)

        # Vectorized compaction: assign positional score; invalid entries pushed past k
        pos = torch.arange(B, device=self.device).unsqueeze(0).expand(N, -1)  # (N, B)
        scores = torch.where(reversed_mask, pos, torch.full((N, B), B, dtype=pos.dtype, device=self.device))
        topk_idx = scores.argsort(dim=1, stable=True)[:, :k]  # (N, k)

        out_ids = torch.gather(reversed_ids, 1, topk_idx)
        out_times = torch.gather(reversed_times, 1, topk_idx)
        out_feats = torch.gather(reversed_feats, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim))
        out_mask = torch.gather(reversed_mask, 1, topk_idx)

        return NeighborData(
            neighbor_ids=out_ids,
            timestamps=out_times,
            edge_feats=out_feats,
            mask=out_mask,
        )

    def co_neighbors(self, src: Tensor, dst: Tensor, times: Tensor, k: int = 32) -> Tensor:
        """Compute co-occurrence counts between src and dst neighbor sets.

        For each (src_i, dst_i) pair, count how many neighbors they share.

        Args:
            src: (B,) source node IDs.
            dst: (B,) destination node IDs.
            times: (B,) query timestamps.
            k: number of recent neighbors to sample per node.

        Returns:
            Tensor (B,) float co-occurrence counts.
        """
        B = src.shape[0]
        # Fused query: sample neighbors for all src and dst in one call
        all_nbrs = self.recent(torch.cat([src, dst]), torch.cat([times, times]), k)
        src_ids = all_nbrs.neighbor_ids[:B]   # (B, k) int32
        dst_ids = all_nbrs.neighbor_ids[B:]   # (B, k) int32
        src_mask = all_nbrs.mask[:B]          # (B, k)
        dst_mask = all_nbrs.mask[B:]          # (B, k)

        # (B, k_src, k_dst): for each pair, which src neighbors appear in dst neighbors
        eq = src_ids.unsqueeze(2) == dst_ids.unsqueeze(1)
        valid = src_mask.unsqueeze(2) & dst_mask.unsqueeze(1)
        # Count unique src neighbors that appear anywhere in dst's neighborhood
        return (eq & valid).any(dim=2).float().sum(dim=1)  # (B,)

    def advance(self, src: Tensor, dst: Tensor, time: Tensor, edge_feat: Optional[Tensor] = None):
        """Append new edges to the graph. Updates ring buffers for both src and dst.

        Args:
            src: (B,) source nodes.
            dst: (B,) destination nodes.
            time: (B,) timestamps.
            edge_feat: (B, d_edge) optional edge features.
        """
        B = src.shape[0]
        if edge_feat is None:
            edge_feat = torch.zeros((B, self.edge_feat_dim), device=self.device)

        # Update src → dst direction
        self._append_edges(src, dst, time, edge_feat)
        # Update dst → src direction (undirected)
        self._append_edges(dst, src, time, edge_feat)
        self._num_edges += B

    def _append_edges(self, from_nodes: Tensor, to_nodes: Tensor, time: Tensor, feat: Tensor):
        """Append directed edges to ring buffers."""
        write_idx = self._write_pos[from_nodes] % self.buffer_size
        self._neighbor_ids[from_nodes, write_idx] = to_nodes.to(torch.int32)
        self._neighbor_times[from_nodes, write_idx] = time.to(torch.float64)
        self._neighbor_feats[from_nodes, write_idx] = feat
        self._write_pos[from_nodes] += 1

    def snapshot(self) -> Snapshot:
        """Create a lightweight snapshot for eval restore."""
        return Snapshot(
            write_pos=self._write_pos.clone(),
            num_edges=self._num_edges,
        )

    def restore(self, snap: Snapshot):
        """Restore graph state from snapshot."""
        self._write_pos.copy_(snap.write_pos)
        self._num_edges = snap.num_edges

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @classmethod
    def from_dataset(
        cls,
        src: Tensor,
        dst: Tensor,
        time: Tensor,
        edge_feat: Optional[Tensor],
        num_nodes: int,
        buffer_size: int = 64,
        device: str = "cuda",
    ) -> "TemporalGraph":
        """Construct a TemporalGraph by replaying all events."""
        d_edge = edge_feat.shape[1] if edge_feat is not None else 172
        graph = cls(num_nodes, buffer_size, d_edge, device)
        # Replay in chronological order (data assumed sorted by time)
        # Process in large chunks for efficiency
        chunk_size = 10000
        n = src.shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            s = src[start:end].to(device)
            d = dst[start:end].to(device)
            t = time[start:end].to(device)
            f = edge_feat[start:end].to(device) if edge_feat is not None else None
            graph.advance(s, d, t, f)
        return graph
