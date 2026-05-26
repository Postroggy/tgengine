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

        O(NB) — no Python loops. Uses amax-based window selection:
          1. Unroll ring buffer to chronological order.
          2. Find last valid position via amax.
          3. Slice a k-window ending there.

        Args:
            nodes: (N,) node IDs to query.
            times: (N,) query timestamps — only neighbors with t < query_time returned.
            k: number of recent neighbors to return (most-recent-last in output).

        Returns:
            NeighborData with tensors of shape (N, k).
        """
        N = nodes.shape[0]
        B = self.buffer_size

        # Read ring buffers for queried nodes
        nbr_ids = self._neighbor_ids[nodes]   # (N, B)
        nbr_times = self._neighbor_times[nodes]  # (N, B)
        nbr_feats = self._neighbor_feats[nodes]  # (N, B, d)
        write_pos = self._write_pos[nodes]       # (N,)

        # Unroll to chronological order: oldest entry at col 0, newest at col B-1
        offsets = torch.arange(B, device=self.device).unsqueeze(0)          # (1, B)
        chrono_idx = (write_pos.unsqueeze(1) - B + offsets) % B             # (N, B)

        sorted_ids = torch.gather(nbr_ids, 1, chrono_idx)                   # (N, B)
        sorted_times = torch.gather(nbr_times, 1, chrono_idx)               # (N, B)
        sorted_feats = torch.gather(
            nbr_feats, 1, chrono_idx.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim)
        )  # (N, B, d)

        # Validity mask: strictly before query time and not padding
        valid = (sorted_times < times.unsqueeze(1)) & (sorted_ids != self.PADDING_ID)  # (N, B)

        # Find rightmost valid position per node — O(NB) via amax
        col_pos = torch.arange(B, device=self.device, dtype=torch.int64)    # (B,)
        last_valid = torch.where(
            valid.any(dim=1),
            (valid.long() * col_pos).amax(dim=1),                           # (N,)
            torch.full((N,), -1, dtype=torch.int64, device=self.device),
        )

        # Build k-window: [last_valid-k+1 ... last_valid], clamped to -1 for out-of-range
        window = last_valid.unsqueeze(1) - torch.arange(k - 1, -1, -1, device=self.device)  # (N, k)
        window = window.clamp(min=-1)   # (N, k) — -1 means no valid entry
        out_mask = window >= 0          # (N, k)
        safe = window.clamp(min=0)      # safe indices for gather

        out_ids = torch.gather(sorted_ids, 1, safe)     # (N, k)
        out_times = torch.gather(sorted_times, 1, safe) # (N, k)
        out_feats = torch.gather(sorted_feats, 1, safe.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim))  # (N,k,d)

        # Valid if: window index is in range AND content is not padding
        out_mask = (window >= 0) & (out_ids != self.PADDING_ID)

        # Zero-fill slots that are not valid
        out_ids = out_ids.masked_fill(~out_mask, self.PADDING_ID)
        out_times = out_times.masked_fill(~out_mask, 0.0)
        out_feats = out_feats.masked_fill(~out_mask.unsqueeze(-1), 0.0)

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

        # Count unique src neighbors that appear anywhere in dst's neighborhood
        eq = src_ids.unsqueeze(2) == dst_ids.unsqueeze(1)   # (B, k, k)
        valid = src_mask.unsqueeze(2) & dst_mask.unsqueeze(1)
        return (eq & valid).any(dim=2).float().sum(dim=1)    # (B,)

    def advance(self, src: Tensor, dst: Tensor, time: Tensor, edge_feat: Optional[Tensor] = None):
        """Append new edges to the graph (undirected). Updates ring buffers for both endpoints.

        Handles duplicate src/dst nodes within the same batch correctly.

        Args:
            src: (B,) source nodes.
            dst: (B,) destination nodes.
            time: (B,) timestamps.
            edge_feat: (B, d_edge) optional edge features.
        """
        B = src.shape[0]
        if edge_feat is None:
            edge_feat = torch.zeros((B, self.edge_feat_dim), device=self.device)

        self._append_edges(src, dst, time, edge_feat)
        self._append_edges(dst, src, time, edge_feat)
        self._num_edges += B

    def _append_edges(self, from_nodes: Tensor, to_nodes: Tensor, time: Tensor, feat: Tensor):
        """Append directed edges to ring buffers.

        Handles duplicate from_nodes by sorting and computing per-event write offsets,
        matching TGM's approach for correctness with repeated nodes in a batch.
        """
        device = self.device
        n = from_nodes.shape[0]
        B = self.buffer_size

        # Sort by (node, time) so duplicate nodes are grouped chronologically
        sort_key = from_nodes.long() * (time.long().max() + 1) + time.long()
        perm = sort_key.argsort(stable=True)

        s_from = from_nodes[perm]
        s_to = to_nodes[perm]
        s_time = time[perm]
        s_feat = feat[perm]

        # Compute per-event position within each node's group
        _, inv, cnts = torch.unique_consecutive(s_from.long(), return_inverse=True, return_counts=True)
        group_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), cnts.cumsum(0)[:-1]])
        event_offset = torch.arange(n, device=device) - group_start[inv]  # (n,)

        # Keep at most last B events per node (earlier events fall off the ring buffer anyway)
        keep = event_offset >= (cnts[inv] - B)
        s_from = s_from[keep]
        s_to = s_to[keep]
        s_time = s_time[keep]
        s_feat = s_feat[keep]

        # Recompute fresh 0-based offsets within the kept events for each node
        n_kept = s_from.shape[0]
        _, inv2, cnts2 = torch.unique_consecutive(s_from.long(), return_inverse=True, return_counts=True)
        gs2 = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), cnts2.cumsum(0)[:-1]])
        kept_offset = torch.arange(n_kept, device=device) - gs2[inv2]  # (n_kept,) 0,1,2...

        write_idx = (self._write_pos[s_from] + kept_offset) % B

        self._neighbor_ids[s_from, write_idx] = s_to.to(torch.int32)
        self._neighbor_times[s_from, write_idx] = s_time.to(torch.float64)
        self._neighbor_feats[s_from, write_idx] = s_feat

        # Increment write_pos by number of actual writes per node
        ones = torch.ones(n_kept, dtype=torch.int64, device=device)
        self._write_pos.scatter_add_(0, s_from.long(), ones)

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
