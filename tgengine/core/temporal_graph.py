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

        # Sort valid entries to front using mask
        # Approach: compute cumulative valid count, use it as scatter index
        valid_count = reversed_mask.cumsum(dim=1)
        # Take first k positions
        out_ids = torch.full((N, k), self.PADDING_ID, dtype=torch.int32, device=self.device)
        out_times = torch.zeros((N, k), dtype=torch.float64, device=self.device)
        out_feats = torch.zeros((N, k, self.edge_feat_dim), dtype=torch.float32, device=self.device)
        out_mask = torch.zeros((N, k), dtype=torch.bool, device=self.device)

        # Compact valid entries to the left
        for i in range(min(k, B)):
            col_mask = reversed_mask[:, i] & (valid_count[:, i] <= k)
            target_col = valid_count[:, i] - 1  # 0-indexed position
            valid_rows = col_mask & (target_col < k)
            if valid_rows.any():
                target_idx = target_col[valid_rows]
                out_ids[valid_rows, target_idx] = reversed_ids[valid_rows, i]
                out_times[valid_rows, target_idx] = reversed_times[valid_rows, i]
                out_feats[valid_rows, target_idx] = reversed_feats[valid_rows, i]
                out_mask[valid_rows, target_idx] = True

        return NeighborData(
            neighbor_ids=out_ids,
            timestamps=out_times,
            edge_feats=out_feats,
            mask=out_mask,
        )

    def co_neighbors(self, src: Tensor, dst: Tensor, times: Tensor) -> Tensor:
        """Compute co-occurrence counts between src and dst neighbor sets.

        For each (src_i, dst_i) pair, count how many neighbors they share.

        Args:
            src: (B,) source node IDs.
            dst: (B,) destination node IDs.
            times: (B,) query timestamps.

        Returns:
            Tensor (B,) co-occurrence counts.
        """
        # TODO: implement efficient vectorized co-neighbor counting
        raise NotImplementedError("co_neighbors not yet implemented")

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
