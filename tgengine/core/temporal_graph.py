from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .batch import NeighborData
from .kernels import (
    HAS_CUDA_EXT, HAS_TRITON,
    cuda_temporal_recent_k, cuda_temporal_recent_2hop, cuda_co_neighbor_count,
    triton_temporal_recent_k,
)


@dataclass
class Snapshot:
    """Lightweight graph snapshot for eval restore.

    CSR is immutable after freeze — only overflow state needs saving.
    """

    overflow_ids: Tensor
    overflow_times: Tensor
    overflow_feats: Tensor
    overflow_write_pos: Tensor
    num_edges: int


class TemporalGraph:
    """GPU-resident temporal graph with CSR + overflow storage.

    Two-phase lifecycle:
      1. Build: advance() accumulates edges into CPU lists.
      2. Freeze: freeze_csr() sorts and compresses into GPU CSR tensors.
      3. Query: recent() uses searchsorted on CSR, merging overflow if present.

    Post-freeze advance() writes to a small overflow ring buffer (for eval-time
    edges). snapshot/restore only touch overflow state since CSR is immutable.

    Args:
        num_nodes: Total number of nodes in the graph.
        edge_feat_dim: Dimensionality of edge features.
        device: Device for all tensors.
        overflow_size: Ring buffer capacity per node for post-freeze edges.
    """

    PADDING_ID: int = -1

    def __init__(
        self,
        num_nodes: int,
        edge_feat_dim: int = 172,
        device: str | torch.device = "cuda",
        overflow_size: int = 64,
        buffer_size: int | None = None,
        use_triton: bool = True,
    ):
        if buffer_size is not None:
            overflow_size = buffer_size
        self._use_triton = use_triton and (HAS_CUDA_EXT or HAS_TRITON)
        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim
        self.device = torch.device(device)
        self.overflow_size = overflow_size
        self._num_edges = 0

        # Build phase: accumulate edges in CPU lists
        self._build_from: list[int] = []
        self._build_to: list[int] = []
        self._build_time: list[float] = []
        self._build_feat: list[Tensor] = []
        self._frozen = False

        # CSR tensors (populated by freeze_csr)
        self._offsets: Optional[Tensor] = None
        self._nbr_ids: Optional[Tensor] = None
        self._nbr_times: Optional[Tensor] = None
        self._nbr_feats: Optional[Tensor] = None

        # Overflow ring buffer (allocated on first post-freeze advance)
        self._ov_ids: Optional[Tensor] = None
        self._ov_times: Optional[Tensor] = None
        self._ov_feats: Optional[Tensor] = None
        self._ov_write_pos: Optional[Tensor] = None

    def _ensure_overflow(self):
        if self._ov_ids is not None:
            return
        B = self.overflow_size
        self._ov_ids = torch.full(
            (self.num_nodes, B), self.PADDING_ID, dtype=torch.int32, device=self.device
        )
        self._ov_times = torch.zeros(
            (self.num_nodes, B), dtype=torch.float64, device=self.device
        )
        self._ov_feats = torch.zeros(
            (self.num_nodes, B, self.edge_feat_dim), dtype=torch.float32, device=self.device
        )
        self._ov_write_pos = torch.zeros(self.num_nodes, dtype=torch.int64, device=self.device)

    def advance(self, src: Tensor, dst: Tensor, time: Tensor, edge_feat: Optional[Tensor] = None):
        """Append undirected edges. Pre-freeze: accumulate to lists. Post-freeze: overflow buffer."""
        B = src.shape[0]
        if edge_feat is None:
            edge_feat = torch.zeros((B, self.edge_feat_dim), device=self.device)

        if not self._frozen:
            s = src.cpu().tolist()
            d = dst.cpu().tolist()
            t = time.cpu().tolist()
            self._build_from.extend(s)
            self._build_from.extend(d)
            self._build_to.extend(d)
            self._build_to.extend(s)
            self._build_time.extend(t)
            self._build_time.extend(t)
            feat_cpu = edge_feat.cpu()
            self._build_feat.append(feat_cpu)
            self._build_feat.append(feat_cpu)
        else:
            self._ensure_overflow()
            all_from = torch.cat([src, dst])
            all_to = torch.cat([dst, src])
            all_time = torch.cat([time, time])
            all_feat = torch.cat([edge_feat, edge_feat])
            self._append_overflow(all_from, all_to, all_time, all_feat)

        self._num_edges += B

    def freeze_csr(self):
        """Convert accumulated edges into sorted CSR tensors on GPU."""
        n_edges = len(self._build_from)
        if n_edges == 0:
            self._offsets = torch.zeros(self.num_nodes + 1, dtype=torch.int64, device=self.device)
            self._nbr_ids = torch.empty(0, dtype=torch.int32, device=self.device)
            self._nbr_times = torch.empty(0, dtype=torch.float64, device=self.device)
            self._nbr_feats = torch.empty(0, self.edge_feat_dim, dtype=torch.float32, device=self.device)
            self._frozen = True
            self._build_from = []
            self._build_to = []
            self._build_time = []
            self._build_feat = []
            return

        from_t = torch.tensor(self._build_from, dtype=torch.int64)
        to_t = torch.tensor(self._build_to, dtype=torch.int32)
        time_t = torch.tensor(self._build_time, dtype=torch.float64)
        if self._build_feat:
            feat_t = torch.cat(self._build_feat, dim=0)  # (n_edges, d_edge)
        else:
            feat_t = torch.zeros(n_edges, self.edge_feat_dim, dtype=torch.float32)

        # Sort by (from_node, time): stable sort by time first, then by node
        perm = torch.argsort(time_t, stable=True)
        perm = perm[torch.argsort(from_t[perm], stable=True)]

        from_sorted = from_t[perm]
        to_sorted = to_t[perm]
        time_sorted = time_t[perm]
        feat_sorted = feat_t[perm]

        # Build CSR offsets
        counts = torch.zeros(self.num_nodes, dtype=torch.int64)
        counts.scatter_add_(0, from_sorted, torch.ones(n_edges, dtype=torch.int64))
        offsets = torch.zeros(self.num_nodes + 1, dtype=torch.int64)
        offsets[1:] = counts.cumsum(0)

        # Move to GPU
        self._offsets = offsets.to(self.device)
        self._nbr_ids = to_sorted.to(self.device)
        self._nbr_times = time_sorted.to(self.device)
        self._nbr_feats = feat_sorted.to(self.device)

        self._frozen = True
        self._build_from = []
        self._build_to = []
        self._build_time = []
        self._build_feat = []

    def recent(self, nodes: Tensor, times: Tensor, k: int) -> NeighborData:
        """Get most recent k neighbors before query time using searchsorted on CSR."""
        if not self._frozen:
            self.freeze_csr()

        N = nodes.shape[0]
        device = self.device

        # Fused kernel fast path: only when no overflow, on CUDA, and non-empty
        if self._use_triton and self._ov_ids is None and N > 0 and device.type == "cuda":
            safe_nodes = nodes.clamp(min=0)
            is_padding = nodes < 0
            query_nodes = safe_nodes.masked_fill(is_padding, 0).to(torch.int64)
            query_times = times.to(torch.float64)

            if HAS_CUDA_EXT:
                out_ids, out_times, out_feats, out_mask = cuda_temporal_recent_k(
                    self._offsets, self._nbr_times, self._nbr_ids, self._nbr_feats,
                    query_nodes, query_times, k,
                )
            else:
                out_ids, out_times, out_feats, out_mask = triton_temporal_recent_k(
                    self._offsets, self._nbr_times, self._nbr_ids, self._nbr_feats,
                    query_nodes, query_times, k,
                )
            # Zero out padding node results
            if is_padding.any():
                pad_mask = is_padding.unsqueeze(1).expand_as(out_mask)
                out_ids = out_ids.masked_fill(pad_mask, self.PADDING_ID)
                out_times = out_times.masked_fill(pad_mask, 0.0)
                out_feats = out_feats.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                out_mask = out_mask & ~pad_mask

            return NeighborData(
                neighbor_ids=out_ids,
                timestamps=out_times,
                edge_feats=out_feats,
                mask=out_mask,
            )

        N = nodes.shape[0]
        device = self.device

        # Clamp padding node IDs to 0 for safe indexing (they'll get masked out)
        safe_nodes = nodes.clamp(min=0)

        # CSR lookup
        starts = self._offsets[safe_nodes]       # (N,)
        ends = self._offsets[safe_nodes + 1]     # (N,)
        # Nodes that were PADDING_ID get zero-length segments
        is_padding = nodes < 0
        starts = starts.masked_fill(is_padding, 0)
        ends = ends.masked_fill(is_padding, 0)
        max_deg = int((ends - starts).max().item()) if N > 0 else 0

        if max_deg == 0 and self._ov_ids is None:
            return self._empty_neighbor_data(N, k, device)

        # Build per-node time segments and searchsorted
        if max_deg > 0:
            col_idx = torch.arange(max_deg, device=device).unsqueeze(0)  # (1, max_deg)
            abs_idx = starts.unsqueeze(1) + col_idx                       # (N, max_deg)
            valid_mask = col_idx < (ends - starts).unsqueeze(1)           # (N, max_deg)
            safe_idx = abs_idx.clamp(max=len(self._nbr_times) - 1)

            seg_times = self._nbr_times[safe_idx]                         # (N, max_deg)
            seg_times = seg_times.masked_fill(~valid_mask, float('inf'))

            # searchsorted: find first index where seg_times >= query_time
            insert_pos = torch.searchsorted(seg_times, times.unsqueeze(1))  # (N, 1)
            insert_pos = insert_pos.squeeze(1)                              # (N,)

            # Take up to k neighbors before insert_pos
            csr_counts = insert_pos.clamp(max=max_deg)  # how many valid neighbors per node
            csr_take = csr_counts.clamp(max=k)           # take at most k

            # Gather the last `csr_take` entries before insert_pos
            out_ids, out_times, out_feats, out_mask = self._gather_csr_window(
                nodes, starts, csr_counts, csr_take, k, N
            )
        else:
            out_ids = torch.full((N, k), self.PADDING_ID, dtype=torch.int32, device=device)
            out_times = torch.zeros(N, k, dtype=torch.float64, device=device)
            out_feats = torch.zeros(N, k, self.edge_feat_dim, dtype=torch.float32, device=device)
            out_mask = torch.zeros(N, k, dtype=torch.bool, device=device)

        # Merge overflow if present
        if self._ov_ids is not None:
            out_ids, out_times, out_feats, out_mask = self._merge_overflow(
                nodes, times, k, out_ids, out_times, out_feats, out_mask
            )

        return NeighborData(
            neighbor_ids=out_ids,
            timestamps=out_times,
            edge_feats=out_feats,
            mask=out_mask,
        )

    def _gather_csr_window(self, nodes, starts, csr_counts, csr_take, k, N):
        """Gather the most recent `csr_take` neighbors from CSR, right-aligned in k slots."""
        device = self.device

        # Right-aligned: valid entries at positions [k-csr_take, k), padding at [0, k-csr_take)
        col = torch.arange(k, device=device).unsqueeze(0)  # (1, k)
        # Position within the valid window (0-based from oldest valid)
        pos_in_window = col - (k - csr_take.unsqueeze(1))  # (N, k)
        valid = (pos_in_window >= 0) & (pos_in_window < csr_take.unsqueeze(1))

        # Map to CSR segment: start of valid window = csr_counts - csr_take
        seg_offset = (csr_counts - csr_take).unsqueeze(1) + pos_in_window.clamp(min=0)  # (N, k)
        abs_idx = starts.unsqueeze(1) + seg_offset.clamp(min=0)
        abs_idx = abs_idx.clamp(max=max(len(self._nbr_ids) - 1, 0))

        out_ids = self._nbr_ids[abs_idx]
        out_times = self._nbr_times[abs_idx]
        out_feats = self._nbr_feats[abs_idx]

        out_ids = out_ids.masked_fill(~valid, self.PADDING_ID)
        out_times = out_times.masked_fill(~valid, 0.0)
        out_feats = out_feats.masked_fill(~valid.unsqueeze(-1), 0.0)

        return out_ids, out_times, out_feats, valid

    def _merge_overflow(self, nodes, times, k, csr_ids, csr_times, csr_feats, csr_mask):
        """Merge CSR results with overflow ring buffer entries."""
        N = nodes.shape[0]
        device = self.device
        OB = self.overflow_size
        safe_nodes = nodes.clamp(min=0)

        ov_ids = self._ov_ids[safe_nodes]       # (N, OB)
        ov_times = self._ov_times[safe_nodes]   # (N, OB)
        ov_feats = self._ov_feats[safe_nodes]   # (N, OB, d)

        # Valid overflow: not padding AND time < query
        ov_valid = (ov_ids != self.PADDING_ID) & (ov_times < times.unsqueeze(1))

        # Count valid overflow per node
        ov_count = ov_valid.sum(dim=1)  # (N,)
        if ov_count.sum() == 0:
            return csr_ids, csr_times, csr_feats, csr_mask

        # Concatenate CSR + overflow, sort by time, take last k
        all_ids = torch.cat([csr_ids, ov_ids], dim=1)        # (N, k+OB)
        all_times = torch.cat([csr_times, ov_times], dim=1)
        all_feats = torch.cat([csr_feats, ov_feats], dim=1)
        all_mask = torch.cat([csr_mask, ov_valid], dim=1)

        # Set invalid entries to -inf time so they sort first (and get dropped)
        sort_times = all_times.masked_fill(~all_mask, -float('inf'))
        _, sort_idx = sort_times.sort(dim=1)  # ascending

        # Gather sorted
        all_ids = torch.gather(all_ids, 1, sort_idx)
        all_times = torch.gather(all_times, 1, sort_idx)
        all_feats = torch.gather(all_feats, 1, sort_idx.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim))
        all_mask = torch.gather(all_mask, 1, sort_idx)

        # Take last k (most recent)
        total = all_ids.shape[1]
        out_ids = all_ids[:, total - k:]
        out_times = all_times[:, total - k:]
        out_feats = all_feats[:, total - k:]
        out_mask = all_mask[:, total - k:]

        return out_ids, out_times, out_feats, out_mask

    def _append_overflow(self, from_nodes: Tensor, to_nodes: Tensor, time: Tensor, feat: Tensor):
        """Append directed edges to overflow ring buffer."""
        device = self.device
        n = from_nodes.shape[0]
        B = self.overflow_size

        sort_key = from_nodes.long() * (int(time.max().item()) + 2) + torch.arange(n, device=device)
        perm = sort_key.argsort(stable=True)

        s_from = from_nodes[perm]
        s_to = to_nodes[perm]
        s_time = time[perm]
        s_feat = feat[perm]

        _, inv, cnts = torch.unique_consecutive(s_from.long(), return_inverse=True, return_counts=True)
        group_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), cnts.cumsum(0)[:-1]])
        event_offset = torch.arange(n, device=device) - group_start[inv]

        keep = event_offset >= (cnts[inv] - B)
        s_from = s_from[keep]
        s_to = s_to[keep]
        s_time = s_time[keep]
        s_feat = s_feat[keep]

        n_kept = s_from.shape[0]
        _, inv2, cnts2 = torch.unique_consecutive(s_from.long(), return_inverse=True, return_counts=True)
        gs2 = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), cnts2.cumsum(0)[:-1]])
        kept_offset = torch.arange(n_kept, device=device) - gs2[inv2]

        write_idx = (self._ov_write_pos[s_from] + kept_offset) % B

        self._ov_ids[s_from, write_idx] = s_to.to(torch.int32)
        self._ov_times[s_from, write_idx] = s_time.to(torch.float64)
        self._ov_feats[s_from, write_idx] = s_feat

        ones = torch.ones(n_kept, dtype=torch.int64, device=device)
        self._ov_write_pos.scatter_add_(0, s_from.long(), ones)

    def _empty_neighbor_data(self, N, k, device):
        return NeighborData(
            neighbor_ids=torch.full((N, k), self.PADDING_ID, dtype=torch.int32, device=device),
            timestamps=torch.zeros(N, k, dtype=torch.float64, device=device),
            edge_feats=torch.zeros(N, k, self.edge_feat_dim, dtype=torch.float32, device=device),
            mask=torch.zeros(N, k, dtype=torch.bool, device=device),
        )

    def recent_2hop(self, nodes: Tensor, times: Tensor, k1: int, k2: int) -> NeighborData:
        """Get 1-hop and 2-hop neighbors for each query node."""
        if not self._frozen:
            self.freeze_csr()

        N = nodes.shape[0]

        # CUDA fused 2-hop path
        if (self._use_triton and HAS_CUDA_EXT and self._ov_ids is None
                and N > 0 and self.device.type == "cuda"):
            safe_nodes = nodes.clamp(min=0)
            is_padding = nodes < 0
            query_nodes = safe_nodes.masked_fill(is_padding, 0).to(torch.int64)
            query_times = times.to(torch.float64)

            (h1_ids, h1_times, h1_feats, h1_mask,
             h2_ids, h2_times, h2_feats, h2_mask) = cuda_temporal_recent_2hop(
                self._offsets, self._nbr_times, self._nbr_ids, self._nbr_feats,
                query_nodes, query_times, k1, k2,
            )

            # Zero out padding nodes
            if is_padding.any():
                pad1 = is_padding.unsqueeze(1).expand_as(h1_mask)
                h1_ids = h1_ids.masked_fill(pad1, self.PADDING_ID)
                h1_times = h1_times.masked_fill(pad1, 0.0)
                h1_feats = h1_feats.masked_fill(pad1.unsqueeze(-1), 0.0)
                h1_mask = h1_mask & ~pad1

                pad2 = is_padding.unsqueeze(1).unsqueeze(2).expand_as(h2_mask)
                h2_ids = h2_ids.masked_fill(pad2, self.PADDING_ID)
                h2_times = h2_times.masked_fill(pad2, 0.0)
                h2_feats = h2_feats.masked_fill(pad2.unsqueeze(-1), 0.0)
                h2_mask = h2_mask & ~pad2

            # Mask hop2 by hop1 validity
            hop1_valid = h1_mask.unsqueeze(-1)
            h2_mask = h2_mask & hop1_valid
            h2_ids = h2_ids.masked_fill(~h2_mask, self.PADDING_ID)
            h2_times = h2_times.masked_fill(~h2_mask, 0.0)
            h2_feats = h2_feats.masked_fill(~h2_mask.unsqueeze(-1), 0.0)

            hop1 = NeighborData(
                neighbor_ids=h1_ids, timestamps=h1_times,
                edge_feats=h1_feats, mask=h1_mask,
            )
            hop1.hop2_ids = h2_ids
            hop1.hop2_times = h2_times
            hop1.hop2_feats = h2_feats
            hop1.hop2_mask = h2_mask
            return hop1

        # Fallback: two separate recent() calls
        hop1 = self.recent(nodes, times, k1)

        N, K1 = hop1.neighbor_ids.shape
        flat_nbrs = hop1.neighbor_ids.reshape(-1)
        flat_nbr_times = hop1.timestamps.reshape(-1)

        hop2 = self.recent(flat_nbrs, flat_nbr_times, k2)

        hop2_ids = hop2.neighbor_ids.reshape(N, K1, k2)
        hop2_times = hop2.timestamps.reshape(N, K1, k2)
        hop2_feats = hop2.edge_feats.reshape(N, K1, k2, self.edge_feat_dim)
        hop2_mask = hop2.mask.reshape(N, K1, k2)

        hop1_valid = hop1.mask.unsqueeze(-1)
        hop2_mask = hop2_mask & hop1_valid
        hop2_ids = hop2_ids.masked_fill(~hop2_mask, self.PADDING_ID)
        hop2_times = hop2_times.masked_fill(~hop2_mask, 0.0)
        hop2_feats = hop2_feats.masked_fill(~hop2_mask.unsqueeze(-1), 0.0)

        hop1.hop2_ids = hop2_ids
        hop1.hop2_times = hop2_times
        hop1.hop2_feats = hop2_feats
        hop1.hop2_mask = hop2_mask
        return hop1

    def co_neighbors(self, src: Tensor, dst: Tensor, times: Tensor, k: int = 32) -> Tensor:
        """Compute co-occurrence counts between src and dst neighbor sets."""
        if not self._frozen:
            self.freeze_csr()

        # CUDA fast path
        if (self._use_triton and HAS_CUDA_EXT and self._ov_ids is None
                and src.shape[0] > 0 and self.device.type == "cuda"):
            return cuda_co_neighbor_count(
                self._offsets, self._nbr_times, self._nbr_ids,
                src, dst, times, k,
            )

        # Fallback
        B = src.shape[0]
        all_nbrs = self.recent(torch.cat([src, dst]), torch.cat([times, times]), k)
        src_ids = all_nbrs.neighbor_ids[:B]
        dst_ids = all_nbrs.neighbor_ids[B:]
        src_mask = all_nbrs.mask[:B]
        dst_mask = all_nbrs.mask[B:]

        eq = src_ids.unsqueeze(2) == dst_ids.unsqueeze(1)
        valid = src_mask.unsqueeze(2) & dst_mask.unsqueeze(1)
        return (eq & valid).any(dim=2).float().sum(dim=1)

    def reset(self):
        """Reset graph to empty state."""
        self._frozen = False
        self._build_from = []
        self._build_to = []
        self._build_time = []
        self._build_feat = []
        self._offsets = None
        self._nbr_ids = None
        self._nbr_times = None
        self._nbr_feats = None
        self._ov_ids = None
        self._ov_times = None
        self._ov_feats = None
        self._ov_write_pos = None
        self._num_edges = 0

    def snapshot(self) -> Snapshot:
        """Create a lightweight snapshot for eval restore. CSR is immutable."""
        self._ensure_overflow()
        return Snapshot(
            overflow_ids=self._ov_ids.clone(),
            overflow_times=self._ov_times.clone(),
            overflow_feats=self._ov_feats.clone(),
            overflow_write_pos=self._ov_write_pos.clone(),
            num_edges=self._num_edges,
        )

    def restore(self, snap: Snapshot):
        """Restore overflow state from snapshot."""
        self._ov_ids.copy_(snap.overflow_ids)
        self._ov_times.copy_(snap.overflow_times)
        self._ov_feats.copy_(snap.overflow_feats)
        self._ov_write_pos.copy_(snap.overflow_write_pos)
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
        device: str = "cuda",
        **kwargs,
    ) -> "TemporalGraph":
        """Construct a TemporalGraph by replaying all events and freezing."""
        d_edge = edge_feat.shape[1] if edge_feat is not None else 172
        graph = cls(num_nodes, edge_feat_dim=d_edge, device=device, **kwargs)
        chunk_size = 10000
        n = src.shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            s = src[start:end].to(device)
            d = dst[start:end].to(device)
            t = time[start:end].to(device)
            f = edge_feat[start:end].to(device) if edge_feat is not None else None
            graph.advance(s, d, t, f)
        graph.freeze_csr()
        return graph
