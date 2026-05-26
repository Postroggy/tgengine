from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from tgengine.core.batch import NeighborData
from tgengine.core.temporal_graph import TemporalGraph


class HistoricalNegPool:
    """Per-node bounded pool for historically-correct negative sampling.

    Uses reservoir sampling so every historical (src, dst) edge has equal
    probability of being in the pool — semantically equivalent to sampling
    from the full history, but memory is O(num_nodes * pool_size) and
    sampling is O(B) GPU gather regardless of dataset size.

    Usage:
        pool = HistoricalNegPool(num_nodes, pool_size=512, device="cuda")
        # Call alongside graph.advance() for every training batch:
        pool.update(src, dst)
        # Sample negatives:
        neg = pool.sample(src)
    """

    PADDING: int = -1

    def __init__(self, num_nodes: int, pool_size: int = 512, device: str = "cuda"):
        self.num_nodes = num_nodes
        self.pool_size = pool_size
        self.device = torch.device(device)

        # Pool: (num_nodes, pool_size) — PADDING means empty slot
        self._pool = torch.full(
            (num_nodes, pool_size), self.PADDING, dtype=torch.int32, device=self.device
        )
        # Total edges seen per node (for reservoir sampling probability)
        self._count = torch.zeros(num_nodes, dtype=torch.int64, device=self.device)

    def update(self, src: Tensor, dst: Tensor):
        """Add (src_i → dst_i) edges to the pool via reservoir sampling.

        For each edge that is the n-th edge seen for its src node:
          - n <= pool_size: write to slot n-1 (fill phase)
          - n > pool_size: with probability pool_size/n, replace a random slot
        This guarantees uniform coverage across ALL historical edges.
        """
        n = src.shape[0]
        if n == 0:
            return
        device = self.device

        # Sort by src for intra-batch grouping (stable to preserve time order)
        perm = src.long().argsort(stable=True)
        s_src = src[perm].long()
        s_dst = dst[perm].long()

        # Intra-batch position: 0, 1, 2, ... within each src group
        # Use cummax trick: set group boundaries, propagate forward
        if n > 1:
            change = torch.cat([
                torch.ones(1, dtype=torch.bool, device=device),
                s_src[1:] != s_src[:-1],
            ])
        else:
            change = torch.ones(1, dtype=torch.bool, device=device)

        group_start = torch.zeros(n, dtype=torch.int64, device=device)
        change_pos = torch.where(change)[0]
        group_start[change_pos] = change_pos
        group_start, _ = group_start.cummax(0)
        intra_pos = torch.arange(n, device=device) - group_start  # (n,) 0,1,2,...

        # Global 1-indexed count for each edge after it is processed
        counts_before = self._count[s_src]
        edge_n = counts_before + intra_pos + 1

        # Reservoir sampling: decide whether to write
        fill = edge_n <= self.pool_size
        rand_p = torch.rand(n, device=device)
        replace = (~fill) & (rand_p < (self.pool_size / edge_n.float()))
        accept = fill | replace

        write_slot = torch.where(
            fill,
            (edge_n - 1).clamp(max=self.pool_size - 1),          # sequential fill
            torch.randint(0, self.pool_size, (n,), device=device), # random replace
        )

        if accept.any():
            acc_src = s_src[accept]
            acc_dst = s_dst[accept]
            acc_slot = write_slot[accept]
            # index_put_: last write wins on (src, slot) collisions — acceptable
            self._pool[acc_src, acc_slot] = acc_dst.to(torch.int32)

        ones = torch.ones(n, dtype=torch.int64, device=device)
        self._count.scatter_add_(0, s_src, ones)

    def sample(self, src: Tensor) -> Tensor:
        """Sample one random historical negative per src. O(B) GPU gather.

        Uniformly draws from all valid pool slots for each src node.
        Falls back to random sampling for nodes with no history.
        """
        device = src.device
        pool_rows = self._pool[src.long()]              # (B, pool_size)
        valid = pool_rows != self.PADDING               # (B, pool_size)
        valid_counts = valid.sum(dim=1)                 # (B,)

        rand_idx = (torch.rand(src.shape[0], device=device) * valid_counts.float()).long()
        rand_idx = rand_idx.clamp(max=self.pool_size - 1)

        neg = pool_rows.gather(1, rand_idx.unsqueeze(1)).squeeze(1).long()

        no_history = valid_counts == 0
        if no_history.any():
            neg = neg.clone()
            neg[no_history] = torch.randint(
                0, self.num_nodes, (int(no_history.sum()),), device=device
            )
        return neg


class NegativeStrategy(ABC):
    """Base class for negative sampling strategies."""

    @abstractmethod
    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        """Sample negative destination nodes.

        Args:
            src: (B,) source nodes.
            dst: (B,) positive destination nodes.
            time: (B,) timestamps.
            graph: the temporal graph (for historical neg).

        Returns:
            Tensor of shape (B,) or (B, N_neg) negative node IDs.
        """
        ...


class RandomNegative(NegativeStrategy):
    """Uniform random negative sampling. O(1), GPU-native."""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        return torch.randint(0, self.num_nodes, (src.shape[0],), device=src.device)


class HistoricalNegative(NegativeStrategy):
    """Sample negatives from src's historical neighbors (ring buffer).

    Complexity: O(B * k) regardless of dataset size.  Always uses
    graph.recent() — call sample_from_neighbors() inside DataPipeline
    to avoid the redundant src query.
    """

    def __init__(self, num_nodes: int, k: int = 32):
        self.num_nodes = num_nodes
        self.k = k

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        """Standalone sample — makes a separate graph.recent(src) call.

        Prefer DataPipeline.prepare_with_hist_neg() to avoid querying
        src neighbors twice.
        """
        nbrs = graph.recent(src, time, k=self.k)
        return self.sample_from_neighbors(nbrs, self.num_nodes)

    @staticmethod
    def sample_from_neighbors(src_nbrs: NeighborData, num_nodes: int) -> Tensor:
        """Vectorized neg selection from already-queried src neighbors.

        No additional graph.recent() call needed — src_nbrs come from
        the pipeline's existing fused query.

        Args:
            src_nbrs: NeighborData(B, K) — src neighbors already obtained.
            num_nodes: fallback upper bound for random sampling.

        Returns:
            (B,) tensor of negative node IDs.
        """
        device = src_nbrs.neighbor_ids.device
        B, K = src_nbrs.neighbor_ids.shape
        valid_counts = src_nbrs.mask.sum(dim=1)  # (B,)

        # Random index in [0, valid_count) per row — vectorized
        rand_idx = (torch.rand(B, device=device) * valid_counts.float()).long()
        rand_idx = rand_idx.clamp(max=K - 1)  # (B,)

        neg = src_nbrs.neighbor_ids.gather(1, rand_idx.unsqueeze(1)).squeeze(1).long()

        # Nodes with no history: fall back to random
        no_history = valid_counts == 0
        if no_history.any():
            neg[no_history] = torch.randint(
                0, num_nodes, (int(no_history.sum()),), device=device
            )
        return neg


class InductiveNegative(NegativeStrategy):
    """Sample negatives from nodes never seen in training (inductive split).

    Nodes are those that have total degree zero in the training graph.
    """

    def __init__(self, inductive_nodes: Tensor):
        self.inductive_nodes = inductive_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        n = src.shape[0]
        rand_idx = torch.randint(0, len(self.inductive_nodes), (n,), device=src.device)
        return self.inductive_nodes.to(src.device)[rand_idx]


class FixedNegative(NegativeStrategy):
    """Fixed negative lists (for TGB evaluation)."""

    def __init__(self, neg_lists: Tensor):
        self.neg_lists = neg_lists

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        raise NotImplementedError("FixedNegative requires edge_indices in RawBatch")
