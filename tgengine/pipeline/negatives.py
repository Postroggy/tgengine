from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
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
    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        """Sample negative destination nodes.

        Args:
            src: (B,) source nodes.
            dst: (B,) positive destination nodes.
            time: (B,) timestamps.
            graph: the temporal graph (for historical neg).
            edge_indices: (B,) global edge indices (for TGB fixed neg lookup).

        Returns:
            Tensor of shape (B,) or (B, N_neg) negative node IDs.
        """
        ...

    def update(self, src: Tensor, dst: Tensor) -> None:
        """Ingest new edges (called alongside graph.advance()). No-op by default."""


class RandomNegative(NegativeStrategy):
    """Uniform random negative sampling. O(1), GPU-native.

    Args:
        num_nodes: total node count (used as fallback when valid_dst_nodes is None).
        valid_dst_nodes: optional 1-D LongTensor of valid dst node IDs.
            When provided, negatives are sampled from this set instead of [0, num_nodes),
            matching DyGLib's behavior of sampling only from dataset-occurring dst nodes.
    """

    def __init__(self, num_nodes: int, valid_dst_nodes: Optional[Tensor] = None):
        self.num_nodes = num_nodes
        self.valid_dst_nodes = valid_dst_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        if self.valid_dst_nodes is not None:
            nodes = self.valid_dst_nodes.to(src.device)
            idx = torch.randint(0, len(nodes), (src.shape[0],), device=src.device)
            return nodes[idx]
        return torch.randint(0, self.num_nodes, (src.shape[0],), device=src.device)


class HistoricalNegative(NegativeStrategy):
    """Per-src GPU-resident historical negative sampling.

    Uses reservoir sampling so every historical (src, dst) edge has equal
    probability of being in the pool. Sampling is O(B) GPU gather.
    Call update(src, dst) alongside graph.advance() for every training batch.
    """

    def __init__(self, num_nodes: int, pool_size: int = 512, device: str = "cuda"):
        self._pool = HistoricalNegPool(num_nodes, pool_size=pool_size, device=device)
        self.num_nodes = num_nodes

    def update(self, src: Tensor, dst: Tensor) -> None:
        self._pool.update(src, dst)

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        return self._pool.sample(src)


class DyGLibHistoricalNegative(NegativeStrategy):
    """DyGLib-compatible historical negative sampling (CPU, for verification only).

    Semantics: from all unique (src, dst) edge pairs observed before
    batch_start_time, exclude current batch edges, uniformly sample `size`
    destination nodes. If not enough historical edges, fill with
    collision-free random edges from possible_edges - batch_edges.

    WARNING: Uses CPU set operations — slow. Only use for correctness
    verification against DyGLib. For actual training/eval use HistoricalNegative.
    """

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                 interact_times: np.ndarray, seed: int = 42):
        order = np.argsort(interact_times, kind='stable')
        self._src = src_node_ids[order]
        self._dst = dst_node_ids[order]
        self._times = interact_times[order]
        self._n = len(order)

        self._unique_src = np.unique(src_node_ids)
        self._unique_dst = np.unique(dst_node_ids)
        self._possible_edges = set(
            (int(s), int(d)) for s in self._unique_src for d in self._unique_dst
        )
        self._rng = np.random.RandomState(seed)

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        device = src.device
        size = src.shape[0]
        batch_start_time = float(time.min().item())
        batch_end_time = float(time.max().item())

        # Historical edges: unique pairs in [earliest, batch_start_time]
        end_idx = int(np.searchsorted(self._times, batch_start_time, side='right'))
        historical_edges = set(
            zip(self._src[:end_idx].tolist(), self._dst[:end_idx].tolist())
        )

        # Current batch edges
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        current_batch_edges = set(zip(src_np.tolist(), dst_np.tolist()))

        # Candidates
        candidates = list(historical_edges - current_batch_edges)

        if size <= len(candidates):
            indices = self._rng.choice(len(candidates), size=size, replace=False)
            neg_dst = np.array([candidates[i][1] for i in indices], dtype=np.int64)
        else:
            # DyGLib order: random fill first, then historical candidates
            num_fill = size - len(candidates)
            possible_random = list(self._possible_edges - current_batch_edges)
            fill_indices = self._rng.choice(
                len(possible_random), size=num_fill,
                replace=len(possible_random) < num_fill,
            )
            neg_dst_list = [possible_random[i][1] for i in fill_indices]
            neg_dst_list.extend(e[1] for e in candidates)
            neg_dst = np.array(neg_dst_list, dtype=np.int64)

        return torch.from_numpy(neg_dst).long().to(device)


class InductiveNegative(NegativeStrategy):
    """GPU-resident inductive negative sampling from unseen nodes.

    Samples from nodes that never appeared during training. O(B) GPU randint.
    """

    def __init__(self, inductive_nodes: Tensor):
        self.inductive_nodes = inductive_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        n = src.shape[0]
        rand_idx = torch.randint(0, len(self.inductive_nodes), (n,), device=src.device)
        return self.inductive_nodes.to(src.device)[rand_idx]


class DyGLibInductiveNegative(NegativeStrategy):
    """DyGLib-compatible inductive negative sampling (CPU, for verification only).

    Semantics: from all unique (src, dst) edge pairs observed before
    batch_start_time, subtract observed_edges (training period edges) and
    current batch edges. Sample from the remaining "new" edge pairs.

    WARNING: Uses CPU set operations — slow. Only use for correctness
    verification against DyGLib. For actual training/eval use InductiveNegative.
    """

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                 interact_times: np.ndarray, last_observed_time: float,
                 seed: int = 42):
        order = np.argsort(interact_times, kind='stable')
        self._src = src_node_ids[order]
        self._dst = dst_node_ids[order]
        self._times = interact_times[order]

        self._unique_src = np.unique(src_node_ids)
        self._unique_dst = np.unique(dst_node_ids)
        self._possible_edges = set(
            (int(s), int(d)) for s in self._unique_src for d in self._unique_dst
        )

        # observed_edges = unique edges in [earliest, last_observed_time]
        obs_end = int(np.searchsorted(self._times, last_observed_time, side='right'))
        self._observed_edges = set(
            zip(self._src[:obs_end].tolist(), self._dst[:obs_end].tolist())
        )
        self._rng = np.random.RandomState(seed)

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        device = src.device
        size = src.shape[0]
        batch_start_time = float(time.min().item())
        batch_end_time = float(time.max().item())

        # Historical edges up to batch_start_time
        end_idx = int(np.searchsorted(self._times, batch_start_time, side='right'))
        historical_edges = set(
            zip(self._src[:end_idx].tolist(), self._dst[:end_idx].tolist())
        )

        # Current batch edges
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        current_batch_edges = set(zip(src_np.tolist(), dst_np.tolist()))

        # Inductive candidates: historical - observed - current_batch
        candidates = list(historical_edges - self._observed_edges - current_batch_edges)

        if size <= len(candidates):
            indices = self._rng.choice(len(candidates), size=size, replace=False)
            neg_dst = np.array([candidates[i][1] for i in indices], dtype=np.int64)
        else:
            num_fill = size - len(candidates)
            possible_random = list(self._possible_edges - current_batch_edges)
            fill_indices = self._rng.choice(
                len(possible_random), size=num_fill,
                replace=len(possible_random) < num_fill,
            )
            # DyGLib order: random fill first, then inductive candidates
            neg_dst_list = [possible_random[i][1] for i in fill_indices]
            neg_dst_list.extend(e[1] for e in candidates)
            neg_dst = np.array(neg_dst_list, dtype=np.int64)

        return torch.from_numpy(neg_dst).long().to(device)


class FixedNegative(NegativeStrategy):
    """Fixed negative lists (for TGB evaluation).

    neg_lists: (N_total, N_neg) pre-defined negative node IDs.
    Each row corresponds to a global edge index. Uses edge_indices
    from RawBatch to look up the correct negative candidates.
    """

    def __init__(self, neg_lists: Tensor):
        self.neg_lists = neg_lists

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        if edge_indices is None:
            raise ValueError("FixedNegative requires edge_indices in RawBatch")
        neg = self.neg_lists[edge_indices.long()]  # (B, N_neg)
        return neg.to(src.device)


class CollisionFreeNegative(NegativeStrategy):
    """Random negative sampling that avoids edges observed so far.

    Matches DyGLib's random_sample_with_collision_check semantics:
    sampled (src, neg_dst) pairs must NOT appear in the set of observed edges.

    GPU-friendly implementation: sample, check collisions, resample collisions.
    Maintains a hash set of observed edges (on CPU for O(1) lookup).

    Args:
        num_nodes: total node count.
        valid_src_nodes: if provided, only sample from these src nodes.
        valid_dst_nodes: if provided, only sample from these dst nodes.
        max_retries: max resample rounds for collision resolution.
    """

    def __init__(
        self,
        num_nodes: int,
        valid_src_nodes: Optional[Tensor] = None,
        valid_dst_nodes: Optional[Tensor] = None,
        max_retries: int = 10,
    ):
        self.num_nodes = num_nodes
        self.valid_dst_nodes = valid_dst_nodes
        self.max_retries = max_retries
        self._observed: set[tuple[int, int]] = set()

    def update(self, src: Tensor, dst: Tensor) -> None:
        """Register observed edges (call alongside graph.advance)."""
        for s, d in zip(src.cpu().tolist(), dst.cpu().tolist()):
            self._observed.add((s, d))

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        B = src.shape[0]
        device = src.device

        if self.valid_dst_nodes is not None:
            pool = self.valid_dst_nodes.to(device)
            neg = pool[torch.randint(0, len(pool), (B,), device=device)]
        else:
            neg = torch.randint(0, self.num_nodes, (B,), device=device)

        # Collision check + resample
        src_cpu = src.cpu().tolist()
        for _ in range(self.max_retries):
            neg_cpu = neg.cpu().tolist()
            collisions = []
            for i in range(B):
                if (src_cpu[i], neg_cpu[i]) in self._observed:
                    collisions.append(i)
            if not collisions:
                break
            # Resample collisions
            n_coll = len(collisions)
            coll_idx = torch.tensor(collisions, device=device)
            if self.valid_dst_nodes is not None:
                new = pool[torch.randint(0, len(pool), (n_coll,), device=device)]
            else:
                new = torch.randint(0, self.num_nodes, (n_coll,), device=device)
            neg[coll_idx] = new

        return neg
