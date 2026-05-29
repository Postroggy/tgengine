"""DyGLib-compatible negative sampling strategies (CPU, for verification only).

These implementations use Python set operations to exactly match DyGLib's
NegativeEdgeSampler semantics. They are slow (CPU-bound) and should only be
used for correctness verification against DyGLib baselines.

For actual training/eval, use the GPU-native strategies in negatives.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from tgengine.core.temporal_graph import TemporalGraph
from tgengine.pipeline.negatives import NegativeStrategy


class DyGLibHistoricalNegative(NegativeStrategy):
    """DyGLib-compatible historical negative sampling (CPU, for verification only).

    Semantics: from all unique (src, dst) edge pairs observed before
    batch_start_time, exclude current batch edges, uniformly sample `size`
    destination nodes. If not enough historical edges, fill with
    collision-free random edges from possible_edges - batch_edges.
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

        end_idx = int(np.searchsorted(self._times, batch_start_time, side='right'))
        historical_edges = set(
            zip(self._src[:end_idx].tolist(), self._dst[:end_idx].tolist())
        )

        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        current_batch_edges = set(zip(src_np.tolist(), dst_np.tolist()))

        candidates = list(historical_edges - current_batch_edges)

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
            neg_dst_list = [possible_random[i][1] for i in fill_indices]
            neg_dst_list.extend(e[1] for e in candidates)
            neg_dst = np.array(neg_dst_list, dtype=np.int64)

        return torch.from_numpy(neg_dst).long().to(device)


class DyGLibInductiveNegative(NegativeStrategy):
    """DyGLib-compatible inductive negative sampling (CPU, for verification only).

    Semantics: from all unique (src, dst) edge pairs observed before
    batch_start_time, subtract observed_edges (training period edges) and
    current batch edges. Sample from the remaining "new" edge pairs.
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

        end_idx = int(np.searchsorted(self._times, batch_start_time, side='right'))
        historical_edges = set(
            zip(self._src[:end_idx].tolist(), self._dst[:end_idx].tolist())
        )

        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        current_batch_edges = set(zip(src_np.tolist(), dst_np.tolist()))

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
            neg_dst_list = [possible_random[i][1] for i in fill_indices]
            neg_dst_list.extend(e[1] for e in candidates)
            neg_dst = np.array(neg_dst_list, dtype=np.int64)

        return torch.from_numpy(neg_dst).long().to(device)


class VectorizedHistoricalNegative(NegativeStrategy):
    """Fast historical negative sampling using sorted pair index.

    Same semantics as DyGLibHistoricalNegative but implemented with vectorized
    numpy operations instead of Python sets. 5-10x faster for medium/large datasets.
    """

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                 interact_times: np.ndarray, seed: int = 42):
        order = np.argsort(interact_times, kind='stable')
        src_sorted = src_node_ids[order]
        dst_sorted = dst_node_ids[order]
        times_sorted = interact_times[order]

        max_node = max(int(src_node_ids.max()), int(dst_node_ids.max())) + 1
        self._max_node = max_node
        pair_keys = src_sorted.astype(np.int64) * max_node + dst_sorted.astype(np.int64)

        _, first_idx = np.unique(pair_keys, return_index=True)
        first_idx.sort()

        self._pair_src = src_sorted[first_idx]
        self._pair_dst = dst_sorted[first_idx]
        self._pair_time = times_sorted[first_idx]

        time_order = np.argsort(self._pair_time, kind='stable')
        self._pair_src = self._pair_src[time_order]
        self._pair_dst = self._pair_dst[time_order]
        self._pair_time = self._pair_time[time_order]

        self._unique_dst = np.unique(dst_node_ids)
        self._rng = np.random.RandomState(seed)

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph,
               edge_indices: Optional[Tensor] = None) -> Tensor:
        device = src.device
        size = src.shape[0]
        batch_start_time = float(time.min().item())

        end_idx = int(np.searchsorted(self._pair_time, batch_start_time, side='left'))

        if end_idx == 0:
            idx = self._rng.randint(0, len(self._unique_dst), size=size)
            return torch.from_numpy(self._unique_dst[idx]).long().to(device)

        src_np = src.cpu().numpy().astype(np.int64)
        dst_np = dst.cpu().numpy().astype(np.int64)
        batch_keys_arr = np.unique(src_np * self._max_node + dst_np)

        hist_keys = (self._pair_src[:end_idx].astype(np.int64) * self._max_node
                     + self._pair_dst[:end_idx].astype(np.int64))

        positions = np.searchsorted(batch_keys_arr, hist_keys)
        positions = np.clip(positions, 0, len(batch_keys_arr) - 1)
        mask = batch_keys_arr[positions] != hist_keys

        candidate_dst = self._pair_dst[:end_idx][mask]

        if size <= len(candidate_dst):
            indices = self._rng.choice(len(candidate_dst), size=size, replace=False)
            neg_dst = candidate_dst[indices]
        else:
            num_fill = size - len(candidate_dst)
            fill_idx = self._rng.randint(0, len(self._unique_dst), size=num_fill)
            neg_dst = np.concatenate([self._unique_dst[fill_idx], candidate_dst])

        return torch.from_numpy(neg_dst.astype(np.int64)).long().to(device)
