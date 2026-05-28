"""Tests for negative sampling strategies — especially HistoricalNegPool."""

import torch
import pytest

from tgengine.pipeline.negatives import (
    FixedNegative, HistoricalNegPool, HistoricalNegative,
    DyGLibHistoricalNegative, InductiveNegative, DyGLibInductiveNegative,
)
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.batch import RawBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.pipeline import DataPipeline


# ---------------------------------------------------------------------------
# HistoricalNegPool
# ---------------------------------------------------------------------------

def test_pool_fills_correctly():
    """Pool slots are filled in order for nodes with few interactions."""
    pool = HistoricalNegPool(num_nodes=10, pool_size=8, device="cpu")
    src = torch.tensor([0, 0, 0])
    dst = torch.tensor([1, 2, 3])
    pool.update(src, dst)

    row = pool._pool[0]
    # First 3 slots should be filled with 1, 2, 3 (in order)
    filled = row[row != pool.PADDING].tolist()
    assert set(filled) == {1, 2, 3}, f"expected {{1,2,3}}, got {filled}"
    assert pool._count[0].item() == 3


def test_pool_count_tracks_total():
    """_count tracks total edges seen per node across multiple update() calls."""
    pool = HistoricalNegPool(num_nodes=5, pool_size=4, device="cpu")
    pool.update(torch.tensor([0, 1, 0]), torch.tensor([2, 3, 4]))
    assert pool._count[0].item() == 2
    assert pool._count[1].item() == 1
    pool.update(torch.tensor([0]), torch.tensor([1]))
    assert pool._count[0].item() == 3


def test_pool_reservoir_bounded():
    """Pool never exceeds pool_size entries per node."""
    pool_size = 4
    pool = HistoricalNegPool(num_nodes=3, pool_size=pool_size, device="cpu")
    src = torch.zeros(20, dtype=torch.long)
    dst = torch.arange(1, 21)
    pool.update(src, dst)

    valid = (pool._pool[0] != pool.PADDING).sum().item()
    assert valid == pool_size, f"expected {pool_size} entries, got {valid}"


def test_pool_sample_returns_valid_negatives():
    """Sampled negatives must all be valid (in pool, not PADDING)."""
    pool = HistoricalNegPool(num_nodes=100, pool_size=16, device="cpu")
    src = torch.randint(0, 50, (200,))
    dst = torch.randint(50, 100, (200,))
    pool.update(src, dst)

    query_src = torch.randint(0, 50, (32,))
    neg = pool.sample(query_src)
    assert neg.shape == (32,)
    # All negatives must be node IDs in [0, 100)
    assert (neg >= 0).all() and (neg < 100).all()


def test_pool_sample_uniform_distribution():
    """Reservoir sampling should produce approximately uniform distribution.

    If node 0 has seen dst in {1, 2, 3} and pool_size >= 3, sampling from
    the pool should yield each with roughly equal frequency.
    """
    pool = HistoricalNegPool(num_nodes=10, pool_size=8, device="cpu")
    pool.update(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]))

    counts = {1: 0, 2: 0, 3: 0}
    n_trials = 3000
    src = torch.zeros(n_trials, dtype=torch.long)
    neg = pool.sample(src)
    for v in neg.tolist():
        if v in counts:
            counts[v] += 1

    total = sum(counts.values())
    assert total == n_trials, f"unexpected fallback sampling: {total} != {n_trials}"
    for v, c in counts.items():
        freq = c / n_trials
        assert 0.25 < freq < 0.45, f"node {v} freq={freq:.3f}, expected ~0.333"


def test_pool_no_history_fallback():
    """Nodes with no history should fall back to random in [0, num_nodes)."""
    pool = HistoricalNegPool(num_nodes=100, pool_size=8, device="cpu")
    # No update() calls — pool is empty
    neg = pool.sample(torch.tensor([0, 1, 2, 3]))
    assert neg.shape == (4,)
    assert (neg >= 0).all() and (neg < 100).all()


def test_pool_reservoir_coverage_improves_with_size():
    """Larger pool_size → higher fraction of all-time interactions covered."""
    n_edges = 100
    src = torch.zeros(n_edges, dtype=torch.long)
    dst = torch.arange(1, n_edges + 1)

    pool_small = HistoricalNegPool(num_nodes=n_edges + 2, pool_size=10, device="cpu")
    pool_large = HistoricalNegPool(num_nodes=n_edges + 2, pool_size=50, device="cpu")

    pool_small.update(src, dst)
    pool_large.update(src, dst)

    small_valid = (pool_small._pool[0] != pool_small.PADDING).sum().item()
    large_valid = (pool_large._pool[0] != pool_large.PADDING).sum().item()

    assert small_valid == 10
    assert large_valid == 50


# ---------------------------------------------------------------------------
# co_occurrence fusion in DataPipeline
# ---------------------------------------------------------------------------

def test_co_occur_fusion_matches_graph_method():
    """DataPipeline._co_occur_from_neighbors should match graph.co_neighbors()."""
    device = "cpu"
    graph = TemporalGraph(num_nodes=20, buffer_size=8, edge_feat_dim=4, device=device)

    # Create edges so nodes 0,1 share some neighbors
    graph.advance(
        torch.tensor([0, 0, 1, 1, 2]),
        torch.tensor([5, 6, 5, 7, 8]),
        torch.tensor([1.0, 2.0, 1.5, 2.5, 3.0], dtype=torch.float64),
        torch.randn(5, 4),
    )

    spec = GatherSpec(neighbors=NeighborSpec(k=8, strategy="recency"), co_occurrence=True)
    pipeline = DataPipeline(spec, graph)

    src = torch.tensor([0, 0, 1])
    dst = torch.tensor([1, 2, 2])
    t = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    raw = RawBatch(src=src, dst=dst, time=t, neg=torch.tensor([3, 3, 3]))

    # Get co_occur via fused pipeline
    batch = pipeline.prepare(raw)
    fused_co = batch.co_occurrence

    # Get co_occur via direct graph method
    direct_co = graph.co_neighbors(src, dst, t, k=8)

    assert torch.allclose(fused_co.float(), direct_co.float()), (
        f"co_occurrence mismatch:\n  fused:  {fused_co}\n  direct: {direct_co}"
    )


def test_co_occur_fusion_no_extra_kernel_call():
    """Verify prepare() uses _co_occur_from_neighbors when src/dst nbrs are available."""
    from tgengine.core.batch import NeighborData
    import torch

    B, k, d = 4, 8, 4
    src_ids = torch.randint(0, 20, (B, k), dtype=torch.int32)
    dst_ids = torch.randint(0, 20, (B, k), dtype=torch.int32)
    src_mask = torch.ones(B, k, dtype=torch.bool)
    dst_mask = torch.ones(B, k, dtype=torch.bool)

    src_nbrs = NeighborData(src_ids, torch.zeros(B, k, dtype=torch.float64),
                            torch.zeros(B, k, d), src_mask)
    dst_nbrs = NeighborData(dst_ids, torch.zeros(B, k, dtype=torch.float64),
                            torch.zeros(B, k, d), dst_mask)

    result = DataPipeline._co_occur_from_neighbors(src_nbrs, dst_nbrs)
    assert result.shape == (B,)
    assert (result >= 0).all()
    # With random IDs in [0, 20) and k=8, some overlap is expected
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# HistoricalNegative (per-src reservoir, fast approximate)
# ---------------------------------------------------------------------------

def test_historical_negative_update_and_sample():
    """HistoricalNegative.update() + sample() returns valid historical nodes."""
    num_nodes = 50
    strategy = HistoricalNegative(num_nodes=num_nodes, pool_size=64, device="cpu")

    src = torch.tensor([0, 0, 0, 1, 1])
    dst = torch.tensor([10, 20, 30, 40, 41])
    strategy.update(src, dst)

    neg = strategy.sample(src[:3], dst[:3], torch.zeros(3), graph=None)
    assert neg.shape == (3,)
    assert set(neg.tolist()).issubset({10, 20, 30}), f"unexpected negatives: {neg.tolist()}"


def test_historical_negative_uniform_distribution():
    """HistoricalNegative samples uniformly from full history."""
    num_nodes = 200
    pool_size = 100
    strategy = HistoricalNegative(num_nodes=num_nodes, pool_size=pool_size, device="cpu")

    src = torch.zeros(3, dtype=torch.long)
    dst = torch.tensor([1, 2, 3])
    strategy.update(src, dst)

    counts = {1: 0, 2: 0, 3: 0}
    n_trials = 3000
    neg = strategy.sample(torch.zeros(n_trials, dtype=torch.long),
                          torch.zeros(n_trials, dtype=torch.long),
                          torch.zeros(n_trials), graph=None)
    for v in neg.tolist():
        if v in counts:
            counts[v] += 1

    total = sum(counts.values())
    assert total == n_trials, f"unexpected fallback sampling: {total} != {n_trials}"
    for v, c in counts.items():
        freq = c / n_trials
        assert 0.25 < freq < 0.45, f"node {v} freq={freq:.3f}, expected ~0.333"


def test_historical_negative_covers_beyond_ring_buffer():
    """HistoricalNegative samples from the FULL history."""
    num_nodes = 150
    k_recent = 32
    pool_size = 128
    strategy = HistoricalNegative(num_nodes=num_nodes, pool_size=pool_size, device="cpu")

    n_interactions = 100
    src = torch.zeros(n_interactions, dtype=torch.long)
    dst = torch.arange(1, n_interactions + 1)
    strategy.update(src, dst)

    n_trials = 5000
    neg = strategy.sample(torch.zeros(n_trials, dtype=torch.long),
                          torch.zeros(n_trials, dtype=torch.long),
                          torch.zeros(n_trials), graph=None)

    seen = set(neg.tolist())
    old_neighbors = set(range(1, n_interactions - k_recent + 1))
    overlap = seen & old_neighbors
    assert len(overlap) > 0


# ---------------------------------------------------------------------------
# FixedNegative (TGB-style fixed neg lists)
# ---------------------------------------------------------------------------

def test_fixed_negative_lookup():
    """FixedNegative.sample() returns correct rows from neg_lists via edge_indices."""
    neg_lists = torch.arange(100).reshape(10, 10)  # 10 edges, 10 negs each
    strategy = FixedNegative(neg_lists)

    # Sample edge 0, 3, 7
    edge_indices = torch.tensor([0, 3, 7])
    neg = strategy.sample(
        torch.zeros(3), torch.zeros(3), torch.zeros(3), graph=None,
        edge_indices=edge_indices,
    )
    assert neg.shape == (3, 10)
    assert torch.equal(neg[0], neg_lists[0])  # edge 0 → row 0
    assert torch.equal(neg[1], neg_lists[3])  # edge 3 → row 3
    assert torch.equal(neg[2], neg_lists[7])  # edge 7 → row 7


def test_fixed_negative_requires_edge_indices():
    """FixedNegative.sample() raises ValueError when edge_indices is None."""
    neg_lists = torch.arange(100).reshape(10, 10)
    strategy = FixedNegative(neg_lists)
    with pytest.raises(ValueError, match="edge_indices"):
        strategy.sample(torch.zeros(3), torch.zeros(3), torch.zeros(3), graph=None)


# ---------------------------------------------------------------------------
# HistoricalNegative (DyGLib-compatible, global edge set)
# ---------------------------------------------------------------------------

import numpy as np


def test_dyglib_historical_basic():
    """DyGLibHistoricalNegative returns dst from historical edge set."""
    src = np.array([0, 1, 2, 0, 1, 3, 2, 0, 4, 1], dtype=np.int64)
    dst = np.array([1, 2, 3, 2, 0, 4, 1, 3, 0, 3], dtype=np.int64)
    times = np.arange(10, dtype=np.float64)

    strategy = DyGLibHistoricalNegative(src, dst, times, seed=42)

    # Query at time=8 means historical edges are those with t < 8 (indices 0..7)
    batch_src = torch.tensor([0, 1])
    batch_dst = torch.tensor([4, 3])  # current batch edges: (0,4), (1,3)
    batch_time = torch.tensor([8.0, 8.5])

    neg = strategy.sample(batch_src, batch_dst, batch_time, graph=None)
    assert neg.shape == (2,)
    # Neg dst should come from historical edges' dst values (excluding batch)
    historical_before_8 = set(zip(src[:8].tolist(), dst[:8].tolist()))
    current = {(0, 4), (1, 3)}
    valid_dst = {e[1] for e in (historical_before_8 - current)}
    assert all(int(v) in valid_dst for v in neg.tolist()), \
        f"neg {neg.tolist()} not in valid_dst {valid_dst}"


def test_dyglib_historical_fallback():
    """DyGLibHistoricalNegative falls back to random when not enough historical edges."""
    # Only 2 edges total
    src = np.array([0, 1], dtype=np.int64)
    dst = np.array([1, 0], dtype=np.int64)
    times = np.array([1.0, 2.0])

    strategy = DyGLibHistoricalNegative(src, dst, times, seed=42)

    # Query at time=1.5: only 1 historical edge (0,1). Need 5 samples.
    batch_src = torch.tensor([0, 1, 2, 3, 4])
    batch_dst = torch.tensor([2, 3, 4, 0, 1])
    batch_time = torch.tensor([1.5, 1.5, 1.5, 1.5, 1.5])

    neg = strategy.sample(batch_src, batch_dst, batch_time, graph=None)
    assert neg.shape == (5,)


# ---------------------------------------------------------------------------
# DyGLibInductiveNegative (DyGLib-compatible, edge-level inductive)
# ---------------------------------------------------------------------------

def test_dyglib_inductive_basic():
    """DyGLibInductiveNegative returns dst from new edge pairs (not in training)."""
    src = np.array([0, 1, 2, 0, 1, 3, 2, 0, 4, 1, 0, 2], dtype=np.int64)
    dst = np.array([1, 2, 3, 2, 0, 4, 1, 3, 0, 3, 4, 4], dtype=np.int64)
    times = np.arange(12, dtype=np.float64)

    # Training period: edges 0-7 (last_observed_time=7)
    last_observed_time = 7.0
    strategy = DyGLibInductiveNegative(src, dst, times, last_observed_time=last_observed_time, seed=42)

    # Query at time=10: historical edges = edges 0..9
    # observed_edges = unique pairs in [0, 7] = edges 0..7
    # inductive candidates = (edges 0..9 unique pairs) - observed - current_batch
    batch_src = torch.tensor([0, 2])
    batch_dst = torch.tensor([4, 4])  # these are edges 10, 11
    batch_time = torch.tensor([10.0, 11.0])

    neg = strategy.sample(batch_src, batch_dst, batch_time, graph=None)
    assert neg.shape == (2,)


def test_dyglib_inductive_no_candidates_fallback():
    """DyGLibInductiveNegative falls back when no inductive edges exist."""
    src = np.array([0, 1], dtype=np.int64)
    dst = np.array([1, 0], dtype=np.int64)
    times = np.array([1.0, 2.0])

    # last_observed_time covers everything
    strategy = DyGLibInductiveNegative(src, dst, times, last_observed_time=3.0, seed=42)

    batch_src = torch.tensor([0, 1])
    batch_dst = torch.tensor([1, 0])
    batch_time = torch.tensor([2.5, 2.5])

    # No inductive edges (all historical = observed), should fallback to random
    neg = strategy.sample(batch_src, batch_dst, batch_time, graph=None)
    assert neg.shape == (2,)

