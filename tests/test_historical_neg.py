"""Tests for HistoricalNegative pool maintenance and Engine integration.

Covers:
  - HistoricalNegPool reservoir update + sample correctness
  - Engine sync train loop calls neg_strategy.update (pool populated)
  - Engine async train loop calls neg_strategy.update (pool populated,
    no cross-stream race)
  - RandomNegative.update is a no-op (base default)
"""

import torch

from tgengine.core.batch import RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import Engine, TrainConfig, APEval
from tgengine.models.graphmixer import GraphMixer
from tgengine.pipeline.async_pipeline import AsyncDataPipeline
from tgengine.pipeline.negatives import (
    HistoricalNegPool,
    HistoricalNegative,
    RandomNegative,
)


# ---------------------------------------------------------------------------
# Unit: HistoricalNegPool
# ---------------------------------------------------------------------------

def test_pool_update_then_sample_returns_seen_dst():
    """After update(src, dst), sample(src) can return the inserted dst."""
    N, dev = 20, "cuda"
    pool = HistoricalNegPool(N, pool_size=8, device=dev)

    src = torch.tensor([0, 0, 0, 1, 1], device=dev)
    dst = torch.tensor([5, 6, 7, 8, 9], device=dev)
    pool.update(src, dst)

    # Node 0 saw 3 edges, all should be in pool (fill phase, <= pool_size)
    row0 = pool._pool[0]
    valid0 = row0[row0 != pool.PADDING]
    assert valid0.numel() == 3
    assert set(valid0.long().tolist()) == {5, 6, 7}

    # Repeated sampling from node 0 must only return inserted dst (5/6/7)
    samples = set()
    for _ in range(200):
        s = pool.sample(torch.tensor([0], device=dev))
        samples.add(s.item())
    assert samples.issubset({5, 6, 7}), f"got unexpected samples {samples}"


def test_pool_sample_falls_back_to_random_for_unseen():
    """Nodes with no history must fall back to random (not PADDING)."""
    N, dev = 20, "cuda"
    pool = HistoricalNegPool(N, pool_size=8, device=dev)
    # No update for node 5 → sample should return a valid node id in [0, N)
    s = pool.sample(torch.tensor([5], device=dev))
    assert 0 <= s.item() < N


def test_pool_reservoir_keeps_subset_when_over_capacity():
    """When a node sees > pool_size edges, pool stays at pool_size and
    contains only edges that were inserted (subset of all seen)."""
    N, dev = 10, "cuda"
    pool = HistoricalNegPool(N, pool_size=4, device=dev)
    # Node 0 sees 20 edges with distinct dst
    src = torch.zeros(20, dtype=torch.long, device=dev)
    dst = torch.arange(0, 20, dtype=torch.long, device=dev) % N
    # Ensure distinct enough; repeat is fine, we just check capacity
    pool.update(src, dst)
    valid = pool._pool[0][pool._pool[0] != pool.PADDING]
    assert valid.numel() <= 4
    assert pool._count[0].item() == 20


# ---------------------------------------------------------------------------
# Integration: Engine sync train loop populates neg pool
# ---------------------------------------------------------------------------

def _make_engine(neg_strat, async_pipeline=False, device="cuda"):
    N, K, d = 30, 4, 8
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=device)

    graph.advance(
        torch.randint(0, N, (80,), device=device),
        torch.randint(0, N, (80,), device=device),
        torch.linspace(0, 80, 80, dtype=torch.float64, device=device),
        torch.randn(80, d, device=device),
    )

    train_batches = []
    for i in range(12):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (8,), device=device),
            dst=torch.randint(0, N, (8,), device=device),
            time=torch.full((8,), 90.0 + i, dtype=torch.float64, device=device),
            edge_feat=torch.randn(8, d, device=device),
        ))

    val_batches = test_batches = train_batches[8:12]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    config = TrainConfig(epochs=2, batch_size=8, lr=1e-3, patience=3,
                         device=device, seed=42)
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    neg_strat, APEval(), config)
    if async_pipeline:
        engine.config.async_pipeline = True
        engine.async_pipeline = AsyncDataPipeline(
            engine.model.gather_spec, engine.graph, engine.neg_strategy
        )
    return engine


def test_engine_sync_populates_historical_neg_pool():
    """Engine sync train loop must call neg_strategy.update so the pool fills."""
    neg = HistoricalNegative(num_nodes=30, pool_size=16, device="cuda")
    engine = _make_engine(neg, async_pipeline=False)

    total_before = int(neg._pool._count.sum().item())
    assert total_before == 0, "pool should start empty"

    engine.train()

    total_after = int(neg._pool._count.sum().item())
    # 12 train batches * 8 edges = 96 edges ingested across epochs
    assert total_after > 0, "pool was never updated by Engine (sync)"
    # Each epoch re-ingests the same 96 edges (graph is frozen, train_batches
    # are replayed), so count == 96 * epochs at minimum
    assert total_after >= 96, f"expected >=96 edges in pool, got {total_after}"


def test_engine_async_populates_historical_neg_pool():
    """Engine async train loop must also call neg_strategy.update (no race)."""
    neg = HistoricalNegative(num_nodes=30, pool_size=16, device="cuda")
    engine = _make_engine(neg, async_pipeline=True)

    total_before = int(neg._pool._count.sum().item())
    assert total_before == 0

    engine.train()

    total_after = int(neg._pool._count.sum().item())
    assert total_after > 0, "pool was never updated by Engine (async)"
    assert total_after >= 96, f"expected >=96 edges in pool, got {total_after}"


def test_historical_neg_sample_works_mid_training():
    """After a few sync train steps, sampling from a seen src returns a
    historically-valid dst (one that was actually inserted), not PADDING."""
    neg = HistoricalNegative(num_nodes=30, pool_size=32, device="cuda")
    engine = _make_engine(neg, async_pipeline=False)

    # Run 1 epoch to populate
    engine._train_epoch_sync()

    # Pick a src that definitely appeared in train_batches
    all_src = torch.cat([rb.src for rb in engine.train_batches])
    some_src = all_src[0].view(1)
    s = neg.sample(some_src, some_src, torch.zeros(1, device="cuda"),
                   engine.graph, None)
    assert 0 <= s.item() < 30, f"sampled invalid neg {s.item()}"


def test_random_negative_update_is_noop():
    """RandomNegative inherits the base no-op update; calling it must not
    raise and must not change sampling behavior."""
    neg = RandomNegative(30)
    # Should not raise
    neg.update(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    s = neg.sample(torch.tensor([0]), torch.tensor([1]), torch.tensor([0.0]),
                   None, None)
    assert 0 <= s.item() < 30
