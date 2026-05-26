"""Tests for AsyncDataPipeline.

Verifies:
  - Output matches synchronous DataPipeline exactly
  - Double-buffer ordering is correct (graph.advance before start_prefetch)
  - Works end-to-end in a training loop
"""

import torch
import pytest

from tgengine.core.batch import RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.dygformer import DyGFormer
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.async_pipeline import AsyncDataPipeline
from tgengine.pipeline.negatives import RandomNegative


NUM_NODES = 50
D_EDGE = 8
K = 8


def _make_graph():
    g = TemporalGraph(NUM_NODES, buffer_size=K, edge_feat_dim=D_EDGE, device="cpu")
    src = torch.randint(0, NUM_NODES, (100,))
    dst = torch.randint(0, NUM_NODES, (100,))
    ts = torch.linspace(0, 100, 100, dtype=torch.float64)
    ef = torch.randn(100, D_EDGE)
    g.advance(src, dst, ts, ef)
    return g


def _make_batches(n: int, B: int = 8):
    batches = []
    for i in range(n):
        batches.append(RawBatch(
            src=torch.randint(0, NUM_NODES, (B,)),
            dst=torch.randint(0, NUM_NODES, (B,)),
            time=torch.full((B,), 110.0 + i, dtype=torch.float64),
            edge_feat=torch.randn(B, D_EDGE),
        ))
    return batches


def test_async_pipeline_instantiation():
    g = _make_graph()
    model = DyGFormer(d_model=32, d_edge=D_EDGE, d_time=8, d_channel=16, K=K, n_layers=1)
    neg = RandomNegative(NUM_NODES)
    pipe = AsyncDataPipeline(model.gather_spec, g, neg)
    assert pipe.spec.neighbors.k == K


def test_async_produces_valid_batch():
    g = _make_graph()
    model = DyGFormer(d_model=32, d_edge=D_EDGE, d_time=8, d_channel=16, K=K, n_layers=1)
    neg_strat = RandomNegative(NUM_NODES)
    batches = _make_batches(3)

    pipe = AsyncDataPipeline(model.gather_spec, g, neg_strat)
    pipe.start_prefetch(batches[0])
    rb, prepared = pipe.get()

    assert prepared.src.shape == (8,)
    assert prepared.src_neighbors.neighbor_ids.shape == (8, K)
    assert prepared.neg is not None
    assert prepared.neg_neighbors is not None


def test_async_matches_sync_output():
    """AsyncDataPipeline must produce identical neighbor data as sync DataPipeline.

    Requires identical graph state and identical query nodes.
    We use a stub neg strategy that always returns the same tensor.
    """

    class _ConstNeg:
        def __init__(self, neg):
            self.neg = neg
        def sample(self, *_):
            return self.neg

    def make_fixed_graph():
        torch.manual_seed(0)
        g = TemporalGraph(NUM_NODES, buffer_size=K, edge_feat_dim=D_EDGE, device="cpu")
        src = torch.randint(0, NUM_NODES, (100,))
        dst = torch.randint(0, NUM_NODES, (100,))
        ts = torch.linspace(0, 100, 100, dtype=torch.float64)
        ef = torch.randn(100, D_EDGE)
        g.advance(src, dst, ts, ef)
        return g

    g_sync = make_fixed_graph()
    g_async = make_fixed_graph()

    model = DyGFormer(d_model=32, d_edge=D_EDGE, d_time=8, d_channel=16, K=K, n_layers=1)

    torch.manual_seed(1)
    src_q = torch.randint(0, NUM_NODES, (4,))
    dst_q = torch.randint(0, NUM_NODES, (4,))
    neg_q = torch.randint(0, NUM_NODES, (4,))
    time_q = torch.full((4,), 110.0, dtype=torch.float64)
    edge_q = torch.randn(4, D_EDGE)
    rb = RawBatch(src=src_q, dst=dst_q, time=time_q, edge_feat=edge_q)

    # Sync: manually set neg
    rb_sync = RawBatch(src=src_q, dst=dst_q, time=time_q, edge_feat=edge_q, neg=neg_q)
    sync_pipe = DataPipeline(model.gather_spec, g_sync)
    prepared_sync = sync_pipe.prepare(rb_sync)

    # Async: use stub strategy that always returns neg_q
    async_pipe = AsyncDataPipeline(model.gather_spec, g_async, _ConstNeg(neg_q))
    async_pipe.start_prefetch(rb)
    rb_async, prepared_async = async_pipe.get()

    # Both graphs are identical → same src/dst query → same neighbor IDs
    assert torch.equal(prepared_sync.src_neighbors.neighbor_ids,
                       prepared_async.src_neighbors.neighbor_ids), \
        "src_neighbors should match for identical graph state + same query"
    assert torch.equal(prepared_sync.dst_neighbors.neighbor_ids,
                       prepared_async.dst_neighbors.neighbor_ids), \
        "dst_neighbors should match for identical graph state + same query"


def test_async_training_loop():
    """Full training loop using async pipeline should not crash and produce finite loss."""
    g = _make_graph()
    model = DyGFormer(d_model=32, d_edge=D_EDGE, d_time=8, d_channel=16, K=K, n_layers=1)
    neg_strat = RandomNegative(NUM_NODES)
    batches = _make_batches(5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    pipe = AsyncDataPipeline(model.gather_spec, g, neg_strat)
    pipe.start_prefetch(batches[0])
    losses = []

    for i in range(len(batches)):
        rb, prepared = pipe.get()
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
        g.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
        if i + 1 < len(batches):
            pipe.start_prefetch(batches[i + 1])

    assert all(torch.isfinite(torch.tensor(losses))), f"Non-finite losses: {losses}"
    assert len(losses) == 5


def test_async_ordering_constraint():
    """Prefetch of batch i+1 should see graph state after advance(batch_i).

    We verify this by checking that the neighbor data for batch i+1
    includes edges inserted by advance(batch_0) — specifically, the src/dst
    nodes from batch 0 should appear as neighbors after advance.
    """
    g = TemporalGraph(NUM_NODES, buffer_size=K, edge_feat_dim=D_EDGE, device="cpu")
    model = DyGFormer(d_model=32, d_edge=D_EDGE, d_time=8, d_channel=16, K=K, n_layers=1)
    neg_strat = RandomNegative(NUM_NODES)

    # Batch 0: edges src=1 → dst=2 at time=1
    b0 = RawBatch(
        src=torch.tensor([1]),
        dst=torch.tensor([2]),
        time=torch.tensor([1.0], dtype=torch.float64),
        edge_feat=torch.randn(1, D_EDGE),
    )
    # Batch 1: query node 1's neighbors at time=2 (should see batch_0's edge)
    b1 = RawBatch(
        src=torch.tensor([1]),
        dst=torch.tensor([3]),
        time=torch.tensor([2.0], dtype=torch.float64),
        edge_feat=torch.randn(1, D_EDGE),
    )

    pipe = AsyncDataPipeline(model.gather_spec, g, neg_strat)
    pipe.start_prefetch(b0)
    rb0, prepared0 = pipe.get()

    # Advance: node 1 now has neighbor 2
    g.advance(rb0.src, rb0.dst, rb0.time, rb0.edge_feat)

    # Now prefetch b1 — should see the advanced graph
    pipe.start_prefetch(b1)
    rb1, prepared1 = pipe.get()

    # Node 1's neighbors at time=2 should include node 2 (from advance)
    src1_nbrs = prepared1.src_neighbors.neighbor_ids[0]  # (K,)
    assert 2 in src1_nbrs, (
        f"Node 1's neighbors should contain node 2 after advance, got: {src1_nbrs}"
    )
