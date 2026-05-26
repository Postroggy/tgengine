"""Comprehensive correctness tests for DyGFormer.

Covers:
  - Forward pass shapes
  - Gradient flow through all modules
  - Loss decreases over training steps
  - Full pipeline (TemporalGraph → DataPipeline → DyGFormer)
  - Co-occurrence encoding logic
  - Different patch_size values
  - Eval protocol integration
"""

import torch
import torch.nn as nn

from tgengine.core.batch import NeighborData, PreparedBatch, RawBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import APEval, TrainConfig
from tgengine.models.dygformer import DyGFormer, _CoOccurrenceEncoder
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nbrs(B: int, K: int, d: int, all_valid: bool = True) -> NeighborData:
    mask = torch.ones(B, K, dtype=torch.bool) if all_valid else (torch.rand(B, K) > 0.3)
    ids = torch.where(mask, torch.randint(0, 20, (B, K), dtype=torch.int32),
                      torch.full((B, K), -1, dtype=torch.int32))
    return NeighborData(
        neighbor_ids=ids,
        timestamps=torch.rand(B, K, dtype=torch.float64),
        edge_feats=torch.randn(B, K, d),
        mask=mask,
    )


def _make_batch(B: int, K: int, d: int, all_valid: bool = True) -> PreparedBatch:
    return PreparedBatch(
        src=torch.arange(B),
        dst=torch.arange(B) + B,
        neg=torch.arange(B) + 2 * B,
        time=torch.rand(B, dtype=torch.float64) * 100 + 100,
        src_neighbors=_make_nbrs(B, K, d, all_valid),
        dst_neighbors=_make_nbrs(B, K, d, all_valid),
        neg_neighbors=_make_nbrs(B, K, d, all_valid),
    )


# ---------------------------------------------------------------------------
# Test: instantiation and gather_spec
# ---------------------------------------------------------------------------

def test_dygformer_instantiation():
    model = DyGFormer(d_model=32, d_edge=16, d_time=8, d_channel=8, K=8)
    assert model.gather_spec.neighbors.k == 8
    assert not model.gather_spec.co_occurrence
    assert not model.supports_independent_encode


def test_dygformer_patch_size():
    """patch_size=2 should halve the number of patches."""
    model = DyGFormer(d_model=32, d_edge=16, d_time=8, d_channel=8, K=8, patch_size=2)
    assert model.n_patches == 4  # 8 / 2


def test_dygformer_invalid_patch_size():
    import pytest
    with pytest.raises(ValueError):
        DyGFormer(K=9, patch_size=2)  # 9 not divisible by 2


# ---------------------------------------------------------------------------
# Test: forward pass output shapes
# ---------------------------------------------------------------------------

def test_forward_shapes():
    B, K, d = 4, 8, 16
    model = DyGFormer(d_model=32, d_edge=d, d_time=8, d_channel=8, K=K)
    batch = _make_batch(B, K, d)

    out = model(batch)
    assert out.pos_score.shape == (B,), f"pos_score shape {out.pos_score.shape}"
    assert out.neg_score.shape == (B,), f"neg_score shape {out.neg_score.shape}"
    assert out.loss.ndim == 0, "loss should be scalar"
    assert out.loss.item() > 0, "loss should be positive"


def test_forward_shapes_patch2():
    """patch_size=2 should still produce correct output shapes."""
    B, K, d = 4, 8, 16
    model = DyGFormer(d_model=32, d_edge=d, d_time=8, d_channel=8, K=K, patch_size=2)
    batch = _make_batch(B, K, d)
    out = model(batch)
    assert out.pos_score.shape == (B,)


def test_forward_sparse_mask():
    """Model should handle mostly-padded neighbor sequences (few valid neighbors)."""
    B, K, d = 6, 16, 8
    model = DyGFormer(d_model=16, d_edge=d, d_time=8, d_channel=8, K=K)
    batch = _make_batch(B, K, d, all_valid=False)
    out = model(batch)
    assert out.pos_score.shape == (B,)
    assert not torch.isnan(out.loss)


# ---------------------------------------------------------------------------
# Test: gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Every parameter must receive a gradient on backward."""
    B, K, d = 4, 8, 16
    model = DyGFormer(d_model=32, d_edge=d, d_time=8, d_channel=8, K=K)
    batch = _make_batch(B, K, d)

    out = model(batch)
    out.loss.backward()

    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"


# ---------------------------------------------------------------------------
# Test: loss decreases under optimization
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """A few gradient steps should reduce loss on fixed data."""
    B, K, d = 8, 8, 16
    model = DyGFormer(d_model=32, d_edge=d, d_time=8, d_channel=8, K=K)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _make_batch(B, K, d)

    losses = []
    for _ in range(10):
        opt.zero_grad()
        out = model(batch)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())

    # Loss should decrease over 10 steps
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# Test: co-occurrence encoding correctness
# ---------------------------------------------------------------------------

def test_co_occurrence_self_count():
    """When a node appears multiple times in its own list, self_count > 1."""
    co_enc = _CoOccurrenceEncoder(d_out=8)

    # a_ids: node 5 appears twice, b_ids: no overlap with 5
    a_ids = torch.tensor([[5, 5, 1, 2]], dtype=torch.int32)   # (1, 4)
    b_ids = torch.tensor([[3, 4, 6, 7]], dtype=torch.int32)

    a_feat, b_feat = co_enc(a_ids.long(), b_ids.long())
    assert a_feat.shape == (1, 4, 8)
    assert b_feat.shape == (1, 4, 8)

    # For a_ids, positions 0 and 1 (both = node 5) should have self_count=2 (appear twice)
    # So a_feat[0,0] and a_feat[0,1] should be identical
    assert torch.allclose(a_feat[0, 0], a_feat[0, 1], atol=1e-6), \
        "Identical neighbor IDs should produce identical co-occurrence features"


def test_co_occurrence_cross_count():
    """Shared neighbors between a and b should produce non-zero cross_count."""
    co_enc = _CoOccurrenceEncoder(d_out=8)

    # Node 5 is in both a and b — cross_count for position of node 5 should be > 0
    a_ids = torch.tensor([[5, 1, 2, 3]], dtype=torch.int32)
    b_ids = torch.tensor([[5, 4, 6, 7]], dtype=torch.int32)

    a_feat_shared, _ = co_enc(a_ids.long(), b_ids.long())
    a_feat_no_share, _ = co_enc(
        a_ids.long(), torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    )

    # Position 0 (node 5) should differ between shared and non-shared scenarios
    assert not torch.allclose(a_feat_shared[0, 0], a_feat_no_share[0, 0], atol=1e-6), \
        "Shared neighbors should change co-occurrence features"


def test_co_occurrence_padding_masked():
    """PADDING_ID (-1) positions should produce zero features."""
    co_enc = _CoOccurrenceEncoder(d_out=8)
    a_ids = torch.tensor([[-1, 1, 2, 3]], dtype=torch.int32)
    b_ids = torch.tensor([[-1, 4, 5, 6]], dtype=torch.int32)

    a_feat, b_feat = co_enc(a_ids.long(), b_ids.long())
    # Position 0 is padding — feature should be zero
    assert torch.all(a_feat[0, 0] == 0), "Padding positions should have zero features"
    assert torch.all(b_feat[0, 0] == 0)


# ---------------------------------------------------------------------------
# Test: full pipeline integration (TemporalGraph → DataPipeline → DyGFormer)
# ---------------------------------------------------------------------------

def test_full_pipeline_integration():
    """End-to-end: seed a graph, run batches through DataPipeline → DyGFormer."""
    device = "cpu"
    num_nodes, K, d_edge = 50, 8, 16
    batch_size = 10

    # Build and seed the graph
    graph = TemporalGraph(num_nodes=num_nodes, buffer_size=K, edge_feat_dim=d_edge, device=device)
    n_seed = 200
    graph.advance(
        src=torch.randint(0, num_nodes, (n_seed,)),
        dst=torch.randint(0, num_nodes, (n_seed,)),
        time=torch.arange(n_seed, dtype=torch.float64),
        edge_feat=torch.randn(n_seed, d_edge),
    )

    model = DyGFormer(d_model=32, d_edge=d_edge, d_time=8, d_channel=8, K=K)
    pipeline = DataPipeline(model.gather_spec, graph)
    neg_strategy = RandomNegative(num_nodes)

    # Generate a batch with negatives
    raw = RawBatch(
        src=torch.randint(0, num_nodes, (batch_size,)),
        dst=torch.randint(0, num_nodes, (batch_size,)),
        time=torch.arange(n_seed, n_seed + batch_size, dtype=torch.float64),
    )
    raw.neg = neg_strategy.sample(raw.src, raw.dst, raw.time, graph)

    prepared = pipeline.prepare(raw)
    out = model(prepared)

    assert out.pos_score.shape == (batch_size,)
    assert not torch.isnan(out.loss)
    assert out.loss.item() > 0


def test_full_pipeline_backward():
    """Backward pass through full pipeline should not crash."""
    device = "cpu"
    num_nodes, K, d_edge = 30, 8, 8

    graph = TemporalGraph(num_nodes=num_nodes, buffer_size=K, edge_feat_dim=d_edge, device=device)
    graph.advance(
        torch.randint(0, num_nodes, (100,)),
        torch.randint(0, num_nodes, (100,)),
        torch.arange(100, dtype=torch.float64),
        torch.randn(100, d_edge),
    )

    model = DyGFormer(d_model=16, d_edge=d_edge, d_time=8, d_channel=8, K=K)
    pipeline = DataPipeline(model.gather_spec, graph)

    raw = RawBatch(
        src=torch.randint(0, num_nodes, (8,)),
        dst=torch.randint(0, num_nodes, (8,)),
        time=torch.arange(100, 108, dtype=torch.float64),
        neg=torch.randint(0, num_nodes, (8,)),
    )
    prepared = pipeline.prepare(raw)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    out = model(prepared)
    out.loss.backward()
    opt.step()

    # Verify parameters were updated
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Test: AP evaluation over full epoch
# ---------------------------------------------------------------------------

def test_ap_eval_integration():
    """APEval should produce a valid AP in [0, 1]."""
    device = "cpu"
    num_nodes, K, d_edge = 30, 8, 8

    graph = TemporalGraph(num_nodes=num_nodes, buffer_size=K, edge_feat_dim=d_edge, device=device)
    graph.advance(
        torch.randint(0, num_nodes, (100,)),
        torch.randint(0, num_nodes, (100,)),
        torch.arange(100, dtype=torch.float64),
        torch.randn(100, d_edge),
    )

    model = DyGFormer(d_model=16, d_edge=d_edge, d_time=8, d_channel=8, K=K)
    pipeline = DataPipeline(model.gather_spec, graph)
    eval_protocol = APEval()

    # Build eval batches
    eval_batches = []
    for i in range(5):
        t = torch.arange(100 + i * 5, 105 + i * 5, dtype=torch.float64)
        b = RawBatch(
            src=torch.randint(0, num_nodes, (5,)),
            dst=torch.randint(0, num_nodes, (5,)),
            time=t,
            neg=torch.randint(0, num_nodes, (5,)),
        )
        eval_batches.append(b)

    snap = graph.snapshot()
    metrics = eval_protocol.evaluate(model, pipeline, eval_batches, graph)
    graph.restore(snap)

    assert "ap" in metrics
    assert 0.0 <= metrics["ap"] <= 1.0, f"AP out of range: {metrics['ap']}"


# ---------------------------------------------------------------------------
# Test: training converges on separable synthetic data
# ---------------------------------------------------------------------------

def test_overfit_separable():
    """DyGFormer should overfit when positive and negative examples are clearly separable.

    We use a graph where positive dst nodes have many shared neighbors with src,
    while negative dst nodes have no shared neighbors.
    """
    device = "cpu"
    num_nodes = 100
    K = 8
    d_edge = 16
    batch_size = 16

    # Create a graph where nodes 0-9 form a dense cluster
    graph = TemporalGraph(num_nodes=num_nodes, buffer_size=16, edge_feat_dim=d_edge, device=device)
    cluster = list(range(10))
    for t, (a, b) in enumerate(zip(cluster[:-1], cluster[1:])):
        graph.advance(
            torch.tensor([a]),
            torch.tensor([b]),
            torch.tensor([float(t)]),
            torch.randn(1, d_edge),
        )

    # Also add outside edges (nodes 50-90 connected to each other)
    for t in range(30):
        u, v = 50 + t, 60 + t
        graph.advance(
            torch.tensor([u % num_nodes]),
            torch.tensor([v % num_nodes]),
            torch.tensor([10.0 + t]),
            torch.randn(1, d_edge),
        )

    model = DyGFormer(d_model=32, d_edge=d_edge, d_time=8, d_channel=8, K=K)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    pipeline = DataPipeline(model.gather_spec, graph)

    # Positive: pairs from dense cluster (high co-occurrence)
    # Negative: random pairs from outside cluster (low co-occurrence)
    t_query = torch.full((batch_size,), 50.0, dtype=torch.float64)
    pos_src = torch.randint(0, 8, (batch_size,))      # cluster nodes
    pos_dst = torch.randint(0, 8, (batch_size,))      # cluster nodes (positive)
    neg_dst = torch.randint(20, 50, (batch_size,))    # outside cluster (negative)

    raw = RawBatch(
        src=pos_src, dst=pos_dst, time=t_query,
        edge_feat=torch.randn(batch_size, d_edge),
        neg=neg_dst,
    )
    prepared = pipeline.prepare(raw)

    initial_loss = None
    for step in range(30):
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        if initial_loss is None:
            initial_loss = out.loss.item()

    final_loss = out.loss.item()
    assert final_loss < initial_loss, \
        f"Model did not converge: {initial_loss:.4f} → {final_loss:.4f}"


if __name__ == "__main__":
    test_dygformer_instantiation()
    test_dygformer_patch_size()
    test_forward_shapes()
    test_forward_shapes_patch2()
    test_forward_sparse_mask()
    test_gradient_flow()
    test_loss_decreases()
    test_co_occurrence_self_count()
    test_co_occurrence_cross_count()
    test_co_occurrence_padding_masked()
    test_full_pipeline_integration()
    test_full_pipeline_backward()
    test_ap_eval_integration()
    test_overfit_separable()
    print("\nAll DyGFormer tests passed!")
