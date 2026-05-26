"""Basic smoke test for TGEngine core components."""

import torch

from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.batch import RawBatch, NeighborData
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.pipeline import DataPipeline
from tgengine.models.dygmamba import DyGMamba
from tgengine.models.dygformer import DyGFormer
from tgengine.models.tgn import TGN


def test_temporal_graph_basic():
    """Test basic graph operations: advance + recent."""
    device = "cpu"
    graph = TemporalGraph(num_nodes=10, buffer_size=8, edge_feat_dim=4, device=device)

    # Add some edges
    src = torch.tensor([0, 0, 0, 1, 1], device=device)
    dst = torch.tensor([1, 2, 3, 2, 3], device=device)
    time = torch.tensor([1.0, 2.0, 3.0, 1.5, 2.5], dtype=torch.float64, device=device)
    feat = torch.randn(5, 4, device=device)

    graph.advance(src, dst, time, feat)
    assert graph.num_edges == 5

    # Query recent neighbors for node 0 before time 4.0
    query_nodes = torch.tensor([0], device=device)
    query_times = torch.tensor([4.0], dtype=torch.float64, device=device)
    result = graph.recent(query_nodes, query_times, k=3)

    assert result.neighbor_ids.shape == (1, 3)
    assert result.mask.shape == (1, 3)
    # Node 0 has 3 neighbors: 1, 2, 3 — all should be valid
    assert result.mask.all()


def test_recent_time_filter():
    """Ensure recent() respects the time filter (no future neighbors)."""
    device = "cpu"
    graph = TemporalGraph(num_nodes=5, buffer_size=4, edge_feat_dim=2, device=device)
    graph.advance(
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 2, 3]),
        torch.tensor([1.0, 5.0, 10.0], dtype=torch.float64),
        torch.randn(3, 2),
    )
    # Query at time 6.0 — should only see neighbors at t=1.0 and t=5.0
    result = graph.recent(torch.tensor([0]), torch.tensor([6.0], dtype=torch.float64), k=3)
    assert result.mask.sum() == 2
    # Most recent valid neighbor is at t=5.0 (node 2)
    assert result.neighbor_ids[0, 0].item() == 2


def test_co_neighbors():
    """Test co-occurrence counting."""
    device = "cpu"
    graph = TemporalGraph(num_nodes=10, buffer_size=8, edge_feat_dim=2, device=device)
    # Both node 0 and node 1 interact with node 2 and 3 — co-occurrence = 2
    graph.advance(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2, 3, 2, 3]),
        torch.tensor([1.0, 2.0, 1.5, 2.5], dtype=torch.float64),
        torch.randn(4, 2),
    )
    co = graph.co_neighbors(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([10.0], dtype=torch.float64),
    )
    assert co.shape == (1,)
    assert co[0].item() == 2.0  # both share neighbors 2 and 3


def test_snapshot_restore():
    """Test graph snapshot and restore."""
    device = "cpu"
    graph = TemporalGraph(num_nodes=5, buffer_size=4, edge_feat_dim=2, device=device)

    # Add edges
    graph.advance(
        torch.tensor([0, 1], device=device),
        torch.tensor([1, 2], device=device),
        torch.tensor([1.0, 2.0], dtype=torch.float64, device=device),
        torch.randn(2, 2, device=device),
    )

    snap = graph.snapshot()
    assert graph.num_edges == 2

    # Add more edges
    graph.advance(
        torch.tensor([2], device=device),
        torch.tensor([3], device=device),
        torch.tensor([3.0], dtype=torch.float64, device=device),
        torch.randn(1, 2, device=device),
    )
    assert graph.num_edges == 3

    # Restore
    graph.restore(snap)
    assert graph.num_edges == 2


def test_model_instantiation():
    """Test that DyGMamba can be instantiated and has correct gather_spec."""
    model = DyGMamba(d_model=64, d_edge=4, n_layers=1)
    assert model.gather_spec.neighbors.k == 32
    assert model.supports_independent_encode is True
    assert sum(p.numel() for p in model.parameters()) > 0


def test_dygmamba_forward():
    """End-to-end forward pass for DyGMamba."""
    device = "cpu"
    B, K, d = 4, 8, 16

    def _dummy_nbrs():
        ids = torch.randint(0, 10, (B, K), dtype=torch.int32)
        ts = torch.rand(B, K, dtype=torch.float64)
        feats = torch.randn(B, K, d)
        mask = torch.ones(B, K, dtype=torch.bool)
        return NeighborData(ids, ts, feats, mask)

    model = DyGMamba(d_model=d, d_edge=d, n_layers=1)
    batch = __import__('tgengine').PreparedBatch(
        src=torch.arange(B),
        dst=torch.arange(B) + B,
        neg=torch.arange(B) + 2 * B,
        time=torch.ones(B, dtype=torch.float64),
        src_neighbors=_dummy_nbrs(),
        dst_neighbors=_dummy_nbrs(),
        neg_neighbors=_dummy_nbrs(),
    )
    output = model(batch)
    assert output.pos_score.shape == (B,)
    assert output.neg_score.shape == (B,)
    assert output.loss.item() > 0


def test_dygformer_forward():
    """End-to-end forward pass for DyGFormer."""
    device = "cpu"
    B, K, d = 4, 8, 16

    def _dummy_nbrs():
        return NeighborData(
            torch.randint(0, 10, (B, K), dtype=torch.int32),
            torch.rand(B, K, dtype=torch.float64),
            torch.randn(B, K, d),
            torch.ones(B, K, dtype=torch.bool),
        )

    model = DyGFormer(d_model=d, d_edge=d, n_layers=1, n_heads=2)
    batch = __import__('tgengine').PreparedBatch(
        src=torch.arange(B),
        dst=torch.arange(B) + B,
        neg=torch.arange(B) + 2 * B,
        time=torch.ones(B, dtype=torch.float64),
        src_neighbors=_dummy_nbrs(),
        dst_neighbors=_dummy_nbrs(),
        neg_neighbors=_dummy_nbrs(),
        co_occurrence=torch.randint(0, 5, (B,)).float(),
    )
    output = model(batch)
    assert output.pos_score.shape == (B,)
    assert output.loss.item() > 0


def test_tgn_forward_and_lifecycle():
    """Test TGN forward pass + evolve/freeze/thaw lifecycle."""
    device = "cpu"
    B, K, d, N = 4, 8, 16, 20

    def _dummy_nbrs():
        return NeighborData(
            torch.randint(0, N, (B, K), dtype=torch.int32),
            torch.rand(B, K, dtype=torch.float64),
            torch.randn(B, K, d),
            torch.ones(B, K, dtype=torch.bool),
        )

    model = TGN(num_nodes=N, d_model=d, d_edge=d, n_gru_layers=1)
    batch = __import__('tgengine').PreparedBatch(
        src=torch.arange(B),
        dst=torch.arange(B) + B,
        neg=torch.arange(B) + 2 * B,
        time=torch.ones(B, dtype=torch.float64),
        src_neighbors=_dummy_nbrs(),
        dst_neighbors=_dummy_nbrs(),
        neg_neighbors=_dummy_nbrs(),
    )

    output = model(batch)
    assert output.pos_score.shape == (B,)

    # evolve should update memory
    state = model.freeze()
    model.evolve(
        src=torch.arange(B),
        dst=torch.arange(B) + B,
        time=torch.ones(B, dtype=torch.float64),
        edge_feat=torch.randn(B, d),
    )
    # memory should have changed
    assert not torch.allclose(model.memory.memory[:B], state[:B])

    # thaw should restore original state
    model.thaw(state)
    assert torch.allclose(model.memory.memory, state)


def test_pipeline_fused_query():
    """DataPipeline produces correct PreparedBatch from RawBatch."""
    device = "cpu"
    graph = TemporalGraph(num_nodes=20, buffer_size=8, edge_feat_dim=4, device=device)
    # Seed the graph with some history
    graph.advance(
        torch.randint(0, 20, (50,)),
        torch.randint(0, 20, (50,)),
        torch.arange(50, dtype=torch.float64),
        torch.randn(50, 4),
    )

    spec = GatherSpec(neighbors=NeighborSpec(k=4))
    pipeline = DataPipeline(spec, graph)

    raw = RawBatch(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([3, 4, 5]),
        time=torch.tensor([60.0, 61.0, 62.0], dtype=torch.float64),
        neg=torch.tensor([6, 7, 8]),
    )
    prepared = pipeline.prepare(raw)
    assert prepared.src_neighbors.neighbor_ids.shape == (3, 4)
    assert prepared.dst_neighbors.neighbor_ids.shape == (3, 4)
    assert prepared.neg_neighbors.neighbor_ids.shape == (3, 4)


if __name__ == "__main__":
    test_temporal_graph_basic()
    test_recent_time_filter()
    test_co_neighbors()
    test_snapshot_restore()
    test_model_instantiation()
    test_dygmamba_forward()
    test_dygformer_forward()
    test_tgn_forward_and_lifecycle()
    test_pipeline_fused_query()
    print("\nAll tests passed!")

