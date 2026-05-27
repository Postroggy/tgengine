"""Basic smoke test for TGEngine core components."""

import pytest
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
    # Most-recent-last ordering: node 2 (t=5.0) is at index k-1=2, node 1 at index 1
    assert result.neighbor_ids[0, 2].item() == 2
    assert result.neighbor_ids[0, 1].item() == 1
    assert result.neighbor_ids[0, 0].item() == -1  # padding


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
    """End-to-end forward pass for DyGMamba (CPU → GRU fallback)."""
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


def test_mamba_gpu_forward():
    """DyGMamba forward pass on GPU uses Mamba SSM (not GRU fallback)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    B, K, d = 4, 8, 16

    def _dummy_nbrs():
        return NeighborData(
            torch.randint(0, 10, (B, K), dtype=torch.int32, device=device),
            torch.rand(B, K, dtype=torch.float64, device=device),
            torch.randn(B, K, d, device=device),
            torch.ones(B, K, dtype=torch.bool, device=device),
        )

    model = DyGMamba(d_model=d, d_edge=d, n_layers=1).to(device)
    assert model.encoder._has_mamba, "Mamba should be available on CUDA"

    batch = __import__('tgengine').PreparedBatch(
        src=torch.arange(B, device=device),
        dst=torch.arange(B, device=device) + B,
        neg=torch.arange(B, device=device) + 2 * B,
        time=torch.ones(B, dtype=torch.float64, device=device),
        src_neighbors=_dummy_nbrs(),
        dst_neighbors=_dummy_nbrs(),
        neg_neighbors=_dummy_nbrs(),
    )
    output = model(batch)
    assert output.pos_score.shape == (B,)
    assert output.loss.item() > 0
    output.loss.backward()
    # Verify Mamba params received gradients
    for name, p in model.named_parameters():
        if "mamba" in name and p.requires_grad:
            assert p.grad is not None, f"Mamba param {name} has no gradient"


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

    model = DyGFormer(d_model=d, d_edge=d, K=K, d_time=8, d_channel=8, n_layers=1, n_heads=2)
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
    state_mem, state_times = state  # checkpoint() now returns (memory, last_updated_times)
    assert not torch.allclose(model.memory.memory[:B], state_mem[:B])

    # thaw should restore original state
    model.thaw(state)
    assert torch.allclose(model.memory.memory, state_mem)


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


# ---------------------------------------------------------------------------
# 2-hop neighbor sampling
# ---------------------------------------------------------------------------


def _build_two_hop_graph():
    """Build a graph with known 2-hop structure.

    Node 0 → Node 1 (t=1) → Node 10 (t=0.5)
    Node 0 → Node 2 (t=2) → Node 11 (t=1.5)
    Node 3 → Node 4 (t=3) → Node 12 (t=2.5)

    Each src has exactly 1 1-hop neighbor, and each 1-hop neighbor has 1 friend.
    """
    device = "cpu"
    graph = TemporalGraph(num_nodes=20, buffer_size=8, edge_feat_dim=2, device=device)
    graph.advance(
        torch.tensor([1, 10, 2, 11, 4, 12]),
        torch.tensor([10, 1, 11, 2, 12, 4]),
        torch.tensor([0.5, 0.5, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64),
        torch.randn(6, 2),
    )
    # Now add the "query" edges — these happen AFTER the second-hop history
    graph.advance(
        torch.tensor([0, 0, 3]),
        torch.tensor([1, 2, 4]),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        torch.randn(3, 2),
    )
    return graph


def test_2hop_basic():
    """recent_2hop returns 1-hop and 2-hop neighbors with correct shapes."""
    graph = _build_two_hop_graph()

    result = graph.recent_2hop(
        torch.tensor([0, 3]),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
        k1=4, k2=4,
    )

    assert result.neighbor_ids.shape == (2, 4)
    assert result.hop2_ids.shape == (2, 4, 4)
    assert result.hop2_mask.shape == (2, 4, 4)
    assert result.hop2_feats.shape == (2, 4, 4, 2)


def test_2hop_temporal_causality():
    """2-hop neighbors must have timestamps before the 1-hop interaction time."""
    graph = _build_two_hop_graph()

    result = graph.recent_2hop(
        torch.tensor([0]),
        torch.tensor([10.0], dtype=torch.float64),
        k1=4, k2=4,
    )

    # Node 0's 1-hop neighbor at position with t=2.0 is node 2
    # Node 2's only valid 2-hop neighbor should be node 11 (interaction at t=1.0 < t=2.0)
    for i in range(4):
        if result.mask[0, i]:
            hop1_time = result.timestamps[0, i].item()
            for j in range(4):
                if result.hop2_mask[0, i, j]:
                    hop2_time = result.hop2_times[0, i, j].item()
                    assert hop2_time < hop1_time, \
                        f"2-hop time {hop2_time} >= 1-hop time {hop1_time}"


def test_2hop_padding_masked():
    """Invalid 1-hop positions should have fully masked 2-hop."""
    graph = _build_two_hop_graph()

    result = graph.recent_2hop(
        torch.tensor([0]),
        torch.tensor([10.0], dtype=torch.float64),
        k1=8, k2=4,
    )

    # Node 0 only has 2 valid 1-hop neighbors
    valid_1hop = result.mask[0]
    num_valid = valid_1hop.sum().item()
    assert num_valid == 2, f"expected 2 valid 1-hop, got {num_valid}"

    # Padding positions (mask=False) must have all 2-hop masked
    padding_positions = ~valid_1hop
    if padding_positions.any():
        assert not result.hop2_mask[0, padding_positions].any(), \
            "padding 1-hop positions should have all 2-hop masked"


def test_2hop_pipeline_integration():
    """DataPipeline with k2>0 produces NeighborData with 2-hop fields."""
    graph = _build_two_hop_graph()
    spec = GatherSpec(neighbors=NeighborSpec(k=4, k2=4))
    pipeline = DataPipeline(spec, graph)

    raw = RawBatch(
        src=torch.tensor([0, 3]),
        dst=torch.tensor([1, 4]),
        time=torch.tensor([10.0, 10.0], dtype=torch.float64),
        neg=torch.tensor([5, 6]),
    )
    prepared = pipeline.prepare(raw)

    assert prepared.src_neighbors.hop2_ids is not None
    assert prepared.src_neighbors.hop2_ids.shape == (2, 4, 4)
    assert prepared.dst_neighbors.hop2_ids is not None
    assert prepared.neg_neighbors.hop2_ids is not None


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

