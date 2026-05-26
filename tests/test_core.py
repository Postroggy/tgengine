"""Basic smoke test for TGEngine core components."""

import torch

from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.batch import RawBatch, NeighborData
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.pipeline import DataPipeline
from tgengine.models.dygmamba import DyGMamba


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
    print(f"Node 0 neighbors: {result.neighbor_ids}")
    print(f"Mask: {result.mask}")


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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    test_temporal_graph_basic()
    test_snapshot_restore()
    test_model_instantiation()
    print("\nAll tests passed!")
