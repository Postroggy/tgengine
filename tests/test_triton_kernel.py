"""Tests for Triton temporal neighbor sampling kernel correctness."""

import pytest
import torch

from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.kernels import HAS_TRITON


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _build_graph(num_nodes=100, num_edges=500, seed=42):
    """Build a small graph and return it frozen."""
    rng = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=rng)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=rng)
    time = torch.sort(torch.rand(num_edges, generator=rng, dtype=torch.float64))[0]
    feat = torch.randn(num_edges, 16, generator=rng)
    return src, dst, time, feat, num_nodes


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
class TestTritonKernel:
    """Verify Triton kernel produces identical output to PyTorch path."""

    def test_basic_correctness(self):
        """Triton output matches PyTorch output on a small graph."""
        src, dst, time, feat, num_nodes = _build_graph()
        k = 10

        # Build graph with Triton disabled (PyTorch reference)
        g_ref = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        # Build graph with Triton enabled
        g_tri = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g_tri.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_tri.freeze_csr()

        # Query
        query_nodes = torch.randint(0, num_nodes, (50,), device="cuda")
        query_times = torch.rand(50, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        tri = g_tri.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, tri.neighbor_ids), \
            f"IDs mismatch:\nref={ref.neighbor_ids[:3]}\ntri={tri.neighbor_ids[:3]}"
        assert torch.equal(ref.mask, tri.mask), "Mask mismatch"
        assert torch.allclose(ref.timestamps, tri.timestamps), "Timestamps mismatch"
        assert torch.allclose(ref.edge_feats, tri.edge_feats, atol=1e-6), "Feats mismatch"

    def test_empty_nodes(self):
        """Nodes with no neighbors should produce all-padding output."""
        src, dst, time, feat, num_nodes = _build_graph(num_nodes=200, num_edges=100)
        k = 8

        g = TemporalGraph(200, edge_feat_dim=16, device="cuda", use_triton=True)
        g.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g.freeze_csr()

        # Query nodes that likely have no edges (high IDs in sparse graph)
        query_nodes = torch.arange(150, 200, device="cuda")
        query_times = torch.ones(50, dtype=torch.float64, device="cuda")

        result = g.recent(query_nodes, query_times, k)

        # Nodes with no neighbors: all padding
        for i in range(50):
            node = query_nodes[i].item()
            if result.mask[i].sum() == 0:
                assert (result.neighbor_ids[i] == -1).all()

    def test_time_filtering(self):
        """Only neighbors with time < query_time should be returned."""
        num_nodes = 10
        # All edges from node 0 at times 0.1, 0.2, ..., 1.0
        src = torch.zeros(10, dtype=torch.long)
        dst = torch.arange(1, 11, dtype=torch.long) % num_nodes
        time = torch.arange(1, 11, dtype=torch.float64) * 0.1
        feat = torch.randn(10, 16)

        g = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g.freeze_csr()

        # Query node 0 at time 0.55 — should see neighbors at 0.1..0.5 (5 neighbors)
        query_nodes = torch.tensor([0], device="cuda", dtype=torch.long)
        query_times = torch.tensor([0.55], device="cuda", dtype=torch.float64)

        result = g.recent(query_nodes, query_times, k=10)
        valid_count = result.mask[0].sum().item()
        assert valid_count == 5, f"Expected 5 valid neighbors, got {valid_count}"
        # All valid timestamps < 0.55
        valid_times = result.timestamps[0][result.mask[0]]
        assert (valid_times < 0.55).all()

    def test_padding_nodes_input(self):
        """Negative node IDs (padding) should produce empty results."""
        src, dst, time, feat, num_nodes = _build_graph()
        k = 8

        g = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g.freeze_csr()

        query_nodes = torch.tensor([-1, -1, 5, 10], device="cuda", dtype=torch.long)
        query_times = torch.tensor([0.5, 0.5, 0.5, 0.5], device="cuda", dtype=torch.float64)

        result = g.recent(query_nodes, query_times, k)
        # First two nodes are padding — should be all empty
        assert (result.neighbor_ids[0] == -1).all()
        assert (result.neighbor_ids[1] == -1).all()
        assert result.mask[0].sum() == 0
        assert result.mask[1].sum() == 0

    def test_large_batch(self):
        """Stress test with larger batch to catch index errors."""
        src, dst, time, feat, num_nodes = _build_graph(num_nodes=1000, num_edges=5000)
        k = 32

        g_ref = TemporalGraph(1000, edge_feat_dim=16, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_tri = TemporalGraph(1000, edge_feat_dim=16, device="cuda", use_triton=True)
        g_tri.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_tri.freeze_csr()

        query_nodes = torch.randint(0, 1000, (2000,), device="cuda")
        query_times = torch.rand(2000, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        tri = g_tri.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, tri.neighbor_ids)
        assert torch.equal(ref.mask, tri.mask)
        assert torch.allclose(ref.timestamps, tri.timestamps)
        assert torch.allclose(ref.edge_feats, tri.edge_feats, atol=1e-6)
