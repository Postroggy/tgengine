"""Tests for CUDA temporal neighbor sampling kernels correctness."""

import pytest
import torch

from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.kernels import HAS_CUDA_EXT


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not HAS_CUDA_EXT, reason="CUDA extension not compiled"),
]


def _build_graph(num_nodes=100, num_edges=500, d_edge=16, seed=42):
    rng = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=rng)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=rng)
    time = torch.sort(torch.rand(num_edges, generator=rng, dtype=torch.float64))[0]
    feat = torch.randn(num_edges, d_edge, generator=rng)
    return src, dst, time, feat, num_nodes


class TestCUDA1Hop:
    """Verify CUDA 1-hop kernel matches PyTorch reference."""

    def test_basic_correctness(self):
        src, dst, time, feat, num_nodes = _build_graph()
        k = 10

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, num_nodes, (50,), device="cuda")
        query_times = torch.rand(50, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        cuda = g_cuda.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, cuda.neighbor_ids)
        assert torch.equal(ref.mask, cuda.mask)
        assert torch.allclose(ref.timestamps, cuda.timestamps)
        assert torch.allclose(ref.edge_feats, cuda.edge_feats, atol=1e-6)

    def test_d_edge_172(self):
        """Full-size feature dim (172) works correctly."""
        src, dst, time, feat, num_nodes = _build_graph(d_edge=172)
        k = 32

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=172, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=172, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, num_nodes, (100,), device="cuda")
        query_times = torch.rand(100, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        cuda = g_cuda.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, cuda.neighbor_ids)
        assert torch.allclose(ref.edge_feats, cuda.edge_feats, atol=1e-6)

    def test_odd_d_edge(self):
        """Non-power-of-2, non-4-aligned feature dim works."""
        src, dst, time, feat, num_nodes = _build_graph(d_edge=37)
        k = 10

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=37, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=37, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, num_nodes, (30,), device="cuda")
        query_times = torch.rand(30, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        cuda = g_cuda.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, cuda.neighbor_ids)
        assert torch.allclose(ref.edge_feats, cuda.edge_feats, atol=1e-6)

    def test_large_k(self):
        """k > 256 (thread striding)."""
        src, dst, time, feat, num_nodes = _build_graph(num_edges=2000, d_edge=16)
        k = 512

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, num_nodes, (20,), device="cuda")
        query_times = torch.ones(20, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        cuda = g_cuda.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, cuda.neighbor_ids)
        assert torch.equal(ref.mask, cuda.mask)

    def test_stress(self):
        """Large batch, large graph."""
        src, dst, time, feat, num_nodes = _build_graph(
            num_nodes=5000, num_edges=50000, d_edge=172
        )
        k = 32

        g_ref = TemporalGraph(5000, edge_feat_dim=172, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(5000, edge_feat_dim=172, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, 5000, (2000,), device="cuda")
        query_times = torch.rand(2000, dtype=torch.float64, device="cuda")

        ref = g_ref.recent(query_nodes, query_times, k)
        cuda = g_cuda.recent(query_nodes, query_times, k)

        assert torch.equal(ref.neighbor_ids, cuda.neighbor_ids)
        assert torch.equal(ref.mask, cuda.mask)
        assert torch.allclose(ref.timestamps, cuda.timestamps)
        assert torch.allclose(ref.edge_feats, cuda.edge_feats, atol=1e-6)


class TestCUDA2Hop:
    """Verify CUDA 2-hop matches PyTorch reference."""

    def test_basic_correctness(self):
        src, dst, time, feat, num_nodes = _build_graph(num_edges=1000, d_edge=16)
        k1, k2 = 10, 5

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, num_nodes, (30,), device="cuda")
        query_times = torch.rand(30, dtype=torch.float64, device="cuda")

        ref = g_ref.recent_2hop(query_nodes, query_times, k1, k2)
        cuda = g_cuda.recent_2hop(query_nodes, query_times, k1, k2)

        # Hop1
        assert torch.equal(ref.neighbor_ids, cuda.neighbor_ids)
        assert torch.equal(ref.mask, cuda.mask)
        assert torch.allclose(ref.timestamps, cuda.timestamps)
        assert torch.allclose(ref.edge_feats, cuda.edge_feats, atol=1e-6)

        # Hop2
        assert torch.equal(ref.hop2_ids, cuda.hop2_ids)
        assert torch.equal(ref.hop2_mask, cuda.hop2_mask)
        assert torch.allclose(ref.hop2_times, cuda.hop2_times)
        assert torch.allclose(ref.hop2_feats, cuda.hop2_feats, atol=1e-6)

    def test_full_feature_dim(self):
        """2-hop with d_edge=172."""
        src, dst, time, feat, num_nodes = _build_graph(num_edges=2000, d_edge=172)
        k1, k2 = 20, 10

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=172, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=172, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        query_nodes = torch.randint(0, num_nodes, (50,), device="cuda")
        query_times = torch.rand(50, dtype=torch.float64, device="cuda")

        ref = g_ref.recent_2hop(query_nodes, query_times, k1, k2)
        cuda = g_cuda.recent_2hop(query_nodes, query_times, k1, k2)

        assert torch.equal(ref.hop2_ids, cuda.hop2_ids)
        assert torch.equal(ref.hop2_mask, cuda.hop2_mask)
        assert torch.allclose(ref.hop2_feats, cuda.hop2_feats, atol=1e-6)


class TestCUDACoNeighbor:
    """Verify CUDA co-neighbor counting matches PyTorch reference."""

    def test_basic_correctness(self):
        src, dst, time, feat, num_nodes = _build_graph(num_edges=2000, d_edge=16)
        k = 32

        g_ref = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=False)
        g_ref.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_ref.freeze_csr()

        g_cuda = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g_cuda.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g_cuda.freeze_csr()

        q_src = torch.randint(0, num_nodes, (50,), device="cuda")
        q_dst = torch.randint(0, num_nodes, (50,), device="cuda")
        q_times = torch.rand(50, dtype=torch.float64, device="cuda")

        ref = g_ref.co_neighbors(q_src, q_dst, q_times, k)
        cuda = g_cuda.co_neighbors(q_src, q_dst, q_times, k)

        assert torch.allclose(ref, cuda), f"Max diff: {(ref - cuda).abs().max().item()}"

    def test_no_overlap(self):
        """Two nodes with completely disjoint neighbors should have count=0."""
        # Create star graph: node 0 connects to 1..10, node 11 connects to 12..21
        num_nodes = 30
        src = torch.cat([torch.zeros(10, dtype=torch.long), torch.full((10,), 11, dtype=torch.long)])
        dst = torch.cat([torch.arange(1, 11), torch.arange(12, 22)])
        time = torch.arange(20, dtype=torch.float64) * 0.01
        feat = torch.randn(20, 16)

        g = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g.freeze_csr()

        q_src = torch.tensor([0], device="cuda")
        q_dst = torch.tensor([11], device="cuda")
        q_times = torch.tensor([1.0], device="cuda", dtype=torch.float64)

        count = g.co_neighbors(q_src, q_dst, q_times, k=10)
        assert count.item() == 0.0

    def test_full_overlap(self):
        """Same node as src and dst should have count = number of valid neighbors."""
        num_nodes = 20
        src = torch.zeros(5, dtype=torch.long)
        dst = torch.arange(1, 6)
        time = torch.arange(5, dtype=torch.float64) * 0.1
        feat = torch.randn(5, 16)

        g = TemporalGraph(num_nodes, edge_feat_dim=16, device="cuda", use_triton=True)
        g.advance(src.cuda(), dst.cuda(), time.cuda(), feat.cuda())
        g.freeze_csr()

        q_src = torch.tensor([0], device="cuda")
        q_dst = torch.tensor([0], device="cuda")
        q_times = torch.tensor([1.0], device="cuda", dtype=torch.float64)

        count = g.co_neighbors(q_src, q_dst, q_times, k=10)
        # Node 0's neighbors: 1,2,3,4,5. Self-comparison = 5 common neighbors
        assert count.item() == 5.0
