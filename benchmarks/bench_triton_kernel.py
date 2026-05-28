"""Benchmark: Triton kernel vs PyTorch vectorized for TemporalGraph.recent()."""

import time
import torch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.kernels import HAS_TRITON


def bench_recent(num_nodes, num_edges, k, batch_size, warmup=5, repeats=50):
    """Compare Triton vs PyTorch recent() performance."""
    rng = torch.Generator().manual_seed(42)
    src = torch.randint(0, num_nodes, (num_edges,), generator=rng)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=rng)
    t = torch.sort(torch.rand(num_edges, generator=rng, dtype=torch.float64))[0]
    feat = torch.randn(num_edges, 172, generator=rng)

    g_pt = TemporalGraph(num_nodes, edge_feat_dim=172, device="cuda", use_triton=False)
    g_pt.advance(src.cuda(), dst.cuda(), t.cuda(), feat.cuda())
    g_pt.freeze_csr()

    g_tr = TemporalGraph(num_nodes, edge_feat_dim=172, device="cuda", use_triton=True)
    g_tr.advance(src.cuda(), dst.cuda(), t.cuda(), feat.cuda())
    g_tr.freeze_csr()

    query_nodes = torch.randint(0, num_nodes, (batch_size,), device="cuda")
    query_times = torch.rand(batch_size, dtype=torch.float64, device="cuda") * 0.9 + 0.05

    # Warmup
    for _ in range(warmup):
        g_pt.recent(query_nodes, query_times, k)
        g_tr.recent(query_nodes, query_times, k)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        g_pt.recent(query_nodes, query_times, k)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - start) / repeats * 1000

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        g_tr.recent(query_nodes, query_times, k)
    torch.cuda.synchronize()
    tr_time = (time.perf_counter() - start) / repeats * 1000

    speedup = pt_time / tr_time
    print(f"  nodes={num_nodes:>7,} edges={num_edges:>9,} k={k:>3} batch={batch_size:>5} | "
          f"PyTorch={pt_time:.3f}ms  Triton={tr_time:.3f}ms  speedup={speedup:.2f}x")
    return pt_time, tr_time


if __name__ == "__main__":
    assert HAS_TRITON, "Triton required"
    print("Benchmark: TemporalGraph.recent() — Triton vs PyTorch\n")

    configs = [
        # (num_nodes, num_edges, k, batch_size)
        (1_000,     10_000,     10,  200),   # tiny
        (10_000,    100_000,    32,  200),   # small (UCI-scale)
        (10_000,    100_000,    32,  600),   # medium batch
        (10_000,    100_000,    64,  200),   # larger k
        (100_000,   1_000_000,  32,  200),   # Wikipedia-scale
        (100_000,   1_000_000,  32,  600),   # Wikipedia larger batch
        (100_000,   1_000_000,  64,  200),   # Wikipedia large k
    ]

    for num_nodes, num_edges, k, batch_size in configs:
        bench_recent(num_nodes, num_edges, k, batch_size)
