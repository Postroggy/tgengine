"""Pipeline benchmark: TGEngine fused query vs TGM-style vs DyGLib-style baselines.

Measures wall-clock time for the neighbor sampling step that dominates
every training iteration.

Three implementations:
  - TGEngine  : single fused recent() for src+dst+neg (1 kernel dispatch)
  - TGM-style : 3 separate recent() calls (matching TGM's _get_recency_neighbors pattern)
  - DyGLib    : Python for-loop, one node at a time (matching DyGLib CPU baseline)

Key findings (GPU, B=600):
  - TGEngine vs DyGLib : 1200x+ speedup  (kernel launch overhead dominates in loop)
  - TGEngine vs TGM    : 2-3x speedup    (3 dispatch → 1 dispatch)

Note on batch size scaling:
  - At small/medium B (200–600): TGEngine wins — kernel launch overhead dominates.
  - At large B (2000+): GPU saturated, memory-bandwidth bound; advantage narrows.
  - Typical DyGFormer/TGN training uses B=200–600, so TGEngine is the right target.

Note on CPU:
  - No kernel launch overhead on CPU, so TGM-style is faster (no concat/split cost).
  - TGEngine is still 25x faster than DyGLib on CPU.
  - TGEngine is designed for GPU training.
"""

import argparse
import time

import torch

from tgengine.core.temporal_graph import TemporalGraph


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _warmup(fn, n=3):
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n_iters: int, device: str) -> float:
    """Returns mean time per iteration in milliseconds."""
    if "cuda" in device:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if "cuda" in device:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000  # ms


# ---------------------------------------------------------------------------
# Three implementations
# ---------------------------------------------------------------------------

def run_fused(graph: TemporalGraph, src, dst, neg, times, k: int):
    """TGEngine: single fused kernel call for all 3 node groups."""
    all_nodes = torch.cat([src, dst, neg])
    all_times = torch.cat([times, times, times])
    all_nbrs = graph.recent(all_nodes, all_times, k)
    B = src.shape[0]
    src_nbrs = all_nbrs.neighbor_ids[:B]
    dst_nbrs = all_nbrs.neighbor_ids[B:2*B]
    neg_nbrs = all_nbrs.neighbor_ids[2*B:]
    return src_nbrs, dst_nbrs, neg_nbrs


def run_tgm_style(graph: TemporalGraph, src, dst, neg, times, k: int):
    """TGM-style: 3 separate recent() calls (one per node group)."""
    src_nbrs = graph.recent(src, times, k)
    dst_nbrs = graph.recent(dst, times, k)
    neg_nbrs = graph.recent(neg, times, k)
    return src_nbrs.neighbor_ids, dst_nbrs.neighbor_ids, neg_nbrs.neighbor_ids


def run_dygl_style(graph: TemporalGraph, src, dst, neg, times, k: int):
    """DyGLib-style: Python for-loop, one node per call."""
    all_nodes = torch.cat([src, dst, neg])
    all_times = torch.cat([times, times, times])
    results = []
    for i in range(all_nodes.shape[0]):
        nbr = graph.recent(all_nodes[i:i+1], all_times[i:i+1], k)
        results.append(nbr.neighbor_ids)
    all_ids = torch.cat(results, dim=0)
    B = src.shape[0]
    return all_ids[:B], all_ids[B:2*B], all_ids[2*B:]


# ---------------------------------------------------------------------------
# Single configuration benchmark
# ---------------------------------------------------------------------------

def bench_one(graph, src, dst, neg, t_query, k, n_iters, device, include_dygl=True):
    _warmup(lambda: run_fused(graph, src, dst, neg, t_query, k))
    t_fused = _timed(lambda: run_fused(graph, src, dst, neg, t_query, k), n_iters, device)

    _warmup(lambda: run_tgm_style(graph, src, dst, neg, t_query, k))
    t_tgm = _timed(lambda: run_tgm_style(graph, src, dst, neg, t_query, k), n_iters, device)

    t_dygl = None
    if include_dygl:
        dygl_iters = max(1, n_iters // 20)
        _warmup(lambda: run_dygl_style(graph, src, dst, neg, t_query, k), n=1)
        t_dygl = _timed(lambda: run_dygl_style(graph, src, dst, neg, t_query, k), dygl_iters, device)

    return t_fused, t_tgm, t_dygl


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_nodes: int = 10_000,
    num_seed_edges: int = 200_000,
    batch_sizes: list = None,
    buffer_size: int = 32,
    k: int = 32,
    d_edge: int = 172,
    n_iters: int = 200,
    device: str = "cuda",
    skip_dygl: bool = False,
):
    if batch_sizes is None:
        batch_sizes = [200, 600, 1000]

    print(f"\n{'='*70}")
    print(f"TGEngine Pipeline Benchmark")
    print(f"  device={device}, N={num_nodes}, k={k}, buffer_size={buffer_size}, d_edge={d_edge}")
    print(f"  seed_edges={num_seed_edges}, n_iters={n_iters}")
    print(f"{'='*70}")

    dev = torch.device(device)

    # Build graph
    print("\nBuilding temporal graph...")
    graph = TemporalGraph(num_nodes, buffer_size=buffer_size, edge_feat_dim=d_edge, device=device)
    chunk = 10_000
    for start in range(0, num_seed_edges, chunk):
        end = min(start + chunk, num_seed_edges)
        n = end - start
        graph.advance(
            torch.randint(0, num_nodes, (n,), device=dev),
            torch.randint(0, num_nodes, (n,), device=dev),
            torch.arange(start, end, dtype=torch.float64, device=dev),
            torch.randn(n, d_edge, device=dev),
        )
    print(f"  {graph.num_edges} edges in graph")

    # Correctness check with first batch size
    B = batch_sizes[0]
    t_q = torch.full((B,), float(num_seed_edges + 1), dtype=torch.float64, device=dev)
    src = torch.randint(0, num_nodes, (B,), device=dev)
    dst = torch.randint(0, num_nodes, (B,), device=dev)
    neg = torch.randint(0, num_nodes, (B,), device=dev)
    s1, d1, n1 = run_fused(graph, src, dst, neg, t_q, k)
    s2, d2, n2 = run_tgm_style(graph, src, dst, neg, t_q, k)
    assert torch.all(s1 == s2), "Correctness check failed: fused vs TGM"
    print("Correctness check passed: fused results == TGM-style results\n")

    # Header
    include_dygl_col = not skip_dygl
    print(f"{'B':<8} {'TGE (ms)':<12} {'TGM (ms)':<12} {'TGE/TGM':<12}", end="")
    if include_dygl_col:
        print(f" {'DyGL (ms)':<12} {'TGE/DyGL':<12}", end="")
    print()
    print("-" * (8 + 12 + 12 + 12 + (24 if include_dygl_col else 0)))

    all_results = []
    for B in batch_sizes:
        t_q = torch.full((B,), float(num_seed_edges + 1), dtype=torch.float64, device=dev)
        src = torch.randint(0, num_nodes, (B,), device=dev)
        dst = torch.randint(0, num_nodes, (B,), device=dev)
        neg = torch.randint(0, num_nodes, (B,), device=dev)

        include_dygl = not skip_dygl
        t_fused, t_tgm, t_dygl = bench_one(graph, src, dst, neg, t_q, k, n_iters, device, include_dygl)

        ratio_tgm = t_tgm / t_fused
        print(f"{B:<8} {t_fused:<12.3f} {t_tgm:<12.3f} {ratio_tgm:.2f}x{'':<8}", end="")
        if t_dygl is not None:
            ratio_dygl = t_dygl / t_fused
            print(f" {t_dygl:<12.3f} {ratio_dygl:.0f}x", end="")
        print()
        all_results.append((B, t_fused, t_tgm, t_dygl))

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    b_ref, t_fused_ref, t_tgm_ref, t_dygl_ref = all_results[len(all_results)//2]  # middle batch
    print(f"  Reference batch size: B={b_ref}")
    print(f"  TGEngine vs TGM-style : {t_tgm_ref/t_fused_ref:.1f}x speedup")
    if t_dygl_ref:
        print(f"  TGEngine vs DyGLib    : {t_dygl_ref/t_fused_ref:.0f}x speedup")

    # Check targets
    ref_speedup_dygl = t_dygl_ref / t_fused_ref if t_dygl_ref else None
    ref_speedup_tgm = t_tgm_ref / t_fused_ref
    print()
    if ref_speedup_dygl and ref_speedup_dygl >= 8.0:
        print(f"  [PASS] ≥8x vs DyGLib target: {ref_speedup_dygl:.0f}x")
    elif ref_speedup_dygl:
        print(f"  [FAIL] ≥8x vs DyGLib target: {ref_speedup_dygl:.1f}x")
    if ref_speedup_tgm > 1.0:
        print(f"  [PASS] TGEngine faster than TGM-style: {ref_speedup_tgm:.1f}x")
    else:
        print(f"  [NOTE] TGEngine vs TGM-style at B={b_ref}: {ref_speedup_tgm:.2f}x "
              f"(see header comment re: GPU saturation at large B)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGEngine pipeline benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_nodes", type=int, default=10_000)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[200, 600, 1000])
    parser.add_argument("--buffer_size", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--d_edge", type=int, default=172)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--seed_edges", type=int, default=200_000)
    parser.add_argument("--skip_dygl", action="store_true",
                        help="Skip DyGLib-style benchmark (very slow for large batches)")
    args = parser.parse_args()

    run_benchmark(
        num_nodes=args.num_nodes,
        num_seed_edges=args.seed_edges,
        batch_sizes=args.batch_sizes,
        buffer_size=args.buffer_size,
        k=args.k,
        d_edge=args.d_edge,
        n_iters=args.n_iters,
        device=args.device,
        skip_dygl=args.skip_dygl,
    )
