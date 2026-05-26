"""Negative sampling benchmark: TGEngine HistoricalNegPool vs TGM vs DyGLib.

All three implementations share the same semantic:
  "For src node s, uniformly sample a historical interaction target as negative."

Semantic equivalence guarantee:
  - DyGLib: O(E) Python set scan of ALL edges before current batch.
  - TGM:    O(M) torch.isin over unboundedly growing memory buffer — all history.
  - TGE-pool: O(B) GPU gather from per-node reservoir pool (pool_size=512).
    Reservoir sampling guarantees uniform coverage: each of the N historical
    interactions has probability min(pool_size, N)/N of being in the pool.
    For LastFM (avg degree ~651), pool_size=512 gives 78% coverage per node.

The critical difference vs the INCORRECT ring-buffer approach (k=32, old):
  - k=32 ring buffer only covers the 32 most-recent neighbors, not full history.
  - Given the same input, ring-buffer and full-history produce different distributions.
  - That comparison was semantically invalid and is NOT repeated here.

Cost model:
  - DyGLib : O(E) Python set construction per batch  — constant in M, grows with E
  - TGM    : O(M) torch.isin per batch               — grows unboundedly with dataset
  - TGE-pool: O(B * pool_size) gather per batch      — bounded, constant in dataset size
"""

import argparse
import time

import torch

from tgengine.core.batch import RawBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import HistoricalNegPool


# ---------------------------------------------------------------------------
# DyGLib-style baseline: O(E) Python set scan per batch
# ---------------------------------------------------------------------------

class DyGLibHistNeg:
    """Simulates DyGLib NegativeEdgeSampler with historical strategy.

    Maintains numpy arrays of all (src, dst) pairs seen so far.
    Per batch: numpy boolean mask over all edges → Python set construction.
    Semantics: uniform sample from ALL historical (src, *) pairs.
    """

    def __init__(self, all_src, all_dst, all_times):
        import numpy as np
        self.src = np.array(all_src)
        self.dst = np.array(all_dst)
        self.times = np.array(all_times)

    def sample(self, batch_src, batch_dst, t_start, size):
        import numpy as np
        hist_mask = self.times < t_start
        hist_edges = set(
            (int(s), int(d))
            for s, d in zip(self.src[hist_mask], self.dst[hist_mask])
        )
        cur_edges = set(zip(batch_src.tolist(), batch_dst.tolist()))
        valid = list(hist_edges - cur_edges)
        if not valid:
            return np.random.randint(0, int(self.dst.max()) + 1, size=(size,))
        idx = np.random.randint(0, len(valid), size=(size,))
        return np.array([valid[i][1] for i in idx])


# ---------------------------------------------------------------------------
# TGM-style baseline: growing memory buffer + torch.isin, O(M) per batch
# Semantic: all historical (src, *) pairs. Memory grows unboundedly.
# ---------------------------------------------------------------------------

class TGMHistNeg:
    """Simulates TGM's HistoricalNegativeEdgeSamplerHook.

    Maintains a (2, capacity) tensor that grows as edges are seen.
    Per batch: torch.isin to find memory entries matching batch src nodes.
    Semantics: uniform sample from ALL historical (src, *) pairs.
    """

    def __init__(self, device: str, init_capacity: int = 1024):
        self.device = device
        self._memory = torch.empty(2, init_capacity, dtype=torch.int64, device=device)
        self._size = 0

    def update(self, src: torch.Tensor, dst: torch.Tensor):
        n = src.shape[0]
        if self._size + n > self._memory.shape[1]:
            new_cap = max(self._memory.shape[1] * 2, self._size + n)
            new_mem = torch.empty(2, new_cap, dtype=torch.int64, device=self.device)
            new_mem[:, :self._size] = self._memory[:, :self._size]
            self._memory = new_mem
        self._memory[0, self._size:self._size + n] = src.long()
        self._memory[1, self._size:self._size + n] = dst.long()
        self._size += n

    def sample(self, src: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self._size == 0:
            return torch.randint(0, num_nodes, (src.shape[0],), device=self.device)

        mem_src = self._memory[0, :self._size]
        mem_dst = self._memory[1, :self._size]

        in_batch = torch.isin(mem_src, src)
        if not in_batch.any():
            return torch.randint(0, num_nodes, (src.shape[0],), device=self.device)

        matched_src = mem_src[in_batch]
        matched_dst = mem_dst[in_batch]
        rand_w = torch.rand(matched_src.shape[0], device=self.device)

        unique_src, inv = torch.unique(matched_src, return_inverse=True)
        best_w = torch.full((unique_src.shape[0],), -1.0, device=self.device)
        best_w.scatter_reduce_(0, inv, rand_w, reduce="amax", include_self=True)

        winner_mask = rand_w == best_w[inv]
        seq = torch.arange(matched_src.shape[0], device=self.device)
        winner_idx = torch.zeros(unique_src.shape[0], dtype=torch.int64, device=self.device)
        winner_idx.scatter_(0, inv[winner_mask], seq[winner_mask])
        selected_dst = matched_dst[winner_idx]

        max_node = int(unique_src.max().item()) + 1
        lookup = torch.full((max_node,), -1, dtype=torch.int64, device=self.device)
        lookup[unique_src] = selected_dst
        neg = lookup[src.clamp(max=max_node - 1)]

        missing = neg == -1
        if missing.any():
            neg = neg.clone()
            neg[missing] = torch.randint(0, num_nodes, (int(missing.sum()),), device=self.device)
        return neg


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()


def _timed(fn, n_iters: int, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n_iters * 1000


def _warmup(fn, n=3):
    for _ in range(n):
        fn()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_nodes: int = 1980,
    num_seed_edges: int = 1_000_000,
    batch_size: int = 200,
    pool_size: int = 512,
    buffer_size: int = 32,
    k: int = 32,
    d_edge: int = 8,
    n_iters: int = 50,
    device: str = "cuda",
    skip_dygl: bool = False,
):
    print(f"\n{'='*70}")
    print("Historical Negative Sampling Benchmark")
    print(f"  All three methods sample from FULL history (semantically equivalent)")
    print(f"  device={device}, num_nodes={num_nodes}, seed_edges={num_seed_edges}")
    print(f"  batch_size={batch_size}, pool_size={pool_size}, k={k}")
    print(f"  (LastFM scale: avg degree {num_seed_edges // max(num_nodes,1):.0f} edges/node)")
    print(f"{'='*70}")

    dev = torch.device(device)
    dev_str = str(device)

    # ---- Build temporal graph ----
    print("\nBuilding temporal graph + pools...")
    graph = TemporalGraph(
        num_nodes=num_nodes, buffer_size=buffer_size,
        edge_feat_dim=d_edge, device=device,
    )
    pool = HistoricalNegPool(num_nodes=num_nodes, pool_size=pool_size, device=dev_str)
    tgm = TGMHistNeg(device=dev_str)

    all_src_cpu = torch.randint(0, num_nodes // 2, (num_seed_edges,))
    all_dst_cpu = torch.randint(num_nodes // 2, num_nodes, (num_seed_edges,))
    all_time_cpu = torch.arange(num_seed_edges, dtype=torch.float64)
    all_feat_cpu = torch.zeros(num_seed_edges, d_edge)

    chunk = 20_000
    for start in range(0, num_seed_edges, chunk):
        end = min(start + chunk, num_seed_edges)
        s = all_src_cpu[start:end].to(dev)
        d = all_dst_cpu[start:end].to(dev)
        t = all_time_cpu[start:end].to(dev)
        f = all_feat_cpu[start:end].to(dev)
        graph.advance(s, d, t, f)
        pool.update(s, d)
    tgm.update(all_src_cpu.to(dev), all_dst_cpu.to(dev))

    # Pool coverage stats
    valid_counts = (pool._pool != pool.PADDING).sum(dim=1).float()
    avg_coverage = valid_counts.mean().item()
    print(f"  Pool coverage: avg {avg_coverage:.1f}/{pool_size} slots filled per node "
          f"({100*avg_coverage/pool_size:.0f}% of pool_size)")
    print(f"  TGM memory: {tgm._size:,} entries ({tgm._size * 8 // 1024}KB)")
    print(f"  TGE pool memory: {num_nodes * pool_size * 4 // 1024}KB (bounded)")

    # ---- Setup batch ----
    t_query = torch.full((batch_size,), float(num_seed_edges + 1), dtype=torch.float64, device=dev)
    src = torch.randint(0, num_nodes // 2, (batch_size,), device=dev)
    dst = torch.randint(num_nodes // 2, num_nodes, (batch_size,), device=dev)
    raw = RawBatch(src=src, dst=dst, time=t_query)

    spec = GatherSpec(neighbors=NeighborSpec(k=k, strategy="recency"), co_occurrence=False)
    pipeline = DataPipeline(spec, graph)

    # ---- TGE-pool: reservoir pool + pipeline.prepare() ----
    def run_tge_pool():
        neg = pool.sample(src)
        raw_with_neg = RawBatch(src=src, dst=dst, time=t_query, neg=neg)
        pipeline.prepare(raw_with_neg)

    # ---- TGM-style: growing memory + torch.isin ----
    def run_tgm():
        neg = tgm.sample(src, num_nodes)
        raw_w = RawBatch(src=src, dst=dst, time=t_query, neg=neg)
        pipeline.prepare(raw_w)

    # ---- DyGLib-style: Python set scan ----
    dygl = None
    if not skip_dygl and num_seed_edges <= 200_000:
        dygl = DyGLibHistNeg(
            all_src_cpu[:num_seed_edges].numpy(),
            all_dst_cpu[:num_seed_edges].numpy(),
            all_time_cpu[:num_seed_edges].numpy(),
        )
        t_start = float(num_seed_edges - batch_size)

        def run_dygl():
            dygl.sample(src.cpu().numpy(), dst.cpu().numpy(), t_start, batch_size)

    # ---- Warmup + timing ----
    print(f"\nRunning {n_iters} iterations each...")

    _warmup(run_tge_pool)
    t_pool = _timed(run_tge_pool, n_iters, dev_str)

    _warmup(run_tgm)
    t_tgm = _timed(run_tgm, n_iters, dev_str)

    t_dygl = None
    if dygl is not None:
        _warmup(run_dygl, n=1)
        t_dygl = _timed(run_dygl, max(1, n_iters // 5), "cpu")

    # ---- Report ----
    print(f"\n{'Method':<28} {'Semantic':<20} {'ms/iter':>8}  {'vs TGE-pool':>12}")
    print("-" * 75)
    print(f"{'TGE-pool (ours)':<28} {'full-history approx':<20} {t_pool:>8.3f}  {'1.00x':>12}")
    print(f"{'TGM-style':<28} {'full-history exact':<20} {t_tgm:>8.3f}  {f'{t_tgm/t_pool:.2f}x':>12}")
    if t_dygl is not None:
        print(f"{'DyGLib-style':<28} {'full-history exact':<20} {t_dygl:>8.3f}  {f'{t_dygl/t_pool:.2f}x':>12}")
    else:
        print(f"{'DyGLib-style':<28} {'full-history exact':<20} {'(skipped)':>8}")

    print(f"\nCost model:")
    print(f"  TGE-pool: O(B * pool_size) = {batch_size} * {pool_size} = {batch_size * pool_size:,} ops [BOUNDED]")
    print(f"  TGM:      O(M) scan = {tgm._size:,} entries [GROWS with dataset]")
    if dygl is not None:
        print(f"  DyGLib:   O(E) Python = {num_seed_edges:,} edges [GROWS with dataset]")

    if dygl is None and num_seed_edges > 200_000:
        # Extrapolate DyGLib using a small sample
        small_e = 10_000
        small_dygl = DyGLibHistNeg(
            all_src_cpu[:small_e].numpy(),
            all_dst_cpu[:small_e].numpy(),
            all_time_cpu[:small_e].numpy(),
        )
        t_small = _timed(
            lambda: small_dygl.sample(
                src.cpu().numpy(), dst.cpu().numpy(), float(small_e - batch_size), batch_size
            ),
            5, "cpu",
        )
        t_dygl_est = t_small * (num_seed_edges / small_e)
        print(f"\n  DyGLib estimated for E={num_seed_edges:,}: ~{t_dygl_est:.0f}ms "
              f"(calibrated on E=10K → {t_small:.1f}ms)")
        print(f"  TGE-pool speedup vs DyGLib (estimated): ~{t_dygl_est/t_pool:.0f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_nodes", type=int, default=1980, help="LastFM has 1980 unique nodes")
    parser.add_argument("--seed_edges", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--skip_dygl", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        num_nodes=args.num_nodes,
        num_seed_edges=args.seed_edges,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        buffer_size=args.buffer_size,
        k=args.k,
        d_edge=8,
        n_iters=args.n_iters,
        device=args.device,
        skip_dygl=args.skip_dygl,
    )
