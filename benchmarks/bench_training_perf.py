"""Training performance analysis across dataset scales.

Measures:
  - End-to-end training throughput (edges/sec, batches/sec)
  - GPU utilization proxy (CUDA kernel time / wall time)
  - Pipeline component breakdown:
      T_sample   = graph.recent() (GPU neighbor sampling)
      T_forward  = model.forward()
      T_backward = loss.backward() + optimizer.step()
      T_other    = neg sampling, Python overhead, etc.
  - TGE-fused (1 kernel) vs TGM-style (3 separate kernels) comparison

GPU utilization note:
  CUDA events measure time on the GPU stream.
  Ratio (total_cuda_ms / wall_ms) is an *upper bound* on true SM utilization
  (a single lightweight kernel still counts as "GPU busy").
  Values >100% are impossible; values near 100% mean minimal CPU↔GPU sync gap.

Usage:
    python benchmarks/bench_training_perf.py                      # all datasets
    python benchmarks/bench_training_perf.py --datasets wiki      # just Wikipedia
    python benchmarks/bench_training_perf.py --n_profile 50       # faster run
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"

DATASETS = {
    "wiki":   ("wikipedia", DATA_ROOT),
    "reddit": ("reddit",    DATA_ROOT),
    "lastfm": ("lastfm",    DATA_ROOT),
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class EventRecorder:
    """Accumulate CUDA event pairs; sync and sum once at end."""

    def __init__(self):
        self._pairs: list[tuple] = []

    @contextmanager
    def record(self):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        yield
        e.record()
        self._pairs.append((s, e))

    def total_ms(self) -> float:
        """Sync once, return total ms across all recorded pairs."""
        if not self._pairs:
            return 0.0
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self._pairs)

    def reset(self):
        self._pairs.clear()


@dataclass
class PerfStats:
    dataset: str
    num_edges: int
    num_nodes: int
    profiled_batches: int
    batch_size: int

    # Throughput pass (no instrumentation)
    wall_sec: float = 0.0

    # Breakdown pass (CUDA event recording)
    cuda_sample_ms: float = 0.0
    cuda_forward_ms: float = 0.0
    cuda_backward_ms: float = 0.0

    # TGM-style comparison
    wall_tgm_sec: float = 0.0
    cuda_sample_tgm_ms: float = 0.0

    # Async pipeline comparison
    wall_async_sec: float = 0.0

    @property
    def edges_per_sec(self) -> float:
        if self.wall_sec == 0:
            return 0.0
        return self.profiled_batches * self.batch_size / self.wall_sec

    @property
    def batches_per_sec(self) -> float:
        return self.profiled_batches / max(self.wall_sec, 1e-9)

    @property
    def total_cuda_ms(self) -> float:
        return self.cuda_sample_ms + self.cuda_forward_ms + self.cuda_backward_ms

    @property
    def gpu_util_pct(self) -> float:
        wall_ms = self.wall_sec * 1000
        if wall_ms <= 0:
            return 0.0
        return min(100.0, self.total_cuda_ms / wall_ms * 100)

    @property
    def speedup_vs_tgm(self) -> float:
        return self.wall_tgm_sec / max(self.wall_sec, 1e-9)

    @property
    def speedup_async(self) -> float:
        return self.wall_sec / max(self.wall_async_sec, 1e-9)

    def breakdown_pct(self) -> dict[str, float]:
        wall_ms = self.wall_sec * 1000
        return {
            "sample":   self.cuda_sample_ms / wall_ms * 100,
            "forward":  self.cuda_forward_ms / wall_ms * 100,
            "backward": self.cuda_backward_ms / wall_ms * 100,
            "other":    max(0.0, (wall_ms - self.total_cuda_ms) / wall_ms * 100),
        }


# ---------------------------------------------------------------------------
# TGM-style pipeline (3 separate graph.recent calls)
# ---------------------------------------------------------------------------

class TGMStylePipeline:
    """Simulates TGM's 3-call approach: separate graph.recent for src/dst/neg."""

    def __init__(self, spec, graph):
        self.spec = spec
        self.graph = graph

    def prepare(self, raw_batch):
        from tgengine.core.batch import NeighborData, PreparedBatch
        k = self.spec.neighbors.k
        t = raw_batch.time

        s = self.graph.recent(raw_batch.src, t, k)
        d = self.graph.recent(raw_batch.dst, t, k)
        n = self.graph.recent(raw_batch.neg, t, k)

        return PreparedBatch(
            src=raw_batch.src, dst=raw_batch.dst, neg=raw_batch.neg, time=t,
            src_neighbors=NeighborData(s.neighbor_ids, s.timestamps, s.edge_feats, s.mask),
            dst_neighbors=NeighborData(d.neighbor_ids, d.timestamps, d.edge_feats, d.mask),
            neg_neighbors=NeighborData(n.neighbor_ids, n.timestamps, n.edge_feats, n.mask),
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _warmup(model, pipeline, batches, neg_strat, graph, opt, n=5):
    """Run a few batches to warm up CUDA kernels and caches."""
    from tgengine.core.batch import RawBatch
    model.train()
    for i in range(min(n, len(batches))):
        rb = batches[i]
        neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
        rb = RawBatch(src=rb.src, dst=rb.dst, time=rb.time, edge_feat=rb.edge_feat, neg=neg)
        prepared = pipeline.prepare(rb)
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    torch.cuda.synchronize()


def _throughput_pass(model, pipeline, batches, neg_strat, graph, opt, n_profile) -> float:
    """Run n_profile batches with no instrumentation; return wall seconds."""
    from tgengine.core.batch import RawBatch
    model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_profile):
        rb = batches[i % len(batches)]
        neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
        rb = RawBatch(src=rb.src, dst=rb.dst, time=rb.time, edge_feat=rb.edge_feat, neg=neg)
        prepared = pipeline.prepare(rb)
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _breakdown_pass(model, pipeline, batches, neg_strat, graph, opt, n_profile
                    ) -> tuple[float, float, float]:
    """Run n_profile batches with CUDA event recording; return (sample, fwd, bwd) ms totals."""
    from tgengine.core.batch import RawBatch
    model.train()
    rec_s = EventRecorder()
    rec_f = EventRecorder()
    rec_b = EventRecorder()

    for i in range(n_profile):
        rb = batches[i % len(batches)]
        neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
        rb = RawBatch(src=rb.src, dst=rb.dst, time=rb.time, edge_feat=rb.edge_feat, neg=neg)

        with rec_s.record():
            prepared = pipeline.prepare(rb)

        opt.zero_grad()
        with rec_f.record():
            out = model(prepared)

        with rec_b.record():
            out.loss.backward()
            opt.step()

        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

    return rec_s.total_ms(), rec_f.total_ms(), rec_b.total_ms()


def _async_throughput_pass(model, spec, neg_strat, graph, opt, batches, n_profile) -> float:
    """Async-pipeline throughput pass. Uses AsyncDataPipeline double-buffer."""
    from tgengine.core.batch import RawBatch
    from tgengine.pipeline.async_pipeline import AsyncDataPipeline

    pipe = AsyncDataPipeline(spec, graph, neg_strat)
    model.train()
    actual = min(n_profile, len(batches))

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    pipe.start_prefetch(batches[0 % len(batches)])
    for i in range(actual):
        rb, prepared = pipe.get()
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
        if i + 1 < actual:
            pipe.start_prefetch(batches[(i + 1) % len(batches)])

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_dataset(
    name: str, dataset_name: str, data_path: str,
    batch_size: int = 200, n_profile: int = 100, device: str = "cuda",
) -> PerfStats:
    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Dataset: {name.upper()} ({dataset_name})")
    print(sep)

    print("  Loading dataset ...", end="", flush=True)
    ds = load_dataset(dataset_name, data_path)
    print(f" {ds.num_edges:,} edges, {ds.num_nodes:,} nodes, d_edge={ds.edge_feat_dim}")

    # Estimate GPU memory for ring buffer; fall back to smaller buffer if needed.
    buf = 32
    mem_gb = ds.num_nodes * buf * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 8.0:
        buf = max(4, int(32 * 8.0 / mem_gb))
        print(f"  Ring buffer reduced {32}→{buf} (estimated {mem_gb:.1f} GB > 8 GB limit)")

    graph = TemporalGraph(ds.num_nodes, buffer_size=buf,
                          edge_feat_dim=ds.edge_feat_dim, device=device)
    train_batches = ds.get_batches("train", batch_size, device)
    print(f"  Train batches: {len(train_batches)}")

    # Pre-populate graph with first 10% so sampling has data
    warmup_end = max(10, len(train_batches) // 10)
    for rb in train_batches[:warmup_end]:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    print(f"  Graph pre-populated with {warmup_end * batch_size:,} edges")

    model = DyGFormer(
        d_model=172, d_edge=ds.edge_feat_dim, d_time=100,
        d_channel=50, K=buf, n_layers=2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: DyGFormer  ({n_params:,} params)")

    pipeline = DataPipeline(model.gather_spec, graph)
    tgm_pipeline = TGMStylePipeline(model.gather_spec, graph)
    neg_strat = RandomNegative(ds.num_nodes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    profile_batches = train_batches[warmup_end: warmup_end + n_profile + 20]
    actual = min(n_profile, len(profile_batches))

    # ---- TGE-fused: warmup + throughput pass + breakdown pass ----
    print(f"  [TGE-fused]  warmup ...", end="", flush=True)
    _warmup(model, pipeline, profile_batches, neg_strat, graph, opt, n=8)
    print(f" profiling {actual} batches ...", end="", flush=True)
    wall = _throughput_pass(model, pipeline, profile_batches, neg_strat, graph, opt, actual)
    s_ms, f_ms, b_ms = _breakdown_pass(model, pipeline, profile_batches, neg_strat, graph, opt, actual)
    print(f" done  ({wall:.2f}s)")

    stats = PerfStats(
        dataset=name.upper(), num_edges=ds.num_edges, num_nodes=ds.num_nodes,
        profiled_batches=actual, batch_size=batch_size,
        wall_sec=wall, cuda_sample_ms=s_ms, cuda_forward_ms=f_ms, cuda_backward_ms=b_ms,
    )

    # ---- TGM-style: warmup + throughput + sample breakdown ----
    print(f"  [TGM-style]  warmup ...", end="", flush=True)
    _warmup(model, tgm_pipeline, profile_batches, neg_strat, graph, opt, n=8)
    print(f" profiling {actual} batches ...", end="", flush=True)
    tgm_wall = _throughput_pass(model, tgm_pipeline, profile_batches, neg_strat, graph, opt, actual)
    tgm_s_ms, _, _ = _breakdown_pass(model, tgm_pipeline, profile_batches, neg_strat, graph, opt, actual)
    print(f" done  ({tgm_wall:.2f}s)")

    stats.wall_tgm_sec = tgm_wall
    stats.cuda_sample_tgm_ms = tgm_s_ms

    # ---- Async pipeline comparison ----
    print(f"  [async]      warmup ...", end="", flush=True)
    _warmup(model, pipeline, profile_batches, neg_strat, graph, opt, n=8)
    print(f" profiling {actual} batches ...", end="", flush=True)
    async_wall = _async_throughput_pass(
        model, model.gather_spec, neg_strat, graph, opt, profile_batches, actual
    )
    print(f" done  ({async_wall:.2f}s)")
    stats.wall_async_sec = async_wall

    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_stats: list[PerfStats]):
    W = 74
    print(f"\n{'='*W}")
    print("  TGEngine — Training Performance Analysis  (DyGFormer, B=200)")
    print(f"{'='*W}")

    # Throughput table
    print(f"\n{'Dataset':<10} {'Edges':>12} {'Nodes':>8} "
          f"{'edges/s':>10} {'batches/s':>11} {'GPU util':>9} {'vs TGM':>8} {'async':>7}")
    print(f"{'─'*10} {'─'*12} {'─'*8} {'─'*10} {'─'*11} {'─'*9} {'─'*8} {'─'*7}")
    for s in all_stats:
        print(f"{s.dataset:<10} {s.num_edges:>12,} {s.num_nodes:>8,} "
              f"{s.edges_per_sec:>10,.0f} {s.batches_per_sec:>11.1f} "
              f"{s.gpu_util_pct:>8.1f}% {s.speedup_vs_tgm:>7.2f}x {s.speedup_async:>6.2f}x")

    # Pipeline breakdown
    print(f"\nWall-time breakdown (% of throughput-pass wall time):")
    print(f"{'Dataset':<10} {'T_sample':>10} {'T_forward':>10} {'T_backward':>12} {'T_other':>9}")
    print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*9}")
    for s in all_stats:
        bd = s.breakdown_pct()
        print(f"{s.dataset:<10} {bd['sample']:>9.1f}% {bd['forward']:>9.1f}% "
              f"{bd['backward']:>11.1f}% {bd['other']:>8.1f}%")

    print(f"\n  Note: T_other = Python overhead, neg sampling, data transfer.")
    print(f"  GPU util proxy = CUDA kernel time / wall time (breakdown pass).")
    print(f"  async = speedup of AsyncDataPipeline vs sync DataPipeline.")
    print(f"  Values >100% not possible; breakdown pass has more overhead than throughput pass.")

    # Sampling comparison
    print(f"\nGraph sampling kernel time (ms / batch):")
    print(f"{'Dataset':<10} {'TGE (fused)':>13} {'TGM (3 calls)':>14} {'speedup':>9}")
    print(f"{'─'*10} {'─'*13} {'─'*14} {'─'*9}")
    for s in all_stats:
        if s.profiled_batches > 0:
            tge = s.cuda_sample_ms / s.profiled_batches
            tgm = s.cuda_sample_tgm_ms / s.profiled_batches
            sp  = tgm / max(tge, 1e-9)
            print(f"{s.dataset:<10} {tge:>12.3f}ms {tgm:>13.3f}ms {sp:>8.2f}x")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=["wiki", "reddit", "lastfm"],
                        choices=list(DATASETS))
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_profile", type=int, default=100,
                        help="Batches to profile per dataset (each dataset runs 4 passes)")
    args = parser.parse_args()

    all_stats: list[PerfStats] = []
    for name in args.datasets:
        dataset_name, data_path = DATASETS[name]
        try:
            s = run_dataset(name, dataset_name, data_path,
                            args.batch_size, args.n_profile, args.device)
            all_stats.append(s)
        except Exception as exc:
            import traceback
            print(f"\n  [FAILED] {name}: {exc}")
            traceback.print_exc()

    if all_stats:
        print_report(all_stats)


if __name__ == "__main__":
    main()
