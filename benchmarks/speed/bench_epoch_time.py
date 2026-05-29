"""Measure per-epoch wall time and true avg GPU utilization via pynvml.

Usage:
    python benchmarks/bench_epoch_time.py --datasets wiki uci --batch_size 200
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import torch

from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.dygformer import DyGFormer
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"

DATASETS = {
    "wiki": ("wikipedia", DATA_ROOT),
    "uci":  ("uci",       DATA_ROOT),
}


def _gpu_monitor_thread(results: list, stop_event: threading.Event, interval: float = 0.05):
    """Poll GPU utilization via pynvml at given interval. Appends to results list."""
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
        nvmlInit()
        # CUDA_VISIBLE_DEVICES=0 maps to physical GPU 1 (RTX 4080) on scnu
        # pynvml uses nvidia-smi ordering, so index 1 = RTX 4080
        handle = nvmlDeviceGetHandleByIndex(1)
        while not stop_event.is_set():
            util = nvmlDeviceGetUtilizationRates(handle)
            results.append(util.gpu)
            time.sleep(interval)
        nvmlShutdown()
    except Exception as e:
        results.append(-1.0)  # sentinel for error
        results.append(float(str(e)) if e else 0.0)


def run_one_epoch(
    dataset_name: str, data_path: str, batch_size: int = 200, device: str = "cuda",
) -> dict:
    # ---- load -----------------------------------------------------------
    ds = load_dataset(dataset_name, data_path)
    n_train = ds.train_end
    n_batches = (n_train + batch_size - 1) // batch_size
    print(f"  {dataset_name}: {ds.num_edges:,} edges, train={n_train:,} "
          f"→ {n_batches} batches (B={batch_size})")

    # ---- graph ----------------------------------------------------------
    buf = min(32, ds.num_nodes)
    mem_gb = ds.num_nodes * buf * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 6.0:
        buf = max(4, int(32 * 6.0 / mem_gb))
        print(f"  Ring buffer: 32→{buf} (est {mem_gb:.1f} GB > 6 GB limit)")

    graph = TemporalGraph(ds.num_nodes, buffer_size=buf,
                          edge_feat_dim=ds.edge_feat_dim, device=device)

    # Pre-populate
    batches = ds.get_batches("train", batch_size, device)
    warmup_n = max(5, n_batches // 10)
    for rb in batches[:warmup_n]:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    print(f"  Pre-populated {warmup_n * batch_size:,} edges")

    # ---- model ----------------------------------------------------------
    model = DyGFormer(
        d_model=172, d_edge=ds.edge_feat_dim, d_time=100,
        d_channel=50, K=31, n_layers=2,
        node_feat=ds.node_feat,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: DyGFormer ({n_params:,} params)")

    pipeline = DataPipeline(model.gather_spec, graph)

    # Build valid dst node set for negative sampling (matching DyGLib)
    train_dst = ds.dst[:ds.train_end]
    valid_dst_nodes = torch.unique(train_dst)
    neg_strat = RandomNegative(ds.num_nodes, valid_dst_nodes=valid_dst_nodes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- warmup ---------------------------------------------------------
    print("  Warming up (10 batches)...", end="", flush=True)
    model.train()
    for i in range(min(10, n_batches - warmup_n)):
        rb = batches[warmup_n + i]
        rb.neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
        prepared = pipeline.prepare(rb)
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    torch.cuda.synchronize()
    print(" done")

    # ---- start pynvml GPU monitor in background thread ------------------
    gpu_samples: list[float] = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_gpu_monitor_thread, args=(gpu_samples, stop_event), daemon=True
    )
    monitor.start()
    time.sleep(0.5)  # let pynvml initialize

    # ---- one epoch ------------------------------------------------------
    model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(warmup_n, len(batches)):
        rb = batches[i]
        rb.neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
        prepared = pipeline.prepare(rb)
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # ---- stop monitor ---------------------------------------------------
    time.sleep(0.3)  # capture tail-end GPU activity
    stop_event.set()
    monitor.join(timeout=2)

    # Filter sentinels
    valid = [v for v in gpu_samples if 0.0 <= v <= 100.0]
    avg_gpu = sum(valid) / len(valid) if valid else 0.0

    return {
        "dataset": dataset_name,
        "train_edges": n_train,
        "num_batches": n_batches - warmup_n,
        "batch_size": batch_size,
        "epoch_wall_sec": wall,
        "edges_per_sec": n_train / wall,
        "avg_gpu_pct": avg_gpu,
        "n_gpu_samples": len(valid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wiki", "uci"])
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Epoch Time & GPU Utilization  (DyGFormer, B={args.batch_size}, RTX 4080)")
    print(f"{sep}\n")

    results = []
    for name in args.datasets:
        dataset_name, data_path = DATASETS[name]
        try:
            r = run_one_epoch(dataset_name, data_path, args.batch_size, args.device)
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"  [FAILED] {name}: {exc}")
            traceback.print_exc()

    if results:
        W = 72
        print(f"\n{'─' * W}")
        print(f"{'Dataset':<12} {'Train':>10} {'Batches':>8} "
              f"{'Epoch':>10} {'edges/s':>10} {'GPU%':>8} {'Samples':>8}")
        print(f"{'─' * W}")
        for r in results:
            print(f"{r['dataset']:<12} {r['train_edges']:>10,} {r['num_batches']:>8} "
                  f"{r['epoch_wall_sec']:>8.1f}s {r['edges_per_sec']:>10,.0f} "
                  f"{r['avg_gpu_pct']:>7.1f}% {r['n_gpu_samples']:>8}")
        print(f"{'─' * W}")
        print(f"  GPU% = pynvml nvmlDeviceGetUtilizationRates() sampled every 50ms")
        print(f"  This is SM utilization: % of sample period where >=1 kernel was active")
        print()


if __name__ == "__main__":
    main()
