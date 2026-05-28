"""Batch size sweep experiment on UCI dataset.

Sweeps batch_size from 200 to 10000 (step 200), measuring for each:
  - Total training wall time
  - Per-epoch wall time
  - Best validation AP
  - Test AP at best epoch
  - Average GPU utilization (pynvml, ~50ms sampling)

Config: DyGFormer, max 50 epochs, early stopping (patience=5).

Usage:
    python benchmarks/bench_bs_sweep.py
    python benchmarks/bench_bs_sweep.py --start 200 --end 2000 --step 200
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import torch

from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import Engine, TrainConfig, APEval
from tgengine.models.dygformer import DyGFormer
from tgengine.pipeline.negatives import RandomNegative

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"


# ---------------------------------------------------------------------------
# GPU monitor (pynvml background thread)
# ---------------------------------------------------------------------------

def _gpu_monitor_thread(samples: list, stop: threading.Event, interval: float = 0.05):
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(1)  # physical GPU 1 = RTX 4080
        while not stop.is_set():
            util = nvmlDeviceGetUtilizationRates(handle)
            samples.append(util.gpu)
            time.sleep(interval)
        nvmlShutdown()
    except Exception:
        pass


def run_sweep_one(
    ds, batch_size: int, device: str = "cuda",
    node_feat=None, valid_dst_nodes=None,
) -> dict | None:
    """Train DyGFormer on UCI at given batch_size. Returns metrics dict or None on OOM."""

    n_train = ds.train_end
    n_nodes = ds.num_nodes
    d_edge = ds.edge_feat_dim

    # ---- graph ----------------------------------------------------------
    buf = min(32, n_nodes)
    mem_gb = n_nodes * buf * max(d_edge, 1) * 4 / 1e9
    if mem_gb > 6.0:
        buf = max(4, int(32 * 6.0 / mem_gb))

    graph = TemporalGraph(n_nodes, buffer_size=buf,
                          edge_feat_dim=d_edge, device=device)
    batches = ds.get_batches("train", batch_size, device)
    n = len(batches)
    # 70/15/15 split with minimum 1 batch each for val/test
    n_val = max(1, n // 7)
    n_test = max(1, n // 7)
    n_train = n - n_val - n_test
    train_batches = batches[:n_train]
    val_batches = batches[n_train:n_train + n_val]
    test_batches = batches[n_train + n_val:]

    # Pre-populate
    warmup_n = max(5, n_train // 10)
    for rb in train_batches[:warmup_n]:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

    # ---- model ----------------------------------------------------------
    model = DyGFormer(
        d_model=172, d_edge=d_edge, d_time=100,
        d_channel=50, K=31, n_layers=2,
        node_feat=node_feat,
    )

    neg_strat = RandomNegative(n_nodes, valid_dst_nodes=valid_dst_nodes)
    eval_proto = APEval()
    config = TrainConfig(
        epochs=50, batch_size=batch_size, lr=1e-4,
        patience=5, device=device, seed=42,
    )

    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    neg_strat, eval_proto, config)

    # ---- start GPU monitor ----------------------------------------------
    gpu_samples: list[float] = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_gpu_monitor_thread, args=(gpu_samples, stop_event), daemon=True
    )
    monitor.start()
    time.sleep(0.3)

    # ---- train ----------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        test_metrics = engine.train()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    except RuntimeError as e:
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if "out of memory" in str(e).lower():
            stop_event.set()
            monitor.join(timeout=1)
            torch.cuda.empty_cache()
            return {"batch_size": batch_size, "oom": True, "error": str(e)}
        raise

    # ---- stop monitor ---------------------------------------------------
    time.sleep(0.2)
    stop_event.set()
    monitor.join(timeout=2)

    valid_gpu = [v for v in gpu_samples if 0.0 <= v <= 100.0]
    avg_gpu = sum(valid_gpu) / len(valid_gpu) if valid_gpu else 0.0

    n_epochs = engine._current_epoch
    avg_epoch_time = wall / n_epochs if n_epochs > 0 else 0.0
    n_edges_per_epoch = len(train_batches) * batch_size
    throughput = n_edges_per_epoch * n_epochs / wall if wall > 0 else 0.0

    return {
        "batch_size": batch_size,
        "oom": False,
        "n_epochs": n_epochs,
        "total_wall_sec": round(wall, 2),
        "avg_epoch_sec": round(avg_epoch_time, 2),
        "throughput_edges_per_sec": round(throughput, 0),
        "best_test_ap": round(test_metrics.get("ap", 0.0), 4),
        "avg_gpu_util_pct": round(avg_gpu, 1),
        "n_gpu_samples": len(valid_gpu),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="bench_bs_sweep.json")
    args = parser.parse_args()

    device = args.device
    batch_sizes = list(range(args.start, args.end + 1, args.step))

    print("=" * 72)
    print(f"  UCI Batch Size Sweep  (DyGFormer, max 50 epochs, early stop patience=5)")
    print(f"  BS: {args.start} → {args.end}, step={args.step}  ({len(batch_sizes)} points)")
    print("=" * 72)

    # ---- load dataset once -----------------------------------------------
    print("\nLoading UCI dataset...", end="", flush=True)
    ds = load_dataset("uci", DATA_ROOT)
    print(f" {ds.num_edges:,} edges, {ds.num_nodes:,} nodes, d_edge={ds.edge_feat_dim}")

    # Compute valid dst nodes from training set (matching DyGLib)
    train_dst = ds.dst[:ds.train_end]
    valid_dst_nodes = torch.unique(train_dst)
    node_feat = ds.node_feat  # may be None if no node features
    if node_feat is not None:
        print(f"  Node features: {node_feat.shape}, valid dst nodes: {len(valid_dst_nodes)}")
    else:
        print(f"  No node features, valid dst nodes: {len(valid_dst_nodes)}")

    results = []
    for i, bs in enumerate(batch_sizes):
        label = f"[{i+1}/{len(batch_sizes)}] BS={bs}"
        print(f"\n{label} ", end="", flush=True)
        start_t = time.perf_counter()

        r = run_sweep_one(ds, bs, device, node_feat=node_feat, valid_dst_nodes=valid_dst_nodes)
        if r is None:
            print("ERROR")
            continue

        elapsed = time.perf_counter() - start_t
        if r.get("oom"):
            print(f"OOM ({elapsed:.0f}s) — stopping sweep")
            results.append(r)
            break

        print(f"epochs={r['n_epochs']} wall={r['total_wall_sec']}s "
              f"ap={r['best_test_ap']} gpu={r['avg_gpu_util_pct']}% ({elapsed:.0f}s real)")
        results.append(r)

        # Save incrementally
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # ---- summary ---------------------------------------------------------
    valid = [r for r in results if not r.get("oom")]
    if valid:
        print(f"\n{'=' * 72}")
        print(f"  Summary ({len(valid)} valid points)")
        print(f"{'=' * 72}")
        print(f"{'BS':>8} {'Epochs':>7} {'TotWall':>9} {'EpochTime':>10} "
              f"{'e/s':>8} {'TestAP':>8} {'GPU%':>7}")
        print(f"{'─' * 62}")
        for r in valid:
            print(f"{r['batch_size']:>8} {r['n_epochs']:>7} {r['total_wall_sec']:>8.1f}s "
                  f"{r['avg_epoch_sec']:>8.1f}s {r['throughput_edges_per_sec']:>8,.0f} "
                  f"{r['best_test_ap']:>8.4f} {r['avg_gpu_util_pct']:>6.1f}%")

        # Best AP
        best = max(valid, key=lambda r: r["best_test_ap"])
        print(f"\n  Best AP: BS={best['batch_size']} → AP={best['best_test_ap']}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
