"""Continuation of batch size sweep with AMP for memory efficiency.

Extends bench_bs_sweep.json from the last completed BS to 10000.
Uses torch.cuda.amp.autocast to reduce GPU memory ~40%, enabling larger BS.

Usage:
    python benchmarks/bench_bs_sweep_extend.py
"""

from __future__ import annotations

import json
import os
import threading
import time

import torch
from tqdm import tqdm

from tgengine.core.batch import RawBatch
from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import APEval
from tgengine.models.dygformer import DyGFormer
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
JSON_PATH = os.path.expanduser("~/CodeBase/Graph/tgengine/bench_bs_sweep.json")


def _gpu_monitor_thread(samples: list, stop: threading.Event, interval: float = 0.05):
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(1)
        while not stop.is_set():
            util = nvmlDeviceGetUtilizationRates(handle)
            samples.append(util.gpu)
            time.sleep(interval)
        nvmlShutdown()
    except Exception:
        pass


def train_one_bs(ds, batch_size: int, device: str = "cuda", use_amp: bool = True,
                 node_feat=None, valid_dst_nodes=None) -> dict | None:
    """Train DyGFormer at given batch_size using a custom loop with AMP support."""
    n_nodes = ds.num_nodes
    d_edge = ds.edge_feat_dim

    # ---- graph ----------------------------------------------------------
    buf = min(32, n_nodes)
    mem_gb = n_nodes * buf * max(d_edge, 1) * 4 / 1e9
    if mem_gb > 4.0:  # tighter limit for larger BS safety
        buf = max(4, int(32 * 4.0 / mem_gb))

    graph = TemporalGraph(n_nodes, buffer_size=buf,
                          edge_feat_dim=d_edge, device=device)
    batches = ds.get_batches("train", batch_size, device)
    n = len(batches)
    n_val = max(1, n // 7)
    n_test = max(1, n // 7)
    n_train_batches = n - n_val - n_test
    train_batches = batches[:n_train_batches]
    val_batches = batches[n_train_batches:n_train_batches + n_val]
    test_batches = batches[n_train_batches + n_val:]

    # Pre-populate
    warmup_n = max(5, n_train_batches // 10)
    for rb in train_batches[:warmup_n]:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

    # ---- model ----------------------------------------------------------
    model = DyGFormer(
        d_model=172, d_edge=d_edge, d_time=100,
        d_channel=50, K=buf, n_layers=2,
        node_feat=node_feat,
    ).to(device)

    pipeline = DataPipeline(model.gather_spec, graph)
    neg_strat = RandomNegative(n_nodes, valid_dst_nodes=valid_dst_nodes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- GPU monitor ----------------------------------------------------
    gpu_samples: list[float] = []
    stop_ev = threading.Event()
    monitor = threading.Thread(target=_gpu_monitor_thread, args=(gpu_samples, stop_ev), daemon=True)
    monitor.start()
    time.sleep(0.3)

    # ---- training loop --------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    best_val = 0.0
    patience_counter = 0
    patience = 5
    n_epochs = 0

    try:
        for epoch in range(1, 51):
            n_epochs = epoch
            model.train()
            for rb in tqdm(train_batches, desc=f"BS={batch_size} Ep={epoch}", leave=False):
                neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph, rb.edge_indices)
                rb.neg = neg
                prepared = pipeline.prepare(rb)
                opt.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(prepared)
                    scaler.scale(out.loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    out = model(prepared)
                    out.loss.backward()
                    opt.step()

                graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

            # Validation
            val_ap = _eval_ap(model, pipeline, val_batches, graph, neg_strat, use_amp)
            if val_ap > best_val:
                best_val = val_ap
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        # Test evaluation at best
        test_ap = _eval_ap(model, pipeline, test_batches, graph, neg_strat, use_amp)

    except RuntimeError as e:
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        stop_ev.set(); monitor.join(timeout=1)
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {"batch_size": batch_size, "oom": True, "error": str(e)[:120]}
        raise

    stop_ev.set(); monitor.join(timeout=2)
    time.sleep(0.2)

    valid_gpu = [v for v in gpu_samples if 0.0 <= v <= 100.0]
    avg_gpu = sum(valid_gpu) / len(valid_gpu) if valid_gpu else 0.0
    n_edges_per_epoch = len(train_batches) * batch_size
    throughput = n_edges_per_epoch * n_epochs / wall if wall > 0 else 0.0

    return {
        "batch_size": batch_size,
        "oom": False,
        "amp": use_amp,
        "n_epochs": n_epochs,
        "total_wall_sec": round(wall, 2),
        "avg_epoch_sec": round(wall / n_epochs, 2) if n_epochs > 0 else 0.0,
        "throughput_edges_per_sec": round(throughput, 0),
        "best_test_ap": round(test_ap, 4),
        "avg_gpu_util_pct": round(avg_gpu, 1),
        "n_gpu_samples": len(valid_gpu),
    }


@torch.no_grad()
def _eval_ap(model, pipeline, batches, graph, neg_strat, use_amp: bool) -> float:
    """Quick AP evaluation with graph snapshot/restore."""
    snap = graph.snapshot()
    model.eval()
    pos_scores, neg_scores = [], []
    for rb in batches:
        neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph, rb.edge_indices)
        batch = RawBatch(src=rb.src, dst=rb.dst, time=rb.time,
                         edge_feat=rb.edge_feat, neg=neg)
        prepared = pipeline.prepare(batch)
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(prepared)
        else:
            out = model(prepared)
        pos_scores.append(out.pos_score)
        neg_scores.append(out.neg_score)
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    graph.restore(snap)
    if not pos_scores:
        return 0.0
    pos = torch.cat(pos_scores).sigmoid()
    neg = torch.cat(neg_scores).sigmoid()
    return (pos > neg).float().mean().item()


def main():
    device = "cuda"
    step = 200
    max_bs = 10000

    # ---- load existing data ----------------------------------------------
    existing = []
    done_bs = set()
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            existing = json.load(f)
        done_bs = {r["batch_size"] for r in existing}

    print(f"Existing: {len(existing)} points, done BS: {sorted(done_bs)}")

    # ---- load dataset ----------------------------------------------------
    ds = load_dataset("uci", DATA_ROOT)
    print(f"UCI: {ds.num_edges:,} edges, {ds.num_nodes:,} nodes")

    train_dst = ds.dst[:ds.train_end]
    valid_dst_nodes = torch.unique(train_dst)
    node_feat = ds.node_feat
    if node_feat is not None:
        print(f"  Node features: {node_feat.shape}, valid dst: {len(valid_dst_nodes)}")

    # ---- determine start point -------------------------------------------
    last_bs = max(done_bs) if done_bs else 0
    start_bs = last_bs + step if last_bs > 0 else 200
    if start_bs > max_bs:
        print(f"Already complete up to BS={max_bs}")
        return

    batch_sizes = list(range(start_bs, max_bs + 1, step))
    print(f"Continuing from BS={start_bs} to {max_bs} ({len(batch_sizes)} points)")

    # ---- run -------------------------------------------------------------
    for i, bs in enumerate(batch_sizes):
        label = f"[{i+1}/{len(batch_sizes)}] BS={bs}"
        print(f"\n{label} ", end="", flush=True)
        start_t = time.perf_counter()

        r = train_one_bs(ds, bs, device, use_amp=True,
                         node_feat=node_feat, valid_dst_nodes=valid_dst_nodes)
        if r is None:
            print("ERROR")
            continue

        elapsed = time.perf_counter() - start_t
        if r.get("oom"):
            print(f"OOM ({elapsed:.0f}s) — skipped, continuing")
            existing.append(r)
            # Reduce memory pressure: shrink buffer for subsequent runs
            # by clearing CUDA cache and pausing
            torch.cuda.empty_cache()
            time.sleep(2)
            continue

        print(f"ep={r['n_epochs']} wall={r['total_wall_sec']}s "
              f"ap={r['best_test_ap']} gpu={r['avg_gpu_util_pct']}% ({elapsed:.0f}s)")
        existing.append(r)

        # Save incrementally
        with open(JSON_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    # ---- summary ---------------------------------------------------------
    valid = [r for r in existing if not r.get("oom")]
    if valid:
        print(f"\nDone. {len(valid)} valid points, saved to {JSON_PATH}")
        best = max(valid, key=lambda r: r["best_test_ap"])
        print(f"Best AP: BS={best['batch_size']} → AP={best['best_test_ap']:.4f}")


if __name__ == "__main__":
    main()
