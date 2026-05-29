"""Ablation: buf size (K = ring buffer / sequence length) vs test AP.

Trains DyGFormer on Wikipedia with K in [4, 8, 16, 24, 32, 48, 64].
Reports val AP and test AP for each K so we can quantify the accuracy
cost of reducing K (e.g. to fit in GPU memory).

Usage:
    python benchmarks/bench_buf_ablation.py
    python benchmarks/bench_buf_ablation.py --k_values 4 8 16 32 --epochs 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"


@dataclass
class KResult:
    k: int
    best_val_ap: float
    best_test_ap: float
    final_val_ap: float
    final_test_ap: float
    epochs_run: int
    total_sec: float
    mem_gb: float        # estimated ring buffer GPU memory for Reddit
    oom_reddit: bool     # would OOM on Reddit (15.6 GB GPU)?


def train_one_k(
    k: int,
    dataset_name: str,
    data_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    patience: int,
) -> KResult:
    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(2020)
    t_start = time.perf_counter()

    ds = load_dataset(dataset_name, data_path)
    graph = TemporalGraph(ds.num_nodes, buffer_size=k,
                          edge_feat_dim=ds.edge_feat_dim, device=device)
    train_batches = ds.get_batches("train", batch_size, device)
    val_batches   = ds.get_batches("val",   batch_size, device)
    test_batches  = ds.get_batches("test",  batch_size, device)

    model = DyGFormer(
        d_model=172, d_edge=ds.edge_feat_dim, d_time=100,
        d_channel=50, K=k, n_layers=2, patch_size=1,
    ).to(device)

    pipeline = DataPipeline(model.gather_spec, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    neg_strat = RandomNegative(ds.num_nodes)

    # Reddit ring buffer memory estimate at this K
    reddit_nodes = 672447
    reddit_d_edge = 172
    mem_gb = reddit_nodes * k * reddit_d_edge * 4 / 1e9
    oom_reddit = mem_gb > 12.0  # leave 3.6 GB headroom on 15.6 GB GPU

    def _eval(batches):
        snap = graph.snapshot()
        model.eval()
        pos_scores, neg_scores = [], []
        with torch.no_grad():
            for rb in batches:
                neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
                rb.neg = neg
                prepared = pipeline.prepare(rb)
                out = model(prepared)
                pos_scores.append(out.pos_score)
                neg_scores.append(out.neg_score)
                graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
        graph.restore(snap)
        pos = torch.cat(pos_scores).sigmoid()
        neg_s = torch.cat(neg_scores).sigmoid()
        return (pos > neg_s).float().mean().item()

    best_val, best_test = 0.0, 0.0
    patience_cnt = 0
    val_ap = test_ap = 0.0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for rb in train_batches:
            neg = neg_strat.sample(rb.src, rb.dst, rb.time, graph)
            rb.neg = neg
            prepared = pipeline.prepare(rb)
            optimizer.zero_grad()
            out = model(prepared)
            out.loss.backward()
            optimizer.step()
            graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

        val_ap  = _eval(val_batches)
        test_ap = _eval(test_batches)
        epochs_run = epoch

        if val_ap > best_val:
            best_val  = val_ap
            best_test = test_ap
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    total_sec = time.perf_counter() - t_start
    return KResult(
        k=k,
        best_val_ap=best_val,
        best_test_ap=best_test,
        final_val_ap=val_ap,
        final_test_ap=test_ap,
        epochs_run=epochs_run,
        total_sec=total_sec,
        mem_gb=mem_gb,
        oom_reddit=oom_reddit,
    )


def print_table(results: list[KResult]):
    ref_ap = 0.974  # DyGLib reference for Wikipedia K=32
    k32_ap = next((r.best_test_ap for r in results if r.k == 32), None)

    print("\n" + "=" * 82)
    print("  Wikipedia DyGFormer — buf size (K) vs Test AP")
    print("=" * 82)

    hdr = (f"{'K':>5}  {'Best val AP':>11}  {'Best test AP':>12}  "
           f"{'vs K=32':>8}  {'Reddit mem':>11}  {'Reddit OOM':>10}  {'time':>6}")
    print(hdr)
    print("─" * 82)

    for r in results:
        delta = ""
        if k32_ap is not None and r.k != 32:
            diff = r.best_test_ap - k32_ap
            delta = f"{diff:+.4f}"
        elif r.k == 32:
            delta = "(baseline)"

        oom_str = "OOM ✗" if r.oom_reddit else "fits ✓"
        print(f"  {r.k:>3}  {r.best_val_ap:>11.4f}  {r.best_test_ap:>12.4f}  "
              f"{delta:>10}  {r.mem_gb:>9.1f} GB  {oom_str:>10}  {r.total_sec:>5.0f}s")

    print("─" * 82)
    print(f"  DyGLib reference (K=32, CPU):  test AP = {ref_ap:.4f}")

    # Analysis
    print("\n── Analysis ─────────────────────────────────────────────────────────────────")
    if len(results) >= 2:
        sorted_r = sorted(results, key=lambda r: r.k)
        k32 = next((r for r in sorted_r if r.k == 32), sorted_r[-1])

        # Degradation per halving of K
        pairs = []
        for i in range(len(sorted_r) - 1):
            a, b = sorted_r[i], sorted_r[i+1]
            pairs.append((a.k, b.k, b.best_test_ap - a.best_test_ap))

        print(f"\n  K=32 test AP   : {k32.best_test_ap:.4f}  (ref: {ref_ap:.4f}, "
              f"gap: {k32.best_test_ap - ref_ap:+.4f})")

        # Find the lowest K where AP drop is < 1%
        for r in sorted_r:
            if k32_ap and abs(r.best_test_ap - k32_ap) < 0.01:
                print(f"  Smallest K with <1% drop vs K=32: K={r.k}")
                break

        # Reddit-safe K
        reddit_safe = [r for r in sorted_r if not r.oom_reddit]
        if reddit_safe:
            best_reddit_safe = max(reddit_safe, key=lambda r: r.k)
            drop = (k32.best_test_ap - best_reddit_safe.best_test_ap)
            print(f"  Largest K that fits on Reddit GPU : K={best_reddit_safe.k} "
                  f"(AP drop: {drop:+.4f})")

        print(f"\n  Per-step AP change:")
        for ka, kb, delta in pairs:
            print(f"    K={ka}→{kb}: {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", nargs="+", type=int,
                        default=[4, 8, 16, 24, 32, 48, 64])
    parser.add_argument("--dataset", default="wikipedia")
    parser.add_argument("--data_path", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"K ablation on {args.dataset}  epochs={args.epochs}  "
          f"K values: {args.k_values}")

    results = []
    for k in args.k_values:
        print(f"\n  ── K={k} ", end="", flush=True)
        try:
            r = train_one_k(
                k, args.dataset, args.data_path,
                args.epochs, args.batch_size, args.lr,
                args.device, args.patience,
            )
            results.append(r)
            print(f"test AP={r.best_test_ap:.4f}  ({r.total_sec:.0f}s)")
        except Exception as exc:
            import traceback
            print(f"\n  FAILED K={k}: {exc}")
            traceback.print_exc()

    if results:
        print_table(results)


if __name__ == "__main__":
    main()
