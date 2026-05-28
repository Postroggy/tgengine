"""Accuracy benchmark: train DyGFormer / GraphMixer / TGN on Wikipedia/Reddit/LastFM.

Uses DyGLib-aligned hyperparameters for fair comparison.
Reports train loss, val AP, test AP per epoch, plus best test AP.

Reference numbers from DyGLib paper (random neg, same 70/15/15 split):
  Wikipedia : DyGFormer=0.974, GraphMixer=0.963, TGN=0.986
  Reddit    : DyGFormer=0.990, GraphMixer=0.971, TGN=0.985
  LastFM    : DyGFormer=0.773, GraphMixer=0.758, TGN=0.765

Usage:
    python benchmarks/bench_accuracy.py                          # DyGFormer, all 3 datasets
    python benchmarks/bench_accuracy.py --model graphmixer --datasets wiki
    python benchmarks/bench_accuracy.py --model tgn --datasets wiki reddit --epochs 50
    python benchmarks/bench_accuracy.py --model all --datasets wiki
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
from torch import Tensor

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
DATASETS = {
    "wiki":   ("wikipedia", DATA_ROOT),
    "reddit": ("reddit",    DATA_ROOT),
    "lastfm": ("lastfm",    DATA_ROOT),
}

# DyGLib reference AP (random neg, transductive split)
REFERENCE_AP = {
    ("dygformer", "wiki"):   0.974,
    ("dygformer", "reddit"): 0.990,
    ("dygformer", "lastfm"): 0.773,
    ("graphmixer", "wiki"):  0.963,
    ("graphmixer", "reddit"): 0.971,
    ("graphmixer", "lastfm"): 0.758,
    ("tgn", "wiki"):  0.986,
    ("tgn", "reddit"): 0.985,
    ("tgn", "lastfm"): 0.765,
}

MODELS = ["dygformer", "graphmixer", "tgn"]


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_ap: float
    test_ap: float
    epoch_sec: float


def _load_node_feat(data_path: str, dataset_name: str):
    """Load and standardize static node features. Returns None if all-zero."""
    import os
    import numpy as np
    path = os.path.join(data_path, dataset_name, f"ml_{dataset_name}.npy")
    if not os.path.exists(path):
        return None
    nf = np.load(path).astype(np.float32)
    if nf.shape[0] == 0 or (nf == 0).all():
        print(f"  Node features: {nf.shape}  (all zeros, skipping)")
        return None
    mu = nf.mean(0, keepdims=True)
    sigma = nf.std(0, keepdims=True) + 1e-6
    nf = (nf - mu) / sigma
    print(f"  Node features: {nf.shape}  (standardized, adding to model)")
    return torch.from_numpy(nf)


def _build_model(model_name: str, ds, buf: int, node_feat_tensor, device: str):
    """Build model by name. Returns (model, is_stateful)."""
    if model_name == "dygformer":
        from tgengine.models.dygformer import DyGFormer
        model = DyGFormer(
            d_model=172, d_edge=ds.edge_feat_dim, d_time=100,
            d_channel=50, K=buf, n_layers=2, patch_size=1,
            node_feat=node_feat_tensor,
        ).to(device)
        return model, False

    elif model_name == "graphmixer":
        from tgengine.models.graphmixer import GraphMixer
        # GraphMixer reference uses K=20, time_gap=2000; we use K=buf for fairness
        model = GraphMixer(
            d_model=172, d_edge=ds.edge_feat_dim, d_time=100,
            K=buf, num_layers=2, dropout=0.1,
        ).to(device)
        return model, False

    elif model_name == "tgn":
        from tgengine.models.tgn import TGN
        model = TGN(
            num_nodes=ds.num_nodes,
            d_model=172, d_edge=ds.edge_feat_dim,
        ).to(device)
        return model, True

    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_dataset(
    model_name: str,
    dataset_key: str,
    dataset_name: str,
    data_path: str,
    epochs: int = 100,
    batch_size: int = 200,
    lr: float = 1e-4,
    device: str = "cuda",
    patience: int = 20,
    eval_neg: str = "random",  # "random" | "historical"
) -> list[EpochLog]:
    import numpy as np
    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import HistoricalNegative, RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(2020)

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {model_name.upper()} on {dataset_key.upper()} ({dataset_name})"
          f"  epochs={epochs}  lr={lr}  B={batch_size}")
    print(sep)

    print("  Loading ...", end="", flush=True)
    ds = load_dataset(dataset_name, data_path)
    print(f" {ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}")

    node_feat_tensor = _load_node_feat(data_path, dataset_name)

    # Adaptive ring buffer to avoid OOM on large datasets (Reddit)
    buf = 32
    mem_gb = ds.num_nodes * buf * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 8.0:
        buf = max(4, int(32 * 8.0 / mem_gb))
        print(f"  buffer_size auto-adjusted 32→{buf} ({mem_gb:.1f} GB est.)")

    graph = TemporalGraph(ds.num_nodes, buffer_size=buf,
                          edge_feat_dim=ds.edge_feat_dim, device=device)
    train_batches = ds.get_batches("train", batch_size, device)
    val_batches   = ds.get_batches("val",   batch_size, device)
    test_batches  = ds.get_batches("test",  batch_size, device)

    model, is_stateful = _build_model(model_name, ds, buf, node_feat_tensor, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_name}  K={buf}  {n_params:,} params  stateful={is_stateful}")
    print(f"  Splits  train={ds.train_end:,}  val={ds.val_end - ds.train_end:,}  "
          f"test={ds.num_edges - ds.val_end:,}")

    pipeline = DataPipeline(model.gather_spec, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_neg = RandomNegative(ds.num_nodes)

    # Eval neg strategy: random (fast) or historical (DyGLib-equivalent semantics)
    if eval_neg == "historical":
        eval_neg_strat = HistoricalNegative(ds.num_nodes, pool_size=512, device=device)
        mem_mb = ds.num_nodes * 512 * 4 / 1e6
        print(f"  eval neg: HistoricalNegative (pool_size=512, ~{mem_mb:.0f} MB)")
    else:
        eval_neg_strat = RandomNegative(ds.num_nodes)
        print(f"  eval neg: RandomNegative")

    logs: list[EpochLog] = []
    best_val = 0.0
    best_test = 0.0
    patience_cnt = 0

    def _evaluate(batches) -> float:
        graph_snap = graph.snapshot()
        mem_snap = model.freeze() if is_stateful else None
        model.eval()
        pos_scores, neg_scores = [], []
        with torch.no_grad():
            for rb in batches:
                neg = eval_neg_strat.sample(rb.src, rb.dst, rb.time, graph)
                rb.neg = neg
                prepared = pipeline.prepare(rb)
                out = model(prepared)
                pos_scores.append(out.pos_score)
                neg_scores.append(out.neg_score)
                graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
                # Stateful models do NOT call evolve() during eval
        graph.restore(graph_snap)
        if is_stateful:
            model.thaw(mem_snap)
        model.train()
        pos = torch.cat(pos_scores).sigmoid()
        neg_s = torch.cat(neg_scores).sigmoid()
        return (pos > neg_s).float().mean().item()

    print(f"\n  {'Epoch':>5}  {'loss':>8}  {'val AP':>8}  {'test AP':>9}  {'time':>6}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*6}")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0

        for rb in train_batches:
            neg = train_neg.sample(rb.src, rb.dst, rb.time, graph)
            rb.neg = neg
            prepared = pipeline.prepare(rb)
            optimizer.zero_grad()
            out = model(prepared)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
            eval_neg_strat.update(rb.src, rb.dst)  # no-op for RandomNegative
            if is_stateful:
                with torch.no_grad():
                    model.evolve(rb.src, rb.dst, rb.time, rb.edge_feat)

        train_loss = total_loss / len(train_batches)
        val_ap  = _evaluate(val_batches)
        test_ap = _evaluate(test_batches)
        elapsed = time.perf_counter() - t0

        log = EpochLog(epoch, train_loss, val_ap, test_ap, elapsed)
        logs.append(log)

        marker = ""
        if val_ap > best_val:
            best_val = val_ap
            best_test = test_ap
            patience_cnt = 0
            marker = " ←best"
        else:
            patience_cnt += 1

        print(f"  {epoch:>5}  {train_loss:>8.4f}  {val_ap:>8.4f}  {test_ap:>9.4f}  "
              f"{elapsed:>5.1f}s{marker}")

        if patience_cnt >= patience:
            print(f"  Early stop at epoch {epoch} (patience={patience})")
            break

    ref = REFERENCE_AP.get((model_name, dataset_key), "?")
    print(f"\n  Best test AP : {best_test:.4f}  (ref DyGLib: {ref})")
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="dygformer",
                        choices=MODELS + ["all"],
                        help="Model to train (or 'all' for all models)")
    parser.add_argument("--datasets", nargs="+", default=["wiki", "reddit", "lastfm"],
                        choices=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval_neg", default="random", choices=["random", "historical"],
                        help="Negative sampling strategy for evaluation. "
                             "'historical' uses HistoricalNegPool (DyGLib-equivalent). "
                             "Memory: ~num_nodes*512*4 bytes extra.")
    args = parser.parse_args()

    models_to_run = MODELS if args.model == "all" else [args.model]

    all_results = {}  # (model, dataset_key) → best_test_ap
    for model_name in models_to_run:
        for dataset_key in args.datasets:
            dataset_name, data_path = DATASETS[dataset_key]
            try:
                logs = train_dataset(
                    model_name, dataset_key, dataset_name, data_path,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    patience=args.patience,
                    eval_neg=args.eval_neg,
                )
                best_test = max(l.test_ap for l in logs)
                all_results[(model_name, dataset_key)] = best_test
            except Exception as exc:
                import traceback
                print(f"\n  [FAILED] {model_name}/{dataset_key}: {exc}")
                traceback.print_exc()

    if all_results:
        print(f"\n{'='*62}")
        print("  Summary: best test AP")
        print(f"{'='*62}")
        print(f"  {'Model':<12} {'Dataset':<10} {'TGEngine':>10} {'DyGLib ref':>12} {'gap':>8}")
        print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")
        for (model_name, dataset_key), ap in sorted(all_results.items()):
            ref = REFERENCE_AP.get((model_name, dataset_key), float("nan"))
            gap = ap - ref
            print(f"  {model_name.upper():<12} {dataset_key.upper():<10} {ap:>10.4f} "
                  f"{ref:>12.4f} {gap:>+8.4f}")


if __name__ == "__main__":
    main()
