"""End-to-end smoke test for all TGEngine models.

Creates a synthetic temporal graph and trains each model for a few epochs,
verifying the full pipeline works without errors:
  TemporalDataset → TemporalGraph → DataPipeline → Model → Engine → APEval

Usage:
    python examples/smoke_test_all_models.py
    python examples/smoke_test_all_models.py --real_data /path/to/dataset/ml_dataset.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import APEval, Engine, TrainConfig
from tgengine.models.dygformer import DyGFormer
from tgengine.models.dygmamba import DyGMamba
from tgengine.models.freedyg import FreeDyG
from tgengine.models.graphmixer import GraphMixer
from tgengine.models.tgn import TGN
from tgengine.pipeline.negatives import RandomNegative
from tgengine.utils import seed_everything


def make_synthetic_dataset(
    n_edges: int = 5000,
    num_nodes: int = 200,
    d_edge: int = 16,
    d_node: int = 8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """Generate a synthetic temporal graph dataset."""
    from dataclasses import dataclass
    from typing import Optional
    from tgengine.core.dataset import TemporalDataset

    rng = np.random.default_rng(42)
    src = torch.from_numpy(rng.integers(0, num_nodes, n_edges)).long()
    dst = torch.from_numpy(rng.integers(0, num_nodes, n_edges)).long()
    time = torch.from_numpy(np.sort(rng.uniform(0, 1000, n_edges))).double()
    edge_feat = torch.randn(n_edges, d_edge)
    node_feat = torch.randn(num_nodes, d_node)

    n = n_edges
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    return TemporalDataset(
        src=src, dst=dst, time=time,
        edge_feat=edge_feat, node_feat=node_feat,
        num_nodes=num_nodes, num_edges=n,
        train_end=train_end, val_end=val_end,
    )


def run_model(name: str, model, dataset, device: str, epochs: int = 3, batch_size: int = 128):
    """Train and evaluate one model. Returns AP and wall time."""
    seed_everything(42)

    graph = TemporalGraph(
        dataset.num_nodes,
        buffer_size=32,
        edge_feat_dim=dataset.edge_feat_dim,
        device=device,
    )

    train_batches = dataset.get_batches("train", batch_size, device)
    val_batches   = dataset.get_batches("val",   batch_size, device)
    test_batches  = dataset.get_batches("test",  batch_size, device)

    config = TrainConfig(epochs=epochs, batch_size=batch_size, lr=1e-4,
                         patience=epochs + 1, device=device)

    engine = Engine(
        model=model,
        graph=graph,
        train_batches=train_batches,
        val_batches=val_batches,
        test_batches=test_batches,
        neg_strategy=RandomNegative(dataset.num_nodes),
        eval_protocol=APEval(),
        config=config,
    )

    t0 = time.perf_counter()
    results = engine.train()
    elapsed = time.perf_counter() - t0
    ap = results.get("ap", float("nan"))
    print(f"  {name:<14}  AP={ap:.4f}  ({elapsed:.1f}s)")
    return ap, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_edges", type=int, default=5000,
                        help="Synthetic dataset size (ignored if --data_path given)")
    parser.add_argument("--data_path", default=None,
                        help="Path to DyGLib-format dataset directory (optional)")
    parser.add_argument("--dataset_name", default=None)
    args = parser.parse_args()

    print(f"Device: {args.device}")

    if args.data_path is not None and args.dataset_name is not None:
        print(f"Loading real dataset: {args.dataset_name} from {args.data_path}")
        dataset = load_dataset(args.dataset_name, args.data_path)
    else:
        print(f"Using synthetic dataset ({args.n_edges} edges)")
        dataset = make_synthetic_dataset(n_edges=args.n_edges)

    print(f"Dataset: {dataset.num_nodes} nodes, {dataset.num_edges} edges, "
          f"d_edge={dataset.edge_feat_dim}, d_node={dataset.node_feat_dim}")
    print(f"Split: train={dataset.train_end}, val={dataset.val_end - dataset.train_end}, "
          f"test={dataset.num_edges - dataset.val_end}")
    print()

    d_edge = dataset.edge_feat_dim
    n = dataset.num_nodes
    nf = dataset.node_feat  # may be None

    models = {
        "DyGFormer":  DyGFormer(d_model=64, d_edge=d_edge, d_time=32, d_channel=32,
                                K=32, patch_size=1, n_layers=2),
        "DyGMamba":   DyGMamba(d_model=64, d_edge=d_edge, n_layers=2),
        "GraphMixer": GraphMixer(d_model=64, d_edge=d_edge, d_time=32, K=32,
                                 num_layers=2, node_raw_features=nf),
        "FreeDyG":    FreeDyG(d_model=64, d_edge=d_edge, d_time=32, d_nif=32,
                              K=32, num_layers=2, node_raw_features=nf),
        "TGN":        TGN(num_nodes=n, d_model=64, d_edge=d_edge),
    }

    print(f"{'Model':<14}  {'AP':>8}  {'Time':>8}")
    print("-" * 36)
    results = {}
    for name, model in models.items():
        try:
            ap, t = run_model(name, model, dataset, args.device, args.epochs, args.batch_size)
            results[name] = (ap, t)
        except Exception as e:
            print(f"  {name:<14}  FAILED: {e}")

    print()
    passed = [n for n, (ap, _) in results.items() if ap == ap]  # NaN check
    print(f"All models ran without error: {len(passed)}/{len(models)}")
    if len(passed) == len(models):
        print("[PASS] Smoke test complete.")
    else:
        failed = set(models) - set(passed)
        print(f"[FAIL] Failed models: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
