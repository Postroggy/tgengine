"""Multi-dataset benchmark runner.

Trains a model on multiple datasets and reports AP against DyGLib reference.
Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_multi_dataset.py
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_multi_dataset.py --model dygformer --datasets wiki uci
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_multi_dataset.py --model graphmixer --amp
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"

REFERENCE_AP = {
    "wikipedia": 0.974,
    "reddit": 0.990,
    "lastfm": 0.773,
    "uci": 0.9613,
}

MODEL_REGISTRY = ["dygformer", "graphmixer", "dygmamba", "freedyg"]


@dataclass
class RunResult:
    dataset: str
    model: str
    best_ap: float
    ref_ap: float
    gap: float
    train_time_s: float
    epochs_run: int


def build_model(name: str, ds, K: int, device: str):
    if name == "dygformer":
        from tgengine.models.dygformer import DyGFormer
        return DyGFormer(
            d_model=172, d_edge=ds.edge_feat_dim,
            d_time=100, d_channel=50,
            K=K, n_layers=2, patch_size=2,
            node_feat=ds.node_feat, num_nodes=ds.num_nodes,
        )
    elif name == "graphmixer":
        from tgengine.models.graphmixer import GraphMixer
        return GraphMixer(
            d_model=172, d_edge=ds.edge_feat_dim,
            d_time=100, K=K, num_layers=2,
        )
    elif name == "dygmamba":
        from tgengine.models.dygmamba import DyGMamba
        return DyGMamba(
            d_model=172, d_edge=ds.edge_feat_dim, n_layers=2,
        )
    elif name == "freedyg":
        from tgengine.models.freedyg import FreeDyG
        return FreeDyG(
            d_model=172, d_edge=ds.edge_feat_dim,
            d_time=100, K=K, num_layers=2,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def run_single(model_name: str, dataset_name: str, args) -> RunResult:
    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(0)
    print(f"\n{'='*60}")
    print(f"  {model_name} on {dataset_name}")
    print(f"{'='*60}")

    ds = load_dataset(dataset_name, args.data_root)
    print(f"  {ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}")

    K = 63
    mem_gb = ds.num_nodes * K * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 8.0:
        K = max(4, int(K * 8.0 / mem_gb))
        print(f"  K auto-reduced to {K}")

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=args.device)
    model = build_model(model_name, ds, K, args.device)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    train_dst = ds.dst[:ds.train_end]
    valid_dst_nodes = torch.unique(train_dst)

    config = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=args.device,
        use_amp=args.amp,
        grad_clip=args.grad_clip,
    )

    engine = Engine(
        model=model,
        graph=graph,
        train_batches=ds.get_batches("train", args.batch_size, args.device),
        val_batches=ds.get_batches("val", args.batch_size, args.device),
        test_batches=ds.get_batches("test", args.batch_size, args.device),
        neg_strategy=RandomNegative(ds.num_nodes, valid_dst_nodes=valid_dst_nodes),
        eval_protocol=APEval(),
        config=config,
    )

    t0 = time.time()
    best = engine.train()
    elapsed = time.time() - t0

    best_ap = best.get("ap", 0.0)
    ref_ap = REFERENCE_AP.get(dataset_name, 0.0)
    gap = best_ap - ref_ap

    print(f"\n  Result: AP={best_ap:.4f}  ref={ref_ap:.4f}  gap={gap:+.4f}  time={elapsed:.1f}s")
    return RunResult(
        dataset=dataset_name, model=model_name,
        best_ap=best_ap, ref_ap=ref_ap, gap=gap,
        train_time_s=elapsed, epochs_run=engine._current_epoch,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset benchmark runner")
    parser.add_argument("--model", default="dygformer", choices=MODEL_REGISTRY)
    parser.add_argument("--datasets", nargs="+", default=["wikipedia", "uci"])
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    results: list[RunResult] = []
    for dataset in args.datasets:
        try:
            r = run_single(args.model, dataset, args)
            results.append(r)
        except Exception as e:
            print(f"\n  FAILED on {dataset}: {e}")
            continue

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'AP':>8} {'Ref':>8} {'Gap':>8} {'Time':>8} {'Epochs':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r.dataset:<12} {r.best_ap:>8.4f} {r.ref_ap:>8.4f} {r.gap:>+8.4f} {r.train_time_s:>7.1f}s {r.epochs_run:>6}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
