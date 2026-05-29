"""Multi-model × multi-dataset accuracy benchmark.

Runs DyGFormer, TGN, GraphMixer, FreeDyG on UCI and Wikipedia via Engine.
Reports best test AP.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_accuracy.py
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_accuracy.py --models dygformer tgn
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_accuracy.py --datasets uci
"""

from __future__ import annotations

import argparse
import time

import torch

from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import APEval, Engine, TrainConfig
from tgengine.pipeline.negatives import RandomNegative
from tgengine.utils import seed_everything


DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
DATASETS = ["uci", "wikipedia"]
MODELS = ["dygformer", "tgn", "graphmixer", "freedyg"]


def _build_model(name: str, num_nodes: int, d_edge: int, K: int, node_feat=None,
                  dataset_name: str = ""):
    if name == "dygformer":
        from tgengine.models.dygformer import DyGFormer
        return DyGFormer(
            d_model=172, d_edge=d_edge, d_time=100, K=K,
            patch_size=1, n_layers=2, n_heads=2,
            num_nodes=num_nodes, node_feat=node_feat,
        )
    elif name == "tgn":
        from tgengine.models.tgn import TGN
        return TGN(num_nodes=num_nodes, d_model=172, d_edge=d_edge)
    elif name == "graphmixer":
        from tgengine.models.graphmixer import GraphMixer
        # Match DyGLib per-dataset config
        gm_cfg = {
            "wikipedia": {"K": 30, "dropout": 0.5},
            "reddit": {"K": 10, "dropout": 0.5},
            "uci": {"K": 20, "dropout": 0.4},
            "mooc": {"K": 20, "dropout": 0.4},
            "lastfm": {"K": 10, "dropout": 0.0},
        }.get(dataset_name, {"K": 20, "dropout": 0.1})
        return GraphMixer(d_model=172, d_edge=d_edge, d_time=100,
                          K=gm_cfg["K"], num_layers=2, dropout=gm_cfg["dropout"],
                          node_raw_features=node_feat)
    elif name == "freedyg":
        from tgengine.models.freedyg import FreeDyG
        return FreeDyG(d_model=172, d_edge=d_edge, d_time=100, K=K, num_layers=2)
    else:
        raise ValueError(f"Unknown model: {name}")


def run_one(dataset_name: str, model_name: str, epochs: int = 100, patience: int = 20) -> dict:
    seed_everything(2020)
    device = "cuda"
    K = 32

    ds = load_dataset(dataset_name, DATA_ROOT)

    # Adaptive K for memory
    mem_gb = ds.num_nodes * K * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 8.0:
        K = max(4, int(K * 8.0 / mem_gb))
        print(f"  K auto-reduced to {K}")

    model = _build_model(model_name, ds.num_nodes, ds.edge_feat_dim, K, ds.node_feat,
                          dataset_name=dataset_name)
    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=device)

    # DyGLib uses train_data dst for train neg, full_data dst for eval neg
    # Train neg: from unique train dst
    train_dst = ds.dst[:ds.train_end]
    valid_dst = torch.unique(train_dst)
    neg = RandomNegative(ds.num_nodes, valid_dst_nodes=valid_dst)
    # Eval neg: from ALL unique dst (matching DyGLib val_neg_edge_sampler using full_data)
    all_dst = torch.unique(ds.dst)
    eval_neg = RandomNegative(ds.num_nodes, valid_dst_nodes=all_dst)

    bs = 200
    config = TrainConfig(
        epochs=epochs, batch_size=bs, lr=1e-4, patience=patience,
        device=device, seed=2020, grad_clip=0.0,
    )

    engine = Engine(
        model=model, graph=graph,
        train_batches=ds.get_batches("train", bs, device),
        val_batches=ds.get_batches("val", bs, device),
        test_batches=ds.get_batches("test", bs, device),
        neg_strategy=neg, eval_protocol=APEval(), config=config,
        inductive_edges=ds.inductive_edges,
        eval_neg_strategy=eval_neg,
    )
    engine.logger.dataset_name = dataset_name

    t0 = time.perf_counter()
    best = engine.train()
    elapsed = time.perf_counter() - t0
    return {"ap": best.get("ap", 0.0), "time_s": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    results = {}
    for dataset in args.datasets:
        for model_name in args.models:
            key = f"{model_name}/{dataset}"
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}\n")
            try:
                r = run_one(dataset, model_name, args.epochs, args.patience)
                results[key] = r
                print(f"\n>>> {key}: AP={r['ap']:.4f} in {r['time_s']:.1f}s")
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key] = {"ap": 0.0, "error": str(e)}
                print(f"\n>>> {key}: FAILED — {e}")
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<15} {'Dataset':<12} {'AP':<10} {'Time(s)':<10}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
    for key, r in results.items():
        model, ds = key.split("/")
        ap = r.get("ap", 0.0)
        t = r.get("time_s", 0.0)
        err = r.get("error", "")
        if err:
            print(f"  {model:<15} {ds:<12} {'FAIL':<10} {err[:30]}")
        else:
            print(f"  {model:<15} {ds:<12} {ap:<10.4f} {t:<10.1f}")


if __name__ == "__main__":
    main()
