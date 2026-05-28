"""Standard DyGFormer training script on Wikipedia/Reddit/LastFM.

Reproduces DyGLib reference accuracy using the Engine API.
Usage:
    python examples/train_dygformer.py --dataset wiki
    python examples/train_dygformer.py --dataset reddit --epochs 100
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"

REFERENCE_AP = {"wiki": 0.974, "reddit": 0.990, "lastfm": 0.773, "uci": 0.9613}
DATASET_NAMES = {"wiki": "wikipedia", "reddit": "reddit", "lastfm": "lastfm", "uci": "uci"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wiki", choices=list(DATASET_NAMES))
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(0)
    dataset_name = DATASET_NAMES[args.dataset]

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, args.data_root)
    print(f"  {ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}"
          f"  d_node={ds.node_feat_dim if ds.node_feat is not None else 'N/A'}")

    # Node features from dataset (already loaded by load_dataset from ml_{dataset}_node.npy)
    node_feat = ds.node_feat

    K = 63  # max_input_sequence_length - 1, matching DyGLib (64 total with self)

    # Adaptive buffer: reduce K if graph would OOM (>8 GB estimate)
    mem_gb = ds.num_nodes * K * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 8.0:
        K = max(4, int(K * 8.0 / mem_gb))
        print(f"  K auto-reduced to {K} ({mem_gb:.1f} GB estimated)")

    graph = TemporalGraph(ds.num_nodes,
                          edge_feat_dim=ds.edge_feat_dim, device=args.device)

    model = DyGFormer(
        d_model=172, d_edge=ds.edge_feat_dim,
        d_time=100, d_channel=50,
        K=K, n_layers=2, patch_size=2,
        node_feat=node_feat,
        num_nodes=ds.num_nodes,
    )
    print(f"  {sum(p.numel() for p in model.parameters()):,} params, K={K}")

    # Negative sampling from seen dst nodes (matching DyGLib)
    train_dst = ds.dst[:ds.train_end]
    valid_dst_nodes = torch.unique(train_dst)

    engine = Engine(
        model=model,
        graph=graph,
        train_batches=ds.get_batches("train", args.batch_size, args.device),
        val_batches=ds.get_batches("val", args.batch_size, args.device),
        test_batches=ds.get_batches("test", args.batch_size, args.device),
        neg_strategy=RandomNegative(ds.num_nodes, valid_dst_nodes=valid_dst_nodes),
        eval_protocol=APEval(),
        config=TrainConfig(
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            device=args.device,
        ),
    )

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    best = engine.train()
    best_ap = best.get("ap", 0.0)
    ref = REFERENCE_AP.get(args.dataset, "?")
    print(f"\nBest test AP: {best_ap:.4f}  (DyGLib ref: {ref}  gap: {best_ap - ref:+.4f})")


if __name__ == "__main__":
    main()
