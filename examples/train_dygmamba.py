"""DyGMamba training script.

Usage:
    python examples/train_dygmamba.py --dataset wiki
    python examples/train_dygmamba.py --dataset uci --epochs 100
"""

from __future__ import annotations

import argparse
import torch

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
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
    from tgengine.models.dygmamba import DyGMamba
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(0)
    dataset_name = DATASET_NAMES[args.dataset]

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, args.data_root)
    print(f"  {ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}")

    K = 32
    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=args.device)

    model = DyGMamba(d_model=172, d_edge=ds.edge_feat_dim, n_layers=2)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params, K={K}")

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
    print(f"\nBest test AP: {best.get('ap', 0.0):.4f}")


if __name__ == "__main__":
    main()
