"""CrossMamba single-domain training — in-domain accuracy benchmark.

Trains CrossMamba on one dataset with actual edge features (d_edge=actual_dim).
Useful for measuring in-domain ceiling vs DyGFormer/TGN baselines.

Usage:
    CUDA_VISIBLE_DEVICES=1 python examples/train_crossmamba_single.py --dataset uci
"""

from __future__ import annotations

import argparse
import time

import torch

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"

BASELINES = {
    "uci":         {"DyGFormer": 0.9579, "TGN": 0.9234, "CAWN": 0.9518, "GraphMixer": 0.9325},
    "wikipedia":   {"DyGFormer": 0.9829, "TGN": 0.9689},
    "lastfm":      {"DyGFormer": 0.9300, "TGN": 0.7707},
    "mooc":        {"DyGFormer": 0.8752, "TGN": 0.8915},
    "enron":       {"DyGFormer": 0.9247, "TGN": 0.8653},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="uci")
    parser.add_argument("--data_root",  default=DATA_ROOT)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--K",          type=int,   default=128)
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--n_layers",   type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--warmup",     type=int,   default=500,
                        help="LR warmup steps (0 to disable)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from tgengine import APEval, CrossMamba, MixedDataset, RandomNegative, TrainConfig, load_dataset
    from tgengine.engine import Engine
    from tgengine.utils import seed_everything

    seed_everything(0)

    print(f"\nDataset : {args.dataset}")
    print(f"Device  : {args.device}")
    if args.device.startswith("cuda"):
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    ds = load_dataset(args.dataset, dataset_path=args.data_root)

    # Use actual edge features (no d_edge_target override)
    mixed = MixedDataset([ds], names=[args.dataset])
    print("\n" + mixed.summary())
    print(f"d_edge  : {mixed.d_edge}  (edge features fully used)")

    train_batches = mixed.get_batches("train", args.batch_size, device=args.device)
    val_batches   = mixed.get_batches("val",   args.batch_size, device=args.device)
    test_batches  = mixed.get_batches("test",  args.batch_size, device=args.device)
    graph         = mixed.make_graph(buffer_size=args.K, device=args.device)

    model = CrossMamba(
        d_model=args.d_model,
        K=args.K,
        n_layers=args.n_layers,
        d_edge=mixed.d_edge,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nCrossMamba  {n_params:,} params  d_model={args.d_model}  K={args.K}"
          f"  n_layers={args.n_layers}  d_edge={mixed.d_edge}")

    engine = Engine(
        model=model,
        graph=graph,
        train_batches=train_batches,
        val_batches=val_batches,
        test_batches=test_batches,
        neg_strategy=RandomNegative(mixed.num_nodes),
        eval_protocol=APEval(),
        config=TrainConfig(
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            eval_strategy="all",
            grad_clip=1.0,
            warmup_steps=args.warmup,
        ),
    )

    print(f"\nTraining: {args.epochs} epochs  patience={args.patience}  warmup={args.warmup}")
    print("=" * 60)
    t0 = time.perf_counter()
    results = engine.train()
    elapsed = time.perf_counter() - t0

    ap = results.get("ap", 0.0)
    best_epoch = results.get("best_epoch", "?")

    print(f"\n{'='*60}")
    print(f"CrossMamba ({args.dataset})  AP={ap:.4f}  best_epoch={best_epoch}  time={elapsed:.0f}s")
    print()

    if args.dataset in BASELINES:
        print("Comparison (in-domain, random negative):")
        for model_name, ref_ap in BASELINES[args.dataset].items():
            gap = ap - ref_ap
            marker = "✓" if gap >= 0 else "✗"
            print(f"  {marker} vs {model_name:12s}  {ref_ap:.4f}  Δ={gap:+.4f}")


if __name__ == "__main__":
    main()

