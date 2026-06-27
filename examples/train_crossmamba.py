"""CrossMamba cross-domain training and zero-shot transfer benchmark.

Training domains (all datasets remapped to d_edge=0):
  CanParl, USLegis, BitcoinAlpha, BitcoinOTC,
  enron, mooc, CollegeMsg, uci

Zero-shot transfer (unseen during training):
  wikipedia, lastfm, mathoverflow, Contacts, UNtrade

Records per-epoch AP, final in-domain test AP, and zero-shot transfer AP.

Usage:
    CUDA_VISIBLE_DEVICES=1 python examples/train_crossmamba.py \
        --data_root /mnt/home/gyq/CodeBase/Graph/DG_Data
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import numpy as np
from sklearn.metrics import average_precision_score

from tgengine import (
    APEval,
    CrossMamba,
    MixedDataset,
    RandomNegative,
    TrainConfig,
    load_dataset,
)
from tgengine.engine import Engine
from tgengine.pipeline import DataPipeline


# ── zero-shot evaluation ───────────────────────────────────────────────────

def zero_shot_eval(
    model: torch.nn.Module,
    dataset_name: str,
    data_root: str,
    device: str,
    batch_size: int,
    K: int,
) -> dict:
    """Evaluate trained model on an unseen dataset (d_edge forced to 0)."""
    from tgengine.core.temporal_graph import TemporalGraph

    ds = load_dataset(dataset_name, dataset_path=data_root)
    # wrap through MixedDataset to get same d_edge=0 normalisation pipeline
    wrapped = MixedDataset([ds], names=[dataset_name], d_edge_target=0)

    N = wrapped.num_nodes
    graph = TemporalGraph(N, buffer_size=K, edge_feat_dim=0, device=device)
    for split in ("train", "val", "test"):
        for rb in wrapped.get_batches(split, batch_size, device=device):
            graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    graph.freeze_csr()

    pipeline = DataPipeline(model.gather_spec, graph)
    neg_strategy = RandomNegative(N)
    test_batches = wrapped.get_batches("test", batch_size, device=device)

    model_state = model.freeze()
    pos_scores, neg_scores = [], []
    with torch.no_grad():
        for rb in test_batches:
            rb.neg = neg_strategy.sample(rb.src, rb.dst, rb.time, graph, None)
            prep = pipeline.prepare(rb)
            out = model(prep)
            pos_scores.append(out.pos_score.cpu())
            neg_scores.append(out.neg_score.cpu())
    model.thaw(model_state)

    pos = torch.cat(pos_scores).numpy()
    neg = torch.cat(neg_scores).numpy()
    labels = np.array([1] * len(pos) + [0] * len(neg))
    scores = np.concatenate([pos, neg])
    ap = float(average_precision_score(labels, scores))

    return {
        "ap": ap,
        "n_test": wrapped.test_size,
        "n_nodes": N,
        "n_edges": wrapped.num_edges,
    }


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=200)
    parser.add_argument("--K",          type=int,   default=128)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_layers",   type=int,   default=2)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--balance",    action="store_true", default=False,
                        help="Cap each domain to min(train_size) for balanced mixing")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out",        default="/tmp/crossmamba_results.json")
    args = parser.parse_args()

    t_total_start = time.perf_counter()
    print(f"\nDevice : {args.device}")
    if args.device.startswith("cuda"):
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── training domains ─────────────────────────────────────────────
    # 4 domains covering social/comms, finance, politics, user-content interaction.
    # Deliberately small + diverse to encourage generalizable temporal patterns.
    TRAIN_DOMAINS = [
        "uci",           # social: campus message network
        "CollegeMsg",    # social: college message network
        "BitcoinAlpha",  # finance: trust network
        "wikipedia",     # user-content: edit interactions
    ]
    TEST_DOMAINS = [
        "UNtrade",       # politics: inter-country trade (very different structure)
        "enron",         # social/email (different from campus msg)
        "lastfm",        # user-content: music listening
        "mathoverflow",  # QA: community knowledge graph
        "Contacts",      # proximity: physical contact traces
        "BitcoinOTC",    # finance: trust network (same family as BitcoinAlpha)
        "CanParl",       # politics: Canadian parliament votes
        "USLegis",       # politics: US legislation co-sponsorship
        "mooc",          # education: MOOC action sequences
        "SocialEvo",     # social: campus sensor proximity
    ]

    print(f"\nLoading {len(TRAIN_DOMAINS)} training domains ...")
    train_datasets = []
    for name in TRAIN_DOMAINS:
        ds = load_dataset(name, dataset_path=args.data_root)
        train_datasets.append(ds)

    # Force d_edge=0 for all — pure temporal structure
    mixed = MixedDataset(train_datasets, names=TRAIN_DOMAINS, d_edge_target=0)
    print("\n" + mixed.summary())

    assert mixed.d_edge == 0, "Expected d_edge=0 for CrossMamba"

    # balance=True: round-robin across domains, each capped at min(train_size).
    # Prevents wikipedia (157K) from dominating over BitcoinAlpha (24K).
    train_batches = mixed.get_batches("train", args.batch_size, device=args.device, balance=args.balance)
    val_batches   = mixed.get_batches("val",   args.batch_size, device=args.device)
    test_batches  = mixed.get_batches("test",  args.batch_size, device=args.device)
    graph = mixed.make_graph(buffer_size=args.K, device=args.device)

    # ── model ────────────────────────────────────────────────────────
    model = CrossMamba(
        d_model=args.d_model,
        K=args.K,
        n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nCrossMamba params : {n_params:,}")
    print(f"  d_model={args.d_model}  K={args.K}  n_layers={args.n_layers}")

    # ── engine ───────────────────────────────────────────────────────
    epoch_log: list[dict] = []

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
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            patience=args.patience,
            eval_strategy="all",
            use_amp=False,
            grad_clip=1.0,
        ),
    )

    print(f"\nTraining: {args.epochs} epochs, patience={args.patience}")
    print("=" * 70)

    t_train_start = time.perf_counter()
    results = engine.train()
    t_train = time.perf_counter() - t_train_start

    in_domain_ap = results.get("ap", 0.0)
    print(f"\nIn-domain test AP : {in_domain_ap:.4f}  (train time: {t_train:.1f}s)")

    # ── zero-shot transfer ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ZERO-SHOT TRANSFER TO UNSEEN DOMAINS")
    print("=" * 70)
    zs_results = {}
    for name in TEST_DOMAINS:
        t0 = time.perf_counter()
        try:
            info = zero_shot_eval(
                model=engine.model,
                dataset_name=name,
                data_root=args.data_root,
                device=args.device,
                batch_size=args.batch_size,
                K=args.K,
            )
            elapsed = time.perf_counter() - t0
            zs_results[name] = info
            print(f"  {name:15s}  AP={info['ap']:.4f}  "
                  f"(n_test={info['n_test']:6d}  t={elapsed:.1f}s)")
        except Exception as e:
            print(f"  {name:15s}  ERROR: {e}")
            zs_results[name] = {"ap": None, "error": str(e)}

    # ── summary ─────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_total_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model          : CrossMamba  ({n_params:,} params)")
    print(f"  Train domains  : {', '.join(TRAIN_DOMAINS)}")
    print(f"  Mixed edges    : {mixed.num_edges:,}  (train={mixed.train_size:,})")
    print(f"  Train time     : {t_train:.1f}s")
    print(f"  Total time     : {t_total:.1f}s")
    print(f"  In-domain AP   : {in_domain_ap:.4f}")
    print()
    print(f"  {'Domain':15s}  {'AP':>6}  {'#test':>7}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*7}")
    for name, info in zs_results.items():
        ap = f"{info['ap']:.4f}" if info.get("ap") is not None else "ERROR"
        n  = info.get("n_test", "-")
        print(f"  {name:15s}  {ap:>6}  {n:>7}")

    # ── save JSON ────────────────────────────────────────────────────
    output = {
        "model": "CrossMamba",
        "config": vars(args),
        "n_params": n_params,
        "train_domains": TRAIN_DOMAINS,
        "mixed_dataset": {
            "n_edges": mixed.num_edges,
            "train_size": mixed.train_size,
            "val_size": mixed.val_size,
            "test_size": mixed.test_size,
        },
        "in_domain_ap": in_domain_ap,
        "zero_shot": zs_results,
        "timing": {
            "train_seconds": round(t_train, 1),
            "total_seconds": round(t_total, 1),
        },
    }
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        if hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        return obj

    with open(args.out, "w") as f:
        json.dump(_to_serializable(output), f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
