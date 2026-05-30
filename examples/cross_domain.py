"""Cross-domain generalization experiment with GraphMixer.

Training domains  (d_edge=1, small):
  CanParl       — parliamentary co-sponsorship network
  USLegis       — US legislative co-sponsorship network
  BitcoinAlpha  — Bitcoin trust network (Alpha platform)
  BitcoinOTC    — Bitcoin trust network (OTC platform)

Zero-shot test domains (never seen during training):
  enron         — email communication network (d_edge=32)
  CollegeMsg    — college message network (d_edge=172)
  uci           — UCI message network (d_edge=100)

Research question: does training on a multi-domain mixture produce
a model that generalizes to unseen domains?

Usage:
    CUDA_VISIBLE_DEVICES=1 python examples/cross_domain.py --data_root /mnt/home/gyq/CodeBase/Graph/DG_Data
"""

from __future__ import annotations

import argparse

import torch

from tgengine import (
    APEval,
    GraphMixer,
    MixedDataset,
    RandomNegative,
    TemporalGraph,
    TrainConfig,
    load_dataset,
)
from tgengine.engine import Engine


def zero_shot_eval(
    model: torch.nn.Module,
    dataset_name: str,
    data_root: str,
    d_edge_target: int,
    device: str,
    batch_size: int,
    K: int,
) -> dict[str, float]:
    """Evaluate a trained model on an unseen dataset (zero-shot transfer).

    The model weights are frozen. We build a fresh TemporalGraph for the
    target dataset and run the standard eval protocol (AP).
    """
    from tgengine.core.mixed_dataset import MixedDataset
    from tgengine.pipeline import DataPipeline
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.pipeline.negatives import RandomNegative as RN
    from sklearn.metrics import average_precision_score
    import numpy as np

    print(f"\n  [zero-shot] Loading {dataset_name} ...")
    ds = load_dataset(dataset_name, dataset_path=data_root)

    # Align edge features to match training d_edge
    # Build a single-dataset MixedDataset to get the same normalization pipeline
    wrapped = MixedDataset([ds], names=[dataset_name], d_edge_target=d_edge_target)

    N = wrapped.num_nodes
    graph = TemporalGraph(N, buffer_size=K, edge_feat_dim=d_edge_target, device=device)

    all_batches = wrapped.get_batches("train", batch_size, device=device) + \
                  wrapped.get_batches("val",   batch_size, device=device) + \
                  wrapped.get_batches("test",  batch_size, device=device)

    for rb in all_batches:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    graph.freeze_csr()

    model_copy = model.to(device)
    model_state = model_copy.freeze()

    pipeline = DataPipeline(model_copy.gather_spec, graph)
    neg_strategy = RN(N)

    test_batches = wrapped.get_batches("test", batch_size, device=device)

    all_labels, all_pos_scores, all_neg_scores = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        for rb in test_batches:
            neg = neg_strategy.sample(rb.src, rb.dst, rb.time, graph, None)
            rb.neg = neg
            prepared = pipeline.prepare(rb)
            out = model_copy(prepared)
            all_pos_scores.append(out.pos_score.cpu())
            all_neg_scores.append(out.neg_score.cpu())
            all_labels.extend([1] * rb.src.shape[0])
            all_labels.extend([0] * rb.src.shape[0])

    pos = torch.cat(all_pos_scores).numpy()
    neg = torch.cat(all_neg_scores).numpy()
    scores = np.concatenate([pos, neg])
    labels = np.array(all_labels)
    ap = float(average_precision_score(labels, scores))

    model_copy.thaw(model_state)
    return {"ap": ap, "n_test_edges": len(test_batches) * batch_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # ── 1. Load training domains ──────────────────────────────────────
    TRAIN_DOMAINS = ["CanParl", "USLegis", "BitcoinAlpha", "BitcoinOTC"]
    TEST_DOMAINS  = ["enron", "CollegeMsg", "uci"]

    print("\nLoading training domains ...")
    train_datasets = [load_dataset(n, dataset_path=args.data_root) for n in TRAIN_DOMAINS]

    # ── 2. Mix ────────────────────────────────────────────────────────
    mixed = MixedDataset(train_datasets, names=TRAIN_DOMAINS)
    print("\n" + mixed.summary())

    D_EDGE = mixed.d_edge  # 1 for all four; used also in zero-shot eval

    train_batches = mixed.get_batches("train", args.batch_size, device=args.device)
    val_batches   = mixed.get_batches("val",   args.batch_size, device=args.device)
    test_batches  = mixed.get_batches("test",  args.batch_size, device=args.device)

    graph = mixed.make_graph(buffer_size=args.K, device=args.device)

    # ── 3. Model ──────────────────────────────────────────────────────
    model = GraphMixer(
        d_model=args.d_model,
        d_edge=D_EDGE,
        d_time=64,
        K=args.K,
        num_layers=2,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nGraphMixer params: {n_params:,}  d_model={args.d_model}  d_edge={D_EDGE}")

    # ── 4. Train on mixed dataset ──────────────────────────────────────
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
            lr=1e-4,
            device=args.device,
            patience=args.patience,
            eval_strategy="all",
            use_amp=False,
            grad_clip=1.0,
        ),
    )

    print(f"\nTraining on mixed dataset: {args.epochs} epochs ...")
    print("=" * 60)
    results = engine.train()

    print("\n" + "=" * 60)
    print("IN-DOMAIN TEST RESULTS (mixed dataset):")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")

    # ── 5. Zero-shot transfer to unseen domains ───────────────────────
    print("\n" + "=" * 60)
    print("ZERO-SHOT TRANSFER (unseen domains):")
    print("=" * 60)

    trained_model = engine.model

    zs_results = {}
    for ds_name in TEST_DOMAINS:
        try:
            metrics = zero_shot_eval(
                model=trained_model,
                dataset_name=ds_name,
                data_root=args.data_root,
                d_edge_target=D_EDGE,
                device=args.device,
                batch_size=args.batch_size,
                K=args.K,
            )
            ap = metrics["ap"]
            zs_results[ds_name] = ap
            print(f"  {ds_name:15s}  AP={ap:.4f}")
        except Exception as e:
            print(f"  {ds_name:15s}  ERROR: {e}")

    # ── 6. Baseline: single-domain trained model on same test sets ────
    print("\n" + "=" * 60)
    print("SINGLE-DOMAIN BASELINE (train only on CanParl, test same zero-shot sets):")
    print("=" * 60)

    single_ds = load_dataset("CanParl", dataset_path=args.data_root)
    single_mixed = MixedDataset([single_ds], names=["CanParl"], d_edge_target=D_EDGE)

    single_graph = single_mixed.make_graph(buffer_size=args.K, device=args.device)
    single_model = GraphMixer(
        d_model=args.d_model, d_edge=D_EDGE, d_time=64, K=args.K, num_layers=2, dropout=0.1,
    )
    single_engine = Engine(
        model=single_model,
        graph=single_graph,
        train_batches=single_mixed.get_batches("train", args.batch_size, device=args.device),
        val_batches=single_mixed.get_batches("val",   args.batch_size, device=args.device),
        test_batches=single_mixed.get_batches("test",  args.batch_size, device=args.device),
        neg_strategy=RandomNegative(single_mixed.num_nodes),
        eval_protocol=APEval(),
        config=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=1e-4,
            device=args.device, patience=args.patience,
            eval_strategy="all", use_amp=False, grad_clip=1.0,
        ),
    )
    print("Training single-domain baseline (CanParl only) ...")
    single_engine.train()

    for ds_name in TEST_DOMAINS:
        try:
            metrics = zero_shot_eval(
                model=single_engine.model,
                dataset_name=ds_name,
                data_root=args.data_root,
                d_edge_target=D_EDGE,
                device=args.device,
                batch_size=args.batch_size,
                K=args.K,
            )
            ap_single = metrics["ap"]
            ap_multi  = zs_results.get(ds_name, float("nan"))
            delta = ap_multi - ap_single
            print(f"  {ds_name:15s}  single={ap_single:.4f}  multi={ap_multi:.4f}  Δ={delta:+.4f}")
        except Exception as e:
            print(f"  {ds_name:15s}  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
