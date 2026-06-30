"""Phase 1 pretraining: multi-domain LP BCE baseline.

Trains a TimeAwareMamba backbone on a MixedDataset (multiple small domains
merged) with standard link prediction BCE. This is the baseline to verify
that multi-domain mixed training works and to compare against Phase 2a
(pure self-supervised MTM+NTP) later.

Per the pretraining design (docs/foundation_model_design.md):
  - Phase 1: pure LP BCE + MiNT protocol (order shuffling via balance=True)
  - Datasets: small, low-burstiness, multi-domain (enron/BitcoinAlpha/uci)
  - Goal: verify multi-domain training > single-domain, establish baseline

MiNT protocol:
  - Order shuffling: MixedDataset.get_batches(balance=True) round-robins
    across domains each epoch (each domain contributes equal edges).
  - Context switching: stateless Mamba (no TGN memory) → hidden state
    resets per batch naturally. TGN-style memory would need explicit reset
    on domain boundary (future work).

Launch (glibc239 fast-path):
    scripts/run_mamba.sh examples/train_pretrain_phase1.py \
        --datasets enron BitcoinAlpha uci --epochs 20 --K 32
"""
import argparse
import time

import torch

import sys
import os
# Make benchmarks/ablation importable for _MiniMambaModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ablation"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["enron", "BitcoinAlpha", "uci"],
                   help="Dataset names to mix for pretraining")
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--holdout", default=None,
                   help="If set, hold out this dataset for zero-shot eval "
                        "(train on the rest, eval zero-shot on holdout)")
    args = p.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.core.mixed_dataset import MixedDataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything
    from bench_mamba_async_amp import _MiniMambaModel

    seed_everything(42)

    # --- Load datasets ---
    ds_list = []
    names = []
    for name in args.datasets:
        ds = load_dataset(name, args.data_root)
        ds_list.append(ds)
        names.append(name)
        print(f"  [{name:15s}]  {ds.num_nodes:6d} nodes  {ds.num_edges:7d} edges  "
              f"d_edge={ds.edge_feat_dim}", flush=True)

    # --- Optionally hold out one dataset for zero-shot eval ---
    holdout_ds = None
    holdout_name = None
    if args.holdout is not None:
        assert args.holdout in names, f"holdout {args.holdout} not in datasets {names}"
        idx = names.index(args.holdout)
        holdout_ds = ds_list.pop(idx)
        holdout_name = names.pop(idx)
        print(f"\nHold-out (zero-shot eval): {holdout_name}")
        print(f"Pretrain on: {names}\n")

    # --- Build MixedDataset ---
    mixed = MixedDataset(ds_list, names=names)
    print(mixed.summary(), flush=True)

    # --- Build graph + model ---
    graph = mixed.make_graph(buffer_size=args.K, device=args.device)
    model = _MiniMambaModel(
        mixed.num_nodes, d_model=args.d_model, K=args.K,
        n_layers=args.n_layers, d_state=16,
    ).to(args.device)
    print(f"Model: params={sum(p.numel() for p in model.parameters()):,}  "
          f"backbone=TimeAwareMamba×{args.n_layers}  K={args.K}  d_model={args.d_model}",
          flush=True)

    # --- Batches ---
    # balance=True: round-robin across domains (MiNT order shuffling at batch level)
    train = mixed.get_batches("train", batch_size=args.batch_size,
                              balance=True, device=args.device)
    val = mixed.get_batches("val", batch_size=args.batch_size, device=args.device)
    test = mixed.get_batches("test", batch_size=args.batch_size, device=args.device)
    print(f"Batches: train={len(train)} (balanced)  val={len(val)}  test={len(test)}",
          flush=True)

    # --- Engine ---
    neg = RandomNegative(mixed.num_nodes)
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, device=args.device,
        patience=args.epochs + 5,  # no early stopping for pretraining
        eval_strategy="every_n", eval_every=5,
    )
    engine = Engine(
        model, graph, train, val, test,
        neg_strategy=neg, eval_protocol=APEval(), config=cfg,
    )

    # --- Train ---
    t0 = time.time()
    result = engine.train()
    dt = time.time() - t0
    print(f"\n=== Phase 1 Pretrain ({'+'.join(names)}) ===", flush=True)
    print(f"Total {dt:.1f}s  avg per-epoch {dt/args.epochs:.2f}s", flush=True)
    print(f"Test metrics (merged val/test): {result}", flush=True)
    if args.device.startswith("cuda"):
        print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.3f} GB", flush=True)

    # --- Per-domain eval (key metric: does multi-domain help each domain?) ---
    print(f"\n=== Per-domain eval ===", flush=True)
    for info, ds in zip(mixed._infos, ds_list):
        ap = eval_on_domain(model, ds, info, args)
        print(f"  [{info.name:15s}] val AP={ap:.4f}  (offset={info.node_offset}, "
              f"{ds.num_nodes} nodes)", flush=True)

    # --- Zero-shot eval on holdout ---
    if holdout_ds is not None:
        print(f"\n=== Zero-shot transfer to {holdout_name} ===", flush=True)
        # NOTE: zero-shot needs domain-agnostic features (no node ID).
        # Phase 1 model has node_emb sized for mixed dataset — holdout nodes
        # are out of range. This will be addressed in Phase 2 with structural
        # features. For now, skip zero-shot and note it.
        print(f"  (skipped — Phase 1 uses node_emb, zero-shot needs Phase 2 "
              f"domain-agnostic features)", flush=True)


def eval_on_domain(model, ds, info, args):
    """Evaluate model on a single domain (in-domain, node IDs within range).

    Negatives sampled from THIS domain's node range only (offset, offset+num_nodes)
    so cross-domain "easy negatives" don't inflate AP.
    """
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.engine import APEval
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative

    offset = info.node_offset
    # Graph sized to cover this domain's node range [offset, offset+num_nodes)
    total_nodes = offset + info.num_nodes
    eval_graph = TemporalGraph(
        total_nodes, edge_feat_dim=ds.edge_feat_dim,
        buffer_size=args.K, device=args.device,
    )
    # Negatives: sample within this domain then add offset
    neg = RandomNegative(info.num_nodes)

    # Load all splits of this domain into eval graph (remapped to mixed ID space)
    for split in ["train", "val", "test"]:
        batches = ds.get_batches(split, batch_size=args.batch_size, device=args.device)
        for rb in batches:
            eval_graph.advance(rb.src + offset, rb.dst + offset,
                               rb.time, rb.edge_feat)
    eval_graph.freeze_csr()

    pipeline = DataPipeline(model.gather_spec, eval_graph)

    # Val batches with per-domain neg (sampled in [0, num_nodes) then +offset)
    val_batches = ds.get_batches("val", batch_size=args.batch_size, device=args.device)
    prepped = []
    for rb in val_batches:
        n = neg.sample(rb.src, rb.dst, rb.time, eval_graph, rb.edge_indices) + offset
        prepped.append(RawBatch(
            src=rb.src + offset, dst=rb.dst + offset, time=rb.time,
            edge_feat=rb.edge_feat, neg=n, edge_indices=rb.edge_indices,
        ))

    model.eval()
    protocol = APEval()
    with torch.no_grad():
        metrics = protocol.evaluate(model, pipeline, prepped, eval_graph)
    return metrics.get("ap", 0.0)


if __name__ == "__main__":
    main()
