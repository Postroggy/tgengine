"""Foundation model pretraining: Phase 2a (MTM + NTP) and Phase 2b (+ LP BCE).

Trains the FoundationModel on a MixedDataset with three pretraining tasks:
  - MTM (Masked Token Modeling): reconstruct masked token features
  - NTP (Next Time Prediction): predict next-event time encoding
  - LP BCE (Link Prediction): standard link prediction loss

Architecture: d_model=512, K=64, 10 Mamba + 2 GCA, ~19M params.

Usage (glibc239 fast-path):
    MAMBA_PYTHON=<mamba2>/python3.11 scripts/run_mamba.sh python \
        examples/train_foundation.py \
        --datasets enron BitcoinAlpha uci \
        --epochs 20 --K 64 --d_model 512 --d_state 64 \
        --lr 1e-3 --batch_size 200

Phase 2a (self-supervised only, no LP):
    --phase 2a
Phase 2b (all three tasks):
    --phase 2b  (default)
"""
import argparse
import sys
import os
import time

import torch
import torch.nn as nn

# Make benchmarks/ablation importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ablation"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["enron", "BitcoinAlpha", "uci"])
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_time", type=int, default=64)
    p.add_argument("--n_mamba_layers", type=int, default=10)
    p.add_argument("--gca_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mtm_mask_ratio", type=float, default=0.15)
    p.add_argument("--mtm_block_size", type=int, default=4)
    p.add_argument("--ema_momentum", type=float, default=0.999)
    p.add_argument("--phase", choices=["2a", "2b"], default="2b",
                   help="2a: MTM+NTP only, 2b: MTM+NTP+LP (default)")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Enable mixed precision (default True)")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.core.mixed_dataset import MixedDataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.models.foundation import FoundationModel
    from tgengine.utils import seed_everything

    seed_everything(42)

    # --- Load and mix datasets ---
    ds_list = []
    names = []
    for name in args.datasets:
        ds = load_dataset(name, args.data_root)
        ds_list.append(ds)
        names.append(name)
        print(f"  [{name:15s}]  {ds.num_nodes:6d} nodes  {ds.num_edges:7d} edges  "
              f"d_edge={ds.edge_feat_dim}", flush=True)

    mixed = MixedDataset(ds_list, names=names)
    print(mixed.summary(), flush=True)

    # --- Build graph ---
    graph = mixed.make_graph(buffer_size=args.K, device=args.device)

    # --- Build batches (before preloading, since preload iterates them) ---
    train_batches = mixed.get_batches(
        "train", batch_size=args.batch_size, balance=True, device=args.device,
    )
    val_batches = mixed.get_batches(
        "val", batch_size=args.batch_size, device=args.device,
    )
    print(f"Batches: train={len(train_batches)} (balanced)  val={len(val_batches)}",
          flush=True)

    # Preload all training edges into the graph (CSR build phase).
    # The TemporalGraph starts empty; without preloading, neighbor queries
    # return all-padding (mask=False), producing zero losses. This mirrors
    # what the Engine does at init: static graph during training.
    print("Preloading training edges into graph...", flush=True)
    t0 = time.time()
    for rb in train_batches:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    graph.freeze_csr()
    print(f"  CSR frozen with {graph.num_edges:,} edges in {time.time()-t0:.1f}s",
          flush=True)

    # --- Build model ---
    model = FoundationModel(
        d_edge=mixed.d_edge,
        d_model=args.d_model,
        d_state=args.d_state,
        K=args.K,
        d_time=args.d_time,
        n_mamba_layers=args.n_mamba_layers,
        gca_every=args.gca_every,
    ).to(args.device)
    model.init_ema(momentum=args.ema_momentum)

    n_params = sum(p.numel() for p in model.parameters())
    mamba_count = sum(1 for b in model.blocks if hasattr(b, "ssm"))
    gca_count = sum(1 for b in model.blocks if hasattr(b, "q_proj"))
    print(f"Model: {n_params:,} params  "
          f"{mamba_count} Mamba + {gca_count} GCA = {len(model.blocks)} blocks  "
          f"K={args.K}  d={args.d_model}  phase={args.phase}", flush=True)

    # --- Build pipeline (for neighbor sampling during training) ---
    pipeline = DataPipeline(model.gather_spec, graph)
    neg = RandomNegative(mixed.num_nodes)

    # --- Optimizer + AMP scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # --- Training loop ---
    print(f"\n{'='*60}", flush=True)
    print(f"  Pretraining Phase {args.phase}  ({'+'.join(names)})  "
          f"amp={args.amp}", flush=True)
    print(f"  {'='*60}", flush=True)

    best_val_ap = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        total_mtm = 0.0
        total_ntp = 0.0
        total_lp = 0.0
        n_steps = 0

        for rb in train_batches:
            # Sample negatives
            neg_ids = neg.sample(rb.src, rb.dst, rb.time, graph, rb.edge_indices)
            rb.neg = neg_ids

            # Prepare batch (neighbor sampling via pipeline)
            batch = pipeline.prepare(rb)

            # Forward under autocast for mixed precision
            with torch.amp.autocast("cuda", enabled=args.amp):
                result = model.pretrain_forward(
                    batch,
                    mtm_mask_ratio=args.mtm_mask_ratio,
                    mtm_block_size=args.mtm_block_size,
                )
                if args.phase == "2a":
                    loss = result["mtm_loss"] + result["ntp_loss"]
                else:
                    loss = result["loss"]

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            model.update_ema()

            total_mtm += result["mtm_loss"].item()
            total_ntp += result["ntp_loss"].item()
            total_lp += result["lp_loss"].item()
            n_steps += 1

        dt = time.time() - t0
        avg_mtm = total_mtm / max(n_steps, 1)
        avg_ntp = total_ntp / max(n_steps, 1)
        avg_lp = total_lp / max(n_steps, 1)

        print(f"  Epoch {epoch:3d}: mtm={avg_mtm:.4f}  ntp={avg_ntp:.4f}  "
              f"lp={avg_lp:.4f}  {dt:.1f}s  "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

        # --- Eval every 5 epochs ---
        if epoch % 5 == 0 or epoch == args.epochs:
            val_ap = eval_per_domain(model, mixed, ds_list, args)
            print(f"    val AP (per-domain): {val_ap}", flush=True)

            # Track best (average per-domain AP)
            avg_ap = sum(val_ap.values()) / max(len(val_ap), 1)
            if avg_ap > best_val_ap:
                best_val_ap = avg_ap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print(f"  Phase {args.phase} complete  best avg per-domain AP={best_val_ap:.4f}",
          flush=True)
    print(f"{'='*60}", flush=True)

    # Save best model
    if best_state is not None:
        save_path = f"checkpoints/foundation_phase{args.phase}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(best_state, save_path)
        print(f"  Saved best model to {save_path}", flush=True)


def eval_per_domain(model, mixed, ds_list, args):
    """Evaluate on each domain separately with in-domain negatives."""
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.engine import APEval

    model.eval()
    results = {}

    for info, ds in zip(mixed._infos, ds_list):
        offset = info.node_offset
        total_nodes = offset + info.num_nodes

        # Build per-domain eval graph
        eval_graph = TemporalGraph(
            total_nodes, edge_feat_dim=ds.edge_feat_dim,
            buffer_size=args.K, device=args.device,
        )

        # Normalize timestamps (same as MixedDataset)
        t_all = ds.time.cpu().float()
        t_min, t_max = float(t_all.min()), float(t_all.max())
        t_norm = ((t_all - t_min) / (t_max - t_min) if t_max > t_min
                  else torch.zeros_like(t_all)).to(args.device)

        n = len(t_norm)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        # Load all edges into eval graph
        src_all = ds.src.to(args.device) + offset
        dst_all = ds.dst.to(args.device) + offset
        feat_all = ds.edge_feat.to(args.device) if ds.edge_feat is not None else None
        eval_graph.advance(src_all, dst_all, t_norm, feat_all)
        eval_graph.freeze_csr()

        # Pipeline for this domain's eval graph
        eval_pipeline = DataPipeline(model.gather_spec, eval_graph)

        # Per-domain neg sampler
        domain_neg = RandomNegative(info.num_nodes)

        # Build val batches
        val_src = src_all[train_end:val_end]
        val_dst = dst_all[train_end:val_end]
        val_t = t_norm[train_end:val_end]
        val_feat = feat_all[train_end:val_end] if feat_all is not None else None

        prepped = []
        bs = args.batch_size
        for i in range(0, len(val_src), bs):
            s = slice(i, i + bs)
            src_b, dst_b, t_b = val_src[s], val_dst[s], val_t[s]
            feat_b = val_feat[s] if val_feat is not None else None
            n_b = domain_neg.sample(
                src_b - offset, dst_b - offset, t_b, eval_graph, None,
            ) + offset
            rb = RawBatch(src=src_b, dst=dst_b, time=t_b, edge_feat=feat_b, neg=n_b)
            prepped.append(eval_pipeline.prepare(rb))

        # Run LP eval (using the model's standard forward for scoring)
        protocol = APEval()
        with torch.no_grad():
            all_pos, all_neg = [], []
            for batch in prepped:
                bundle = model.encode(batch)
                pos = (bundle.src * bundle.dst).sum(-1)
                neg_s = (bundle.src * bundle.neg).sum(-1)
                all_pos.append(pos)
                all_neg.append(neg_s)
            pos_cat = torch.cat(all_pos).sigmoid().cpu().numpy()
            neg_cat = torch.cat(all_neg).sigmoid().cpu().numpy()
            import numpy as np
            from sklearn.metrics import average_precision_score
            predicts = np.concatenate([pos_cat, neg_cat])
            labels = np.concatenate([np.ones(len(pos_cat)), np.zeros(len(neg_cat))])
            ap = average_precision_score(labels, predicts)

        results[info.name] = ap

    model.train()
    return results


if __name__ == "__main__":
    main()
