"""Verify attribution hypotheses for poor pretraining results.

3 controlled experiments (small model d=128, K=32, 10 epoch for speed):
  A: proportional mixing (no cap) — tests data-cap hypothesis
  B: pure self-supervised (MTM+NTP, no LP) — tests LP-dominate hypothesis
  C: with node embedding — tests no-node-emb hypothesis

Baseline (from previous run): balanced mix, no node emb, 3 tasks
  enron=0.711, BA=0.732, uci=0.777
"""
import argparse
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ablation"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["enron", "BitcoinAlpha", "uci"])
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--n_mamba_layers", type=int, default=4)
    p.add_argument("--gca_every", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--experiment", choices=["baseline", "A_prop", "B_noss", "C_nodeemb"],
                   required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.core.mixed_dataset import MixedDataset
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.models.foundation import FoundationModel
    from tgengine.utils import seed_everything

    seed_everything(42)

    # --- Load datasets ---
    ds_list = [load_dataset(n, args.data_root) for n in args.datasets]
    mixed = MixedDataset(ds_list, names=args.datasets)
    print(f"Experiment: {args.experiment}", flush=True)
    print(mixed.summary(), flush=True)

    # --- Build graph ---
    graph = mixed.make_graph(buffer_size=args.K, device=args.device)

    # --- Experiment-specific config ---
    pretrain_mode = True
    task_weights = {"mtm": 1.0, "ntp": 1.0, "lp": 1.0}
    mix_mode = "balanced"
    use_node_emb = False

    if args.experiment == "A_prop":
        # Proportional mixing (no cap)
        mix_mode = "proportional"
    elif args.experiment == "B_noss":
        # Pure self-supervised: disable LP
        task_weights = {"mtm": 1.0, "ntp": 1.0, "lp": 0.0}
    elif args.experiment == "C_nodeemb":
        # Add node embedding back
        use_node_emb = True

    # --- Build model ---
    model = FoundationModel(
        d_edge=mixed.d_edge,
        d_model=args.d_model,
        d_state=args.d_state,
        K=args.K,
        d_time=32,
        n_mamba_layers=args.n_mamba_layers,
        gca_every=args.gca_every,
        pretrain_mode=pretrain_mode,
        task_weights=task_weights,
    ).to(args.device)
    model.init_ema(momentum=0.999)

    # Add node embedding if requested
    if use_node_emb:
        model.node_emb = torch.nn.Embedding(mixed.num_nodes, args.d_model).to(args.device)
        # Hook into encode_one to add node_emb
        orig_encode_one = model.encode_one
        def encode_one_with_nodeemb(nbr, query_time, mtm_mask=None, ntp_mask_pos0=False):
            h, st = orig_encode_one(nbr, query_time, mtm_mask, ntp_mask_pos0)
            # Add source node embedding (from neighbor_ids[:, 0] = most recent)
            # Actually we need the query node's own ID, which isn't in NeighborData.
            # Use mean of neighbor embeddings as proxy.
            ids = nbr.neighbor_ids.long().clamp(min=0)
            nbr_emb = model.node_emb(ids).mean(dim=1)  # (B, d_model)
            h = h + nbr_emb.unsqueeze(1)  # broadcast to (B, K, d_model)
            return h, st
        model.encode_one = encode_one_with_nodeemb

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  mix={mix_mode}  weights={task_weights}  "
          f"node_emb={use_node_emb}", flush=True)

    # --- Batches ---
    train_batches = mixed.get_batches(
        "train", batch_size=args.batch_size, balance=True, mode=mix_mode, device=args.device,
    )
    val_batches = mixed.get_batches("val", batch_size=args.batch_size, device=args.device)
    test_batches = mixed.get_batches("test", batch_size=args.batch_size, device=args.device)
    print(f"Batches: train={len(train_batches)} val={len(val_batches)}", flush=True)

    # --- Engine ---
    neg = RandomNegative(mixed.num_nodes)
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, device=args.device,
        patience=args.epochs + 5, grad_clip=1.0,
        eval_strategy="every_n", eval_every=5,
    )
    engine = Engine(
        model, graph, train_batches, val_batches, test_batches,
        neg_strategy=neg, eval_protocol=APEval(), config=cfg,
    )

    # --- Train ---
    t0 = time.time()
    result = engine.train()
    dt = time.time() - t0
    print(f"\n=== {args.experiment} complete in {dt:.1f}s ===", flush=True)
    print(f"  merged test AP: {result}", flush=True)

    # --- Per-domain eval ---
    print(f"\n  Per-domain eval:", flush=True)
    for info, ds in zip(mixed._infos, ds_list):
        ap = eval_on_domain(model, ds, info, args)
        print(f"    [{info.name:15s}] val AP={ap:.4f}", flush=True)


def eval_on_domain(model, ds, info, args):
    """Per-domain eval with in-domain negatives + normalized timestamps."""
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative

    offset = info.node_offset
    total_nodes = offset + info.num_nodes
    eval_graph = TemporalGraph(
        total_nodes, edge_feat_dim=ds.edge_feat_dim,
        buffer_size=args.K, device=args.device,
    )
    domain_neg = RandomNegative(info.num_nodes)

    # Normalize timestamps (same as MixedDataset)
    t_all = ds.time.cpu().float()
    t_min, t_max = float(t_all.min()), float(t_all.max())
    t_norm = ((t_all - t_min) / (t_max - t_min) if t_max > t_min
              else torch.zeros_like(t_all)).to(args.device)

    n = len(t_norm)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    src_all = ds.src.to(args.device) + offset
    dst_all = ds.dst.to(args.device) + offset
    feat_all = ds.edge_feat.to(args.device) if ds.edge_feat is not None else None
    eval_graph.advance(src_all, dst_all, t_norm, feat_all)
    eval_graph.freeze_csr()

    eval_pipeline = DataPipeline(model.gather_spec, eval_graph)

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
        n_b = domain_neg.sample(src_b - offset, dst_b - offset, t_b, eval_graph, None) + offset
        rb = RawBatch(src=src_b, dst=dst_b, time=t_b, edge_feat=feat_b, neg=n_b)
        prepped.append(eval_pipeline.prepare(rb))

    model.eval()
    import numpy as np
    from sklearn.metrics import average_precision_score
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
        predicts = np.concatenate([pos_cat, neg_cat])
        labels = np.concatenate([np.ones(len(pos_cat)), np.zeros(len(neg_cat))])
        return average_precision_score(labels, predicts)


if __name__ == "__main__":
    main()
