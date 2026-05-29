"""GraphMixer on UCI: all downstream tasks in one training run (3 epochs).

Tasks:
  link_pred   — standard link prediction (BCE loss)
  edge_reg    — predict edge-feature L2 norm (communication "strength")
  anomaly     — unsupervised autoencoder anomaly detection (no labels needed)
  node_cls    — 3-class node classification by degree quantile (synthetic label)

Eval protocols per epoch:
  ap_auc     → AP + AUC  (link prediction)
  threeway   → AP_random / AP_historical / AP_inductive  (link prediction)
  edge_reg   → MAE + RMSE
  anomaly    → {} for unsupervised without binary labels (included for completeness)
  node_cls   → accuracy + F1-macro  (synthetic degree-bucket labels)

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/graphmixer_all_tasks.py --dataset uci
"""

from __future__ import annotations

import argparse

import torch

from tgengine import (
    APEval,
    GraphMixer,
    MRREval,
    NodeClassificationHead,
    NodeClsEval,
    LinkPredHead,
    RandomNegative,
    TemporalGraph,
    TrainConfig,
    ThreeWayEval,
    load_dataset,
)
from tgengine.engine import Engine
from tgengine.engine.eval import EdgeRegEval, AnomalyEval
from tgengine.tasks import AnomalyDetectionHead, EdgeRegressionHead


def make_degree_labels(
    src: torch.Tensor, dst: torch.Tensor,
    train_mask: torch.Tensor, num_nodes: int, num_classes: int = 3
) -> torch.Tensor:
    """Assign class label to each node based on training-set degree quantile."""
    train_src = src[train_mask]
    train_dst = dst[train_mask]
    all_nodes = torch.cat([train_src, train_dst])
    degree = torch.zeros(num_nodes, dtype=torch.long)
    for v in all_nodes.cpu():
        degree[v.item()] += 1

    nonzero = degree[degree > 0]
    if len(nonzero) == 0:
        return degree
    quantiles = torch.quantile(nonzero.float(), torch.linspace(0, 1, num_classes + 1))
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        lo = quantiles[c].item()
        hi = quantiles[c + 1].item()
        mask = (degree >= lo) if c == num_classes - 1 else ((degree >= lo) & (degree < hi))
        labels[mask] = c
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="uci")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=172)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, dataset_path=args.data_root)
    N = dataset.num_nodes
    print(f"  nodes={N}  edges={len(dataset.src)}  d_edge={dataset.edge_feat_dim}")

    # --- inductive nodes for ThreeWayEval ---
    train_mask = dataset.train_mask if hasattr(dataset, "train_mask") else None
    if dataset.inductive_edges is not None:
        ie = dataset.inductive_edges
        inductive_nodes = torch.unique(torch.cat([ie["src"], ie["dst"]]))
    elif (dataset.new_node_test_mask is not None):
        # test edges that involve new nodes → extract unique node ids
        test_src = dataset.src[dataset.new_node_test_mask.bool()]
        test_dst = dataset.dst[dataset.new_node_test_mask.bool()]
        inductive_nodes = torch.unique(torch.cat([test_src, test_dst]))
    else:
        inductive_nodes = torch.arange(N)
    inductive_nodes = inductive_nodes.to(args.device)
    print(f"  Inductive nodes: {len(inductive_nodes)}")

    # --- synthetic node labels: degree bucket (3 classes) ---
    # Use raw src/dst arrays (before batching) for efficiency
    all_src = dataset.src.long()   # (num_edges,)
    all_dst = dataset.dst.long()
    # train split: first ~70% of edges
    n_train = int(len(all_src) * 0.70)
    tr_mask = torch.zeros(len(all_src), dtype=torch.bool)
    tr_mask[:n_train] = True
    node_labels = make_degree_labels(all_src, all_dst, tr_mask, N, num_classes=3)
    dataset.node_labels = node_labels
    counts = node_labels.bincount(minlength=3).tolist()
    print(f"  Node label distribution (3 degree classes): {counts}")

    # --- edge regression target: synthetic (edge_feat is all-zero on UCI) ---
    torch.manual_seed(42)
    edge_reg_targets = torch.rand(len(dataset.src))
    dataset.edge_labels = edge_reg_targets
    print(f"  Edge reg target: min={edge_reg_targets.min():.3f} max={edge_reg_targets.max():.3f}")

    # Build batches with labels
    train_batches = dataset.get_batches("train", args.batch_size, device=args.device)
    val_batches   = dataset.get_batches("val",   args.batch_size, device=args.device)
    test_batches  = dataset.get_batches("test",  args.batch_size, device=args.device)

    graph = TemporalGraph(N, buffer_size=args.K,
                          edge_feat_dim=dataset.edge_feat_dim, device=args.device)

    # --- model ---
    model = GraphMixer(
        d_model=args.d_model,
        d_edge=dataset.edge_feat_dim,
        d_time=100,
        K=args.K,
        num_layers=2,
        dropout=0.1,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  GraphMixer params: {num_params:,}")

    # --- task heads ---
    link_head     = LinkPredHead()
    edge_reg_head = EdgeRegressionHead(d_model=args.d_model, output_dim=1,
                                       loss="mse", input_mode="concat")
    anomaly_head  = AnomalyDetectionHead(d_model=args.d_model, mode="unsupervised")
    node_cls_head = NodeClassificationHead(d_model=args.d_model, num_classes=3)

    # --- random neg candidates for MRR (UCI has no pre-computed lists) ---
    # Engine reuses the same MRREval for val and test, so size must cover both.
    n_mrr_neg = 49
    mrr_size = max(dataset.val_size, dataset.test_size)
    torch.manual_seed(0)
    mrr_neg = torch.randint(0, N, (mrr_size, n_mrr_neg))

    # --- eval protocols ---
    # head= routes through model.encode() + head() for downstream task eval
    eval_protocols = {
        "ap_auc":   APEval(include_auc=True),
        "threeway": ThreeWayEval(num_nodes=N, inductive_nodes=inductive_nodes,
                                 device=args.device),
        "mrr":      MRREval(mrr_neg),
        "edge_reg": EdgeRegEval(head=edge_reg_head),
        "anomaly":  AnomalyEval(head=anomaly_head),          # returns {} (no binary labels)
        "node_cls": NodeClsEval(num_classes=3, head=node_cls_head),
    }

    # --- engine ---
    engine = Engine(
        model=model,
        graph=graph,
        train_batches=train_batches,
        val_batches=val_batches,
        test_batches=test_batches,
        neg_strategy=RandomNegative(N),
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-4,
            head_lr=5e-4,
            device=args.device,
            patience=0,
            eval_strategy="all",
            stopping_rule="primary",
            use_amp=False,
            grad_clip=1.0,
        ),
        tasks={
            "link_pred": link_head,
            "edge_reg":  edge_reg_head,
            "anomaly":   anomaly_head,
            "node_cls":  node_cls_head,
        },
        task_weights={
            "link_pred": 1.0,
            "edge_reg":  0.3,
            "anomaly":   0.3,
            "node_cls":  0.5,
        },
        eval_protocols=eval_protocols,
        primary_metric="ap",
        inductive_edges=dataset.inductive_edges,
    )

    print(f"\nStarting training: {args.epochs} epochs, {len(train_batches)} batches/epoch")
    print("=" * 60)
    results = engine.train()

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS (best val epoch):")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
