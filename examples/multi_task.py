"""Multi-task training example: link prediction + node classification.

Demonstrates the new Engine tasks= API — no subclassing required.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/multi_task.py --dataset uci
"""

from __future__ import annotations

import argparse

import torch

import tgengine
from tgengine import (
    APEval,
    GraphMixer,
    NodeClassificationHead,
    NodeClsEval,
    LinkPredHead,
    RandomNegative,
    TrainConfig,
    load_dataset,
)
from tgengine.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="uci")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=172)
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of node label classes (dataset-dependent)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load data ---
    dataset = load_dataset(args.dataset, device=device)
    train_b, val_b, test_b = dataset.get_batches(batch_size=200)
    graph = dataset.build_graph(device=device)

    # --- Model: plain GraphMixer, no subclassing ---
    model = GraphMixer(
        d_model=args.d_model,
        d_edge=dataset.d_edge,
        K=32,
    )

    # --- Task heads ---
    # Link prediction: standard BCE loss
    link_head = LinkPredHead()

    # Node classification: 2-layer MLP on src embedding
    # (requires dataset.node_labels to be set)
    node_cls_head = NodeClassificationHead(
        d_model=args.d_model,
        num_classes=args.num_classes,
    )

    # --- Engine with multi-task config ---
    engine = Engine(
        model=model,
        graph=graph,
        train_batches=train_b,
        val_batches=val_b,
        test_batches=test_b,
        neg_strategy=RandomNegative(dataset.num_nodes),
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=200,
            device=device,
            patience=10,
        ),
        # Multi-task: Engine handles encode() → heads automatically
        tasks={
            "link_pred": link_head,
            "node_cls": node_cls_head,
        },
        task_weights={
            "link_pred": 1.0,
            "node_cls": 0.5,   # secondary task gets lower weight
        },
        eval_protocols={
            "link_pred": APEval(),
            "node_cls": NodeClsEval(num_classes=args.num_classes),
        },
        primary_metric="ap",   # early stopping driven by AP
    )

    results = engine.train()
    print("\nFinal test metrics:", results)


if __name__ == "__main__":
    main()
