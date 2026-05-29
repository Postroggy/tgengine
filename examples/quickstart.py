"""Quick start example: train DyGFormer on a dataset.

Usage:
    python examples/quickstart.py --dataset wikipedia --data_root datasets
    python examples/quickstart.py --dataset uci --epochs 50
"""

import argparse
import torch

from tgengine import (
    DyGFormer,
    Engine,
    TrainConfig,
    APEval,
    RandomNegative,
    TemporalGraph,
    load_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="TGEngine Quick Start")
    parser.add_argument("--dataset", default="wikipedia")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--result_dir", default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, dataset_path=args.data_root)
    K = 32
    graph = TemporalGraph(
        dataset.num_nodes,
        buffer_size=K,
        edge_feat_dim=dataset.edge_feat_dim,
        device=args.device,
    )

    model = DyGFormer(
        d_edge=dataset.edge_feat_dim,
        d_time=100,
        K=K,
        num_layers=2,
        num_heads=2,
    )

    engine = Engine(
        model=model,
        graph=graph,
        train_batches=dataset.get_batches("train", 200, device=args.device),
        val_batches=dataset.get_batches("val", 200, device=args.device),
        test_batches=dataset.get_batches("test", 200, device=args.device),
        neg_strategy=RandomNegative(dataset.num_nodes),
        eval_protocol=APEval(include_auc=True),
        config=TrainConfig(
            epochs=args.epochs,
            lr=1e-4,
            device=args.device,
            eval_strategy="adaptive",
            result_dir=args.result_dir,
        ),
        inductive_edges=dataset.inductive_edges,
    )

    results = engine.train()
    print(f"\nFinal test metrics: {results}")


if __name__ == "__main__":
    main()
