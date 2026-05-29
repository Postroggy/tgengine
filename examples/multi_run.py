"""Example: run a multi-seed experiment and get mean +- std results.

Usage:
    python examples/multi_run.py --dataset wikipedia --data_root datasets --n_runs 3
"""

import argparse

import torch

from tgengine import (
    DyGFormer, Engine, TrainConfig, APEval, RandomNegative,
    TemporalGraph, load_dataset,
)
from tgengine.engine import run_experiment


def make_build_fn(args):
    """Create a builder function that constructs a fresh Engine for each seed."""

    def build(seed: int) -> Engine:
        torch.manual_seed(seed)
        dataset = load_dataset(args.dataset, dataset_path=args.data_root)
        graph = TemporalGraph(
            dataset.num_nodes, buffer_size=32,
            edge_feat_dim=dataset.edge_feat_dim, device=args.device,
        )
        model = DyGFormer(
            d_edge=dataset.edge_feat_dim, d_time=100,
            K=32, num_layers=2, num_heads=2,
        )
        return Engine(
            model=model, graph=graph,
            train_batches=dataset.get_batches("train", 200, device=args.device),
            val_batches=dataset.get_batches("val", 200, device=args.device),
            test_batches=dataset.get_batches("test", 200, device=args.device),
            neg_strategy=RandomNegative(dataset.num_nodes),
            eval_protocol=APEval(include_auc=True),
            config=TrainConfig(
                epochs=args.epochs, lr=1e-4, seed=seed,
                device=args.device, eval_strategy="adaptive",
            ),
            inductive_edges=dataset.inductive_edges,
        )

    return build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikipedia")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--result_dir", default="results/")
    args = parser.parse_args()

    results = run_experiment(
        build_fn=make_build_fn(args),
        n_runs=args.n_runs,
        result_dir=args.result_dir,
    )

    print("\n" + "=" * 60)
    print("Aggregated Results:")
    for metric, stats in results["aggregated"].items():
        print(f"  {metric}: {stats['mean']:.4f} +- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
