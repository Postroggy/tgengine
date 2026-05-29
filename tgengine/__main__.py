"""TGEngine CLI entry point.

Usage:
    python -m tgengine train configs/dygformer_wiki.yaml
    python -m tgengine download wikipedia reddit
    python -m tgengine list-models
    python -m tgengine list-datasets
"""

from __future__ import annotations

import argparse
import sys


def cmd_train(args):
    """Run training from a YAML config."""
    from tgengine.run import main as run_main
    sys.argv = ["tgengine", "--config", args.config]
    if args.overrides:
        sys.argv.extend(args.overrides)
    run_main()


def cmd_download(args):
    """Download one or more datasets."""
    from tgengine.utils.download import download_dataset, list_available_datasets

    if not args.datasets:
        print("Available datasets:", ", ".join(list_available_datasets()))
        return

    for name in args.datasets:
        try:
            download_dataset(name, dest_dir=args.dest)
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")


def cmd_list_models(args):
    """List available built-in models."""
    models = [
        ("DyGFormer", "NeurIPS 2023", "Patched neighbor attention with co-occurrence"),
        ("TGN", "ICML 2020", "Memory-augmented GNN with GRU message passing"),
        ("GraphMixer", "ICLR 2023", "MLP-Mixer on temporal neighbor sequences"),
        ("DyGMamba", "2024", "Mamba state-space model for temporal encoding"),
        ("FreeDyG", "AAAI 2024", "Frequency-domain temporal encoding"),
        ("EdgeBank", "-", "Heuristic baseline (no learning)"),
    ]
    print(f"\n{'Model':<14} {'Paper':<14} Description")
    print("-" * 60)
    for name, paper, desc in models:
        print(f"{name:<14} {paper:<14} {desc}")
    print()


def cmd_list_datasets(args):
    """List datasets available for auto-download."""
    from tgengine.utils.download import list_datasets_by_family
    families = list_datasets_by_family()

    print("\nDatasets available for auto-download:\n")
    for family, datasets in families.items():
        label = {
            "dyglib": "DyGLib/DGB (link prediction, AP eval)",
            "tgb": "TGB (link prediction, MRR eval)",
            "tgbseq": "TGB-Seq (sequential dynamics, MRR eval)",
        }.get(family, family)
        print(f"  {label}:")
        for d in datasets:
            print(f"    - {d}")
        print()

    total = sum(len(v) for v in families.values())
    print(f"Total: {total} datasets")
    print(f"\nUsage: python -m tgengine download wikipedia")
    print(f"Or: load_dataset(\"wikipedia\") — downloads automatically.\n")


def main():
    parser = argparse.ArgumentParser(
        prog="tgengine",
        description="TGEngine: High-Performance Temporal Graph Learning",
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    p_train = subparsers.add_parser("train", help="Train a model from YAML config")
    p_train.add_argument("config", help="Path to YAML config file")
    p_train.add_argument("overrides", nargs="*", help="Override config values (key=value)")
    p_train.set_defaults(func=cmd_train)

    # download
    p_dl = subparsers.add_parser("download", help="Download datasets")
    p_dl.add_argument("datasets", nargs="*", help="Dataset names (e.g., wikipedia reddit)")
    p_dl.add_argument("--dest", default="datasets", help="Destination directory")
    p_dl.set_defaults(func=cmd_download)

    # list-models
    p_lm = subparsers.add_parser("list-models", help="List available models")
    p_lm.set_defaults(func=cmd_list_models)

    # list-datasets
    p_ld = subparsers.add_parser("list-datasets", help="List downloadable datasets")
    p_ld.set_defaults(func=cmd_list_datasets)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
