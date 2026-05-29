"""YAML-driven training runner.

Usage:
    python -m tgengine.run configs/dygformer/uci.yaml
    python -m tgengine.run configs/dygformer/uci.yaml --override training.lr=5e-4
    python -m tgengine.run configs/dygformer/uci.yaml --override training.use_amp=true
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import torch

from tgengine.utils import load_config, merge_config, seed_everything
from tgengine.utils.logging import check_gpu_memory


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model(cfg: dict, dataset_info: dict) -> "TemporalModel":
    """Instantiate a model from config dict."""
    from tgengine.models.dygformer import DyGFormer
    from tgengine.models.dygmamba import DyGMamba
    from tgengine.models.freedyg import FreeDyG
    from tgengine.models.graphmixer import GraphMixer
    from tgengine.models.tgn import TGN

    name = cfg.pop("name")
    cfg.setdefault("d_edge", dataset_info["edge_feat_dim"])

    registry = {
        "dygformer": DyGFormer,
        "graphmixer": GraphMixer,
        "dygmamba": DyGMamba,
        "freedyg": FreeDyG,
        "tgn": TGN,
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(registry)}")

    cls = registry[name]

    # Inject dataset-level params
    if name in ("dygformer", "tgn"):
        cfg.setdefault("num_nodes", dataset_info["num_nodes"])
    if name == "dygformer":
        cfg.setdefault("node_feat", dataset_info.get("node_feat"))

    # Filter to only params the constructor accepts
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in cfg.items() if k in valid_params}

    return cls(**filtered)


# ---------------------------------------------------------------------------
# Negative strategy registry
# ---------------------------------------------------------------------------

def _build_neg_strategy(cfg: dict, dataset_info: dict, device: str):
    """Instantiate a negative sampling strategy from config."""
    from tgengine.pipeline.negatives import (
        HistoricalNegative, InBatchNegative, RandomNegative,
    )

    name = cfg.get("name", "random")
    num_nodes = dataset_info["num_nodes"]

    if name == "random":
        valid_dst = dataset_info.get("valid_dst_nodes")
        return RandomNegative(num_nodes, valid_dst_nodes=valid_dst)
    elif name == "in_batch":
        mix_random = cfg.get("mix_random", 0.0)
        valid_dst = dataset_info.get("valid_dst_nodes")
        return InBatchNegative(num_nodes, mix_random=mix_random,
                               valid_dst_nodes=valid_dst)
    elif name == "historical":
        pool_size = cfg.get("pool_size", 512)
        return HistoricalNegative(num_nodes, pool_size=pool_size, device=device)
    else:
        raise ValueError(f"Unknown neg strategy: {name}")


# ---------------------------------------------------------------------------
# Eval protocol registry
# ---------------------------------------------------------------------------

def _build_eval_protocol(cfg: dict, dataset_info: dict, device: str):
    """Instantiate an eval protocol from config."""
    from tgengine.engine import APEval, MRREval, ThreeWayEval

    name = cfg.get("name", "ap")

    if name == "ap":
        return APEval()
    elif name == "three_way":
        inductive_nodes = dataset_info.get("inductive_nodes",
                                           torch.empty(0, dtype=torch.long))
        return ThreeWayEval(dataset_info["num_nodes"], inductive_nodes, device=device)
    elif name == "mrr":
        neg_lists = dataset_info.get("neg_lists")
        if neg_lists is None:
            raise ValueError("MRREval requires neg_lists in dataset_info")
        return MRREval(neg_lists)
    else:
        raise ValueError(f"Unknown eval protocol: {name}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def train_from_config(cfg: dict) -> dict[str, float]:
    """Run full training pipeline from a config dict.

    Config structure:
        data:
            root: str
            dataset: str
        model:
            name: str
            ...model params...
        training:
            epochs: int
            batch_size: int
            lr: float
            patience: int
            device: str
            seed: int
            use_amp: bool
            grad_clip: float
            warmup_steps: int
            compile_model: bool
            async_pipeline: bool
            checkpoint_dir: str | null
        negative_sampling:
            name: random | in_batch | historical
            ...strategy params...
        eval:
            name: ap | three_way | mrr

    Returns:
        Best test metrics dict.
    """
    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import Engine, TrainConfig

    # --- Data ---
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    device = train_cfg.get("device", "cuda")
    seed = train_cfg.get("seed", 42)
    seed_everything(seed)

    ds = load_dataset(data_cfg["dataset"], data_cfg.get("root", "."))
    print(f"Dataset: {data_cfg['dataset']} — {ds.num_edges:,} edges, "
          f"{ds.num_nodes:,} nodes, d_edge={ds.edge_feat_dim}")

    batch_size = train_cfg.get("batch_size", 200)

    # Dataset info for builders
    train_dst = ds.dst[:ds.train_end]
    valid_dst_nodes = torch.unique(train_dst)
    dataset_info: dict[str, Any] = {
        "num_nodes": ds.num_nodes,
        "edge_feat_dim": ds.edge_feat_dim,
        "node_feat": ds.node_feat,
        "valid_dst_nodes": valid_dst_nodes,
    }

    # Auto-detect fixed neg samples → default to MRR eval
    if ds.test_neg_candidates is not None:
        dataset_info["neg_lists"] = ds.test_neg_candidates
        eval_cfg = cfg.get("eval", {"name": "mrr"})
        if "eval" not in cfg:
            print("  Auto-selected MRR eval (fixed negative candidates detected)")
    else:
        eval_cfg = cfg.get("eval", {"name": "ap"})

    # --- Model ---
    model_cfg = dict(cfg.get("model", {}))
    K = model_cfg.get("K", 63)
    # Adaptive K reduction for memory
    mem_gb = ds.num_nodes * K * max(ds.edge_feat_dim, 1) * 4 / 1e9
    if mem_gb > 8.0:
        K = max(4, int(K * 8.0 / mem_gb))
        model_cfg["K"] = K
        print(f"  K auto-reduced to {K}")

    model = _build_model(model_cfg, dataset_info)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']} — {n_params:,} params")

    # GPU memory warning
    warning = check_gpu_memory(ds.num_nodes, K, ds.edge_feat_dim, device)
    if warning:
        print(f"⚠ {warning}")

    # --- Graph ---
    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=device)

    # --- Neg strategy ---
    neg_cfg = cfg.get("negative_sampling", {"name": "random"})
    neg_strategy = _build_neg_strategy(neg_cfg, dataset_info, device)

    # --- Eval protocol ---
    eval_protocol = _build_eval_protocol(eval_cfg, dataset_info, device)

    # --- TrainConfig ---
    config = TrainConfig(
        epochs=train_cfg.get("epochs", 100),
        batch_size=batch_size,
        lr=train_cfg.get("lr", 1e-4),
        patience=train_cfg.get("patience", 20),
        device=device,
        seed=seed,
        async_pipeline=train_cfg.get("async_pipeline", False),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        use_amp=train_cfg.get("use_amp", False),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        compile_model=train_cfg.get("compile_model", False),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        wandb_project=train_cfg.get("wandb_project"),
        wandb_run_name=train_cfg.get("wandb_run_name"),
    )

    # --- Engine ---
    engine = Engine(
        model=model,
        graph=graph,
        train_batches=ds.get_batches("train", batch_size, device),
        val_batches=ds.get_batches("val", batch_size, device),
        test_batches=ds.get_batches("test", batch_size, device),
        neg_strategy=neg_strategy,
        eval_protocol=eval_protocol,
        config=config,
    )
    engine.logger.dataset_name = data_cfg['dataset']

    # --- Train ---
    best = engine.train()
    return best


def _parse_overrides(override_strs: list[str]) -> dict:
    """Parse 'a.b.c=value' overrides into nested dict."""
    result: dict = {}
    for s in override_strs:
        if "=" not in s:
            raise ValueError(f"Override must be key=value, got: {s}")
        key, val = s.split("=", 1)
        # Auto-cast value
        parsed_val: Any
        if val.lower() in ("true", "yes"):
            parsed_val = True
        elif val.lower() in ("false", "no"):
            parsed_val = False
        elif val.lower() in ("null", "none"):
            parsed_val = None
        else:
            try:
                parsed_val = int(val)
            except ValueError:
                try:
                    parsed_val = float(val)
                except ValueError:
                    parsed_val = val

        # Build nested dict from dotted key
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = parsed_val

    return result


def main():
    parser = argparse.ArgumentParser(description="TGEngine YAML-driven trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--override", "-o", nargs="*", default=[],
                        help="Override config values: key.path=value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        overrides = _parse_overrides(args.override)
        cfg = merge_config(cfg, overrides)

    train_from_config(cfg)


if __name__ == "__main__":
    main()
