"""Utility functions: seeding, config loading, logging."""

import os
import re
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch import Tensor


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# YAML 1.1 safe_load doesn't parse scientific notation (1e-4) as float.
# Add an implicit resolver for it.
_SCI_NOTATION_RE = re.compile(
    r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)$'
)


class _SciFloatLoader(yaml.SafeLoader):
    pass


_SciFloatLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    _SCI_NOTATION_RE,
    list('-+0123456789.'),
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.load(f, Loader=_SciFloatLoader)
    return cfg if cfg is not None else {}


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into base config (override wins on conflict)."""
    result = base.copy()
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


def prepare_ranking_negs(
    num_nodes: int,
    size: int,
    n_neg: int = 49,
    cache_path: Optional[str] = None,
    seed: int = 0,
) -> Tensor:
    """Generate (or load cached) negative candidate lists for MRR / Hits@K eval.

    Negative candidates are sampled once and optionally persisted to disk.
    Loading from cache ensures the same negative set is used across runs,
    making MRR numbers directly comparable across different model configs.

    Args:
        num_nodes: total number of nodes (sampling range [0, num_nodes)).
        size: number of rows — use ``max(dataset.val_size, dataset.test_size)``
            so the same tensor covers both val and test evaluation.
        n_neg: negatives per positive edge. TGB standard is 49.
        cache_path: path to a ``.pt`` file. If it exists, loads from there;
            otherwise generates and saves. ``None`` disables caching.
        seed: random seed used when generating (ignored when loading from cache).

    Returns:
        LongTensor of shape ``(size, n_neg)`` on CPU.

    Example::

        neg = prepare_ranking_negs(
            num_nodes=1900,
            size=max(dataset.val_size, dataset.test_size),
            n_neg=49,
            cache_path=f"neg_cache/{args.dataset}_mrr49.pt",
        )
        engine = Engine(..., eval_protocols={"mrr": MRREval(neg)})
    """
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            return torch.load(cache_file, weights_only=True)

    gen = torch.Generator()
    gen.manual_seed(seed)
    neg = torch.randint(0, num_nodes, (size, n_neg), generator=gen)

    if cache_path is not None:
        os.makedirs(Path(cache_path).parent, exist_ok=True)
        torch.save(neg, cache_path)
        print(f"Saved {size}×{n_neg} neg candidates to {cache_path}")

    return neg
