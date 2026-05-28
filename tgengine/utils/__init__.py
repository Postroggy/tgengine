"""Utility functions: seeding, config loading, logging."""

import re
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


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
