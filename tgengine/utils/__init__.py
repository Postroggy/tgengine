"""Utility functions: seeding, config loading, logging."""

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


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
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
