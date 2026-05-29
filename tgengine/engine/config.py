from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 200
    lr: float = 1e-4
    patience: int = 0
    device: str = "cuda"
    seed: int = 42
    async_pipeline: bool = False
    checkpoint_dir: Optional[str] = None
    use_amp: bool = False
    grad_clip: float = 1.0
    compile_model: bool = False
    warmup_steps: int = 0

    # Eval scheduling
    eval_strategy: Literal["adaptive", "every_n", "all"] = "adaptive"
    eval_every: int = 1
    min_eval_gap: int = 1
    max_eval_gap: int = 10
    loss_threshold: float = 0.02

    # Logging & tracking
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    result_dir: Optional[str] = None
