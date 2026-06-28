from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 200
    lr: float = 1e-4
    # Optional separate learning rate for task heads.
    # When set, encoder (model) uses `lr` and heads use `head_lr`.
    # When None, all parameters share `lr`.
    head_lr: Optional[float] = None
    patience: int = 5
    device: str = "cuda"
    seed: int = 42
    async_pipeline: bool = False
    checkpoint_dir: Optional[str] = None
    use_amp: bool = False
    grad_clip: float = 1.0
    compile_model: bool = False
    warmup_steps: int = 0

    # Distributed training (DDP). When True, Engine wraps the model with
    # DistributedDataParallel and uses a DistributedSampler to shard batches
    # across ranks. The process group must be initialized externally (e.g. via
    # `torchrun`); Engine detects torch.distributed.is_initialized() and adapts.
    distributed: bool = False
    dist_backend: str = "nccl"
    find_unused_parameters: bool = False

    # Eval scheduling
    eval_strategy: Literal["adaptive", "every_n", "all"] = "adaptive"
    eval_every: int = 1
    min_eval_gap: int = 1
    max_eval_gap: int = 10
    loss_threshold: float = 0.02

    # Multi-task early stopping rule:
    #   "primary"     — stop when primary_metric stops improving (default)
    #   "all_improve" — stop only when ALL tracked val metrics stop improving
    stopping_rule: Literal["primary", "all_improve"] = "primary"

    # Logging & tracking
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    result_dir: Optional[str] = None

    def __post_init__(self):
        if self.use_amp and not self.device.startswith("cuda"):
            raise ValueError(
                f"use_amp=True requires a CUDA device, but device='{self.device}'. "
                "Set use_amp=False when running on CPU."
            )
