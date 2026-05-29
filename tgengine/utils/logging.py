"""Structured logging for TGEngine training runs."""

from __future__ import annotations

import time
from typing import Optional

import torch


class TrainLogger:
    """Lightweight logger that prints structured training progress.

    Tracks per-epoch timing, GPU memory, and metrics.
    """

    def __init__(self, model_name: str = "", dataset_name: str = ""):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._epoch_start: float = 0.0

    def log_start(self, config) -> None:
        """Print a startup banner with config summary."""
        device = config.device
        lines = [
            "=" * 60,
            f"  TGEngine — {self.model_name} on {self.dataset_name}",
            "=" * 60,
            f"  device: {device}",
        ]
        if device.startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append(f"  GPU: {gpu_name} ({mem_total:.1f} GB)")
        lines += [
            f"  epochs: {config.epochs}, patience: {config.patience}",
            f"  lr: {config.lr}, batch_size: {config.batch_size}",
            f"  amp: {config.use_amp}, grad_clip: {config.grad_clip}",
            "=" * 60,
        ]
        print("\n".join(lines))

    def epoch_start(self) -> None:
        self._epoch_start = time.perf_counter()

    def epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_score: float,
        is_best: bool,
        best_test: Optional[dict] = None,
        patience_counter: int = 0,
    ) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        mem_str = ""
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            mem_str = f" mem={mem_mb:.0f}MB"

        if is_best:
            test_str = " ".join(f"{k}={v:.4f}" for k, v in (best_test or {}).items())
            print(f"  Epoch {epoch}: loss={train_loss:.4f} val={val_score:.4f} "
                  f"test=[{test_str}] {elapsed:.1f}s{mem_str} *")
        else:
            print(f"  Epoch {epoch}: loss={train_loss:.4f} val={val_score:.4f} "
                  f"patience={patience_counter} {elapsed:.1f}s{mem_str}")

    def log_finish(self, best_test: dict, total_epochs: int) -> None:
        print(f"\nTraining complete after {total_epochs} epochs.")
        print(f"Best test: {best_test}")


def validate_config(config) -> None:
    """Validate TrainConfig fields. Raises ValueError with actionable messages."""
    if config.lr <= 0:
        raise ValueError(f"lr must be positive, got {config.lr}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}")
    if config.patience <= 0:
        raise ValueError(f"patience must be positive, got {config.patience}")
    if config.grad_clip < 0:
        raise ValueError(f"grad_clip must be non-negative, got {config.grad_clip}")
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {config.warmup_steps}")

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device='{config.device}' but CUDA is not available. "
            "Use device='cpu' or check your CUDA installation."
        )


def check_gpu_memory(num_nodes: int, k: int, d_edge: int, device: str) -> Optional[str]:
    """Estimate graph memory usage and warn if it might OOM.

    Returns a warning string if memory looks tight, None otherwise.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None

    est_gb = num_nodes * k * max(d_edge, 1) * 4 / 1e9
    free_mem = torch.cuda.mem_get_info()[0] / 1e9

    if est_gb > free_mem * 0.8:
        return (
            f"Estimated graph memory ({est_gb:.1f} GB) exceeds 80% of free GPU memory "
            f"({free_mem:.1f} GB). Consider reducing K or using a smaller dataset."
        )
    return None
