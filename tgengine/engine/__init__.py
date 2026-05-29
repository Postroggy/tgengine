from __future__ import annotations

import json
import math
import os
import time as time_module
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from tgengine.core.batch import RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.base import TemporalModel
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.async_pipeline import AsyncDataPipeline
from tgengine.pipeline.negatives import NegativeStrategy
from tgengine.utils.logging import TrainLogger, validate_config

from .config import TrainConfig
from .eval import APEval, AUCEval, EvalProtocol, HitsEval, MRREval, RankingEval, ThreeWayEval

__all__ = [
    "APEval",
    "AUCEval",
    "Engine",
    "EvalProtocol",
    "HitsEval",
    "MRREval",
    "RankingEval",
    "ThreeWayEval",
    "TrainConfig",
    "run_experiment",
]


class Engine:
    """Main training and evaluation engine.

    Manages the full lifecycle:
    - Builds DataPipeline from model's GatherSpec
    - Runs training loop with async prefetch
    - Handles eval with graph snapshot/restore
    - Manages stateful model lifecycle (evolve/freeze/thaw)
    """

    def __init__(
        self,
        model: TemporalModel,
        graph: TemporalGraph,
        train_batches: list[RawBatch],
        val_batches: list[RawBatch],
        test_batches: list[RawBatch],
        neg_strategy: NegativeStrategy,
        eval_protocol: EvalProtocol,
        config: TrainConfig,
        inductive_edges: Optional[dict] = None,
        eval_neg_strategy: Optional[NegativeStrategy] = None,
    ):
        validate_config(config)

        self.model = model.to(config.device)
        self.graph = graph
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.neg_strategy = neg_strategy
        self.eval_neg_strategy = eval_neg_strategy or neg_strategy
        self.eval_protocol = eval_protocol
        self.config = config
        self.inductive_edges = inductive_edges
        self.logger = TrainLogger(
            model_name=model.__class__.__name__,
            dataset_name="",
        )

        self.pipeline = DataPipeline(model.gather_spec, graph)
        self.async_pipeline = (
            AsyncDataPipeline(model.gather_spec, graph, neg_strategy)
            if config.async_pipeline and config.device.startswith("cuda")
            else None
        )
        # Preload all training edges into graph and freeze CSR
        for rb in train_batches:
            graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
        graph.freeze_csr()

        # Build eval graph once: full CSR with ALL edges (train+val+test+inductive).
        # DyGLib uses full_neighbor_sampler during eval — overflow ring buffer can't
        # hold enough edges for popular nodes, so we freeze everything into CSR.
        self._eval_graph = TemporalGraph(
            graph.num_nodes, edge_feat_dim=graph.edge_feat_dim,
            device=config.device,
            buffer_size=model.gather_spec.neighbors.k if model.gather_spec.neighbors else 32,
        )
        for rb in train_batches + val_batches + test_batches:
            self._eval_graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
        if inductive_edges is not None:
            ie = inductive_edges
            self._eval_graph.advance(
                ie["src"].to(config.device), ie["dst"].to(config.device),
                ie["time"].to(config.device),
                ie["edge_feat"].to(config.device) if ie["edge_feat"] is not None else None,
            )
        self._eval_graph.freeze_csr()
        self._eval_pipeline = DataPipeline(model.gather_spec, self._eval_graph)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
        self.scheduler = self._build_scheduler() if config.warmup_steps > 0 else None
        if config.compile_model:
            self.model = torch.compile(self.model)
        self._current_epoch = 0
        self._best_val = 0.0
        self._last_eval_epoch = 0
        self._loss_at_last_eval = float("inf")

        # W&B integration
        self._wandb = None
        if config.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config={
                        "epochs": config.epochs,
                        "batch_size": config.batch_size,
                        "lr": config.lr,
                        "patience": config.patience,
                        "use_amp": config.use_amp,
                        "grad_clip": config.grad_clip,
                        "warmup_steps": config.warmup_steps,
                        "model_params": sum(p.numel() for p in model.parameters()),
                    },
                )
                self._wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, skipping logging")

    def _build_scheduler(self):
        total_steps = self.config.epochs * len(self.train_batches)
        warmup = self.config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": self._current_epoch,
            "best_val": self._best_val,
            "config": self.config,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._current_epoch = checkpoint["epoch"]
        self._best_val = checkpoint["best_val"]
        return self._current_epoch

    def _should_eval(self, epoch: int, train_loss: float) -> bool:
        """Decide whether to run validation this epoch."""
        strategy = self.config.eval_strategy

        if strategy == "all":
            return True
        if strategy == "every_n":
            return epoch % self.config.eval_every == 0 or epoch == self.config.epochs

        # adaptive strategy
        if epoch == 1 or epoch == self.config.epochs:
            return True
        gap = epoch - self._last_eval_epoch
        if gap < self.config.min_eval_gap:
            return False
        if gap >= self.config.max_eval_gap:
            return True
        relative_change = abs(train_loss - self._loss_at_last_eval) / (abs(self._loss_at_last_eval) + 1e-8)
        return relative_change > self.config.loss_threshold

    def train(self, resume: bool = False) -> dict[str, float]:
        self.logger.log_start(self.config)
        best_val = self._best_val if resume else 0.0
        best_test: dict[str, float] = {}
        best_epoch = 0
        patience_counter = 0
        eval_count = 0
        t_start = time_module.time()

        for epoch in range(self._current_epoch + 1, self.config.epochs + 1):
            self._current_epoch = epoch
            self.logger.epoch_start()

            try:
                train_loss = self._train_epoch()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM at epoch {epoch}. Try reducing batch_size "
                    f"(current: {self.config.batch_size}) or neighbor K."
                )

            # Evaluate
            if self._should_eval(epoch, train_loss):
                self._last_eval_epoch = epoch
                self._loss_at_last_eval = train_loss
                eval_count += 1

                val_metrics = self._evaluate(self.val_batches)
                val_score = val_metrics.get("ap", val_metrics.get("mrr", 0.0))

                is_best = val_score > best_val
                if is_best:
                    best_val = val_score
                    self._best_val = best_val
                    best_test = self._evaluate(self.test_batches)
                    best_epoch = epoch
                    patience_counter = 0

                    if self.config.checkpoint_dir is not None:
                        save_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                        self.save_checkpoint(save_path)
                else:
                    patience_counter += 1

                self.logger.epoch_end(
                    epoch, train_loss, val_score, is_best,
                    best_test=best_test, patience_counter=patience_counter,
                )
            else:
                self.logger.epoch_end(epoch, train_loss, None, False)

            if self._wandb:
                log = {"epoch": epoch, "train_loss": train_loss}
                if epoch == self._last_eval_epoch:
                    log["val_score"] = val_score
                    if best_test:
                        log.update({f"best_test_{k}": v for k, v in best_test.items()})
                if self.scheduler:
                    log["lr"] = self.scheduler.get_last_lr()[0]
                self._wandb.log(log)

            if self.config.patience > 0 and patience_counter >= self.config.patience:
                break

        elapsed = time_module.time() - t_start
        self.logger.log_finish(best_test, self._current_epoch)

        # Structured result output
        result = {
            "model": self.model.__class__.__name__,
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "grad_clip": self.config.grad_clip,
                "use_amp": self.config.use_amp,
                "eval_strategy": self.config.eval_strategy,
                "seed": self.config.seed,
            },
            "result": {
                "best_val": best_val,
                "best_epoch": best_epoch,
                "test_metrics": best_test,
            },
            "stats": {
                "total_epochs": self._current_epoch,
                "eval_count": eval_count,
                "elapsed_seconds": round(elapsed, 1),
                "stopped_by": "patience" if (self.config.patience > 0 and patience_counter >= self.config.patience) else "completed",
            },
        }

        if self.config.result_dir:
            os.makedirs(self.config.result_dir, exist_ok=True)
            path = os.path.join(self.config.result_dir, "result.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)

        if self._wandb:
            self._wandb.finish()
        return best_test

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        if self.async_pipeline is not None:
            total_loss = self._train_epoch_async()
        else:
            total_loss = self._train_epoch_sync()

        return total_loss / len(self.train_batches)

    def _train_epoch_sync(self) -> float:
        total_loss = 0.0
        amp_enabled = self.config.use_amp
        for raw_batch in tqdm(self.train_batches, desc="Training"):
            neg = self.neg_strategy.sample(
                raw_batch.src, raw_batch.dst, raw_batch.time, self.graph,
                raw_batch.edge_indices,
            )
            raw_batch.neg = neg
            prepared = self.pipeline.prepare(raw_batch)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = self.model(prepared)
            self.scaler.scale(output.loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += output.loss.item()
            self.model.evolve(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)
        return total_loss

    def _train_epoch_async(self) -> float:
        pipe = self.async_pipeline
        batches = self.train_batches
        total_loss = 0.0
        amp_enabled = self.config.use_amp

        if not batches:
            return 0.0

        pipe.start_prefetch(batches[0])

        for i in tqdm(range(len(batches)), desc="Training (async)"):
            rb, prepared = pipe.get()

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = self.model(prepared)
            self.scaler.scale(output.loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += output.loss.item()

            self.model.evolve(rb.src, rb.dst, rb.time, rb.edge_feat)

            if i + 1 < len(batches):
                pipe.start_prefetch(batches[i + 1])

        return total_loss

    def _evaluate(self, eval_batches: list[RawBatch]) -> dict[str, float]:
        """Run evaluation using pre-built full-data eval graph."""
        model_state = self.model.freeze()

        prepped = []
        for rb in eval_batches:
            if rb.neg is None:
                neg = self.eval_neg_strategy.sample(rb.src, rb.dst, rb.time,
                                                    self._eval_graph, rb.edge_indices)
                rb = RawBatch(src=rb.src, dst=rb.dst, time=rb.time,
                              edge_feat=rb.edge_feat, neg=neg,
                              edge_indices=rb.edge_indices)
            prepped.append(rb)

        with torch.amp.autocast("cuda", enabled=self.config.use_amp):
            metrics = self.eval_protocol.evaluate(
                self.model, self._eval_pipeline, prepped, self._eval_graph
            )

        self.model.thaw(model_state)
        return metrics


def run_experiment(
    build_fn,
    seeds: list[int] | None = None,
    n_runs: int = 5,
    result_dir: str | None = None,
) -> dict:
    """Run multiple training runs with different seeds and aggregate results.

    Args:
        build_fn: callable(seed: int) -> Engine. Must construct a fresh model,
            graph, dataset, and Engine each call.
        seeds: explicit seed list. If None, uses [0, 1, ..., n_runs-1].
        n_runs: number of runs (ignored if seeds is provided).
        result_dir: directory to write per-run and aggregated results.

    Returns:
        Dict with per-run results and aggregated mean/std.
    """
    if seeds is None:
        seeds = list(range(n_runs))

    all_results: list[dict[str, float]] = []
    per_run: list[dict] = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")

        engine = build_fn(seed)
        test_metrics = engine.train()
        all_results.append(test_metrics)
        per_run.append({"seed": seed, "test_metrics": test_metrics})

        # Clean up GPU memory between runs
        del engine
        torch.cuda.empty_cache()

    # Aggregate
    metric_keys = list(all_results[0].keys()) if all_results else []
    aggregated = {}
    for k in metric_keys:
        vals = [r[k] for r in all_results if k in r]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        aggregated[k] = {"mean": round(mean, 4), "std": round(std, 4)}
        print(f"{k}: {mean:.4f} ± {std:.4f}")

    output = {
        "n_runs": len(seeds),
        "seeds": seeds,
        "per_run": per_run,
        "aggregated": aggregated,
    }

    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        path = os.path.join(result_dir, "experiment.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {path}")

    return output
