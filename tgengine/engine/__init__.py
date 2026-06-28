from __future__ import annotations

import copy
import json
import math
import os
import time as time_module
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from tgengine.core.batch import RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.base import EmbeddingBundle, ModelOutput, TemporalModel
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.async_pipeline import AsyncDataPipeline
from tgengine.pipeline.negatives import NegativeStrategy
from tgengine.tasks.base import TaskHead
from tgengine.utils.logging import TrainLogger, validate_config

from .config import TrainConfig
from .eval import (
    APEval,
    AUCEval,
    AnomalyEval,
    EdgeClsEval,
    EdgeRegEval,
    EvalProtocol,
    HitsEval,
    MRREval,
    NodeClsEval,
    NodeRegEval,
    RankingEval,
    ThreeWayEval,
)

__all__ = [
    "APEval",
    "AUCEval",
    "AnomalyEval",
    "EdgeClsEval",
    "EdgeRegEval",
    "Engine",
    "EvalProtocol",
    "HitsEval",
    "MRREval",
    "NodeClsEval",
    "NodeRegEval",
    "RankingEval",
    "ThreeWayEval",
    "TrainConfig",
    "run_experiment",
]


class Engine:
    """Training and evaluation engine.

    ## Simple usage (link prediction, backward-compatible):

        engine = Engine(
            model=GraphMixer(d_model=172, ...),
            graph=graph,
            train_batches=train, val_batches=val, test_batches=test,
            neg_strategy=RandomNegative(num_nodes),
            eval_protocol=APEval(),
            config=TrainConfig(),
        )
        engine.train()

    ## Multi-task usage:

        engine = Engine(
            model=GraphMixer(d_model=172, ...),
            ...
            tasks={
                "link_pred": LinkPredHead(),
                "node_cls": NodeClassificationHead(172, num_classes=7),
            },
            task_weights={"link_pred": 1.0, "node_cls": 0.5},
            eval_protocols={
                "link_pred": APEval(),
                "node_cls": NodeClsEval(num_classes=7),
            },
            primary_metric="ap",        # which val metric drives early stopping
        )

    When `tasks` is provided the engine calls `model.encode()` to get
    `EmbeddingBundle`, routes src/dst embeddings into each head, and sums the
    weighted losses. `eval_protocol` (single) is still supported for backward
    compatibility; `eval_protocols` (dict) is the new multi-task form.

    Either `eval_protocol` or `eval_protocols` must be provided.
    """

    def __init__(
        self,
        model: TemporalModel,
        graph: TemporalGraph,
        train_batches: list[RawBatch],
        val_batches: list[RawBatch],
        test_batches: list[RawBatch],
        neg_strategy: NegativeStrategy,
        # eval_protocol before config to preserve positional backward-compat
        eval_protocol: Optional[EvalProtocol] = None,
        config: Optional[TrainConfig] = None,
        # --- multi-task ---
        tasks: Optional[Dict[str, Union[TaskHead, Tuple[TaskHead, List[RawBatch]]]]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        eval_protocols: Optional[Dict[str, EvalProtocol]] = None,
        primary_metric: Optional[str] = None,
        # --- misc ---
        inductive_edges: Optional[dict] = None,
        eval_neg_strategy: Optional[NegativeStrategy] = None,
    ):
        if eval_protocol is None and eval_protocols is None:
            raise ValueError("Provide either eval_protocol (single) or eval_protocols (dict).")

        if config is None:
            config = TrainConfig()
        validate_config(config)

        # Normalize to dict form internally
        if eval_protocols is not None:
            self._eval_protocols: Dict[str, EvalProtocol] = eval_protocols
        else:
            self._eval_protocols = {"default": eval_protocol}  # type: ignore[dict-item]

        # Keep single-protocol reference for backward compat
        self.eval_protocol = eval_protocol or next(iter(self._eval_protocols.values()))

        self.model = model.to(config.device)
        self.graph = graph
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.neg_strategy = neg_strategy
        self.eval_neg_strategy = eval_neg_strategy or neg_strategy
        self.config = config
        self.inductive_edges = inductive_edges

        # Task heads (optional multi-task).
        # Each entry can be:
        #   TaskHead                        — shares train_batches
        #   (TaskHead, List[RawBatch])      — uses its own batch source
        self._tasks: Dict[str, TaskHead] = {}
        self._task_batches: Dict[str, Optional[List[RawBatch]]] = {}
        if tasks:
            for k, v in tasks.items():
                if isinstance(v, tuple):
                    head, extra_batches = v
                    self._tasks[k] = head.to(config.device)
                    self._task_batches[k] = extra_batches
                else:
                    self._tasks[k] = v.to(config.device)
                    self._task_batches[k] = None
        self._task_weights: Dict[str, float] = task_weights or {}
        self._primary_metric = primary_metric or self._infer_primary_metric()
        self._use_tasks = bool(self._tasks)

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

        # Build full eval graph (all splits) to match DyGLib eval protocol
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

        # Optimizer: encoder and heads can use different LRs
        if config.head_lr is not None and self._tasks:
            param_groups = [
                {"params": list(model.parameters()), "lr": config.lr},
                {"params": [p for h in self._tasks.values() for p in h.parameters()],
                 "lr": config.head_lr},
            ]
            self.optimizer = torch.optim.Adam(param_groups)
        else:
            all_params = list(model.parameters())
            for head in self._tasks.values():
                all_params.extend(head.parameters())
            self.optimizer = torch.optim.Adam(all_params, lr=config.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
        self.scheduler = self._build_scheduler() if config.warmup_steps > 0 else None
        if config.compile_model:
            self.model = torch.compile(self.model)

        # Distributed (DDP) setup. Opt-in via config.distributed; requires an
        # externally initialized process group (e.g. via torchrun).
        self._distributed = config.distributed and torch.distributed.is_available() \
            and torch.distributed.is_initialized()
        self._rank = torch.distributed.get_rank() if self._distributed else 0
        self._world_size = torch.distributed.get_world_size() if self._distributed else 1
        if self._distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[config.device] if config.device.startswith("cuda") else None,
                find_unused_parameters=config.find_unused_parameters,
            )
        # Expose model for evolve/encode calls (unwrap DDP when needed)
        self._raw_model = self.model.module if self._distributed else self.model

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
                        "tasks": list(self._tasks.keys()),
                    },
                )
                self._wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, skipping logging")

    def _infer_primary_metric(self) -> str:
        """Pick a sensible default primary metric from the eval protocols."""
        for preferred in ("ap", "mrr", "auc", "acc", "f1_macro", "auroc"):
            for proto in self._eval_protocols.values():
                if hasattr(proto, "_primary_hint") and proto._primary_hint == preferred:
                    return preferred
        # Fall back to first metric key of first protocol (resolved at eval time)
        return "ap"

    def _build_scheduler(self):
        total_steps = self.config.epochs * len(self.train_batches)
        warmup = self.config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Core: step a single prepared batch through model + task heads
    # ------------------------------------------------------------------

    def _step(self, prepared) -> ModelOutput:
        """Run one forward pass: either multi-task (encode→heads) or legacy forward()."""
        if self._use_tasks:
            return self._step_multitask(prepared)
        return self.model(prepared)

    def _step_multitask(self, prepared) -> ModelOutput:
        """Multi-task step: encode() → route shared heads → sum losses.

        Heads with independent batch sources are NOT run here; they are stepped
        separately in _train_epoch_sync via _step_independent_heads().
        """
        bundle: EmbeddingBundle = self._raw_model.encode(prepared)

        total_loss = torch.tensor(0.0, device=bundle.src.device)
        merged = ModelOutput(
            loss=total_loss,
            pos_score=torch.zeros(bundle.src.shape[0], device=bundle.src.device),
            neg_score=torch.zeros(bundle.src.shape[0], device=bundle.src.device),
        )

        for name, head in self._tasks.items():
            if self._task_batches.get(name) is not None:
                continue  # independent-batch heads handled separately
            w = self._task_weights.get(name, 1.0)
            out = self._route_to_head(head, name, bundle, prepared)
            total_loss = total_loss + w * out.loss

            # Merge non-None fields (last write wins)
            if out.pos_score is not None and out.pos_score.any():
                merged.pos_score = out.pos_score
                merged.neg_score = out.neg_score
            if out.node_pred is not None:
                merged.node_pred = out.node_pred
                merged.node_labels = out.node_labels
            if out.edge_pred is not None:
                merged.edge_pred = out.edge_pred
                merged.edge_labels = out.edge_labels
            if out.anomaly_score is not None:
                merged.anomaly_score = out.anomaly_score

        merged.loss = total_loss
        return merged

    def _step_independent_heads(self, step_idx: int) -> torch.Tensor:
        """Run heads that have their own batch sources.

        Cycles through the head's batch list using step_idx % len(batches).
        Returns the weighted sum of their losses (scalar tensor).
        """
        total = torch.tensor(0.0, device=self.config.device)
        for name, head in self._tasks.items():
            extra = self._task_batches.get(name)
            if extra is None or len(extra) == 0:
                continue
            rb = extra[step_idx % len(extra)]
            # Encode src nodes from the independent batch
            neg = self.neg_strategy.sample(rb.src, rb.dst, rb.time, self.graph,
                                            rb.edge_indices)
            rb_with_neg = RawBatch(src=rb.src, dst=rb.dst, time=rb.time,
                                   edge_feat=rb.edge_feat, neg=neg,
                                   edge_indices=rb.edge_indices)
            prep = self.pipeline.prepare(rb_with_neg)
            bundle = self._raw_model.encode(prep)
            w = self._task_weights.get(name, 1.0)
            out = self._route_to_head(head, name, bundle, prep)
            total = total + w * out.loss
        return total

    def _route_to_head(self, head: TaskHead, name: str, bundle: EmbeddingBundle, prepared) -> ModelOutput:
        """Route embeddings to a task head, passing the right kwargs per head type."""
        from tgengine.tasks.link_pred import LinkPredHead
        from tgengine.tasks.node_cls import NodeClassificationHead, NodeBinaryClassificationHead
        from tgengine.tasks.node_reg import NodeRegressionHead
        from tgengine.tasks.edge_cls import EdgeClassificationHead, EdgeBinaryClassificationHead
        from tgengine.tasks.edge_reg import EdgeRegressionHead
        from tgengine.tasks.anomaly import AnomalyDetectionHead

        src = bundle.src
        src_for_neg = bundle.src_for_neg if bundle.src_for_neg is not None else bundle.src

        if isinstance(head, LinkPredHead):
            return head(src, src_emb=src, dst_emb=bundle.dst, neg_emb=bundle.neg)

        if isinstance(head, (EdgeClassificationHead, EdgeBinaryClassificationHead, EdgeRegressionHead)):
            return head(src, labels=prepared.edge_labels, dst_emb=bundle.dst)

        if isinstance(head, (NodeClassificationHead, NodeBinaryClassificationHead, NodeRegressionHead)):
            return head(src, labels=prepared.node_labels)

        if isinstance(head, AnomalyDetectionHead):
            labels = prepared.node_labels if prepared.node_labels is not None else prepared.edge_labels
            return head(src, labels=labels)

        # Generic fallback: pass src emb + node_labels
        return head(src, labels=prepared.node_labels)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": self._current_epoch,
            "best_val": self._best_val,
            "config": self.config,
        }
        if self._tasks:
            checkpoint["task_heads"] = {k: v.state_dict() for k, v in self._tasks.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self._raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "task_heads" in checkpoint:
            for k, sd in checkpoint["task_heads"].items():
                if k in self._tasks:
                    self._tasks[k].load_state_dict(sd)
        self._current_epoch = checkpoint["epoch"]
        self._best_val = checkpoint["best_val"]
        return self._current_epoch

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _should_eval(self, epoch: int, train_loss: float) -> bool:
        strategy = self.config.eval_strategy
        if strategy == "all":
            return True
        if strategy == "every_n":
            return epoch % self.config.eval_every == 0 or epoch == self.config.epochs
        # adaptive
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
        best_model_state = None
        patience_counter = 0
        eval_count = 0
        # For "all_improve" rule: track per-metric best values
        _all_improve_bests: dict[str, float] = {}
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

            if self._should_eval(epoch, train_loss):
                self._last_eval_epoch = epoch
                self._loss_at_last_eval = train_loss
                eval_count += 1

                val_metrics = self._evaluate(self.val_batches)
                val_score = val_metrics.get(self._primary_metric, 0.0)
                # Fall back to first available metric
                if val_score == 0.0 and val_metrics:
                    val_score = next(iter(val_metrics.values()))

                # DDP: average val_score across ranks so all ranks agree on is_best
                if self._distributed:
                    val_score_t = torch.tensor(val_score, device=self.config.device)
                    torch.distributed.all_reduce(val_score_t, op=torch.distributed.ReduceOp.AVG)
                    val_score = val_score_t.item()

                # Determine is_best based on stopping_rule
                if self.config.stopping_rule == "all_improve":
                    if not _all_improve_bests:
                        # First eval — initialize and treat as best
                        _all_improve_bests = dict(val_metrics)
                        is_best = True
                    else:
                        # Best only if EVERY tracked metric improved or held
                        is_best = all(
                            val_metrics.get(k, 0.0) >= _all_improve_bests.get(k, 0.0)
                            for k in _all_improve_bests
                        )
                        if is_best:
                            _all_improve_bests.update(val_metrics)
                else:
                    is_best = val_score > best_val

                if is_best:
                    best_val = val_score
                    self._best_val = best_val
                    best_test = self._evaluate(self.test_batches)
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self._raw_model.state_dict())
                    patience_counter = 0

                    if self._rank == 0 and self.config.checkpoint_dir is not None:
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

        if best_model_state is not None:
            self._raw_model.load_state_dict(best_model_state)

        self.logger.log_finish(best_test, self._current_epoch)

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

        if self._rank == 0 and self.config.result_dir:
            os.makedirs(self.config.result_dir, exist_ok=True)
            path = os.path.join(self.config.result_dir, "result.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)

        if self._wandb:
            self._wandb.finish()
        # DDP: ensure all ranks finish before returning
        if self._distributed:
            torch.distributed.barrier()
        return best_test

    def _train_epoch(self) -> float:
        self.model.train()
        for head in self._tasks.values():
            head.train()
        total_loss = 0.0

        if self.async_pipeline is not None:
            total_loss = self._train_epoch_async()
        else:
            total_loss = self._train_epoch_sync()

        return total_loss / len(self._sharded_batches())

    def _sharded_batches(self) -> list[RawBatch]:
        """Batches for this rank (DDP shard). Single-GPU returns all batches."""
        if not self._distributed:
            return self.train_batches
        return self.train_batches[self._rank::self._world_size]

    def _train_epoch_sync(self) -> float:
        total_loss = 0.0
        amp_enabled = self.config.use_amp
        batches = self._sharded_batches()
        iterator = tqdm(batches, desc="Training") if self._rank == 0 else batches
        for step_idx, raw_batch in enumerate(iterator):
            neg = self.neg_strategy.sample(
                raw_batch.src, raw_batch.dst, raw_batch.time, self.graph,
                raw_batch.edge_indices,
            )
            raw_batch.neg = neg
            prepared = self.pipeline.prepare(raw_batch)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = self._step(prepared)
                # Add loss from independent-batch heads (if any)
                if self._use_tasks:
                    indep_loss = self._step_independent_heads(step_idx)
                    total_step_loss = output.loss + indep_loss
                else:
                    total_step_loss = output.loss
            self.scaler.scale(total_step_loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + [p for h in self._tasks.values() for p in h.parameters()],
                    self.config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += total_step_loss.item()
            self._raw_model.evolve(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)
        return total_loss

    def _train_epoch_async(self) -> float:
        pipe = self.async_pipeline
        batches = self._sharded_batches()
        total_loss = 0.0
        amp_enabled = self.config.use_amp

        if not batches:
            return 0.0

        pipe.start_prefetch(batches[0])

        iterator = range(len(batches))
        if self._rank == 0:
            iterator = tqdm(iterator, desc="Training (async)")
        for i in iterator:
            rb, prepared = pipe.get()

            # Issue prefetch for next batch BEFORE compute, so it overlaps with
            # forward/backward of the current batch. Safe because the training graph
            # is static (frozen CSR; train loop calls model.evolve, not graph.advance),
            # so prefetch(i+1) has no dependency on compute(i).
            if i + 1 < len(batches):
                pipe.start_prefetch(batches[i + 1])

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = self._step(prepared)
                if self._use_tasks:
                    indep_loss = self._step_independent_heads(i)
                    total_step_loss = output.loss + indep_loss
                else:
                    total_step_loss = output.loss
            self.scaler.scale(total_step_loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + [p for h in self._tasks.values() for p in h.parameters()],
                    self.config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += total_step_loss.item()
            self._raw_model.evolve(rb.src, rb.dst, rb.time, rb.edge_feat)

        return total_loss

    def _evaluate(self, eval_batches: list[RawBatch]) -> dict[str, float]:
        """Run all registered eval protocols; return merged metrics dict."""
        model_state = self._raw_model.freeze()
        for head in self._tasks.values():
            head.eval()

        prepped = []
        for rb in eval_batches:
            if rb.neg is None:
                neg = self.eval_neg_strategy.sample(rb.src, rb.dst, rb.time,
                                                    self._eval_graph, rb.edge_indices)
                rb = RawBatch(src=rb.src, dst=rb.dst, time=rb.time,
                              edge_feat=rb.edge_feat, neg=neg,
                              edge_indices=rb.edge_indices,
                              node_labels=rb.node_labels,
                              edge_labels=rb.edge_labels)
            prepped.append(rb)

        all_metrics: dict[str, float] = {}
        with torch.amp.autocast("cuda", enabled=self.config.use_amp):
            for proto in self._eval_protocols.values():
                metrics = proto.evaluate(
                    self.model, self._eval_pipeline, prepped, self._eval_graph
                )
                all_metrics.update(metrics)

        self._raw_model.thaw(model_state)
        for head in self._tasks.values():
            head.train()
        return all_metrics

    # ------------------------------------------------------------------
    # Probe mode: freeze encoder, train only task heads
    # ------------------------------------------------------------------

    def probe(
        self,
        task_name: str,
        epochs: int = 20,
        lr: Optional[float] = None,
    ) -> dict[str, float]:
        """Freeze the encoder and train only the named task head.

        This is the standard "linear probing" evaluation: representation
        quality is measured by how well a freshly-trained head performs
        on top of a frozen encoder.

        Args:
            task_name: Key in `self._tasks` to probe.
            epochs: Number of epochs to train the head.
            lr: Learning rate for the head (defaults to config.head_lr or config.lr).

        Returns:
            Val metrics dict after probe training.
        """
        if task_name not in self._tasks:
            raise ValueError(f"Task '{task_name}' not found. Available: {list(self._tasks)}")

        head = self._tasks[task_name]
        head_lr = lr or self.config.head_lr or self.config.lr

        # Freeze encoder
        for p in self.model.parameters():
            p.requires_grad_(False)

        head_params = list(head.parameters())
        has_params = len(head_params) > 0
        if has_params:
            probe_opt = torch.optim.Adam(head_params, lr=head_lr)
        amp_enabled = self.config.use_amp

        head.train()
        for _ in range(epochs):
            for step_idx, raw_batch in enumerate(self.train_batches):
                neg = self.neg_strategy.sample(
                    raw_batch.src, raw_batch.dst, raw_batch.time, self.graph,
                    raw_batch.edge_indices,
                )
                raw_batch.neg = neg
                prepared = self.pipeline.prepare(raw_batch)

                if has_params:
                    probe_opt.zero_grad()
                # No torch.no_grad() here: encoder params are frozen (requires_grad=False)
                # but the computation graph still needs to flow through head params.
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    bundle = self._raw_model.encode(prepared)
                    out = self._route_to_head(head, task_name, bundle, prepared)
                if has_params and out.loss.requires_grad:
                    out.loss.backward()
                    probe_opt.step()

        # Unfreeze encoder
        for p in self.model.parameters():
            p.requires_grad_(True)

        return self._evaluate(self.val_batches)


def run_experiment(
    build_fn,
    seeds: list[int] | None = None,
    n_runs: int = 5,
    result_dir: str | None = None,
) -> dict:
    """Run multiple training runs with different seeds and aggregate results.

    Args:
        build_fn: callable(seed: int) -> Engine.
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

        del engine
        torch.cuda.empty_cache()

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

