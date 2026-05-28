from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch import Tensor
from tqdm import tqdm

from tgengine.core.batch import NeighborData, PreparedBatch, RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.async_pipeline import AsyncDataPipeline
from tgengine.pipeline.negatives import (
    HistoricalNegative,
    InductiveNegative,
    NegativeStrategy,
    RandomNegative,
)


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 200
    lr: float = 1e-4
    patience: int = 10
    device: str = "cuda"
    seed: int = 42
    async_pipeline: bool = False   # prefetch batch i+1 while computing batch i
    checkpoint_dir: Optional[str] = None  # if set, auto-save best checkpoint here
    use_amp: bool = False  # enable mixed precision (fp16) training
    grad_clip: float = 1.0  # max gradient norm (0 to disable)
    compile_model: bool = False  # torch.compile() the model forward pass
    warmup_steps: int = 0  # linear warmup steps (0 to disable scheduler)
    wandb_project: Optional[str] = None  # if set, log to W&B with this project name
    wandb_run_name: Optional[str] = None


class EvalProtocol(ABC):
    """Base class for evaluation protocols."""

    @abstractmethod
    def evaluate(
        self,
        model: TemporalModel,
        pipeline: DataPipeline,
        eval_batches: list[RawBatch],
        graph: TemporalGraph,
    ) -> dict[str, float]:
        ...


class APEval(EvalProtocol):
    """Standard Average Precision evaluation (1 pos + 1 neg per edge).

    When neg_strategy and graph are provided, negatives are sampled automatically.
    Otherwise, the caller must set raw_batch.neg before passing batches in.
    """

    def evaluate(self, model, pipeline, eval_batches, graph, neg_strategy=None):
        model.eval()
        all_pos_scores = []
        all_neg_scores = []

        with torch.no_grad():
            for raw_batch in eval_batches:
                if neg_strategy is not None and raw_batch.neg is None:
                    neg = neg_strategy.sample(raw_batch.src, raw_batch.dst,
                                              raw_batch.time, graph,
                                              raw_batch.edge_indices)
                    raw_batch = RawBatch(
                        src=raw_batch.src, dst=raw_batch.dst,
                        time=raw_batch.time, edge_feat=raw_batch.edge_feat,
                        neg=neg, edge_indices=raw_batch.edge_indices,
                    )
                prepared = pipeline.prepare(raw_batch)
                output = model(prepared)
                all_pos_scores.append(output.pos_score)
                all_neg_scores.append(output.neg_score)
                graph.advance(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

        pos = torch.cat(all_pos_scores).sigmoid()
        neg = torch.cat(all_neg_scores).sigmoid()
        # sklearn average_precision_score, matching DyGLib evaluation
        predicts = torch.cat([pos, neg]).cpu().numpy()
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        ap = float(average_precision_score(y_true=labels, y_score=predicts))
        return {"ap": ap}


class ThreeWayEval(EvalProtocol):
    """Evaluation with three negative types: random, historical, inductive.

    Each type is evaluated independently with its own graph pass.
    Reports ap_random, ap_historical, ap_inductive.

    Args:
        num_nodes: total node count.
        inductive_nodes: 1-D tensor of node IDs unseen during training.
    """

    def __init__(self, num_nodes: int, inductive_nodes: Tensor, device: str = "cuda"):
        self._strategies = {
            "random": RandomNegative(num_nodes),
            "historical": HistoricalNegative(num_nodes, device=device),
            "inductive": InductiveNegative(inductive_nodes),
        }

    def evaluate(self, model, pipeline, eval_batches, graph):
        results = {}
        base_snap = graph.snapshot()

        for name, strategy in self._strategies.items():
            graph.restore(base_snap)
            model.eval()
            all_pos_scores: list[Tensor] = []
            all_neg_scores: list[Tensor] = []

            with torch.no_grad():
                for raw_batch in eval_batches:
                    neg = strategy.sample(raw_batch.src, raw_batch.dst, raw_batch.time, graph,
                                         raw_batch.edge_indices)
                    batch = RawBatch(
                        src=raw_batch.src,
                        dst=raw_batch.dst,
                        time=raw_batch.time,
                        edge_feat=raw_batch.edge_feat,
                        neg=neg,
                    )
                    prepared = pipeline.prepare(batch)
                    output = model(prepared)
                    all_pos_scores.append(output.pos_score)
                    all_neg_scores.append(output.neg_score)
                    graph.advance(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

            pos = torch.cat(all_pos_scores).sigmoid()
            neg_scores = torch.cat(all_neg_scores).sigmoid()
            predicts = torch.cat([pos, neg_scores]).cpu().numpy()
            labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_scores))])
            results[f"ap_{name}"] = float(average_precision_score(y_true=labels, y_score=predicts))

        return results


class MRREval(EvalProtocol):
    """Mean Reciprocal Rank evaluation with fixed negative candidate lists (TGB-style).

    Supports two compute paths:
    - Fast path (supports_independent_encode=True): encode all unique nodes once,
      score all pairs in one shot.
    - Slow path: expand batch to cover all (src, candidate) pairs.

    Args:
        neg_lists: (N_eval_edges, N_neg) pre-loaded negative node IDs aligned to eval_batches.
    """

    def __init__(self, neg_lists: Tensor):
        self.neg_lists = neg_lists

    def evaluate(self, model, pipeline, eval_batches, graph):
        model.eval()
        all_rr: list[Tensor] = []
        edge_offset = 0

        with torch.no_grad():
            for raw_batch in eval_batches:
                B = raw_batch.batch_size
                neg = self.neg_lists[edge_offset : edge_offset + B].to(raw_batch.device)  # (B, N_neg)
                edge_offset += B

                if model.supports_independent_encode:
                    rr = self._mrr_encode(model, pipeline, raw_batch, neg)
                else:
                    rr = self._mrr_pairwise(model, pipeline, raw_batch, neg)

                all_rr.append(rr)
                graph.advance(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

        return {"mrr": torch.cat(all_rr).mean().item()}

    def _mrr_encode(
        self, model: TemporalModel, pipeline: DataPipeline, raw_batch: RawBatch, neg: Tensor
    ) -> Tensor:
        """Fast path: one fused neighbor query for all candidate nodes."""
        B, N_neg = neg.shape
        k = pipeline.spec.neighbors.k

        # Concatenate pos dst and neg candidates: (B, 1+N_neg)
        all_cands = torch.cat([raw_batch.dst.unsqueeze(1), neg], dim=1)
        cands_flat = all_cands.reshape(-1)  # (B*(1+N_neg),)
        times_cand = raw_batch.time.unsqueeze(1).expand(-1, 1 + N_neg).reshape(-1)

        # Fused neighbor query
        all_nodes = torch.cat([raw_batch.src, cands_flat])
        all_times = torch.cat([raw_batch.time, times_cand])
        all_nbrs = pipeline.graph.recent(all_nodes, all_times, k)

        src_nbrs = NeighborData(
            all_nbrs.neighbor_ids[:B],
            all_nbrs.timestamps[:B],
            all_nbrs.edge_feats[:B],
            all_nbrs.mask[:B],
        )
        cand_nbrs = NeighborData(
            all_nbrs.neighbor_ids[B:],
            all_nbrs.timestamps[B:],
            all_nbrs.edge_feats[B:],
            all_nbrs.mask[B:],
        )

        src_emb = model.encode_nodes(src_nbrs, raw_batch.time)  # (B, d)
        cand_emb = model.encode_nodes(cand_nbrs, times_cand)    # (B*(1+N_neg), d)
        cand_emb = cand_emb.view(B, 1 + N_neg, -1)             # (B, 1+N_neg, d)

        src_emb_exp = src_emb.unsqueeze(1).expand(-1, 1 + N_neg, -1)  # (B, 1+N_neg, d)
        scores = model.score_pairs(src_emb_exp, cand_emb)  # (B, 1+N_neg)
        pos_s = scores[:, 0:1]  # (B, 1)
        rank = (scores >= pos_s).sum(dim=1).float()  # (B,) 1-indexed rank
        return 1.0 / rank

    def _mrr_pairwise(
        self, model: TemporalModel, pipeline: DataPipeline, raw_batch: RawBatch, neg: Tensor
    ) -> Tensor:
        """Slow path: expand to all (src, candidate) pairs and forward once."""
        B, N_neg = neg.shape

        all_cands = torch.cat([raw_batch.dst.unsqueeze(1), neg], dim=1)  # (B, 1+N_neg)
        src_exp = raw_batch.src.unsqueeze(1).expand(-1, 1 + N_neg).reshape(-1)
        cands_flat = all_cands.reshape(-1)
        times_exp = raw_batch.time.unsqueeze(1).expand(-1, 1 + N_neg).reshape(-1)

        big_batch = RawBatch(
            src=src_exp,
            dst=cands_flat,
            time=times_exp,
            edge_feat=None,
            neg=cands_flat,  # dummy neg to satisfy PreparedBatch contract
        )
        prepared = pipeline.prepare(big_batch)
        scores_flat = model(prepared).pos_score  # (B*(1+N_neg),)
        scores = scores_flat.view(B, 1 + N_neg)

        pos_s = scores[:, 0:1]
        rank = (scores >= pos_s).sum(dim=1).float()
        return 1.0 / rank


class HitsEval(EvalProtocol):
    """Hits@K evaluation: fraction of positives ranked in top-K among negatives.

    Uses the same fixed negative candidate lists as MRREval (TGB-style).
    Reports hits@1, hits@3, hits@10 by default.

    Args:
        neg_lists: (N_eval_edges, N_neg) pre-loaded negative node IDs.
        ks: list of K values to compute Hits@K for.
    """

    def __init__(self, neg_lists: Tensor, ks: list[int] | None = None):
        self.neg_lists = neg_lists
        self.ks = ks or [1, 3, 10]

    def evaluate(self, model, pipeline, eval_batches, graph):
        model.eval()
        all_ranks: list[Tensor] = []
        edge_offset = 0

        with torch.no_grad():
            for raw_batch in eval_batches:
                B = raw_batch.batch_size
                neg = self.neg_lists[edge_offset : edge_offset + B].to(raw_batch.device)
                edge_offset += B

                if model.supports_independent_encode:
                    rr = self._rank_encode(model, pipeline, raw_batch, neg)
                else:
                    rr = self._rank_pairwise(model, pipeline, raw_batch, neg)

                all_ranks.append(rr)
                graph.advance(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

        ranks = torch.cat(all_ranks)  # (N,) 1-indexed ranks
        results = {}
        for k in self.ks:
            results[f"hits@{k}"] = (ranks <= k).float().mean().item()
        return results

    def _rank_encode(self, model, pipeline, raw_batch, neg) -> Tensor:
        """Fast path via encode_nodes."""
        B, N_neg = neg.shape
        k = pipeline.spec.neighbors.k
        all_cands = torch.cat([raw_batch.dst.unsqueeze(1), neg], dim=1)
        cands_flat = all_cands.reshape(-1)
        times_cand = raw_batch.time.unsqueeze(1).expand(-1, 1 + N_neg).reshape(-1)

        all_nodes = torch.cat([raw_batch.src, cands_flat])
        all_times = torch.cat([raw_batch.time, times_cand])
        all_nbrs = pipeline.graph.recent(all_nodes, all_times, k)

        src_nbrs = NeighborData(
            all_nbrs.neighbor_ids[:B], all_nbrs.timestamps[:B],
            all_nbrs.edge_feats[:B], all_nbrs.mask[:B],
        )
        cand_nbrs = NeighborData(
            all_nbrs.neighbor_ids[B:], all_nbrs.timestamps[B:],
            all_nbrs.edge_feats[B:], all_nbrs.mask[B:],
        )
        src_emb = model.encode_nodes(src_nbrs, raw_batch.time)
        cand_emb = model.encode_nodes(cand_nbrs, times_cand).view(B, 1 + N_neg, -1)
        src_emb_exp = src_emb.unsqueeze(1).expand(-1, 1 + N_neg, -1)
        scores = model.score_pairs(src_emb_exp, cand_emb)
        pos_s = scores[:, 0:1]
        return (scores >= pos_s).sum(dim=1).float()

    def _rank_pairwise(self, model, pipeline, raw_batch, neg) -> Tensor:
        """Slow path: expand to all pairs."""
        B, N_neg = neg.shape
        all_cands = torch.cat([raw_batch.dst.unsqueeze(1), neg], dim=1)
        src_exp = raw_batch.src.unsqueeze(1).expand(-1, 1 + N_neg).reshape(-1)
        cands_flat = all_cands.reshape(-1)
        times_exp = raw_batch.time.unsqueeze(1).expand(-1, 1 + N_neg).reshape(-1)

        big_batch = RawBatch(
            src=src_exp, dst=cands_flat, time=times_exp,
            edge_feat=None, neg=cands_flat,
        )
        prepared = pipeline.prepare(big_batch)
        scores_flat = model(prepared).pos_score
        scores = scores_flat.view(B, 1 + N_neg)
        pos_s = scores[:, 0:1]
        return (scores >= pos_s).sum(dim=1).float()


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
    ):
        self.model = model.to(config.device)
        self.graph = graph
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.neg_strategy = neg_strategy
        self.eval_protocol = eval_protocol
        self.config = config

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

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
        self.scheduler = self._build_scheduler() if config.warmup_steps > 0 else None
        if config.compile_model:
            self.model = torch.compile(self.model)
        self._current_epoch = 0
        self._best_val = 0.0

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
        """Save full training state to a checkpoint file.

        Saves: model weights, optimizer state, epoch, best_val, config.
        """
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
        """Restore model, optimizer, and training state from checkpoint.

        Returns the epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._current_epoch = checkpoint["epoch"]
        self._best_val = checkpoint["best_val"]
        return self._current_epoch

    def train(self, resume: bool = False) -> dict[str, float]:
        """Run full training loop. Returns best test metrics.

        If config.checkpoint_dir is set, automatically saves the best
        checkpoint to ``{checkpoint_dir}/best.pt``.
        """
        best_val = self._best_val if resume else 0.0
        best_test: dict[str, float] = {}
        patience_counter = 0

        for epoch in range(self._current_epoch + 1, self.config.epochs + 1):
            self._current_epoch = epoch
            train_loss = self._train_epoch()

            # Evaluate
            val_metrics = self._evaluate(self.val_batches)
            val_score = val_metrics.get("ap", val_metrics.get("mrr", 0.0))

            if val_score > best_val:
                best_val = val_score
                self._best_val = best_val
                best_test = self._evaluate(self.test_batches)
                patience_counter = 0

                # Auto-save best checkpoint
                if self.config.checkpoint_dir is not None:
                    save_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                    self.save_checkpoint(save_path)
                print(f"  Epoch {epoch}: loss={train_loss:.4f} val={val_score:.4f} test={best_test} *")
            else:
                patience_counter += 1
                print(f"  Epoch {epoch}: loss={train_loss:.4f} val={val_score:.4f} patience={patience_counter}")

            if self._wandb:
                log = {"epoch": epoch, "train_loss": train_loss, "val_score": val_score}
                if best_test:
                    log.update({f"best_test_{k}": v for k, v in best_test.items()})
                if self.scheduler:
                    log["lr"] = self.scheduler.get_last_lr()[0]
                self._wandb.log(log)

            if patience_counter >= self.config.patience:
                break

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
        """Training loop with double-buffered async prefetch."""
        pipe = self.async_pipeline
        batches = self.train_batches
        total_loss = 0.0
        amp_enabled = self.config.use_amp

        if not batches:
            return 0.0

        # Prime: prefetch first batch (graph is empty, no advance needed yet)
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

            # Graph state advance not needed (CSR is static for training)
            self.model.evolve(rb.src, rb.dst, rb.time, rb.edge_feat)

            if i + 1 < len(batches):
                pipe.start_prefetch(batches[i + 1])

        return total_loss

    def _evaluate(self, eval_batches: list[RawBatch]) -> dict[str, float]:
        """Run evaluation with proper snapshot/restore.

        Preloads ALL val+test edges into graph before evaluation to match DyGLib's
        full_neighbor_sampler semantics (eval sees all edges in the dataset).
        """
        graph_snap = self.graph.snapshot()
        model_state = self.model.freeze()

        # Preload all val+test edges so neighbor queries see full dataset (DyGLib semantics)
        for rb in self.val_batches:
            self.graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
        for rb in self.test_batches:
            self.graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)

        # Ensure every eval batch has negatives sampled (use train neg strategy)
        prepped = []
        for rb in eval_batches:
            if rb.neg is None:
                neg = self.neg_strategy.sample(rb.src, rb.dst, rb.time, self.graph,
                                                rb.edge_indices)
                rb = RawBatch(src=rb.src, dst=rb.dst, time=rb.time,
                              edge_feat=rb.edge_feat, neg=neg,
                              edge_indices=rb.edge_indices)
            prepped.append(rb)

        with torch.amp.autocast("cuda", enabled=self.config.use_amp):
            metrics = self.eval_protocol.evaluate(
                self.model, self.pipeline, prepped, self.graph
            )

        self.graph.restore(graph_snap)
        self.model.thaw(model_state)
        return metrics
