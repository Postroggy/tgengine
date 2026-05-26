from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from tgengine.core.batch import PreparedBatch, RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import NegativeStrategy


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 200
    lr: float = 1e-4
    patience: int = 10
    device: str = "cuda"
    seed: int = 42


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
    """Standard Average Precision evaluation (1 pos + 1 neg per edge)."""

    def evaluate(self, model, pipeline, eval_batches, graph):
        model.eval()
        all_pos_scores = []
        all_neg_scores = []

        with torch.no_grad():
            for raw_batch in eval_batches:
                prepared = pipeline.prepare(raw_batch)
                output = model(prepared)
                all_pos_scores.append(output.pos_score)
                all_neg_scores.append(output.neg_score)
                graph.advance(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

        pos = torch.cat(all_pos_scores).sigmoid()
        neg = torch.cat(all_neg_scores).sigmoid()
        # AP = fraction of times pos > neg
        ap = (pos > neg).float().mean().item()
        return {"ap": ap}


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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    def train(self) -> dict[str, float]:
        """Run full training loop. Returns best test metrics."""
        best_val = 0.0
        best_test = {}
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()

            # Evaluate
            val_metrics = self._evaluate(self.val_batches)
            val_score = val_metrics.get("ap", 0.0)

            if val_score > best_val:
                best_val = val_score
                best_test = self._evaluate(self.test_batches)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                break

        return best_test

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for raw_batch in tqdm(self.train_batches, desc="Training"):
            # Sample negatives
            neg = self.neg_strategy.sample(
                raw_batch.src, raw_batch.dst, raw_batch.time, self.graph
            )
            raw_batch.neg = neg

            # Pipeline: prepare all data in one fused pass
            prepared = self.pipeline.prepare(raw_batch)

            # Forward
            self.optimizer.zero_grad()
            output = self.model(prepared)
            output.loss.backward()
            self.optimizer.step()

            total_loss += output.loss.item()

            # Graph evolves: new edges become visible
            self.graph.advance(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

            # Stateful model update
            self.model.evolve(raw_batch.src, raw_batch.dst, raw_batch.time, raw_batch.edge_feat)

        return total_loss / len(self.train_batches)

    def _evaluate(self, eval_batches: list[RawBatch]) -> dict[str, float]:
        """Run evaluation with proper snapshot/restore."""
        graph_snap = self.graph.snapshot()
        model_state = self.model.freeze()

        metrics = self.eval_protocol.evaluate(
            self.model, self.pipeline, eval_batches, self.graph
        )

        self.graph.restore(graph_snap)
        self.model.thaw(model_state)
        return metrics
