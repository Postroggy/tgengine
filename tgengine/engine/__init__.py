from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from tgengine.core.batch import NeighborData, PreparedBatch, RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.base import ModelOutput, TemporalModel
from tgengine.pipeline import DataPipeline
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


class ThreeWayEval(EvalProtocol):
    """Evaluation with three negative types: random, historical, inductive.

    Each type is evaluated independently with its own graph pass.
    Reports ap_random, ap_historical, ap_inductive.

    Args:
        num_nodes: total node count.
        inductive_nodes: 1-D tensor of node IDs unseen during training.
    """

    def __init__(self, num_nodes: int, inductive_nodes: Tensor):
        self._strategies = {
            "random": RandomNegative(num_nodes),
            "historical": HistoricalNegative(num_nodes),
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
                    neg = strategy.sample(raw_batch.src, raw_batch.dst, raw_batch.time, graph)
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
            results[f"ap_{name}"] = (pos > neg_scores).float().mean().item()

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

        scores = model.score_pairs(src_emb.unsqueeze(1), cand_emb)  # (B, 1+N_neg)
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
            val_score = val_metrics.get("ap", val_metrics.get("mrr", 0.0))

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
