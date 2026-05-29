from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

from tgengine.core.batch import NeighborData, RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.models.base import TemporalModel
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import (
    HistoricalNegative,
    InductiveNegative,
    NegativeStrategy,
    RandomNegative,
)


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

    Args:
        include_auc: if True, also compute AUC-ROC alongside AP.
    """

    def __init__(self, include_auc: bool = False):
        self.include_auc = include_auc

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

        pos = torch.cat(all_pos_scores).sigmoid()
        neg = torch.cat(all_neg_scores).sigmoid()
        predicts = torch.cat([pos, neg]).cpu().numpy()
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        result = {"ap": float(average_precision_score(y_true=labels, y_score=predicts))}
        if self.include_auc:
            result["auc"] = float(roc_auc_score(y_true=labels, y_score=predicts))
        return result


class AUCEval(EvalProtocol):
    """AUC-ROC evaluation (1 pos + 1 neg per edge)."""

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

        pos = torch.cat(all_pos_scores).sigmoid()
        neg = torch.cat(all_neg_scores).sigmoid()
        predicts = torch.cat([pos, neg]).cpu().numpy()
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return {"auc": float(roc_auc_score(y_true=labels, y_score=predicts))}


class ThreeWayEval(EvalProtocol):
    """Evaluation with three negative types: random, historical, inductive.

    Reports ap_random, ap_historical, ap_inductive.
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

            pos = torch.cat(all_pos_scores).sigmoid()
            neg_scores = torch.cat(all_neg_scores).sigmoid()
            predicts = torch.cat([pos, neg_scores]).cpu().numpy()
            labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_scores))])
            results[f"ap_{name}"] = float(average_precision_score(y_true=labels, y_score=predicts))

        return results


class RankingEval(EvalProtocol):
    """Base class for ranking-based evaluation with fixed negative lists (TGB-style).

    Supports two compute paths:
    - Fast path (supports_independent_encode): encode all unique nodes once.
    - Slow path: expand batch to cover all (src, candidate) pairs.

    Subclasses implement _aggregate_ranks() to produce final metrics.
    """

    def __init__(self, neg_lists: Tensor):
        self.neg_lists = neg_lists

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
                    ranks = self._rank_encode(model, pipeline, raw_batch, neg)
                else:
                    ranks = self._rank_pairwise(model, pipeline, raw_batch, neg)

                all_ranks.append(ranks)

        return self._aggregate_ranks(torch.cat(all_ranks))

    @abstractmethod
    def _aggregate_ranks(self, ranks: Tensor) -> dict[str, float]:
        """Convert 1-indexed ranks tensor to final metrics dict."""
        ...

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


class MRREval(RankingEval):
    """Mean Reciprocal Rank evaluation with fixed negative candidate lists."""

    def _aggregate_ranks(self, ranks: Tensor) -> dict[str, float]:
        return {"mrr": (1.0 / ranks).mean().item()}


class HitsEval(RankingEval):
    """Hits@K evaluation: fraction of positives ranked in top-K."""

    def __init__(self, neg_lists: Tensor, ks: list[int] | None = None):
        super().__init__(neg_lists)
        self.ks = ks or [1, 3, 10]

    def _aggregate_ranks(self, ranks: Tensor) -> dict[str, float]:
        return {f"hits@{k}": (ranks <= k).float().mean().item() for k in self.ks}
