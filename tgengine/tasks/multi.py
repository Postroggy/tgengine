"""Multi-task head: runs multiple TaskHeads with configurable loss weighting."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class MultiTaskHead(TaskHead):
    """Combines several TaskHeads into one joint objective.

    Each sub-head receives the same ``emb`` (and any kwargs you pass in).
    Losses are summed with per-task weights.

    Example::

        head = MultiTaskHead(
            heads={
                "link": LinkPredHead(),
                "node_cls": NodeClassificationHead(128, 5),
            },
            weights={"link": 1.0, "node_cls": 0.5},
        )
        out = head(emb, labels=node_labels, src_emb=s, dst_emb=d, neg_emb=n)

    The returned ``ModelOutput`` has:
    - ``loss``: weighted sum of all sub-head losses
    - All other fields merged from sub-heads (last write wins for scalar fields)
    - ``task_losses``: dict stored in ``ModelOutput.extras`` (if extras exists)

    Args:
        heads: mapping from task name → TaskHead instance.
        weights: per-task loss weights (default 1.0 for any missing task).
    """

    def __init__(
        self,
        heads: Dict[str, TaskHead],
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.heads = torch.nn.ModuleDict(heads)
        self.weights: Dict[str, float] = weights or {}

    def compute(self, emb: Tensor, labels: Optional[Tensor] = None, **kwargs) -> ModelOutput:
        total_loss = torch.tensor(0.0, device=emb.device)
        task_losses: Dict[str, float] = {}

        merged = ModelOutput(
            loss=total_loss,
            pos_score=torch.zeros(emb.shape[0], device=emb.device),
            neg_score=torch.zeros(emb.shape[0], device=emb.device),
        )

        for name, head in self.heads.items():
            # per-task labels can be passed as `<name>_labels` kwarg
            task_labels = kwargs.pop(f"{name}_labels", labels)
            out = head(emb, labels=task_labels, **kwargs)
            w = self.weights.get(name, 1.0)
            total_loss = total_loss + w * out.loss
            task_losses[name] = out.loss.item()

            # merge non-None output fields (last write wins)
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
