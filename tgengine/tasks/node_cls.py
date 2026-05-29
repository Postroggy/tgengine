"""Node classification task heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class NodeClassificationHead(TaskHead):
    """Multi-class node classification head (cross-entropy).

    Args:
        d_model: input embedding dimension.
        num_classes: number of target classes.
        dropout: dropout before classifier.
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def compute(self, emb: Tensor, labels: Optional[Tensor] = None, **_) -> ModelOutput:
        logits = self.classifier(emb)        # (B, C)
        loss = torch.tensor(0.0, device=emb.device)
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
        # pos_score / neg_score kept as zeros for API compatibility
        dummy = torch.zeros(emb.shape[0], device=emb.device)
        return ModelOutput(
            loss=loss,
            pos_score=dummy,
            neg_score=dummy,
            node_pred=logits,
            node_labels=labels,
        )


class NodeBinaryClassificationHead(TaskHead):
    """Binary node classification head (BCE with logits).

    Useful for fraud detection, bot detection, anomalous-node labeling, etc.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def compute(self, emb: Tensor, labels: Optional[Tensor] = None, **_) -> ModelOutput:
        logits = self.classifier(emb).squeeze(-1)  # (B,)
        loss = torch.tensor(0.0, device=emb.device)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        dummy = torch.zeros(emb.shape[0], device=emb.device)
        return ModelOutput(
            loss=loss,
            pos_score=dummy,
            neg_score=dummy,
            node_pred=logits,
            node_labels=labels,
        )
