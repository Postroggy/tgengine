"""Edge / event classification task heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class EdgeClassificationHead(TaskHead):
    """Multi-class edge/event classification head.

    Input: concatenated src+dst embeddings or an interaction embedding.

    Useful for: relation-type prediction, transaction category prediction,
    interaction intent classification.

    Args:
        d_model: embedding dimension of each node.
        num_classes: number of edge class labels.
        input_mode: "concat" (default) uses cat(src, dst); "single" uses a
            single pre-merged embedding of size d_model.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        input_mode: str = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_mode in ("concat", "single")
        d_in = d_model * 2 if input_mode == "concat" else d_model
        self.input_mode = input_mode
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def compute(
        self,
        emb: Tensor,
        labels: Optional[Tensor] = None,
        dst_emb: Optional[Tensor] = None,
        **_,
    ) -> ModelOutput:
        if self.input_mode == "concat":
            assert dst_emb is not None, "EdgeClassificationHead(concat) requires dst_emb kwarg"
            inp = torch.cat([emb, dst_emb], dim=-1)
        else:
            inp = emb
        logits = self.classifier(inp)   # (B, C)
        loss = torch.tensor(0.0, device=emb.device)
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
        dummy = torch.zeros(emb.shape[0], device=emb.device)
        return ModelOutput(
            loss=loss,
            pos_score=dummy,
            neg_score=dummy,
            edge_pred=logits,
            edge_labels=labels,
        )


class EdgeBinaryClassificationHead(TaskHead):
    """Binary edge classification (e.g., fraud detection on transactions).

    Args:
        d_model: node embedding dimension.
        input_mode: "concat" or "single".
    """

    def __init__(self, d_model: int, input_mode: str = "concat", dropout: float = 0.1):
        super().__init__()
        assert input_mode in ("concat", "single")
        d_in = d_model * 2 if input_mode == "concat" else d_model
        self.input_mode = input_mode
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def compute(
        self,
        emb: Tensor,
        labels: Optional[Tensor] = None,
        dst_emb: Optional[Tensor] = None,
        **_,
    ) -> ModelOutput:
        if self.input_mode == "concat":
            assert dst_emb is not None
            inp = torch.cat([emb, dst_emb], dim=-1)
        else:
            inp = emb
        logits = self.classifier(inp).squeeze(-1)   # (B,)
        loss = torch.tensor(0.0, device=emb.device)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        dummy = torch.zeros(emb.shape[0], device=emb.device)
        return ModelOutput(
            loss=loss,
            pos_score=dummy,
            neg_score=dummy,
            edge_pred=logits,
            edge_labels=labels,
        )
