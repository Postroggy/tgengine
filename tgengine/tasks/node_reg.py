"""Node regression task head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class NodeRegressionHead(TaskHead):
    """Node-level regression head (MSE / MAE loss).

    Useful for: user activity forecasting, node attribute prediction,
    temporal property estimation.

    Args:
        d_model: input embedding dimension.
        output_dim: number of regression targets (1 for scalar).
        loss: "mse" or "mae".
    """

    def __init__(self, d_model: int, output_dim: int = 1, loss: str = "mse", dropout: float = 0.1):
        super().__init__()
        assert loss in ("mse", "mae"), f"Unknown loss: {loss}"
        self.loss_type = loss
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def compute(self, emb: Tensor, labels: Optional[Tensor] = None, **_) -> ModelOutput:
        pred = self.regressor(emb)   # (B, output_dim) or (B, 1)
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)  # (B,)
        loss = torch.tensor(0.0, device=emb.device)
        if labels is not None:
            targets = labels.float()
            if self.loss_type == "mse":
                loss = F.mse_loss(pred, targets)
            else:
                loss = F.l1_loss(pred, targets)
        dummy = torch.zeros(emb.shape[0], device=emb.device)
        return ModelOutput(
            loss=loss,
            pos_score=dummy,
            neg_score=dummy,
            node_pred=pred,
            node_labels=labels,
        )
