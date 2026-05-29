"""Edge / event regression task head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class EdgeRegressionHead(TaskHead):
    """Edge-level regression head.

    Useful for: interaction weight prediction, edge attribute forecasting,
    temporal link strength estimation, traffic volume prediction.

    Args:
        d_model: node embedding dimension.
        output_dim: number of regression outputs per edge.
        input_mode: "concat" (src||dst) or "single" (pre-merged).
        loss: "mse" or "mae".
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int = 1,
        input_mode: str = "concat",
        loss: str = "mse",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_mode in ("concat", "single")
        assert loss in ("mse", "mae")
        d_in = d_model * 2 if input_mode == "concat" else d_model
        self.input_mode = input_mode
        self.loss_type = loss
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
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
        pred = self.regressor(inp)
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        loss = torch.tensor(0.0, device=emb.device)
        if labels is not None:
            targets = labels.float()
            loss = F.mse_loss(pred, targets) if self.loss_type == "mse" else F.l1_loss(pred, targets)
        dummy = torch.zeros(emb.shape[0], device=emb.device)
        return ModelOutput(
            loss=loss,
            pos_score=dummy,
            neg_score=dummy,
            edge_pred=pred,
            edge_labels=labels,
        )
