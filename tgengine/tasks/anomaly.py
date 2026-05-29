"""Temporal anomaly detection task head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class AnomalyDetectionHead(TaskHead):
    """Temporal graph anomaly detection head (unsupervised or supervised).

    Two modes:
    - **supervised**: labels 0/1 provided → binary cross-entropy.
    - **unsupervised** (no labels): reconstruction loss via auto-encoder;
      the anomaly score is the reconstruction error.

    Useful for: fraud detection, intrusion detection, bot-network detection,
    temporal event anomaly scoring.

    Args:
        d_model: node embedding dimension.
        hidden_dim: bottleneck dimension for the auto-encoder path.
        mode: "supervised" (requires labels) or "unsupervised".
        dropout: dropout before scoring head.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        mode: str = "supervised",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert mode in ("supervised", "unsupervised")
        self.mode = mode
        h = hidden_dim or d_model // 2

        if mode == "supervised":
            self.scorer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
        else:
            # auto-encoder: encoder → bottleneck → decoder
            self.encoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, h),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(h, d_model),
            )

    def compute(self, emb: Tensor, labels: Optional[Tensor] = None, **_) -> ModelOutput:
        dummy = torch.zeros(emb.shape[0], device=emb.device)

        if self.mode == "supervised":
            logits = self.scorer(emb).squeeze(-1)       # (B,)
            loss = torch.tensor(0.0, device=emb.device)
            if labels is not None:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            anomaly_score = torch.sigmoid(logits).detach()
            return ModelOutput(
                loss=loss,
                pos_score=dummy,
                neg_score=dummy,
                anomaly_score=anomaly_score,
                node_pred=logits,
                node_labels=labels,
            )
        else:
            z = self.encoder(emb)
            recon = self.decoder(z)
            # per-sample reconstruction error → anomaly score
            recon_err = (emb - recon).pow(2).mean(dim=-1)  # (B,)
            loss = recon_err.mean()
            return ModelOutput(
                loss=loss,
                pos_score=dummy,
                neg_score=dummy,
                anomaly_score=recon_err.detach(),
            )
