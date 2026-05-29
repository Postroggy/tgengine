"""Link prediction task head (wraps existing BCE loss pattern)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.base import ModelOutput
from .base import TaskHead


class LinkPredHead(TaskHead):
    """Standard binary cross-entropy link prediction head.

    Args:
        decoder: scoring module (B,d)x(B,d) -> (B,). If None, uses dot product.
    """

    def __init__(self, decoder: Optional[nn.Module] = None):
        super().__init__()
        self.decoder = decoder

    def compute(
        self,
        emb: Tensor,                  # unused — src/dst/neg passed via kwargs
        labels: Optional[Tensor] = None,
        src_emb: Optional[Tensor] = None,
        dst_emb: Optional[Tensor] = None,
        neg_emb: Optional[Tensor] = None,
    ) -> ModelOutput:
        assert src_emb is not None and dst_emb is not None and neg_emb is not None

        if self.decoder is not None:
            pos_score = self.decoder(src_emb, dst_emb)
            neg_score = self.decoder(src_emb, neg_emb)
        else:
            pos_score = (src_emb * dst_emb).sum(-1)
            neg_score = (src_emb * neg_emb).sum(-1)

        loss = (
            F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        )
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)
