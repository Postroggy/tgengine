"""TaskHead abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from torch import Tensor

from tgengine.models.base import ModelOutput


class TaskHead(nn.Module, ABC):
    """Base class for all task-specific output heads.

    A TaskHead sits on top of a backbone encoder. It receives node or edge
    embeddings and ground-truth labels, computes loss, and fills the
    relevant fields in ModelOutput.

    Subclasses implement compute() and should NOT override forward().
    """

    @abstractmethod
    def compute(
        self,
        emb: Tensor,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """Compute task loss and predictions.

        Args:
            emb: (B, d) node or edge embeddings from the backbone.
            labels: (B,) or (B, C) ground-truth labels. May be None at inference.

        Returns:
            ModelOutput with task-specific fields populated. loss is 0 if labels is None.
        """
        ...

    def forward(self, emb: Tensor, labels: Optional[Tensor] = None, **kwargs) -> ModelOutput:
        return self.compute(emb, labels, **kwargs)
