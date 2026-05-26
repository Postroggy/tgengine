from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CoNeighborEncoder(nn.Module):
    """Encode co-occurrence counts into feature vectors (DyGFormer Section 4.1)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, co_counts: Tensor) -> Tensor:
        """
        Args:
            co_counts: (B,) or (B, K) co-occurrence counts.

        Returns:
            (B, d_model) encoded co-occurrence features.
        """
        if co_counts.ndim == 1:
            co_counts = co_counts.unsqueeze(-1)
        return self.encoder(co_counts.float().unsqueeze(-1)).mean(dim=-2)
