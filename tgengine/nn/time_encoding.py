from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class Time2Vec(nn.Module):
    """Learnable time encoding via Time2Vec (sinusoidal with learnable frequencies).

    Maps scalar time deltas to d_model-dimensional feature vectors.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w = nn.Linear(1, d_model)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, dt: Tensor) -> Tensor:
        """
        Args:
            dt: (...) arbitrary shape of time deltas.

        Returns:
            (..., d_model) time features.
        """
        dt = dt.unsqueeze(-1).float()  # (..., 1)
        out = self.w(dt)  # (..., d_model)
        # First dimension is linear, rest are sinusoidal — use cat to avoid inplace
        return torch.cat([out[..., :1], torch.sin(out[..., 1:])], dim=-1)


class HarmonicEncoder(nn.Module):
    """Fixed-frequency harmonic time encoding (non-learnable)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Log-spaced frequencies
        freqs = torch.exp(torch.linspace(0, math.log(10000), d_model // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, dt: Tensor) -> Tensor:
        """
        Args:
            dt: (...) time deltas.

        Returns:
            (..., d_model) time features.
        """
        dt = dt.unsqueeze(-1).float()  # (..., 1)
        angles = dt * self.freqs  # (..., d_model//2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
