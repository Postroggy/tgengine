from __future__ import annotations

import math

import numpy as np
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


class FixedCosineTimeEncoder(nn.Module):
    """Fixed cosine time encoding matching DyGLib's TimeEncoder exactly.

    Uses frequencies w = 1 / 10^linspace(0, 9, d_model) (fixed, not learnable).
    This is the time encoder used in DyGFormer and GraphMixer reference implementations.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # DyGLib: w = 1/10^linspace(0,9,d) → frequencies from 1 to 1e-9
        w = 1.0 / (10 ** np.linspace(0, 9, d_model, dtype=np.float32))
        # Use nn.Linear with frozen weights (bias=0) to get w*t
        lin = nn.Linear(1, d_model, bias=True)
        lin.weight = nn.Parameter(torch.from_numpy(w).reshape(d_model, 1), requires_grad=False)
        lin.bias = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.linear = lin

    def forward(self, dt: Tensor) -> Tensor:
        """
        Args:
            dt: (...) time deltas.

        Returns:
            (..., d_model) cosine time features.
        """
        return torch.cos(self.linear(dt.float().unsqueeze(-1)))  # (..., d_model)


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
