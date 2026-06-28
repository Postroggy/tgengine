"""Rotary time encoding for continuous-time dynamic graphs.

Based on RoTHP (BDMA 2025): Hawkes-process log-likelihood depends only on the
time gap Δt (translation-invariant), and rotary position encoding naturally
encodes *relative* differences. Applying rotary to Δt therefore gives a
time encoding whose inner products depend only on time gaps — the property
Hawkes likelihood needs — without a learned frequency table like Time2Vec.

Two forms are provided:
  - RotaryTimeEncoder: produces absolute rotary cos/sin tables from Δt,
    usable anywhere a (..., d_model) time feature is expected (drop-in
    replacement for FixedCosineTimeEncoder / Time2Vec).
  - apply_rotary: applies rotary to a feature tensor along the last dim,
    mixing the feature with the Δt rotation (RoPE-style). This is the form
    the foundation model uses to inject time *into* the SSM input projection
    without spending a dedicated channel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def _build_freqs(dt: Tensor, d_model: int, base: float = 10000.0) -> Tensor:
    """Compute rotation angles for each frequency dimension.

    Args:
        dt: (...) time deltas.
        d_model: must be even.
        base: RoPE frequency base.

    Returns:
        (..., d_model // 2) angles θ_i = base^(-2i/d) * dt.
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for rotary encoding, got {d_model}")
    half = d_model // 2
    # Geometric frequency progression: 1, base^(-2/d), base^(-4/d), ...
    inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32, device=dt.device) * 2.0 / d_model)
    # (..., half) = dt[..., None] * inv_freq[None...]
    angles = dt.float().unsqueeze(-1) * inv_freq  # (..., half)
    return angles


class RotaryTimeEncoder(nn.Module):
    """Rotary time encoding for event-stream Δt.

    Unlike Time2Vec (learnable linear+sin, absolute-time) this is a
    *relative* encoding: rotating two features by Δt_a and Δt_b and taking
    their dot product depends only on Δt_a - Δt_b. That translation
    invariance matches Hawkes-process likelihood, which is the theoretical
    basis for time encoding in temporal graph models.

    Δt is used *linearly* (no log/asinh compression). Compression would
    break translation invariance — and since cos/sin are already bounded,
    large Δt cannot produce NaN/Inf, so there is no numerical need to
    compress. For datasets with extreme Δt ranges, scale Δt at the caller
    (e.g. divide by dataset mean gap) rather than here.

    Two usage modes:

    1. Feature mode (default, ``as_feature=True``): forward(dt) returns a
       (..., d_model) feature tensor — concat of [cos θ, sin θ]. Drop-in
       replacement for FixedCosineTimeEncoder.

    2. Rotation mode (``as_feature=False``): forward(dt) returns
       (..., d_model//2, 2) cos/sin pairs for use with ``apply_rotary``.
    """

    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.base = base

    def forward(self, dt: Tensor, as_feature: bool = True) -> Tensor:
        """Encode time deltas as rotary cos/sin.

        Args:
            dt: (...) time deltas (t_now - t_event). Arbitrary shape.
            as_feature: if True, return (..., d_model) feature tensor
                ([cos | sin]); if False, return (..., d_model//2, 2) pairs
                for ``apply_rotary``.

        Returns:
            Encoded time representation.
        """
        angles = _build_freqs(dt, self.d_model, self.base)  # (..., half)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        if as_feature:
            return torch.cat([cos, sin], dim=-1)  # (..., d_model)
        return torch.stack([cos, sin], dim=-1)  # (..., half, 2)


def apply_rotary(x: Tensor, rot: Tensor) -> Tensor:
    """Apply rotary embedding to the second half of each feature pair.

    Splits x into (..., half, 2) pairs and rotates each pair by the angle
    encoded in ``rot`` (produced by RotaryTimeEncoder(as_feature=False)).
    This is the standard GPT-NeoX / Llama RoPE rotation, generalized to
    arbitrary leading dims (here: (B, K, ...)).

    Args:
        x: (..., d_model) feature tensor. d_model must be even.
        rot: (..., d_model//2, 2) cos/sin pairs.

    Returns:
        (..., d_model) rotated features.
    """
    *lead, d_model = x.shape
    if d_model % 2 != 0:
        raise ValueError(f"x last dim must be even, got {d_model}")
    half = d_model // 2
    x_pairs = x.reshape(*lead, half, 2)  # (..., half, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    cos = rot[..., 0]  # (..., half)
    sin = rot[..., 1]
    # Rotate each (x1, x2) pair: (x1*cos - x2*sin, x1*sin + x2*cos)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).reshape(*lead, d_model)
