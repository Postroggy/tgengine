"""MLP-Mixer building blocks for temporal graph models.

Shared by GraphMixer and FreeDyG. Implements:
  - FeedForwardNet: two-layer MLP (GELU, dropout)
  - MLPMixerLayer: token-mixing + channel-mixing (GraphMixer-style)
  - FilterLayer: FFT-based mixing (FreeDyG extension)
  - FreeDyGMixerLayer: FilterLayer + MLPMixerLayer
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FeedForwardNet(nn.Module):
    """Two-layer MLP with GELU activation (identical to reference DyGLib)."""

    def __init__(self, input_dim: int, expansion_factor: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(expansion_factor * input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class MLPMixerLayer(nn.Module):
    """Single MLP-Mixer layer: token-mixing then channel-mixing.

    Input/output shape: (B, T, C)

    Token mixing:  transpose → LayerNorm(T) → FFN(T) → transpose → residual
    Channel mixing: LayerNorm(C) → FFN(C) → residual
    """

    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_expansion: float = 0.5,
        channel_expansion: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_ffn = FeedForwardNet(num_tokens, token_expansion, dropout)
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_ffn = FeedForwardNet(num_channels, channel_expansion, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        # Token mixing
        h = self.token_norm(x.transpose(1, 2))       # (B, C, T)
        h = self.token_ffn(h).transpose(1, 2)         # (B, T, C)
        x = x + h

        # Channel mixing
        h = self.channel_norm(x)                      # (B, T, C)
        x = x + self.channel_ffn(h)
        return x


class FilterLayer(nn.Module):
    """FFT-based frequency filter from FreeDyG.

    Applies a learnable complex weight in the frequency domain and returns
    to the time domain via IRFFT. Adds a residual + dropout.
    """

    def __init__(self, max_seq_len: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        # Learnable complex weight: shape (1, freq_bins, hidden_dim, 2)
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_len // 2 + 1, hidden_dim, 2, dtype=torch.float32)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        freq = torch.fft.rfft(x, n=self.max_seq_len, dim=1, norm="forward")
        weight = torch.view_as_complex(self.complex_weight)         # (1, freq_bins, C)
        freq = freq * weight
        out = torch.fft.irfft(freq, n=self.max_seq_len, dim=1, norm="forward")
        out = out[:, :T, :]          # truncate back to original length
        return x + self.drop(out)


class FreeDyGMixerLayer(nn.Module):
    """MLP-Mixer layer augmented with FFT filter (FreeDyG variant).

    Order: FilterLayer → token-mixing → channel-mixing
    """

    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_expansion: float = 0.5,
        channel_expansion: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.filter = FilterLayer(num_tokens, num_channels, dropout)
        self.mixer = MLPMixerLayer(
            num_tokens, num_channels, token_expansion, channel_expansion, dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mixer(self.filter(x))
