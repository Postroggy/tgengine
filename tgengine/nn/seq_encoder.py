from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class SequenceEncoder(nn.Module, ABC):
    """Base class for sequence-to-representation encoders.

    All sequence encoders share the same interface:
        Input:  (B, K, d_in) sequence of features
                (B, K) boolean mask (True = valid position)
        Output: (B, d_out) fixed-size representation

    This is where model innovation happens. Subclass this to create
    new temporal graph models.
    """

    @abstractmethod
    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        """Encode a padded sequence into a fixed-size representation.

        Args:
            seq: (B, K, d_in) input sequence features.
            mask: (B, K) boolean mask. True for valid positions.

        Returns:
            (B, d_out) encoded representation.
        """
        ...


class TransformerSeqEncoder(SequenceEncoder):
    """Multi-head self-attention encoder (DyGFormer-style)."""

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        # Transformer expects src_key_padding_mask: True = IGNORE
        padding_mask = ~mask
        out = self.transformer(seq, src_key_padding_mask=padding_mask)
        # Mean pool over valid positions
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled


class MambaSeqEncoder(SequenceEncoder):
    """Selective State Space Model encoder (Mamba)."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        # TODO: import and use actual Mamba implementation
        # For now, placeholder with GRU (same interface)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)

    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        # Pack padded sequence for efficiency
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths, batch_first=True, enforce_sorted=False
        )
        output, hidden = self.rnn(packed)
        # Return last hidden state
        return hidden[-1]


class GRUSeqEncoder(SequenceEncoder):
    """GRU-based sequence encoder."""

    def __init__(self, d_model: int, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)

    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden[-1]


class MeanPoolEncoder(SequenceEncoder):
    """Simplest baseline: masked mean pooling."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        mask_expanded = mask.unsqueeze(-1).float()
        return (seq * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
