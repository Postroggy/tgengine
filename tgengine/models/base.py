from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from tgengine.core.batch import PreparedBatch, NeighborData
from tgengine.core.gather_spec import GatherSpec, NeighborSpec


@dataclass
class ModelOutput:
    """Standard model output."""

    loss: Tensor
    pos_score: Tensor  # (B,)
    neg_score: Tensor  # (B,) or (B, N_neg)


class TemporalModel(nn.Module, ABC):
    """Base class for all temporal graph models.

    Subclasses must define:
        - gather_spec: what data the model needs
        - forward(): neural network computation on PreparedBatch

    Optionally override for stateful models (TGN-style):
        - evolve(): update internal state after each batch
        - freeze(): checkpoint state before eval
        - thaw(): restore state after eval

    Optionally override for MRR evaluation:
        - encode_nodes(): encode nodes independently
        - score_pairs(): score (src, dst) pairs
    """

    gather_spec: GatherSpec = GatherSpec()

    @abstractmethod
    def forward(self, batch: PreparedBatch) -> ModelOutput:
        """Forward pass. Receives fully-prepared data, returns loss + scores."""
        ...

    # --- Stateful model lifecycle (override for TGN-style) ---

    def evolve(self, src: Tensor, dst: Tensor, time: Tensor, edge_feat: Optional[Tensor] = None):
        """Update internal state after processing a batch. No-op for stateless models."""
        pass

    def freeze(self) -> Any:
        """Save internal state before evaluation. Returns opaque state object."""
        return None

    def thaw(self, state: Any):
        """Restore internal state after evaluation."""
        pass

    # --- Independent encoding for MRR eval (override if supported) ---

    @property
    def supports_independent_encode(self) -> bool:
        """Whether this model can encode nodes independently of the pair.

        Models with cross-features (co-neighbor) should return False.
        """
        return False

    def encode_nodes(self, neighbors: NeighborData, times: Tensor) -> Tensor:
        """Encode nodes independently. Only called if supports_independent_encode=True.

        Args:
            neighbors: neighbor data for nodes to encode.
            times: (N,) query timestamps.

        Returns:
            (N, d_model) node embeddings.
        """
        raise NotImplementedError

    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        """Score (src, dst) embedding pairs. Only called if supports_independent_encode=True.

        Args:
            src_emb: (B, d) or (B, 1, d) source embeddings.
            dst_emb: (B, d) or (B, N, d) destination embeddings.

        Returns:
            (B,) or (B, N) scores.
        """
        raise NotImplementedError
