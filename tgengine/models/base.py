from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import PreparedBatch, NeighborData
from tgengine.core.gather_spec import GatherSpec, NeighborSpec


@dataclass
class ModelOutput:
    """Standard model output — covers all downstream task types.

    Link prediction (required):
        loss, pos_score, neg_score

    Node-level tasks (optional):
        node_pred   — (B, C) logits for node classification, or (B,) for regression
        node_labels — (B,) or (B, C) ground-truth node labels

    Edge-level tasks (optional):
        edge_pred   — (B, C) logits for edge classification, or (B,) for regression
        edge_labels — (B,) or (B, C) ground-truth edge labels

    Anomaly detection (optional):
        anomaly_score — (B,) per-event anomaly scores (higher = more anomalous)
    """

    loss: Tensor
    pos_score: Tensor        # (B,)
    neg_score: Tensor        # (B,) or (B, N_neg)

    # --- node-level ---
    node_pred: Optional[Tensor] = None    # (B, C) or (B,)
    node_labels: Optional[Tensor] = None  # (B,) or (B, C)

    # --- edge-level ---
    edge_pred: Optional[Tensor] = None    # (B, C) or (B,)
    edge_labels: Optional[Tensor] = None  # (B,) or (B, C)

    # --- anomaly ---
    anomaly_score: Optional[Tensor] = None  # (B,)


@dataclass
class EmbeddingBundle:
    """Node embeddings produced by a model's encode() step.

    For independent-encode models (GraphMixer, TGN, DyGMamba):
        src, dst, neg are all (B, d_model).

    For pair-dependent models (DyGFormer, FreeDyG) where src representation
    changes depending on who it's paired with:
        src_for_dst and neg_src are used instead of a single src.

    The Engine reads these fields to route embeddings to task heads and
    to compute link prediction scores.

    Fields:
        src:          (B, d) source embedding encoded independently.
        dst:          (B, d) destination embedding.
        neg:          (B, d) negative destination embedding.
        src_for_neg:  (B, d) source re-encoded against neg (pair-dependent models only).
        d_model:      embedding dimension.
    """

    src: Tensor                      # (B, d)
    dst: Tensor                      # (B, d)
    neg: Tensor                      # (B, d)
    src_for_neg: Optional[Tensor] = None  # (B, d) — only for pair-dependent models

    @property
    def d_model(self) -> int:
        return self.src.shape[-1]

    @property
    def is_pair_dependent(self) -> bool:
        return self.src_for_neg is not None


class TemporalModel(nn.Module, ABC):
    """Base class for all temporal graph models.

    ## Minimal implementation (independent-encode models):

        class MyModel(TemporalModel):
            gather_spec = GatherSpec(neighbors=NeighborSpec(k=32))

            def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
                src = self._encoder(batch.src_neighbors, batch.time)
                dst = self._encoder(batch.dst_neighbors, batch.time)
                neg = self._encoder(batch.neg_neighbors, batch.time)
                return EmbeddingBundle(src=src, dst=dst, neg=neg)

    ## Pair-dependent models (DyGFormer / FreeDyG style):

        def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
            src, dst = self._pair_encode(batch.src_neighbors, batch.dst_neighbors, ...)
            src_neg, neg = self._pair_encode(batch.src_neighbors, batch.neg_neighbors, ...)
            return EmbeddingBundle(src=src, dst=dst, neg=neg, src_for_neg=src_neg)

    ## Engine integration:

    The Engine calls encode() to get embeddings, then:
    - Routes embeddings to each TaskHead in `engine.tasks`
    - Falls back to the model's own forward() for link prediction loss
      when no explicit link pred task head is provided.

    Backward compatibility: if a model only implements forward() (old API),
    the Engine calls forward() directly — encode() is never called.
    """

    gather_spec: GatherSpec = GatherSpec()

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        """Compute node embeddings from a prepared batch.

        Override this instead of (or alongside) forward() to support
        multi-task learning and downstream tasks via the Engine's tasks= API.

        Default: raises NotImplementedError — models that only implement
        forward() still work via the Engine's backward-compatible code path.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement encode(). "
            "Either implement encode() for multi-task support, or use "
            "the model's own forward() for single-task training."
        )

    @property
    def has_encode(self) -> bool:
        """Whether this model implements the encode() API."""
        return type(self).encode is not TemporalModel.encode

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        """Default forward: encode() + link prediction loss.

        Models that implement encode() get this for free.
        Models that want custom forward logic (e.g. pair-dependent scoring)
        should override forward() directly.
        """
        bundle = self.encode(batch)
        return self._link_pred_loss(bundle)

    def _link_pred_loss(self, bundle: EmbeddingBundle) -> ModelOutput:
        """Standard BCE link prediction loss from an EmbeddingBundle.

        Subclasses with custom decoders should override forward() instead.
        """
        src_for_pos = bundle.src
        src_for_neg = bundle.src_for_neg if bundle.src_for_neg is not None else bundle.src

        pos_score = (src_for_pos * bundle.dst).sum(-1)
        neg_score = (src_for_neg * bundle.neg).sum(-1)

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

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
        """Whether this model can encode nodes independently of the pair."""
        return False

    def encode_nodes(self, neighbors: NeighborData, times: Tensor) -> Tensor:
        """Encode nodes independently. Only called if supports_independent_encode=True."""
        raise NotImplementedError

    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        """Score (src, dst) embedding pairs. Only called if supports_independent_encode=True."""
        raise NotImplementedError
