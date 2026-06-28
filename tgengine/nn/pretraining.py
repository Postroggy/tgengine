"""Pre-training objectives for the dynamic-graph foundation model.

Foundation-model component 4/4. The foundation model is pre-trained on
multi-domain graphs before any downstream task. The pre-training signal
comes from the graph's own structure: predicting which node appears next
in a neighbor sequence. This is the graph analogue of next-token /
next-patch prediction in LLMs — aLLM4TS (ICML 2024) showed next-patch
prediction beats masked reconstruction for time-series, and the same
logic applies to event streams.

NextNeighborPatchObjective
    Splits each node's K-length neighbor sequence (ordered by time) into
    non-overlapping patches of ``patch_size`` neighbors. The model's
    hidden state at patch i is projected to logits over the node vocab and
    trained to predict the neighbor IDs in patch i+1 (next-neighbor-patch).
    Causal: patch i never sees patch i+1's tokens. Padding positions are
    masked out of the CE loss.

The objective is backbone-agnostic: it consumes hidden states (B, n_patches,
d_model) + the raw neighbor_ids (B, K). Any encoder that produces a
per-patch summary (Mamba last-state, mean-pool, Transformer) can plug in.
This keeps the pre-training logic reusable across future backbones and
unit-testable without a full model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _split_into_patches(neighbor_ids: Tensor, mask: Tensor, patch_size: int):
    """Reshape a K-length neighbor sequence into patches.

    Args:
        neighbor_ids: (B, K) long.
        mask: (B, K) bool, True = valid neighbor.
        patch_size: neighbors per patch.

    Returns:
        patch_ids: (B, P, patch_size) long — neighbor IDs grouped into P patches.
        patch_mask: (B, P) bool — True if the patch has ≥1 valid neighbor.
        n_patches: int P = ceil(K / patch_size).
    """
    B, K = neighbor_ids.shape
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    n_patches = (K + patch_size - 1) // patch_size
    # Pad K up to a multiple of patch_size so reshape is exact.
    pad_len = n_patches * patch_size - K
    if pad_len > 0:
        neighbor_ids = F.pad(neighbor_ids, (0, pad_len), value=0)
        mask = F.pad(mask, (0, pad_len), value=False)
    patch_ids = neighbor_ids.view(B, n_patches, patch_size)
    patch_mask = mask.view(B, n_patches, patch_size).any(dim=2)  # (B, P)
    return patch_ids, patch_mask, n_patches


class NextNeighborPatchObjective(nn.Module):
    """Next-neighbor-patch pre-training loss.

    Given per-patch hidden states and the neighbor sequence, compute the
    causal next-patch cross-entropy: hidden[i] predicts the neighbor IDs
    in patch[i+1].

    Args:
        d_model: hidden dim of the per-patch representation.
        vocab_size: number of nodes (node ID vocabulary).
        patch_size: neighbors per patch.
    """

    def __init__(self, d_model: int, vocab_size: int, patch_size: int = 4):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        # LM head: per-patch hidden → logits over node vocab. Shared across
        # positions within the next patch (a patch predicts a distribution,
        # applied to each of its positions).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        hidden: Tensor,
        neighbor_ids: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute the next-neighbor-patch CE loss.

        Args:
            hidden: (B, P, d_model) per-patch representations. Must come
                from a causal encoder so patch i does not attend to patch
                i+1. The caller is responsible for causality.
            neighbor_ids: (B, K) long, time-ordered neighbor IDs.
            mask: (B, K) bool, True = valid neighbor.

        Returns:
            Scalar loss tensor (mean CE over valid next-patch positions).
        """
        B, P, _ = hidden.shape
        patch_ids, patch_mask, n_patches = _split_into_patches(
            neighbor_ids, mask, self.patch_size
        )  # patch_ids (B,P,ps), patch_mask (B,P)
        if P != n_patches:
            raise ValueError(
                f"hidden has {P} patches but neighbor_ids implies {n_patches}; "
                f"the encoder must produce one hidden per patch."
            )

        # Causal shift: hidden[i] predicts patch[i+1]. Drop the last hidden
        # (no next patch to predict) and the first target patch (no predictor).
        pred_hidden = hidden[:, :-1, :]          # (B, P-1, d_model)
        target_patches = patch_ids[:, 1:, :]      # (B, P-1, ps)
        target_patch_mask = patch_mask[:, 1:]     # (B, P-1) — valid target patch?

        logits = self.lm_head(pred_hidden)        # (B, P-1, vocab)
        # Expand logits across patch positions: each patch position shares
        # the patch-level prediction.
        ps = self.patch_size
        logits = logits.unsqueeze(2).expand(-1, -1, ps, -1)  # (B, P-1, ps, vocab)
        # CE over vocab, averaged within valid target positions.
        targets = target_patches.long()           # (B, P-1, ps)
        # Per-position validity: a target position is valid if its patch is
        # valid AND the original neighbor mask at that position was True.
        target_pos_mask = mask.view(B, n_patches, ps)[:, 1:, :]  # (B, P-1, ps)
        valid_pos = target_pos_mask & target_patch_mask.unsqueeze(-1)  # (B, P-1, ps)

        # Flatten and compute CE only on valid positions.
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        valid_flat = valid_pos.reshape(-1)

        if not valid_flat.any():
            return logits_flat.sum() * 0.0  # zero loss with grad graph

        loss = F.cross_entropy(
            logits_flat[valid_flat],
            targets_flat[valid_flat],
            reduction="mean",
        )
        return loss

    @torch.no_grad()
    def next_patch_accuracy(
        self,
        hidden: Tensor,
        neighbor_ids: Tensor,
        mask: Tensor,
    ) -> float:
        """Top-1 accuracy of next-patch neighbor prediction (eval metric)."""
        B, P, _ = hidden.shape
        patch_ids, patch_mask, n_patches = _split_into_patches(
            neighbor_ids, mask, self.patch_size
        )
        pred_hidden = hidden[:, :-1, :]
        target_patches = patch_ids[:, 1:, :].long()
        target_patch_mask = patch_mask[:, 1:]
        logits = self.lm_head(pred_hidden)  # (B, P-1, vocab)
        preds = logits.argmax(dim=-1)       # (B, P-1) — predicted node id per patch

        ps = self.patch_size
        target_pos_mask = mask.view(B, n_patches, ps)[:, 1:, :]
        valid_pos = target_pos_mask & target_patch_mask.unsqueeze(-1)
        # A patch prediction is "correct" if the argmax node equals ALL valid
        # positions in the target patch — strict. Also compute a looser
        # "matches at least one valid position" variant below.
        correct = (preds.unsqueeze(2).expand(-1, -1, ps) == target_patches) & valid_pos
        # at-least-one-match per valid position
        n_valid = valid_pos.sum().item()
        if n_valid == 0:
            return 0.0
        return correct.sum().item() / n_valid
