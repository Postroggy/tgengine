"""CrossMamba: cross-domain temporal link prediction via Mamba v1.

Design principles:
  - No edge features, no node features — purely temporal structure
  - Input to each Mamba layer: K neighbor time embeddings only
  - Enables zero-shot cross-domain transfer without feature alignment
  - Backbone: Mamba v1 (selective SSM) with pre-norm, 2 stacked layers
  - TimeEncoder: FixedCosineTimeEncoder (same as other models in this repo)

Architecture (per node):
  neighbors (K recent interactions) → relative time diffs
    → FixedCosineTimeEncoder → (B, K, d_model)
    → MambaBlock × n_layers  → (B, K, d_model)
    → mean-pool over valid positions → (B, d_model)
  src, dst, neg each independently encoded → link pred score
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import EmbeddingBundle, ModelOutput, TemporalModel
from tgengine.nn import FixedCosineTimeEncoder


# ── Mamba v1 building blocks ───────────────────────────────────────────────


class _SelectiveSSM(nn.Module):
    """S6: Mamba v1 selective state space model (input-dependent A, B, C).

    Runs a sequential scan — efficient enough for short sequences (K ≤ 64).
    """

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = max(1, d_inner // 16)

        # Project x → [dt_raw | B_ssm | C_ssm]
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # dt_raw → dt (d_inner)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)  # softplus → small dt

        # A: (d_inner, d_state), stored as log for stability
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A_init))

        # D: skip connection, one scalar per d_inner channel
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_inner)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, _ = x.shape

        xz = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt_rank = self.dt_proj.in_features
        dt_raw, B_ssm, C_ssm = xz.split([dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)
        A = -torch.exp(self.log_A)              # (d_inner, d_state), negative

        # Discretise: zero-order hold
        # A_bar: (B, L, d_inner, d_state)
        A_bar = torch.exp(A[None, None] * dt.unsqueeze(-1))
        # B_bar: (B, L, d_inner, d_state)  — outer product of dt and B_ssm per position
        B_bar = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

        # Sequential selective scan
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys: list[Tensor] = []
        for l in range(L):
            h = A_bar[:, l] * h + B_bar[:, l] * x[:, l, :, None]
            ys.append((h * C_ssm[:, l, None, :]).sum(-1))  # (B, d_inner)

        y = torch.stack(ys, dim=1)              # (B, L, d_inner)
        y = y + x * self.D[None, None, :]      # skip connection
        return y


class MambaBlock(nn.Module):
    """Pre-norm Mamba v1 block with residual connection.

    x → LayerNorm → expand → [conv1d → SiLU → SSM] ⊗ SiLU(z) → project → + x
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv over the sequence dimension
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        self.ssm = _SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x: (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                      # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)         # each (B, L, d_inner)

        # Causal conv: trim output to length L
        x_branch = x_branch.transpose(1, 2)                   # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :residual.shape[1]]
        x_branch = F.silu(x_branch.transpose(1, 2))           # (B, L, d_inner)

        y = self.ssm(x_branch)                    # (B, L, d_inner)
        y = y * F.silu(z)                         # gating
        return residual + self.out_proj(y)


# ── CrossMamba model ───────────────────────────────────────────────────────


class CrossMamba(TemporalModel):
    """Cross-domain temporal graph model backed by Mamba v1.

    Uses only temporal structure (interaction timestamps) — no edge or node
    features. This makes the model domain-agnostic: the same weights can be
    applied to any dataset without feature-alignment preprocessing.

    Args:
        d_model: Embedding dimension throughout the model.
        K: Number of recent temporal neighbors to attend over.
        n_layers: Number of stacked MambaBlocks (default 2).
        d_state: SSM latent state size (default 16).
        d_conv: Depthwise conv kernel size (default 4).
        expand: Inner dimension expansion factor (default 2).
        dropout: Dropout applied after the final LayerNorm.

    Example::

        model = CrossMamba(d_model=128, K=32)
        engine = Engine(model=model, ..., config=TrainConfig(...))
    """

    def __init__(
        self,
        d_model: int = 128,
        K: int = 32,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.K = K
        self.d_model = d_model

        # GatherSpec: neighbors only — edge features are ignored at runtime
        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, strategy="recency"),
            co_occurrence=False,
        )

        # Time encoder: maps scalar time-delta → d_model vector
        # FixedCosineTimeEncoder is non-learnable → no domain-specific freq tuning
        self.time_enc = FixedCosineTimeEncoder(d_model, learnable=False)

        # Mamba backbone
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Link prediction: bilinear dot-product (parameter-free → less overfitting)
        # score(u, v) = src_emb · dst_emb  (sigmoid applied externally in loss)
        self.score_scale = nn.Parameter(torch.ones(1))  # learnable temperature

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def _encode_neighbors(self, nbrs, query_time: Tensor) -> Tensor:
        """Encode a node from its K temporal neighbors using time only.

        Args:
            nbrs: NeighborData — uses only `timestamps` and `mask`.
            query_time: (B,) event timestamps for relative time computation.

        Returns:
            (B, d_model) node embedding.
        """
        # Relative time diffs: clamp to >= 0 (no future neighbors)
        dt = (query_time.unsqueeze(1).float() - nbrs.timestamps.float()).clamp(min=0)  # (B, K)

        # Time encoding → (B, K, d_model)
        seq = self.time_enc(dt)

        # Zero-out padded positions before Mamba (avoids leaking zeros as signal)
        seq = seq * nbrs.mask.unsqueeze(-1)

        # Mamba layers (pre-norm + residual inside each block)
        for layer in self.layers:
            seq = layer(seq)

        seq = self.out_norm(seq)
        seq = self.dropout(seq)

        # Masked mean pooling over valid neighbors
        mask_f = nbrs.mask.float()                              # (B, K)
        n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)
        emb = (seq * mask_f.unsqueeze(-1)).sum(dim=1) / n_valid # (B, d_model)
        return emb

    # ------------------------------------------------------------------
    # TemporalModel interface
    # ------------------------------------------------------------------

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        src = self._encode_neighbors(batch.src_neighbors, batch.time)
        dst = self._encode_neighbors(batch.dst_neighbors, batch.time)
        neg = self._encode_neighbors(batch.neg_neighbors, batch.time)
        return EmbeddingBundle(src=src, dst=dst, neg=neg)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self._encode_neighbors(batch.src_neighbors, batch.time)
        dst_emb = self._encode_neighbors(batch.dst_neighbors, batch.time)
        neg_emb = self._encode_neighbors(batch.neg_neighbors, batch.time)

        pos_score = self.score_scale * (src_emb * dst_emb).sum(dim=-1)   # (B,)
        neg_score = self.score_scale * (src_emb * neg_emb).sum(dim=-1)   # (B,)

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]), labels
        )
        return ModelOutput(loss=loss, pos_score=pos_score.sigmoid(), neg_score=neg_score.sigmoid())

    # MRR eval support
    @property
    def supports_independent_encode(self) -> bool:
        return True

    def encode_nodes(self, neighbors, times: Tensor) -> Tensor:
        return self._encode_neighbors(neighbors, times)

    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        return (self.score_scale * (src_emb * dst_emb).sum(dim=-1)).sigmoid()
