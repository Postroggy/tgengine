"""CrossMamba: cross-domain temporal link prediction via Mamba v1.

Each neighbor position carries two orthogonal signals:
  1. Temporal: FixedCosineTimeEncoder(query_time - nbr_time)
  2. Structural (domain-agnostic):
       - recency rank: normalized position in K-length window (1=most recent, 0=oldest)
       - repeat freq:  fraction of times this neighbor ID appears in the K-length window
       - co_occur:     1 if this neighbor ID also appears in the counterpart node's K-window

These structural features require no node semantics and transfer across domains.
freq and co_occur are computed by a Triton kernel (no O(B,K,K) intermediate tensor).
Requires: mamba_ssm (selective_scan_fn CUDA kernel) and triton.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import NeighborData, PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import EmbeddingBundle, ModelOutput, TemporalModel
from tgengine.nn import FixedCosineTimeEncoder

import triton
import triton.language as tl
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn


# ── Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def _struct_kernel(
        ids_ptr,          # (B, K) int32
        cpart_ptr,        # (B, K) int32  — counterpart neighbor IDs
        mask_ptr,         # (B, K) bool/uint8
        freq_ptr,         # (B, K) float32  — output: repeat frequency
        co_ptr,           # (B, K) float32  — output: co-occurrence flag
        B: tl.constexpr,
        K: tl.constexpr,
        K_POW2: tl.constexpr,  # next power-of-2 >= K for tl.arange
    ):
        """Each program handles one (batch, position) pair.

        Scans the full K-window to count:
          freq[b,i]    = |{j : ids[b,j]==ids[b,i] AND mask[b,j]}| / K
          co_occur[b,i] = any(cpart[b,:] == ids[b,i] AND mask[b,:])
        """
        b = tl.program_id(0)
        i = tl.program_id(1)

        base = b * K
        query_id = tl.load(ids_ptr + base + i)
        is_valid  = tl.load(mask_ptr + base + i)

        offsets = tl.arange(0, K_POW2)
        valid   = offsets < K

        row_ids   = tl.load(ids_ptr   + base + offsets, mask=valid, other=-2)
        row_masks = tl.load(mask_ptr  + base + offsets, mask=valid, other=0)
        cp_ids    = tl.load(cpart_ptr + base + offsets, mask=valid, other=-2)

        freq_count = tl.sum((row_ids == query_id) & (row_masks > 0), axis=0)
        freq_val   = (freq_count / K) * is_valid

        co_hit  = tl.sum((cp_ids == query_id) & (row_masks > 0), axis=0)
        co_val  = tl.where((co_hit > 0) & (is_valid > 0), 1.0, 0.0)

        tl.store(freq_ptr + base + i, freq_val.to(tl.float32))
        tl.store(co_ptr   + base + i, co_val.to(tl.float32))


def _struct_features_triton(
    ids: Tensor,     # (B, K) int32
    mask: Tensor,    # (B, K) bool
    cpart: Tensor,   # (B, K) int32
    K: int,
) -> tuple[Tensor, Tensor]:
    """Launch Triton kernel. Returns freq (B,K) and co_occur (B,K)."""
    B = ids.shape[0]
    K_pow2 = 1 << int(math.ceil(math.log2(max(K, 1))))

    ids_c   = ids.to(torch.int32).contiguous()
    cpart_c = cpart.to(torch.int32).contiguous()
    mask_c  = mask.to(torch.uint8).contiguous()
    freq    = torch.empty(B, K, dtype=torch.float32, device=ids.device)
    co      = torch.empty(B, K, dtype=torch.float32, device=ids.device)

    grid = (B, K)
    _struct_kernel[grid](
        ids_c, cpart_c, mask_c, freq, co,
        B=B, K=K, K_POW2=K_pow2,
    )
    return freq, co


# ── Unified entry point ────────────────────────────────────────────────────

def _struct_features(
    ids: Tensor,
    mask: Tensor,
    counterpart_ids: Tensor,
) -> Tensor:
    """Compute 3 domain-agnostic structural scalars per neighbor position.

    Args:
        ids:              (B, K) neighbor node IDs, -1 for padding
        mask:             (B, K) bool, True = valid
        counterpart_ids:  (B, K) neighbor IDs of the paired node

    Returns:
        feats: (B, K, 3) — [recency_rank, repeat_freq, co_occur]
    """
    B, K = ids.shape
    device = ids.device

    rank = torch.arange(K, device=device).float() / max(K - 1, 1)
    rank = rank.unsqueeze(0).expand(B, -1) * mask.float()      # (B, K)

    freq, co = _struct_features_triton(ids, mask, counterpart_ids, K)

    return torch.stack([rank, freq, co], dim=-1)                # (B, K, 3)


# ── Mamba v1 building blocks ───────────────────────────────────────────────


class _SelectiveSSM(nn.Module):
    """S6: selective state space model via mamba_ssm CUDA kernel."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: Tensor) -> Tensor:
        xz = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        dt_raw, B_ssm, C_ssm = xz.split([dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.log_A)
        u = x.transpose(1, 2).contiguous()
        delta_no_bias = F.linear(dt_raw, self.dt_proj.weight).transpose(1, 2).contiguous()
        B_t = B_ssm.transpose(1, 2).contiguous()
        C_t = C_ssm.transpose(1, 2).contiguous()
        y = _selective_scan_fn(
            u, delta_no_bias, A, B_t, C_t,
            D=self.D,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return y.transpose(1, 2)


class MambaBlock(nn.Module):
    """Pre-norm Mamba v1 block with residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)
        self.ssm = _SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :residual.shape[1]]
        x_branch = F.silu(x_branch.transpose(1, 2))
        y = self.ssm(x_branch) * F.silu(z)
        return residual + self.out_proj(y)


# ── CrossMamba model ───────────────────────────────────────────────────────


class CrossMamba(TemporalModel):
    """Cross-domain temporal graph model backed by Mamba v1.

    Each neighbor position is encoded as:
      input_proj(time_enc(dt) || struct_proj([rank, freq, co_occur]) [|| edge_proj(edge_feat)])  →  d_model

    Set d_edge=0 for cross-domain zero-shot mode (no edge features).
    Set d_edge=actual_dim for in-domain training (full edge features used).

    Pooling: last valid position in Mamba output (causally richest summary).
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
        d_edge: int = 0,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.d_edge = d_edge

        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, strategy="recency",
                                   include_edge_feat=(d_edge > 0)),
            co_occurrence=False,  # co-occur computed in-model from neighbor IDs
        )

        self.time_enc = FixedCosineTimeEncoder(d_model, learnable=False)

        # 3 structural scalars → d_model//4
        self.struct_proj = nn.Linear(3, d_model // 4, bias=False)
        # optional edge feature projection
        self.edge_proj = nn.Linear(d_edge, d_model // 4, bias=False) if d_edge > 0 else None
        # fused input → d_model
        input_dim = d_model + d_model // 4 + (d_model // 4 if d_edge > 0 else 0)
        self.input_proj  = nn.Linear(input_dim, d_model, bias=False)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.out_norm    = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)
        self.score_scale = nn.Parameter(torch.ones(1))

    # ------------------------------------------------------------------

    def _build_input(
        self,
        nbrs: NeighborData,
        query_time: Tensor,
        counterpart_ids: Tensor,
    ) -> Tensor:
        """Build per-position Mamba input: fused temporal + structural [+ edge] features."""
        dt    = (query_time.unsqueeze(1).float() - nbrs.timestamps.float()).clamp(min=0)
        t_enc = self.time_enc(dt)                                    # (B, K, d_model)

        s     = _struct_features(nbrs.neighbor_ids, nbrs.mask, counterpart_ids)  # (B, K, 3)
        s_enc = self.struct_proj(s)                                  # (B, K, d_model//4)

        parts = [t_enc, s_enc]
        if self.edge_proj is not None:
            parts.append(self.edge_proj(nbrs.edge_feats.float()))    # (B, K, d_model//4)

        seq = self.input_proj(torch.cat(parts, dim=-1))              # (B, K, d_model)
        return seq * nbrs.mask.unsqueeze(-1)                         # zero out padding

    def _encode(
        self,
        nbrs: NeighborData,
        query_time: Tensor,
        counterpart_ids: Tensor,
    ) -> Tensor:
        """Encode a node: build input → Mamba layers → last-valid pooling."""
        seq = self._build_input(nbrs, query_time, counterpart_ids)

        for layer in self.layers:
            seq = layer(seq)

        seq = self.out_norm(seq)
        seq = self.dropout(seq)

        # Last-valid pooling: most recent valid position carries the full causal summary.
        mask_f   = nbrs.mask.float()                                       # (B, K)
        last_idx = (mask_f.cumsum(dim=1) * mask_f).argmax(dim=1)          # (B,)
        has_any  = mask_f.sum(dim=1) > 0                                   # (B,)
        last_emb = seq[torch.arange(seq.size(0), device=seq.device), last_idx]
        return torch.where(has_any.unsqueeze(-1), last_emb,
                           seq.new_zeros(seq.size(0), self.d_model))

    # ------------------------------------------------------------------
    # TemporalModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self._encode(batch.src_neighbors, batch.time,
                               batch.dst_neighbors.neighbor_ids)
        dst_emb = self._encode(batch.dst_neighbors, batch.time,
                               batch.src_neighbors.neighbor_ids)
        neg_emb = self._encode(batch.neg_neighbors, batch.time,
                               batch.src_neighbors.neighbor_ids)

        pos_score = self.score_scale * (src_emb * dst_emb).sum(dim=-1)
        neg_score = self.score_scale * (src_emb * neg_emb).sum(dim=-1)

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss   = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]), labels
        )
        return ModelOutput(loss=loss,
                           pos_score=pos_score.sigmoid(),
                           neg_score=neg_score.sigmoid())

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        src = self._encode(batch.src_neighbors, batch.time,
                           batch.dst_neighbors.neighbor_ids)
        dst = self._encode(batch.dst_neighbors, batch.time,
                           batch.src_neighbors.neighbor_ids)
        neg = self._encode(batch.neg_neighbors, batch.time,
                           batch.src_neighbors.neighbor_ids)
        return EmbeddingBundle(src=src, dst=dst, neg=neg)

    @property
    def supports_independent_encode(self) -> bool:
        return False

    def encode_nodes(self, neighbors, times: Tensor) -> Tensor:
        raise NotImplementedError("CrossMamba requires counterpart neighbors for co-occurrence")

    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        return (self.score_scale * (src_emb * dst_emb).sum(dim=-1)).sigmoid()
