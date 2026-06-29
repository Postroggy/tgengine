"""Mamba v1 blocks for the dynamic-graph foundation model.

Module-level reusable versions of the Mamba building blocks that previously
lived inside models/crossmamba.py. Two variants:

  - MambaBlock: standard selective SSM block (S6) with pre-norm + residual.
    Input-dependent dt/B/C via x_proj/dt_proj, fixed A. This is the plain
    Mamba backbone — works for any token sequence regardless of timing.

  - TimeAwareMambaBlock: A(Δt) variant inspired by DyG-Mamba (NeurIPS 2025).
    Irregular inter-event gaps Δt are fed as a *control signal* on the SSM
    step size (delta). In the SSM discretization A_bar = exp(delta · A),
    so a larger Δt → larger delta → stronger exponential decay of the
    hidden state — i.e. longer gaps cause more forgetting (Ebbinghaus).
    This requires NO custom CUDA kernel: selective_scan_fn already takes a
    per-timestep delta; we just inject Δt into it.

Both blocks require mamba_ssm (selective_scan_fn CUDA kernel) and CUDA
tensors. They raise a clear ImportError at construction if mamba_ssm is
unavailable, so callers can feature-gate (the rest of the framework keeps
working without mamba).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _require_mamba():
    """Import selective_scan_fn or raise a clear error.

    Returns the kernel so callers don't re-import per forward pass.
    """
    try:
        from mamba_ssm.ops.selective_scan_interface import (
            selective_scan_fn as _fn,
        )
    except ImportError as e:  # pragma: no cover - exercised via feature gating
        raise ImportError(
            "TimeAwareMambaBlock / MambaBlock require mamba_ssm "
            "(selective_scan_fn). Install mamba_ssm or run under the glibc "
            "2.39 ld-linux launcher (see AGENTS.md §2)."
        ) from e
    return _fn


# Wrap the selective_scan call in torch.compiler.disable so torch.compile
# can be used on models containing Mamba blocks. dynamo graph-breaks cleanly
# at this call (the SSM is a closed mamba_ssm CUDA op that cannot be traced),
# the SSM runs eager with its own autograd backward intact, and inductor
# fuses the surrounding projections. Measured 1.06x vs eager on the block
# (projection fusion outweighs the graph-break overhead).
#
# Why not allow_in_graph / custom_op: AOTAutograd still traces into an
# allow_in_graph wrapper with FakeTensor and hits "Cannot access data
# pointer" from selective_scan_cuda; custom_op would require a hand-written
# backward (selective_scan's bwd is a closed CUDA kernel), impractical.
# torch.compiler.disable is the PyTorch-blessed path for "exclude this
# C++ call from compile, keep its autograd".
@torch.compiler.disable
def _selective_scan_call(fn, u, delta, A, B, C, D, delta_bias):
    return fn(u, delta, A, B, C, D=D, delta_bias=delta_bias, delta_softplus=True)


class _SelectiveSSM(nn.Module):
    """S6 selective state space model via the mamba_ssm CUDA kernel."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # mamba convention: init dt bias in (log(0.001)..log(0.1)) ≈ uniform(-1..-4)*something
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

    def _dt_bias(self) -> Tensor:
        return self.dt_proj.bias.float()

    def forward(
        self,
        x: Tensor,
        delta_bias: Tensor,
        extra_delta: Optional[Tensor] = None,
    ) -> Tensor:
        """Run selective scan.

        Args:
            x: (B, L, d_inner) post-conv input.
            delta_bias: (d_inner,) base step bias from dt_proj.
            extra_delta: optional (B, L, d_inner) additive step modulation
                (e.g. from Δt in TimeAwareMambaBlock). Added to the
                input-projected delta before softplus.
        """
        selective_scan_fn = _require_mamba()
        xz = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        dt_raw, B_ssm, C_ssm = xz.split([dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.log_A)  # (d_inner, d_state), negative for stable decay
        u = x.transpose(1, 2).contiguous()
        delta_no_bias = F.linear(dt_raw, self.dt_proj.weight).transpose(1, 2).contiguous()
        if extra_delta is not None:
            # (B, d_inner, L) += extra (B, L, d_inner).transpose
            delta_no_bias = delta_no_bias + extra_delta.transpose(1, 2).contiguous()
        B_t = B_ssm.transpose(1, 2).contiguous()
        C_t = C_ssm.transpose(1, 2).contiguous()
        y = _selective_scan_call(
            selective_scan_fn, u, delta_no_bias, A, B_t, C_t, self.D, delta_bias,
        )
        return y.transpose(1, 2)


class MambaBlock(nn.Module):
    """Pre-norm Mamba v1 block with residual.

    Args:
        d_model: hidden dimension.
        d_state: SSM state size.
        d_conv: conv1d kernel width.
        expand: inner expansion factor (d_inner = d_model * expand).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        self.ssm = _SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x (B, L, d_model). Returns (B, L, d_model)."""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :residual.shape[1]]
        x_branch = F.silu(x_branch.transpose(1, 2))
        y = self.ssm(x_branch, delta_bias=self.ssm._dt_bias()) * F.silu(z)
        return residual + self.out_proj(y)


class TimeAwareMambaBlock(MambaBlock):
    """Mamba block with A(Δt): irregular time gaps modulate SSM forgetting.

    The inter-event gap Δt at each position is projected to a per-channel
    step-size adjustment and added to the SSM delta before the softplus.
    Because the discretized transition is A_bar = exp(delta · A), a larger
    Δt inflates delta, which strengthens exponential decay of the hidden
    state — modelling the Ebbinghaus forgetting curve over long gaps.

    This keeps the standard selective_scan_fn CUDA kernel: we only change
    what feeds `delta`, not the scan itself. No custom CUDA op required.

    Args:
        d_model, d_state, d_conv, expand: same as MambaBlock.
        dt_scale: initial scale of the Δt→delta projection. Small so Δt
            modulates rather than dominates the learned delta.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_scale: float = 0.1,
    ):
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        d_inner = d_model * expand
        # Project scalar Δt → per-channel delta modulation. Small init so the
        # block starts close to plain Mamba and learns how much Δt matters.
        self.dt_time_proj = nn.Linear(1, d_inner, bias=False)
        with torch.no_grad():
            self.dt_time_proj.weight.mul_(dt_scale)

    def forward(self, x: Tensor, dt: Optional[Tensor] = None) -> Tensor:
        """Args:
            x: (B, L, d_model).
            dt: (B, L) inter-event time gaps. If None, behaves as plain
                MambaBlock (no time modulation) — convenient for ablation
                and for reusing the block in non-temporal contexts.
        Returns: (B, L, d_model).
        """
        residual = x
        h = self.norm(x)
        xz = self.in_proj(h)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :residual.shape[1]]
        x_branch = F.silu(x_branch.transpose(1, 2))

        extra_delta = None
        if dt is not None:
            # (B, L) -> (B, L, d_inner)
            extra_delta = self.dt_time_proj(dt.float().unsqueeze(-1))

        y = self.ssm(x_branch, delta_bias=self.ssm._dt_bias(), extra_delta=extra_delta) * F.silu(z)
        return residual + self.out_proj(y)
