"""Integration test: foundation-model components wired end-to-end.

Validates that the four foundation-model building blocks compose correctly:
    RotaryTimeEncoder  →  TimeAwareMambaBlock  →  GraphCrossAttention
        →  NextNeighborPatchObjective

Uses a shrunk config (d_model=16, K=8) for fast logic validation — this is
NOT an accuracy test. It checks that tensors flow through the full pipeline
with correct shapes, loss is finite and backpropagates to all components.

Requires mamba_ssm + CUDA (TimeAwareMambaBlock). Skipped otherwise.
"""

import pytest

pytest.importorskip("mamba_ssm")

import torch
import torch.nn as nn

from tgengine.nn import (
    GraphCrossAttention,
    NextNeighborPatchObjective,
    RotaryTimeEncoder,
    TimeAwareMambaBlock,
)


def _cuda():
    assert torch.cuda.is_available(), "CUDA required"
    return "cuda"


def test_foundation_components_pipeline_end_to_end():
    """Wire rotary → mamba → gca → patch-objective and run a training step.

    Models a single node's neighbor sequence:
      - neighbor events at times t_0..t_{K-1}, with edge features
      - per-position structural tokens (co-occurrence-like, 3-dim)
      - inter-event gaps Δt feed the A(Δt) Mamba block
      - rotary Δt rotates the input tokens
      - GCA fuses structural tokens into the temporal hidden
      - patch objective trains next-patch neighbor prediction
    """
    dev = _cuda()
    torch.manual_seed(0)

    B, K = 4, 8
    d_model = 16
    d_edge = 8
    d_struct = 3
    patch_size = 2
    vocab = 30  # node vocab
    n_layers = 2

    # --- components ---
    rotary = RotaryTimeEncoder(d_model=d_model).to(dev)
    edge_proj = nn.Linear(d_edge, d_model, bias=False).to(dev)
    struct_proj = nn.Linear(d_struct, d_model, bias=False).to(dev)
    mamba_layers = nn.ModuleList([
        TimeAwareMambaBlock(d_model, d_state=8, expand=2, dt_scale=0.1).to(dev)
        for _ in range(n_layers)
    ]).to(dev)
    gca = GraphCrossAttention(d_model=d_model, d_kv=d_model, n_heads=4, ff_mult=0).to(dev)
    # Mamba outputs per-position hidden; reduce to per-patch by last-token-in-patch pooling.
    objective = NextNeighborPatchObjective(
        d_model=d_model, vocab_size=vocab, patch_size=patch_size
    ).to(dev)

    # --- synthetic inputs (one node's neighbor sequence per batch row) ---
    # Times increasing so Δt > 0 (causal, gaps = inter-event).
    times = torch.linspace(1.0, 100.0, K, device=dev).unsqueeze(0).expand(B, -1).contiguous()
    dt_gaps = torch.cat(
        [torch.zeros(B, 1, device=dev), times[:, 1:] - times[:, :-1]], dim=1
    )  # (B, K) gap to previous event; position 0 gap=0
    edge_feats = torch.randn(B, K, d_edge, device=dev)
    struct_feats = torch.rand(B, K, d_struct, device=dev)  # e.g. [rank, freq, co_occur]
    mask = torch.ones(B, K, dtype=torch.bool, device=dev)
    neighbor_ids = torch.randint(0, vocab, (B, K), device=dev)

    # --- forward ---
    # 1) rotary injects time into edge-proj tokens
    tokens = edge_proj(edge_feats)  # (B, K, d_model)
    rot = rotary(times, as_feature=False)  # (B, K, half, 2)
    tokens = _apply_rotary_safe(tokens, rot)

    # 2) A(Δt) Mamba layers (time gaps modulate forgetting)
    h = tokens
    for layer in mamba_layers:
        h = layer(h, dt=dt_gaps)

    # 3) GCA: temporal hidden (query) fused with structural tokens (kv)
    struct_tokens = struct_proj(struct_feats)  # (B, K, d_model)
    h = gca(h, struct_tokens, kv_mask=mask)

    # 4) reduce per-position → per-patch (last valid token in each patch)
    n_patches = K // patch_size
    # gather last position of each patch as the patch summary (causal: last
    # token in patch i has seen all of patch i).
    patch_idx = torch.arange(n_patches, device=dev) * patch_size + (patch_size - 1)
    patch_hidden = h[:, patch_idx, :]  # (B, n_patches, d_model)

    # 5) next-neighbor-patch objective
    loss = objective(patch_hidden, neighbor_ids, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() > 0

    # 6) backward reaches every component
    loss.backward()
    for name, p in [("edge_proj", edge_proj.weight),
                    ("struct_proj", struct_proj.weight),
                    ("gca.out_proj", gca.out_proj.weight),
                    ("lm_head", objective.lm_head.weight)]:
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    for i, layer in enumerate(mamba_layers):
        assert layer.dt_time_proj.weight.grad is not None, f"no dt_time_proj grad in layer {i}"
        assert layer.ssm.log_A.grad is not None, f"no log_A grad in layer {i}"


def test_foundation_components_pipeline_loss_decreases():
    """A few training steps should reduce next-patch loss — the full
    pipeline is trainable end-to-end (logic check, not accuracy)."""
    dev = _cuda()
    torch.manual_seed(1)

    B, K = 4, 8
    d_model = 16
    d_edge = 8
    d_struct = 3
    patch_size = 2
    vocab = 30
    n_patches = K // patch_size

    rotary = RotaryTimeEncoder(d_model=d_model).to(dev)
    edge_proj = nn.Linear(d_edge, d_model, bias=False).to(dev)
    struct_proj = nn.Linear(d_struct, d_model, bias=False).to(dev)
    mamba_layers = nn.ModuleList([
        TimeAwareMambaBlock(d_model, d_state=8, expand=2).to(dev) for _ in range(2)
    ]).to(dev)
    gca = GraphCrossAttention(d_model=d_model, d_kv=d_model, n_heads=4).to(dev)
    objective = NextNeighborPatchObjective(d_model, vocab, patch_size=patch_size).to(dev)

    params = (
        list(edge_proj.parameters()) + list(struct_proj.parameters())
        + list(mamba_layers.parameters()) + list(gca.parameters())
        + list(objective.parameters())
    )
    opt = torch.optim.Adam(params, lr=5e-3)

    # Fixed synthetic batch (deterministic next-patch signal).
    times = torch.linspace(1.0, 100.0, K, device=dev).unsqueeze(0).expand(B, -1).contiguous()
    dt_gaps = torch.cat([torch.zeros(B, 1, device=dev), times[:, 1:] - times[:, :-1]], dim=1)
    edge_feats = torch.randn(B, K, d_edge, device=dev)
    struct_feats = torch.rand(B, K, d_struct, device=dev)
    mask = torch.ones(B, K, dtype=torch.bool, device=dev)
    # Synthetic target: patch i's next patch has node id = (i+2) % vocab
    neighbor_ids = torch.zeros(B, K, dtype=torch.long, device=dev)
    for p in range(n_patches):
        neighbor_ids[:, p * patch_size:(p + 1) * patch_size] = (p + 2) % vocab

    losses = []
    for _ in range(40):
        tokens = edge_proj(edge_feats)
        rot = rotary(times, as_feature=False)
        tokens = _apply_rotary_safe(tokens, rot)
        h = tokens
        for layer in mamba_layers:
            h = layer(h, dt=dt_gaps)
        h = gca(h, struct_proj(struct_feats), kv_mask=mask)
        patch_hidden = h[:, torch.arange(n_patches, device=dev) * patch_size + (patch_size - 1), :]
        loss = objective(patch_hidden, neighbor_ids, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f"pipeline loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"


def _apply_rotary_safe(tokens: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Apply rotary; tolerant of the (B,K,half,2) rotation layout."""
    from tgengine.nn.rotary_time import apply_rotary
    return apply_rotary(tokens, rot)
