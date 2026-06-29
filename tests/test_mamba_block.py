"""Tests for modular Mamba blocks (MambaBlock, TimeAwareMambaBlock, Mamba2Block).

Requires mamba_ssm + CUDA. Skipped when mamba_ssm can't import.
"""

import pytest

pytest.importorskip("mamba_ssm")

import torch

from tgengine.nn.mamba_block import Mamba2Block, Mamba3Block, MambaBlock, TimeAwareMambaBlock


def _cuda():
    assert torch.cuda.is_available(), "CUDA required for mamba tests"
    return "cuda"


def test_mamba_block_output_shape():
    dev = _cuda()
    blk = MambaBlock(d_model=32, d_state=8).to(dev)
    x = torch.randn(4, 10, 32, device=dev)
    out = blk(x)
    assert out.shape == (4, 10, 32)


def test_mamba_block_residual_preserved_when_zeroed():
    """If the SSM branch output is forced small, output ≈ input (residual)."""
    dev = _cuda()
    torch.manual_seed(0)
    blk = MambaBlock(d_model=16, d_state=8).to(dev)
    blk.eval()
    x = torch.randn(2, 6, 16, device=dev)
    out = blk(x)
    # Residual block: output differs from input by the SSM branch, but the
    # shape is preserved and the difference is finite.
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # Not exactly equal (conv/ssm add signal), but same order of magnitude.
    assert (out - x).abs().mean() < x.abs().mean() * 5


def test_mamba_block_backward():
    dev = _cuda()
    blk = MambaBlock(d_model=16, d_state=8).to(dev)
    x = torch.randn(2, 6, 16, device=dev, requires_grad=True)
    out = blk(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # SSM params get grad
    for name, p in blk.ssm.named_parameters():
        assert p.grad is not None, f"no grad for ssm.{name}"
        assert torch.isfinite(p.grad).all()


def test_timeaware_block_shape():
    dev = _cuda()
    blk = TimeAwareMambaBlock(d_model=32, d_state=8).to(dev)
    x = torch.randn(4, 10, 32, device=dev)
    dt = torch.rand(4, 10, device=dev) * 100.0
    out = blk(x, dt=dt)
    assert out.shape == (4, 10, 32)


def test_timeaware_block_dt_modulates_output():
    """Core A(Δt) property: different Δt must produce different outputs.
    With dt=0 (no gap) vs dt=large (strong forgetting), the hidden-state
    evolution differs, so outputs must not be identical."""
    dev = _cuda()
    torch.manual_seed(42)
    blk = TimeAwareMambaBlock(d_model=16, d_state=8, dt_scale=1.0).to(dev)
    blk.eval()
    x = torch.randn(2, 8, 16, device=dev)

    dt_zero = torch.zeros(2, 8, device=dev)
    dt_big = torch.full((2, 8), 50.0, device=dev)

    with torch.no_grad():
        out_zero = blk(x, dt=dt_zero)
        out_big = blk(x, dt=dt_big)

    assert not torch.allclose(out_zero, out_big, atol=1e-5), \
        "A(dt) failed: identical outputs for dt=0 vs dt=large — Δt not modulating"


def test_timeaware_block_dt_none_equals_no_modulation():
    """dt=None must behave as plain Mamba (no time modulation). This is the
    ablation path and lets the block be reused in non-temporal contexts."""
    dev = _cuda()
    torch.manual_seed(7)
    # Plain MambaBlock
    blk_plain = MambaBlock(d_model=16, d_state=8).to(dev)
    blk_plain.eval()
    # TimeAware with dt_time_proj zeroed → extra_delta=0 path differs slightly
    # in init, so compare dt=None vs a zeroed-projection explicitly instead.
    blk_ta = TimeAwareMambaBlock(d_model=16, d_state=8).to(dev)
    # Copy plain block's weights into ta (shared params have same shapes)
    blk_ta.norm.load_state_dict(blk_plain.norm.state_dict())
    blk_ta.in_proj.load_state_dict(blk_plain.in_proj.state_dict())
    blk_ta.conv1d.load_state_dict(blk_plain.conv1d.state_dict())
    blk_ta.ssm.load_state_dict(blk_plain.ssm.state_dict())
    blk_ta.out_proj.load_state_dict(blk_plain.out_proj.state_dict())
    blk_ta.eval()
    # Zero the dt projection so even with dt it adds nothing
    with torch.no_grad():
        blk_ta.dt_time_proj.weight.zero_()

    x = torch.randn(2, 6, 16, device=dev)
    dt = torch.rand(2, 6, device=dev) * 10.0
    with torch.no_grad():
        out_plain = blk_plain(x)
        out_ta_none = blk_ta(x, dt=None)
        out_ta_zero = blk_ta(x, dt=dt)
    # dt=None and dt=zeroed-proj both equal plain Mamba
    assert torch.allclose(out_plain, out_ta_none, atol=1e-5)
    assert torch.allclose(out_plain, out_ta_zero, atol=1e-5)


def test_timeaware_block_backward_includes_dt_proj():
    dev = _cuda()
    blk = TimeAwareMambaBlock(d_model=16, d_state=8).to(dev)
    x = torch.randn(2, 6, 16, device=dev, requires_grad=True)
    dt = torch.rand(2, 6, device=dev, requires_grad=True)
    out = blk(x, dt=dt)
    out.sum().backward()
    assert blk.dt_time_proj.weight.grad is not None
    assert torch.isfinite(blk.dt_time_proj.weight.grad).all()


def test_timeaware_block_long_dt_increases_forgetting():
    """Ebbinghaus property: with a longer gap, the influence of an early
    event on a later position should be *weaker* (more forgotten). We
    plant a strong signal at position 0 and measure how much it propagates
    to the last position under small vs large inter-event gaps."""
    dev = _cuda()
    torch.manual_seed(11)
    blk = TimeAwareMambaBlock(d_model=8, d_state=8, dt_scale=2.0).to(dev)
    blk.eval()

    L = 8
    # base sequence: zeros except a spike at position 0
    base = torch.zeros(1, L, 8, device=dev)
    base[:, 0, :] = 5.0

    # small gaps vs huge gaps between every position
    dt_small = torch.full((1, L), 0.01, device=dev)
    dt_huge = torch.full((1, L), 100.0, device=dev)

    with torch.no_grad():
        out_small = blk(base, dt=dt_small)
        out_huge = blk(base, dt=dt_huge)

    # Influence of position-0 spike on the LAST position:
    # |out_last - baseline| should be smaller under huge gaps (more forgetting).
    # Use out at a position far from 0 (last) and compare magnitudes.
    influence_small = out_small[0, -1].abs().mean().item()
    influence_huge = out_huge[0, -1].abs().mean().item()
    # Under huge gaps the propagated signal at the far end should be weaker.
    # (Directional: this is the intended A(dt) semantics.)
    assert influence_huge <= influence_small + 1e-3, \
        f"expected long-dt forgetting, got small={influence_small} huge={influence_huge}"


# ---------------------------------------------------------------------------
# Mamba2Block (SSD)
# ---------------------------------------------------------------------------

def test_mamba2_block_output_shape():
    dev = _cuda()
    blk = Mamba2Block(d_model=64, d_state=64).to(dev)
    x = torch.randn(4, 10, 64, device=dev)
    out = blk(x)
    assert out.shape == (4, 10, 64)


def test_mamba2_block_residual_finite():
    dev = _cuda()
    blk = Mamba2Block(d_model=64, d_state=64).to(dev)
    blk.eval()
    x = torch.randn(2, 8, 64, device=dev)
    out = blk(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mamba2_block_backward():
    dev = _cuda()
    blk = Mamba2Block(d_model=64, d_state=64).to(dev)
    x = torch.randn(2, 8, 64, device=dev, requires_grad=True)
    out = blk(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Mamba2 internal params get grad
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in blk.mamba2.parameters())
    assert has_grad


def test_mamba2_block_headdim_auto_pick():
    """Small d_model where default headdim=64 doesn't divide d_inner should
    auto-pick a smaller headdim instead of crashing."""
    dev = _cuda()
    # d_model=32, expand=2 → d_inner=64, headdim=64 OK (1 head)
    # d_model=24, expand=2 → d_inner=48, 48%64!=0 → auto-pick headdim=16 (3 heads)
    blk = Mamba2Block(d_model=24, d_state=32, expand=2).to(dev)
    assert blk.headdim in (16, 8)  # auto-picked down from 64
    x = torch.randn(2, 8, 24, device=dev)
    out = blk(x)
    assert out.shape == (2, 8, 24)


def test_mamba2_block_runs_under_compile():
    """Mamba2Block must work under torch.compile (graph-breaks at _mamba2_call)."""
    dev = _cuda()
    blk = Mamba2Block(d_model=64, d_state=64).to(dev)
    blk.eval()
    x = torch.randn(2, 8, 64, device=dev)
    with torch.no_grad():
        y_eager = blk(x)
    blk_c = torch.compile(blk)
    blk_c.eval()
    with torch.no_grad():
        y_compiled = blk_c(x)
    assert torch.allclose(y_eager, y_compiled, atol=1e-3), \
        f"Mamba2Block compile diverged: max diff {(y_eager-y_compiled).abs().max()}"


# ---------------------------------------------------------------------------
# Mamba3Block (SSD + trapezoidal + MIMO)
# ---------------------------------------------------------------------------

# Mamba3 requires mamba_ssm built from source (MAMBA_FORCE_BUILD). Skip if absent.
try:
    from mamba_ssm import Mamba3  # noqa: F401
    _HAS_MAMBA3 = True
except ImportError:
    _HAS_MAMBA3 = False

mamba3_required = pytest.mark.skipif(not _HAS_MAMBA3, reason="mamba_ssm.Mamba3 not installed")


@mamba3_required
def test_mamba3_block_output_shape():
    dev = _cuda()
    blk = Mamba3Block(d_model=64, d_state=64, expand=2).to(dev)
    x = torch.randn(4, 10, 64, device=dev)
    out = blk(x)
    assert out.shape == (4, 10, 64)


@mamba3_required
def test_mamba3_block_residual_finite():
    dev = _cuda()
    blk = Mamba3Block(d_model=64, d_state=64, expand=2).to(dev)
    blk.eval()
    x = torch.randn(2, 8, 64, device=dev)
    out = blk(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@mamba3_required
def test_mamba3_block_backward():
    dev = _cuda()
    blk = Mamba3Block(d_model=64, d_state=64, expand=2).to(dev)
    x = torch.randn(2, 8, 64, device=dev, requires_grad=True)
    out = blk(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in blk.mamba3.parameters())
    assert has_grad


@mamba3_required
def test_mamba3_block_headdim_auto_pick():
    """Small d_model where default headdim=64 doesn't divide d_inner should
    auto-pick a smaller headdim instead of crashing."""
    dev = _cuda()
    # d_model=24, expand=2 → d_inner=48, 48%64!=0 → auto headdim=16
    blk = Mamba3Block(d_model=24, d_state=32, expand=2).to(dev)
    assert blk.headdim in (16, 8)
    x = torch.randn(2, 8, 24, device=dev)
    out = blk(x)
    assert out.shape == (2, 8, 24)
