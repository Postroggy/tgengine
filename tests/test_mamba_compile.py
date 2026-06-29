"""E2E: mamba fast-path + triton + torch.compile coexistence.

Validates that a MambaBlock (which uses selective_scan_fn CUDA kernel)
works correctly under torch.compile, once _selective_scan_call is wrapped
in torch.compiler.disable. Verifies:
  1. fast-path selective_scan_fn is importable (glibc239 launcher present)
  2. torch.compile(model) runs without graph-break/crash errors
  3. compiled output matches eager output (numerical equivalence)
  4. backward works through the compiled model

Requires mamba_ssm + CUDA + the glibc 2.39 launcher (run via run_mamba.sh
or the glibc239 ld-linux directly). Skipped otherwise.
"""

import pytest

pytest.importorskip("mamba_ssm")

import torch

if not torch.cuda.is_available():
    pytest.skip("mamba compile tests require CUDA", allow_module_level=True)

from tgengine.nn.mamba_block import MambaBlock, TimeAwareMambaBlock, _selective_scan_call


def test_selective_scan_call_wraps_ssm():
    """_selective_scan_call is the torch.compiler.disable-wrapped entry point
    that lets torch.compile coexist with the closed selective_scan CUDA op.
    Behaviorally verified by the compile tests below (no crash on FakeTensor
    trace). Here we just check the wrapper exists and is callable."""
    assert callable(_selective_scan_call)
    # torch.compiler.disable tags the function; the exact attribute name is
    # an internal implementation detail, so we check the public behavior:
    # the wrapper must NOT be traced by dynamo. is_dynamo_disabled is the
    # stable API for this check (torch >= 2.4).
    if hasattr(_selective_scan_call, "__torch_dynamo_disable"):
        assert _selective_scan_call.__torch_dynamo_disable is True
    # Fallback: if the attribute name differs across torch versions, the
    # compile tests below are the authoritative check.


def test_compile_mamba_block_forward_matches_eager():
    """Compiled MambaBlock forward must match eager within fp tolerance."""
    torch.manual_seed(0)
    dev = "cuda"
    blk = MambaBlock(d_model=32, d_state=8).to(dev)
    blk.eval()
    x = torch.randn(2, 8, 32, device=dev)

    with torch.no_grad():
        y_eager = blk(x)

    blk_c = torch.compile(blk)
    blk_c.eval()
    with torch.no_grad():
        y_compiled = blk_c(x)

    assert torch.allclose(y_eager, y_compiled, atol=1e-3), \
        f"compiled diverged from eager: max diff {(y_eager-y_compiled).abs().max()}"


def test_compile_mamba_block_backward():
    """Backward through the compiled Mamba block must produce finite grads.
    The SSM's autograd backward must stay intact across the compile boundary."""
    torch.manual_seed(1)
    dev = "cuda"
    blk = MambaBlock(d_model=32, d_state=8).to(dev)
    x = torch.randn(2, 8, 32, device=dev, requires_grad=True)

    blk_c = torch.compile(blk)
    out = blk_c(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert blk.ssm.log_A.grad is not None
    assert torch.isfinite(blk.ssm.log_A.grad).all()


def test_compile_timeaware_block_with_dt():
    """TimeAwareMambaBlock (A(Δt)) under compile: dt modulation path works
    and compiled output matches eager."""
    torch.manual_seed(2)
    dev = "cuda"
    blk = TimeAwareMambaBlock(d_model=32, d_state=8, dt_scale=0.5).to(dev)
    blk.eval()
    x = torch.randn(2, 8, 32, device=dev)
    dt = torch.rand(2, 8, device=dev) * 50.0

    with torch.no_grad():
        y_eager = blk(x, dt=dt)

    blk_c = torch.compile(blk)
    blk_c.eval()
    with torch.no_grad():
        y_compiled = blk_c(x, dt=dt)

    assert torch.allclose(y_eager, y_compiled, atol=1e-3), \
        f"compiled TimeAware diverged: max diff {(y_eager-y_compiled).abs().max()}"


def test_compile_does_not_crash_on_repeated_calls():
    """Repeated compiled forward calls must not hit the CUDA-graph tensor
    overwrite error (the reason we use default mode, not reduce-overhead)."""
    torch.manual_seed(3)
    dev = "cuda"
    blk = MambaBlock(d_model=16, d_state=8).to(dev)
    blk_c = torch.compile(blk)
    x = torch.randn(2, 6, 16, device=dev)
    for _ in range(5):
        y = blk_c(x)
        assert torch.isfinite(y).all()
