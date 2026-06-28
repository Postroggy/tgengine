"""Tests for rotary time encoding."""

import math

import pytest
import torch

from tgengine.nn.rotary_time import RotaryTimeEncoder, apply_rotary


def test_output_shape_feature_mode():
    enc = RotaryTimeEncoder(d_model=16)
    dt = torch.rand(4, 8)
    out = enc(dt, as_feature=True)
    assert out.shape == (4, 8, 16)


def test_output_shape_rotation_mode():
    enc = RotaryTimeEncoder(d_model=16)
    dt = torch.rand(4, 8)
    out = enc(dt, as_feature=False)
    assert out.shape == (4, 8, 8, 2)


def test_odd_d_model_raises():
    with pytest.raises(ValueError):
        RotaryTimeEncoder(d_model=15)


def test_translation_invariance_of_inner_product():
    """Core RoTHP property: <rot(f, dt_a), rot(f, dt_b)> depends only on
    dt_a - dt_b. This is what makes rotary suitable for Hawkes-process
    time encoding (likelihood depends only on gaps)."""
    torch.manual_seed(0)
    d_model = 16
    enc = RotaryTimeEncoder(d_model=d_model)
    B, K = 2, 3
    f = torch.randn(B, K, d_model)

    dt_a = torch.full((B, K), 5.0)
    dt_b = torch.full((B, K), 8.0)
    dt_c = torch.full((B, K), 15.0)
    dt_d = torch.full((B, K), 18.0)  # same gap (dt_d - dt_c == dt_b - dt_a == 3)

    rot_a = enc(dt_a, as_feature=False)
    rot_b = enc(dt_b, as_feature=False)
    rot_c = enc(dt_c, as_feature=False)
    rot_d = enc(dt_d, as_feature=False)

    fa = apply_rotary(f, rot_a)
    fb = apply_rotary(f, rot_b)
    fc = apply_rotary(f, rot_c)
    fd = apply_rotary(f, rot_d)

    ip1 = (fa * fb).sum(dim=-1)
    ip2 = (fc * fd).sum(dim=-1)
    # Same relative gap → same inner product (within numerical tolerance)
    assert torch.allclose(ip1, ip2, atol=1e-4), \
        f"rotary inner product not translation-invariant: {ip1} vs {ip2}"


def test_rotation_preserves_norm():
    """Applying a rotation must not change the L2 norm of features."""
    torch.manual_seed(1)
    d_model = 16
    enc = RotaryTimeEncoder(d_model=d_model)
    f = torch.randn(3, 5, d_model)
    rot = enc(torch.rand(3, 5), as_feature=False)
    out = apply_rotary(f, rot)
    norm_before = f.norm(dim=-1)
    norm_after = out.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_zero_dt_rotation_is_identity():
    """Δt=0 → angle 0 → identity rotation."""
    torch.manual_seed(2)
    d_model = 16
    enc = RotaryTimeEncoder(d_model=d_model)
    f = torch.randn(4, d_model)
    rot = enc(torch.zeros(4), as_feature=False)
    out = apply_rotary(f, rot)
    assert torch.allclose(out, f, atol=1e-6)


def test_large_dt_bounded():
    """Huge Δt must not produce NaN/Inf. cos/sin are inherently bounded in
    [-1, 1], so even unbounded Δt stays finite — no compression needed."""
    enc = RotaryTimeEncoder(d_model=16)
    dt = torch.tensor([1e9, 1e12, -1e9, 0.0])
    out = enc(dt, as_feature=True)
    assert torch.isfinite(out).all()
    assert (out.abs() <= 1.0).all()


def test_dropin_replaces_cosine_encoder_shape():
    """Same call signature as FixedCosineTimeEncoder: forward(dt)->(...,d)."""
    from tgengine.nn import FixedCosineTimeEncoder
    d = 32
    rotary = RotaryTimeEncoder(d_model=d)
    cosine = FixedCosineTimeEncoder(d_model=d)
    dt = torch.rand(8, 20)
    assert rotary(dt).shape == cosine(dt).shape == (8, 20, d)


def test_gradient_flows():
    """Rotary features must support backprop (encoder is param-free but
    rotation still needs grad w.r.t. input features)."""
    d_model = 16
    enc = RotaryTimeEncoder(d_model=d_model)
    f = torch.randn(4, d_model, requires_grad=True)
    rot = enc(torch.rand(4), as_feature=False)
    out = apply_rotary(f, rot)
    out.sum().backward()
    assert f.grad is not None
    assert torch.isfinite(f.grad).all()
