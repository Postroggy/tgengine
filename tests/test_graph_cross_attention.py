"""Tests for GraphCrossAttention."""

import pytest
import torch

from tgengine.nn.graph_cross_attention import GraphCrossAttention


def test_output_shape():
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 8)
    out = gca(q, kv)
    assert out.shape == (2, 5, 16)


def test_kv_none_is_noop():
    """When no structural stream is provided, GCA returns the query
    unchanged — so a model can wire GCA conditionally."""
    gca = GraphCrossAttention(d_model=16, n_heads=4)
    q = torch.randn(2, 5, 16)
    out = gca(q, None)
    assert torch.equal(out, q)


def test_residual_added_to_query():
    """Output must differ from input (residual + attention contribution),
    but stay the same shape."""
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    gca.eval()
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 8)
    out = gca(q, kv)
    assert out.shape == q.shape
    # Not identical (attention injects signal)
    assert not torch.allclose(out, q, atol=1e-6)


def test_indivisible_heads_raises():
    with pytest.raises(ValueError):
        GraphCrossAttention(d_model=18, n_heads=4)  # 18 % 4 != 0


def test_kv_mask_hides_tokens():
    """Masked structural tokens must not contribute to the output.

    Verifies the mask is actually applied: attend to a single kv token
    (mask only position 0 valid) and check the result is a function of
    just that one token's value — changing the *masked* tokens must not
    change the output.
    """
    torch.manual_seed(0)
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    gca.eval()
    q = torch.randn(2, 4, 16)
    kv = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 0] = True  # only position 0 attended

    out_a = gca(q, kv, kv_mask=mask)

    # Corrupt the masked positions (1..5) — must not affect output since
    # they're masked out of the softmax.
    kv_corrupted = kv.clone()
    kv_corrupted[:, 1:, :] = torch.randn(2, 5, 8) * 1e3  # huge corruption
    out_b = gca(q, kv_corrupted, kv_mask=mask)

    assert torch.allclose(out_a, out_b, atol=1e-5), \
        "masked kv tokens leaked into output — mask not applied"


def test_all_invalid_kv_row_no_nan():
    """A batch row with all kv masked out must not produce NaN (falls back
    to attending everywhere, degenerate but finite)."""
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    gca.eval()
    q = torch.randn(2, 4, 16)
    kv = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, :] = False  # row 1 fully invalid
    out = gca(q, kv, kv_mask=mask)
    assert torch.isfinite(out).all()


def test_structure_modulates_query():
    """The structural stream must actually change the output: same query,
    different kv → different output."""
    torch.manual_seed(1)
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    gca.eval()
    q = torch.randn(2, 4, 16)
    kv_a = torch.randn(2, 6, 8)
    kv_b = torch.randn(2, 6, 8)
    out_a = gca(q, kv_a)
    out_b = gca(q, kv_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_different_queries_different_outputs():
    """Same kv, different query → different output (query drives attention)."""
    torch.manual_seed(2)
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    gca.eval()
    kv = torch.randn(2, 6, 8)
    q_a = torch.randn(2, 4, 16)
    q_b = torch.randn(2, 4, 16)
    out_a = gca(q_a, kv)
    out_b = gca(q_b, kv)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_backward_grad_flow():
    gca = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4)
    q = torch.randn(2, 4, 16, requires_grad=True)
    kv = torch.randn(2, 6, 8, requires_grad=True)
    out = gca(q, kv)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert kv.grad is not None and torch.isfinite(kv.grad).all()
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        proj = getattr(gca, name)
        assert proj.weight.grad is not None


def test_ffn_option_changes_capacity():
    """ff_mult>0 adds a feedforward; params differ and output still valid."""
    gca_no_ff = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4, ff_mult=0)
    gca_ff = GraphCrossAttention(d_model=16, d_kv=8, n_heads=4, ff_mult=4)
    n1 = sum(p.numel() for p in gca_no_ff.parameters())
    n2 = sum(p.numel() for p in gca_ff.parameters())
    assert n2 > n1
    q = torch.randn(2, 4, 16)
    kv = torch.randn(2, 6, 8)
    out = gca_ff(q, kv)
    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()


def test_runs_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    gca = GraphCrossAttention(d_model=32, d_kv=16, n_heads=4).to("cuda")
    q = torch.randn(4, 8, 32, device="cuda")
    kv = torch.randn(4, 10, 16, device="cuda")
    mask = torch.ones(4, 10, dtype=torch.bool, device="cuda")
    out = gca(q, kv, kv_mask=mask)
    assert out.shape == (4, 8, 32)
    assert out.is_cuda
