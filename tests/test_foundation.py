"""Unit tests for foundation model components.

Tests cover:
  - InputTokenizer: feature computation, padding, shapes
  - Pretraining heads: MTM/NTP/LP forward + loss
  - FoundationModel: encode, forward, pretrain_forward, EMA
"""
import pytest
import torch

from tgengine.core.batch import NeighborData, PreparedBatch
from tgengine.models.foundation import FoundationModel
from tgengine.nn.input_tokenizer import (
    InputTokenizer,
    TrainableSinusoidalTimeEncoding,
    compute_node_context,
    compute_pair_features,
)
from tgengine.nn.pretraining_heads import (
    EMAEncoder,
    LPHead,
    MTMHead,
    NTPHead,
    block_wise_mask,
)


def _make_neighbor_data(B, K, num_nodes=100, d_edge=8, device="cuda"):
    """Create deterministic NeighborData for testing."""
    torch.manual_seed(42)
    # Most-recent-first timestamps (descending)
    ts = torch.sort(torch.rand(B, K, device=device), descending=True).values
    # Ensure some padding: last position invalid for half the batch
    mask = torch.ones(B, K, dtype=torch.bool, device=device)
    mask[B // 2:, K // 2:] = False  # second half of batch has padding
    # Zero out timestamps for padding positions (avoid inf issues)
    ts = ts * mask.float()
    return NeighborData(
        neighbor_ids=torch.randint(0, num_nodes, (B, K), device=device),
        timestamps=ts,
        edge_feats=torch.randn(B, K, d_edge, device=device),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Input tokenizer components
# ---------------------------------------------------------------------------

def test_trainable_sinusoidal_time_encoding():
    """Time encoding produces correct shape and is differentiable."""
    enc = TrainableSinusoidalTimeEncoding(d_time=16)
    dt = torch.rand(4, 8, device="cuda")
    out = enc(dt)
    assert out.shape == (4, 8, 16)
    assert out.requires_grad  # freqs are trainable
    # Backward works
    out.sum().backward()


def test_compute_node_context_shape():
    """Node context produces 5 features per sample."""
    B, K = 4, 8
    nbr = _make_neighbor_data(B, K)
    query_time = torch.ones(B, device="cuda")
    ctx = compute_node_context(nbr.timestamps, nbr.mask, query_time)
    assert ctx.shape == (B, 5)
    # recent_degree should be <= K
    assert ctx[:, 0].max() <= K
    # All-padding nodes should have zero context
    all_pad = ~nbr.mask.any(dim=1)
    if all_pad.any():
        assert ctx[all_pad].abs().max() == 0


def test_compute_node_context_all_padding():
    """All-padding node produces zero context."""
    B, K = 2, 4
    nbr = _make_neighbor_data(B, K)
    # Make all padding
    nbr.mask[:] = False
    query_time = torch.ones(B, device="cuda")
    ctx = compute_node_context(nbr.timestamps, nbr.mask, query_time)
    assert ctx.abs().max() == 0


def test_compute_pair_features_shape():
    """Pair features produce 2 features per position."""
    B, K = 4, 8
    nbr = _make_neighbor_data(B, K)
    pf = compute_pair_features(nbr.neighbor_ids, nbr.timestamps, nbr.mask)
    assert pf.shape == (B, K, 2)
    # pair_count >= 1 for valid positions (at least self)
    valid = nbr.mask
    assert (pf[..., 0][valid] >= 1).all()
    # Padding positions have zero features
    pad = ~nbr.mask
    if pad.any():
        assert (pf[pad] == 0).all()


def test_input_tokenizer_output_shape():
    """InputTokenizer produces correctly shaped tokens + struct token."""
    d_edge, d_model = 8, 32
    tokenizer = InputTokenizer(d_edge=d_edge, d_model=d_model, d_time=16).cuda()
    B, K = 4, 8
    nbr = _make_neighbor_data(B, K, d_edge=d_edge)
    query_time = torch.ones(B, device="cuda")
    tokens, struct = tokenizer(nbr, query_time)
    assert tokens.shape == (B, K, d_model)
    assert struct.shape == (B, 1, d_model)


def test_input_tokenizer_padding_replaced():
    """Padding positions use learned embedding, not computed features."""
    d_edge, d_model = 8, 16
    tokenizer = InputTokenizer(d_edge=d_edge, d_model=d_model, d_time=8).cuda()
    B, K = 2, 4
    nbr = _make_neighbor_data(B, K, d_edge=d_edge)
    query_time = torch.ones(B, device="cuda")
    tokens, _ = tokenizer(nbr, query_time)
    # All padding tokens in a sample should be identical (same embedding)
    pad_mask = ~nbr.mask
    if pad_mask.any():
        # Get first padding position per sample
        for b in range(B):
            pad_pos = pad_mask[b]
            if pad_pos.any():
                pad_tokens = tokens[b][pad_pos]
                # All padding tokens should be equal
                assert torch.allclose(pad_tokens[0], pad_tokens[1:]), \
                    "padding tokens should be identical"


# ---------------------------------------------------------------------------
# Pretraining heads
# ---------------------------------------------------------------------------

def test_mtm_head_shape():
    """MTM decoder produces correct shape."""
    d_model, d_target = 32, 10
    head = MTMHead(d_model=d_model, d_target=d_target).cuda()
    h = torch.randn(4, 8, d_model, device="cuda")
    out = head(h)
    assert out.shape == (4, 8, d_target)


def test_ntp_head_shape():
    """NTP head pools sequence to time encoding vector."""
    d_model, d_time = 32, 16
    head = NTPHead(d_model=d_model, d_time=d_time).cuda()
    h = torch.randn(4, 8, d_model, device="cuda")
    mask = torch.ones(4, 8, dtype=torch.bool, device="cuda")
    out = head(h, mask)
    assert out.shape == (4, d_time)


def test_lp_head_pool_and_score():
    """LP head pools to vector and scores via dot product."""
    d_model = 32
    head = LPHead(d_model=d_model).cuda()
    h = torch.randn(4, 8, d_model, device="cuda")
    mask = torch.ones(4, 8, dtype=torch.bool, device="cuda")
    vec = head.pool(h, mask)
    assert vec.shape == (4, d_model)
    score = head.score(vec, vec)
    assert score.shape == (4,)


def test_block_wise_mask():
    """Block mask produces contiguous blocks."""
    K = 64
    mask = block_wise_mask(K, mask_ratio=0.15, block_size=4, device="cuda")
    assert mask.dtype == torch.bool
    assert mask.shape == (K,)
    # Should mask roughly 15% (at least 1 block)
    assert mask.sum() >= 4  # at least one block of 4


def test_ema_encoder_update():
    """EMA encoder updates toward online encoder."""
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    online = Dummy().cuda()
    ema = EMAEncoder(online, momentum=0.9).cuda()
    original = ema.encoder.lin.weight.clone()
    # Modify online
    with torch.no_grad():
        online.lin.weight.add_(1.0)
    ema.update(online)
    # EMA should have moved toward online (not equal to original)
    assert not torch.allclose(ema.encoder.lin.weight, original)
    # EMA params should be frozen
    assert not ema.encoder.lin.weight.requires_grad


# ---------------------------------------------------------------------------
# Foundation model
# ---------------------------------------------------------------------------

def test_foundation_model_encode():
    """FoundationModel.encode produces EmbeddingBundle with correct shapes."""
    model = FoundationModel(
        d_edge=8, d_model=32, d_state=8, K=4, d_time=8,
        n_mamba_layers=2, gca_every=1,
    ).cuda()
    B = 4
    nbr = _make_neighbor_data(B, K=4, d_edge=8)
    batch = PreparedBatch(
        src=torch.randint(0, 100, (B,), device="cuda"),
        dst=torch.randint(0, 100, (B,), device="cuda"),
        neg=torch.randint(0, 100, (B,), device="cuda"),
        time=torch.ones(B, device="cuda"),
        src_neighbors=nbr, dst_neighbors=nbr, neg_neighbors=nbr,
    )
    bundle = model.encode(batch)
    assert bundle.src.shape == (B, 32)
    assert bundle.dst.shape == (B, 32)
    assert bundle.neg.shape == (B, 32)


def test_foundation_model_forward():
    """Standard forward produces LP BCE loss."""
    model = FoundationModel(
        d_edge=8, d_model=32, d_state=8, K=4, d_time=8,
        n_mamba_layers=2, gca_every=1,
    ).cuda()
    B = 4
    nbr = _make_neighbor_data(B, K=4, d_edge=8)
    batch = PreparedBatch(
        src=torch.randint(0, 100, (B,), device="cuda"),
        dst=torch.randint(0, 100, (B,), device="cuda"),
        neg=torch.randint(0, 100, (B,), device="cuda"),
        time=torch.ones(B, device="cuda"),
        src_neighbors=nbr, dst_neighbors=nbr, neg_neighbors=nbr,
    )
    out = model(batch)
    assert out.loss.dim() == 0  # scalar
    assert out.pos_score.shape == (B,)
    assert out.neg_score.shape == (B,)
    out.loss.backward()


def test_foundation_model_pretrain_forward():
    """Pretrain forward computes all 3 losses."""
    model = FoundationModel(
        d_edge=8, d_model=32, d_state=8, K=8, d_time=8,
        n_mamba_layers=2, gca_every=1,
    ).cuda()
    model.init_ema(momentum=0.9)
    B = 4
    nbr = _make_neighbor_data(B, K=8, d_edge=8)
    batch = PreparedBatch(
        src=torch.randint(0, 100, (B,), device="cuda"),
        dst=torch.randint(0, 100, (B,), device="cuda"),
        neg=torch.randint(0, 100, (B,), device="cuda"),
        time=torch.ones(B, device="cuda"),
        src_neighbors=nbr, dst_neighbors=nbr, neg_neighbors=nbr,
    )
    result = model.pretrain_forward(batch, mtm_mask_ratio=0.25, mtm_block_size=2)
    for key in ["loss", "mtm_loss", "ntp_loss", "lp_loss"]:
        assert key in result
        assert result[key].dim() == 0
    # All losses should be positive and finite
    for key in ["mtm_loss", "ntp_loss", "lp_loss"]:
        assert result[key].item() > 0
        assert torch.isfinite(result[key])
    # Total = sum
    expected = result["mtm_loss"] + result["ntp_loss"] + result["lp_loss"]
    assert torch.allclose(result["loss"], expected, atol=1e-4)
    # Backward works
    result["loss"].backward()


def test_foundation_model_ema_updates():
    """EMA encoder updates after update_ema() call."""
    model = FoundationModel(
        d_edge=8, d_model=32, d_state=8, K=4, d_time=8,
        n_mamba_layers=2, gca_every=1,
    ).cuda()
    model.init_ema(momentum=0.9)
    # Get EMA weight before
    ema_w_before = model._ema.encoder.input_tokenizer.edge_proj.weight.clone()
    # Modify online model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    model.update_ema()
    # EMA should have changed
    ema_w_after = model._ema.encoder.input_tokenizer.edge_proj.weight
    assert not torch.allclose(ema_w_before, ema_w_after)
