"""Tests for CrossMamba model."""

import pytest
import torch

from tgengine import CrossMamba
from tgengine.core.batch import NeighborData, PreparedBatch


def _make_neighbor_data(B: int, K: int, d_edge: int = 0) -> NeighborData:
    """Helper: create dummy NeighborData with some valid and some padded positions."""
    # First K//2 positions valid, rest padding
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[:, : K // 2 + 1] = True
    return NeighborData(
        neighbor_ids=torch.randint(0, 100, (B, K)),
        timestamps=torch.rand(B, K) * 100.0,
        edge_feats=torch.zeros(B, K, max(d_edge, 1)),  # ignored by CrossMamba
        mask=mask,
    )


def _make_batch(B: int = 4, K: int = 8) -> PreparedBatch:
    time = torch.rand(B) * 200.0 + 100.0  # query times > neighbor times
    return PreparedBatch(
        src=torch.randint(0, 50, (B,)),
        dst=torch.randint(0, 50, (B,)),
        neg=torch.randint(0, 50, (B,)),
        time=time,
        src_neighbors=_make_neighbor_data(B, K),
        dst_neighbors=_make_neighbor_data(B, K),
        neg_neighbors=_make_neighbor_data(B, K),
    )


class TestCrossMambaForward:
    def test_output_shapes(self):
        model = CrossMamba(d_model=32, K=8, n_layers=2)
        model.eval()
        batch = _make_batch(B=4, K=8)
        out = model(batch)
        assert out.loss.ndim == 0
        assert out.pos_score.shape == (4,)
        assert out.neg_score.shape == (4,)

    def test_loss_is_finite(self):
        model = CrossMamba(d_model=32, K=8)
        batch = _make_batch(B=4, K=8)
        out = model(batch)
        assert torch.isfinite(out.loss)

    def test_scores_in_unit_interval(self):
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        batch = _make_batch(B=4, K=8)
        out = model(batch)
        assert (out.pos_score >= 0).all() and (out.pos_score <= 1).all()
        assert (out.neg_score >= 0).all() and (out.neg_score <= 1).all()

    def test_backward(self):
        model = CrossMamba(d_model=32, K=8)
        batch = _make_batch(B=4, K=8)
        out = model(batch)
        out.loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"

    def test_no_edge_features_used(self):
        """CrossMamba output must not change when edge_feats change."""
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        batch = _make_batch(B=4, K=8)
        out1 = model(batch)

        # Overwrite all edge_feats with random noise
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.edge_feats = torch.randn_like(nbrs.edge_feats)
        out2 = model(batch)

        torch.testing.assert_close(out1.pos_score, out2.pos_score)
        torch.testing.assert_close(out1.neg_score, out2.neg_score)

    def test_different_d_model_sizes(self):
        for d in (16, 64, 128):
            model = CrossMamba(d_model=d, K=8)
            batch = _make_batch(B=2, K=8)
            out = model(batch)
            assert torch.isfinite(out.loss)

    def test_single_layer(self):
        model = CrossMamba(d_model=32, K=8, n_layers=1)
        out = model(_make_batch(B=2, K=8))
        assert torch.isfinite(out.loss)


class TestCrossMambaEncode:
    def test_encode_returns_bundle(self):
        from tgengine.models.base import EmbeddingBundle
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        batch = _make_batch(B=4, K=8)
        bundle = model.encode(batch)
        assert isinstance(bundle, EmbeddingBundle)
        assert bundle.src.shape == (4, 32)
        assert bundle.dst.shape == (4, 32)
        assert bundle.neg.shape == (4, 32)

    def test_encode_nodes_shape(self):
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        B = 3
        nbrs = _make_neighbor_data(B, 8)
        times = torch.rand(B) * 200.0 + 100.0
        emb = model.encode_nodes(nbrs, times)
        assert emb.shape == (B, 32)

    def test_score_pairs_range(self):
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        src = torch.randn(4, 32)
        dst = torch.randn(4, 32)
        scores = model.score_pairs(src, dst)
        assert scores.shape == (4,)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestCrossMambaGatherSpec:
    def test_gather_spec_has_neighbors(self):
        model = CrossMamba(d_model=32, K=16)
        assert model.gather_spec.neighbors is not None
        assert model.gather_spec.neighbors.k == 16

    def test_gather_spec_no_co_occurrence(self):
        model = CrossMamba(d_model=32, K=8)
        assert not model.gather_spec.co_occurrence


class TestCrossMambaMasking:
    def test_all_valid_positions(self):
        """All K positions valid → should still produce finite output."""
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        B, K = 4, 8
        batch = _make_batch(B, K)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.mask = torch.ones(B, K, dtype=torch.bool)
        out = model(batch)
        assert torch.isfinite(out.loss)

    def test_all_padding_positions(self):
        """All K positions padded → mean pool denominator clamped to 1, no NaN."""
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        B, K = 4, 8
        batch = _make_batch(B, K)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.mask = torch.zeros(B, K, dtype=torch.bool)
        out = model(batch)
        assert torch.isfinite(out.loss)

    def test_single_valid_position(self):
        model = CrossMamba(d_model=32, K=8)
        model.eval()
        B, K = 4, 8
        batch = _make_batch(B, K)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.mask = torch.zeros(B, K, dtype=torch.bool)
            nbrs.mask[:, 0] = True
        out = model(batch)
        assert torch.isfinite(out.loss)
