"""Tests for CrossMamba model."""

import pytest

# CrossMamba requires mamba_ssm (selective_scan_fn CUDA kernel). Skip the
# whole module on environments where mamba_ssm cannot import (e.g. GLIBC
# mismatch on the test server) instead of erroring during collection.
pytest.importorskip("mamba_ssm")

import torch

# _struct_features uses a Triton kernel, which only runs on CUDA. Skip on
# CPU-only environments.
if not torch.cuda.is_available():
    pytest.skip("CrossMamba tests require CUDA (Triton kernel)", allow_module_level=True)

from tgengine import CrossMamba
from tgengine.core.batch import NeighborData, PreparedBatch
from tgengine.models.crossmamba import _struct_features

DEV = "cuda"


def _make_neighbor_data(B: int, K: int) -> NeighborData:
    mask = torch.zeros(B, K, dtype=torch.bool, device=DEV)
    mask[:, : K // 2 + 1] = True
    return NeighborData(
        neighbor_ids=torch.randint(0, 100, (B, K), device=DEV),
        timestamps=torch.rand(B, K, device=DEV) * 100.0,
        edge_feats=torch.zeros(B, K, 1, device=DEV),
        mask=mask,
    )


def _make_batch(B: int = 4, K: int = 8) -> PreparedBatch:
    time = torch.rand(B, device=DEV) * 200.0 + 100.0
    return PreparedBatch(
        src=torch.randint(0, 50, (B,), device=DEV),
        dst=torch.randint(0, 50, (B,), device=DEV),
        neg=torch.randint(0, 50, (B,), device=DEV),
        time=time,
        src_neighbors=_make_neighbor_data(B, K),
        dst_neighbors=_make_neighbor_data(B, K),
        neg_neighbors=_make_neighbor_data(B, K),
    )


def _model(**kwargs) -> CrossMamba:
    """Construct a CrossMamba on CUDA (triton kernel requires GPU)."""
    return CrossMamba(**kwargs).to(DEV)


class TestStructFeatures:
    def test_output_shape(self):
        B, K = 4, 8
        ids = torch.randint(0, 20, (B, K), device=DEV)
        mask = torch.ones(B, K, dtype=torch.bool, device=DEV)
        cpart = torch.randint(0, 20, (B, K), device=DEV)
        feats = _struct_features(ids, mask, cpart)
        assert feats.shape == (B, K, 3)

    def test_rank_monotone(self):
        """Recency rank must be non-decreasing left→right."""
        B, K = 2, 8
        ids = torch.randint(0, 20, (B, K), device=DEV)
        mask = torch.ones(B, K, dtype=torch.bool, device=DEV)
        feats = _struct_features(ids, mask, ids)
        rank = feats[..., 0]
        assert (rank[:, 1:] >= rank[:, :-1]).all()

    def test_rank_zeroed_for_padding(self):
        B, K = 2, 8
        ids = torch.randint(0, 20, (B, K), device=DEV)
        mask = torch.zeros(B, K, dtype=torch.bool, device=DEV)
        feats = _struct_features(ids, mask, ids)
        assert (feats[..., 0] == 0).all()

    def test_co_occur_detected(self):
        """If a neighbor ID appears in counterpart window, co_occur must be 1."""
        B, K = 1, 4
        ids   = torch.tensor([[5, 7, 9, 11]], device=DEV)
        mask  = torch.ones(B, K, dtype=torch.bool, device=DEV)
        cpart = torch.tensor([[99, 7, 99, 99]], device=DEV)  # node 7 is shared
        feats = _struct_features(ids, mask, cpart)
        co = feats[0, :, 2]
        assert co[1] == 1.0   # position 1 holds node 7
        assert co[0] == 0.0   # node 5 not in counterpart

    def test_repeat_freq_hub(self):
        """A node appearing 3 times in K=4 should have freq=0.75."""
        B, K = 1, 4
        ids  = torch.tensor([[42, 42, 42, 7]], device=DEV)
        mask = torch.ones(B, K, dtype=torch.bool, device=DEV)
        feats = _struct_features(ids, mask, torch.zeros(B, K, dtype=torch.long, device=DEV))
        freq = feats[0, :, 1]
        assert abs(float(freq[0]) - 0.75) < 1e-5
        assert abs(float(freq[3]) - 0.25) < 1e-5


class TestCrossMambaForward:
    def test_output_shapes(self):
        model = _model(d_model=32, K=8, n_layers=2)
        model.eval()
        out = model(_make_batch(B=4, K=8))
        assert out.loss.ndim == 0
        assert out.pos_score.shape == (4,)
        assert out.neg_score.shape == (4,)

    def test_loss_is_finite(self):
        model = _model(d_model=32, K=8)
        out = model(_make_batch(B=4, K=8))
        assert torch.isfinite(out.loss)

    def test_scores_in_unit_interval(self):
        model = _model(d_model=32, K=8)
        model.eval()
        out = model(_make_batch(B=4, K=8))
        assert (out.pos_score >= 0).all() and (out.pos_score <= 1).all()
        assert (out.neg_score >= 0).all() and (out.neg_score <= 1).all()

    def test_backward(self):
        model = _model(d_model=32, K=8)
        out = model(_make_batch(B=4, K=8))
        out.loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"

    def test_different_d_model_sizes(self):
        for d in (16, 64, 128):
            model = _model(d_model=d, K=8)
            assert torch.isfinite(model(_make_batch(B=2, K=8)).loss)

    def test_single_layer(self):
        model = _model(d_model=32, K=8, n_layers=1)
        assert torch.isfinite(model(_make_batch(B=2, K=8)).loss)


class TestCrossMambaMasking:
    def test_all_valid(self):
        model = _model(d_model=32, K=8)
        model.eval()
        B, K = 4, 8
        batch = _make_batch(B, K)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.mask = torch.ones(B, K, dtype=torch.bool, device=DEV)
        assert torch.isfinite(model(batch).loss)

    def test_all_padding(self):
        """All K positions padded → last-valid fallback to zero emb, no NaN."""
        model = _model(d_model=32, K=8)
        model.eval()
        B, K = 4, 8
        batch = _make_batch(B, K)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.mask = torch.zeros(B, K, dtype=torch.bool, device=DEV)
        assert torch.isfinite(model(batch).loss)

    def test_single_valid(self):
        model = _model(d_model=32, K=8)
        model.eval()
        B, K = 4, 8
        batch = _make_batch(B, K)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.mask = torch.zeros(B, K, dtype=torch.bool, device=DEV)
            nbrs.mask[:, 0] = True
        assert torch.isfinite(model(batch).loss)


class TestCrossMambaCoOccur:
    def test_co_occur_affects_output(self):
        """Co-occur = all matches vs no matches must produce different embeddings."""
        torch.manual_seed(42)
        model = _model(d_model=32, K=8)
        model.eval()
        batch = _make_batch(B=2, K=8)
        src_ids = batch.src_neighbors.neighbor_ids  # (2, 8)

        with torch.no_grad():
            # counterpart = exact same IDs → co_occur = 1 everywhere (valid positions)
            emb_full_co = model._encode(batch.src_neighbors, batch.time, src_ids)
            # counterpart = completely out-of-range IDs → co_occur = 0 everywhere
            no_co_ids = src_ids + 10000
            emb_no_co = model._encode(batch.src_neighbors, batch.time, no_co_ids)

        assert not torch.allclose(emb_full_co, emb_no_co)

    def test_edge_feats_not_used(self):
        """Changing edge_feats should not affect output (model ignores them)."""
        model = _model(d_model=32, K=8)
        model.eval()
        batch = _make_batch(B=2, K=8)
        out1 = model(batch)
        for nbrs in [batch.src_neighbors, batch.dst_neighbors, batch.neg_neighbors]:
            nbrs.edge_feats = torch.randn_like(nbrs.edge_feats)
        out2 = model(batch)
        torch.testing.assert_close(out1.pos_score, out2.pos_score)
        torch.testing.assert_close(out1.neg_score, out2.neg_score)


class TestCrossMambaGatherSpec:
    def test_gather_spec(self):
        model = _model(d_model=32, K=16)
        assert model.gather_spec.neighbors is not None
        assert model.gather_spec.neighbors.k == 16
        assert not model.gather_spec.co_occurrence

    def test_supports_independent_encode_false(self):
        model = _model(d_model=32, K=8)
        assert not model.supports_independent_encode
