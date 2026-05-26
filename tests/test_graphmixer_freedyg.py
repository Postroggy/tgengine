"""Tests for GraphMixer and FreeDyG models.

Covers:
  - Instantiation and gather_spec
  - Forward pass output shapes
  - Gradient flow (all parameters receive gradients)
  - No-node-features variant
  - With node_raw_features (node encoder enabled)
  - Overfit test (loss decreases on a fixed tiny batch)
  - Full DataPipeline integration
  - APEval integration
"""

import torch
import pytest

from tgengine.core.batch import NeighborData, PreparedBatch, RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import APEval
from tgengine.models.graphmixer import GraphMixer
from tgengine.models.freedyg import FreeDyG
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nbrs(B: int, K: int, d: int) -> NeighborData:
    mask = torch.rand(B, K) > 0.2
    # Ensure at least one valid per row
    mask[:, 0] = True
    ids = torch.where(mask, torch.randint(1, 50, (B, K), dtype=torch.int32),
                      torch.full((B, K), -1, dtype=torch.int32))
    return NeighborData(
        neighbor_ids=ids,
        timestamps=torch.rand(B, K, dtype=torch.float64) * 50.0,
        edge_feats=torch.randn(B, K, d),
        mask=mask,
    )


def _make_batch(B: int, K: int, d: int) -> PreparedBatch:
    return PreparedBatch(
        src=torch.randint(1, 50, (B,)),
        dst=torch.randint(1, 50, (B,)),
        neg=torch.randint(1, 50, (B,)),
        time=torch.rand(B, dtype=torch.float64) * 100 + 200,
        src_neighbors=_make_nbrs(B, K, d),
        dst_neighbors=_make_nbrs(B, K, d),
        neg_neighbors=_make_nbrs(B, K, d),
    )


def _node_features(num_nodes: int, d_node: int) -> torch.Tensor:
    return torch.randn(num_nodes, d_node)


# ===========================================================================
# GraphMixer tests
# ===========================================================================

class TestGraphMixer:

    def test_instantiation_no_node_feat(self):
        model = GraphMixer(d_model=32, d_edge=16, d_time=8, K=8)
        assert model.gather_spec.neighbors.k == 8
        assert not model.gather_spec.co_occurrence
        assert model.supports_independent_encode
        assert model.node_raw_features is None

    def test_instantiation_with_node_feat(self):
        nf = _node_features(50, 12)
        model = GraphMixer(d_model=32, d_edge=16, d_time=8, K=8, node_raw_features=nf)
        assert model.node_raw_features is not None
        assert model.d_node == 12

    def test_forward_shapes(self):
        B, K, d = 8, 8, 16
        model = GraphMixer(d_model=32, d_edge=d, d_time=8, K=K, num_layers=1)
        batch = _make_batch(B, K, d)
        out = model(batch)
        assert out.loss.shape == ()
        assert out.pos_score.shape == (B,)
        assert out.neg_score.shape == (B,)

    def test_forward_shapes_with_node_feat(self):
        B, K, d = 8, 8, 16
        nf = _node_features(100, 12)
        model = GraphMixer(d_model=32, d_edge=d, d_time=8, K=K, num_layers=1,
                           node_raw_features=nf)
        batch = _make_batch(B, K, d)
        out = model(batch)
        assert out.loss.shape == ()
        assert out.pos_score.shape == (B,)

    def test_gradient_flow(self):
        B, K, d = 4, 8, 16
        model = GraphMixer(d_model=32, d_edge=d, d_time=8, K=K, num_layers=1)
        batch = _make_batch(B, K, d)
        out = model(batch)
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        none_grads = [p for p, g in zip(model.parameters(), grads) if g is None and p.requires_grad]
        assert len(none_grads) == 0, f"{len(none_grads)} params with None gradient"

    def test_node_features_frozen(self):
        """Node raw features should not be trainable."""
        nf = _node_features(50, 12)
        model = GraphMixer(d_model=32, d_edge=16, d_time=8, K=8, node_raw_features=nf)
        for name, buf in model.named_buffers():
            if "node_raw_features" in name:
                assert buf.requires_grad is False

    def test_overfit_small_batch(self):
        """Loss should decrease on a fixed tiny batch."""
        B, K, d = 4, 8, 16
        model = GraphMixer(d_model=32, d_edge=d, d_time=8, K=K, num_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch(B, K, d)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            out = model(batch)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_pipeline_integration(self):
        """GraphMixer should work end-to-end through DataPipeline."""
        B, K, d_edge, num_nodes = 4, 8, 16, 30
        model = GraphMixer(d_model=32, d_edge=d_edge, d_time=8, K=K)
        graph = TemporalGraph(num_nodes, buffer_size=K, edge_feat_dim=d_edge, device="cpu")

        # Pre-populate graph
        src = torch.randint(0, num_nodes, (50,))
        dst = torch.randint(0, num_nodes, (50,))
        ts = torch.linspace(0, 100, 50, dtype=torch.float64)
        ef = torch.randn(50, d_edge)
        graph.advance(src, dst, ts, ef)

        pipeline = DataPipeline(model.gather_spec, graph)
        neg_strat = RandomNegative(num_nodes)

        raw = RawBatch(
            src=torch.randint(0, num_nodes, (B,)),
            dst=torch.randint(0, num_nodes, (B,)),
            time=torch.full((B,), 120.0, dtype=torch.float64),
            edge_feat=torch.randn(B, d_edge),
        )
        raw.neg = neg_strat.sample(raw.src, raw.dst, raw.time, graph)
        prepared = pipeline.prepare(raw)
        out = model(prepared)
        assert out.pos_score.shape == (B,)

    def test_ap_eval_integration(self):
        """APEval should run without error."""
        B, K, d_edge, num_nodes = 4, 8, 16, 30
        model = GraphMixer(d_model=32, d_edge=d_edge, d_time=8, K=K)
        graph = TemporalGraph(num_nodes, buffer_size=K, edge_feat_dim=d_edge, device="cpu")
        pipeline = DataPipeline(model.gather_spec, graph)
        neg_strat = RandomNegative(num_nodes)

        raw = RawBatch(
            src=torch.randint(0, num_nodes, (B,)),
            dst=torch.randint(0, num_nodes, (B,)),
            time=torch.full((B,), 50.0, dtype=torch.float64),
            edge_feat=torch.randn(B, d_edge),
        )
        raw.neg = neg_strat.sample(raw.src, raw.dst, raw.time, graph)
        eval_batches = [raw]
        metrics = APEval().evaluate(model, pipeline, eval_batches, graph)
        assert "ap" in metrics
        assert 0.0 <= metrics["ap"] <= 1.0


# ===========================================================================
# FreeDyG tests
# ===========================================================================

class TestFreeDyG:

    def test_instantiation_no_node_feat(self):
        model = FreeDyG(d_model=32, d_edge=16, d_time=8, d_nif=16, K=8)
        assert model.gather_spec.neighbors.k == 8
        assert not model.supports_independent_encode
        assert not model._has_node_feat

    def test_instantiation_with_node_feat(self):
        nf = _node_features(50, 12)
        model = FreeDyG(d_model=32, d_edge=16, d_time=8, d_nif=16, K=8, node_raw_features=nf)
        assert model._has_node_feat
        assert model.node_raw_features.shape == (51, 12)  # +1 for padding row

    def test_forward_shapes(self):
        B, K, d = 8, 8, 16
        model = FreeDyG(d_model=32, d_edge=d, d_time=8, d_nif=16, K=K, num_layers=1)
        batch = _make_batch(B, K, d)
        out = model(batch)
        assert out.loss.shape == ()
        assert out.pos_score.shape == (B,)
        assert out.neg_score.shape == (B,)

    def test_forward_with_node_feat(self):
        B, K, d = 8, 8, 16
        nf = _node_features(100, 12)
        model = FreeDyG(d_model=32, d_edge=d, d_time=8, d_nif=16, K=K, num_layers=1,
                        node_raw_features=nf)
        batch = _make_batch(B, K, d)
        out = model(batch)
        assert out.pos_score.shape == (B,)

    def test_gradient_flow(self):
        B, K, d = 4, 8, 16
        model = FreeDyG(d_model=32, d_edge=d, d_time=8, d_nif=16, K=K, num_layers=1)
        batch = _make_batch(B, K, d)
        out = model(batch)
        out.loss.backward()
        none_grads = [name for name, p in model.named_parameters()
                      if p.requires_grad and p.grad is None]
        assert len(none_grads) == 0, f"no gradient for: {none_grads}"

    def test_overfit_small_batch(self):
        B, K, d = 4, 8, 16
        model = FreeDyG(d_model=32, d_edge=d, d_time=8, d_nif=16, K=K, num_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch(B, K, d)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            out = model(batch)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_nif_changes_with_context(self):
        """NIF features should differ when paired with different target nodes."""
        B, K, d = 2, 8, 16
        model = FreeDyG(d_model=32, d_edge=d, d_time=8, d_nif=16, K=K, num_layers=1)
        model.eval()
        with torch.no_grad():
            # Same src neighbors, different dst neighbors
            src_nbrs = _make_nbrs(B, K, d)
            dst_nbrs_a = _make_nbrs(B, K, d)
            dst_nbrs_b = _make_nbrs(B, K, d)
            time = torch.full((B,), 100.0, dtype=torch.float64)

            emb_src_a, _ = model._encode_pair(src_nbrs, dst_nbrs_a, time)
            emb_src_b, _ = model._encode_pair(src_nbrs, dst_nbrs_b, time)

        # src embeddings differ because NIF changes with dst context
        assert not torch.allclose(emb_src_a, emb_src_b), \
            "src embeddings should differ when dst neighbors differ (NIF is pair-dependent)"

    def test_pipeline_integration(self):
        B, K, d_edge, num_nodes = 4, 8, 16, 30
        model = FreeDyG(d_model=32, d_edge=d_edge, d_time=8, d_nif=16, K=K)
        graph = TemporalGraph(num_nodes, buffer_size=K, edge_feat_dim=d_edge, device="cpu")

        src = torch.randint(0, num_nodes, (50,))
        dst = torch.randint(0, num_nodes, (50,))
        ts = torch.linspace(0, 100, 50, dtype=torch.float64)
        ef = torch.randn(50, d_edge)
        graph.advance(src, dst, ts, ef)

        pipeline = DataPipeline(model.gather_spec, graph)
        neg_strat = RandomNegative(num_nodes)

        raw = RawBatch(
            src=torch.randint(0, num_nodes, (B,)),
            dst=torch.randint(0, num_nodes, (B,)),
            time=torch.full((B,), 120.0, dtype=torch.float64),
            edge_feat=torch.randn(B, d_edge),
        )
        raw.neg = neg_strat.sample(raw.src, raw.dst, raw.time, graph)
        prepared = pipeline.prepare(raw)
        out = model(prepared)
        assert out.pos_score.shape == (B,)

    def test_filter_layer_output_shape(self):
        """FilterLayer should preserve (B, T, C) shape."""
        from tgengine.nn.mlp_mixer import FilterLayer
        B, T, C = 4, 32, 16
        fl = FilterLayer(max_seq_len=T, hidden_dim=C)
        x = torch.randn(B, T, C)
        out = fl(x)
        assert out.shape == (B, T, C)

    def test_mlp_mixer_layer_shape(self):
        """MLPMixerLayer should preserve (B, T, C) shape."""
        from tgengine.nn.mlp_mixer import MLPMixerLayer
        B, T, C = 4, 16, 32
        layer = MLPMixerLayer(num_tokens=T, num_channels=C)
        x = torch.randn(B, T, C)
        out = layer(x)
        assert out.shape == (B, T, C)
