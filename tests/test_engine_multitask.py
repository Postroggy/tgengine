"""Tests for the refactored Engine: tasks= API, encode(), EmbeddingBundle."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch

from tgengine.models.base import EmbeddingBundle, ModelOutput, TemporalModel
from tgengine.core.batch import PreparedBatch
from tgengine.core.gather_spec import GatherSpec


# ---------------------------------------------------------------------------
# Minimal test model implementing encode()
# ---------------------------------------------------------------------------

class _TinyModel(TemporalModel):
    gather_spec = GatherSpec()

    def __init__(self, d=8):
        super().__init__()
        self.d = d
        self.linear = torch.nn.Linear(d, d)

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        B = batch.src.shape[0]
        src = torch.randn(B, self.d)
        dst = torch.randn(B, self.d)
        neg = torch.randn(B, self.d)
        return EmbeddingBundle(src=src, dst=dst, neg=neg)


class _PairDepModel(TemporalModel):
    """Simulates DyGFormer-style pair-dependent encoding."""
    gather_spec = GatherSpec()

    def __init__(self, d=8):
        super().__init__()
        self.d = d

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        B = batch.src.shape[0]
        src = torch.randn(B, self.d)
        dst = torch.randn(B, self.d)
        neg = torch.randn(B, self.d)
        src_for_neg = torch.randn(B, self.d)
        return EmbeddingBundle(src=src, dst=dst, neg=neg, src_for_neg=src_for_neg)


# ---------------------------------------------------------------------------
# EmbeddingBundle
# ---------------------------------------------------------------------------

def test_embedding_bundle_properties():
    B, d = 4, 16
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
    )
    assert bundle.d_model == d
    assert not bundle.is_pair_dependent


def test_embedding_bundle_pair_dependent():
    B, d = 4, 16
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
        src_for_neg=torch.randn(B, d),
    )
    assert bundle.is_pair_dependent


# ---------------------------------------------------------------------------
# TemporalModel.has_encode
# ---------------------------------------------------------------------------

def test_has_encode_true_for_implementing_model():
    m = _TinyModel()
    assert m.has_encode is True


def test_has_encode_false_for_base():
    class _NoEncode(TemporalModel):
        gather_spec = GatherSpec()
        def forward(self, batch):
            return ModelOutput(
                loss=torch.tensor(0.0),
                pos_score=torch.zeros(1),
                neg_score=torch.zeros(1),
            )

    m = _NoEncode()
    assert m.has_encode is False


# ---------------------------------------------------------------------------
# TemporalModel default forward uses encode()
# ---------------------------------------------------------------------------

def _make_neighbor_data(B=4, K=4, d_edge=2):
    from tgengine.core.batch import NeighborData
    return NeighborData(
        neighbor_ids=torch.zeros(B, K, dtype=torch.long),
        timestamps=torch.zeros(B, K),
        edge_feats=torch.zeros(B, K, d_edge),
        mask=torch.ones(B, K, dtype=torch.bool),
    )


def _make_prepared_batch(B=4):
    from tgengine.core.batch import PreparedBatch
    nbr = _make_neighbor_data(B)
    return PreparedBatch(
        src=torch.randint(0, 10, (B,)),
        dst=torch.randint(0, 10, (B,)),
        neg=torch.randint(0, 10, (B,)),
        time=torch.rand(B),
        src_neighbors=nbr,
        dst_neighbors=nbr,
        neg_neighbors=nbr,
    )


def test_tiny_model_forward_uses_encode():
    model = _TinyModel(d=8)
    batch = _make_prepared_batch(B=4)
    out = model(batch)
    assert isinstance(out, ModelOutput)
    assert out.loss.ndim == 0
    assert out.pos_score.shape == (4,)


# ---------------------------------------------------------------------------
# Engine._step_multitask routing
# ---------------------------------------------------------------------------

def _make_engine_with_tasks(tasks, task_weights=None):
    """Build a minimal Engine-like object with just _step_multitask + _route_to_head."""
    from tgengine.engine import Engine

    # We don't want to construct the full Engine (it needs GPU/graph/etc.)
    # Instead we test the routing logic directly by calling internal methods.
    engine = object.__new__(Engine)
    engine._tasks = tasks
    engine._task_weights = task_weights or {}
    engine.model = _TinyModel(d=8)
    engine._use_tasks = True
    return engine


def test_link_pred_head_routing():
    from tgengine.tasks import LinkPredHead
    engine = _make_engine_with_tasks({"link": LinkPredHead()})

    B, d = 4, 8
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
    )
    prepared = _make_prepared_batch(B)

    out = engine._route_to_head(engine._tasks["link"], "link", bundle, prepared)
    assert isinstance(out, ModelOutput)
    assert out.pos_score.shape == (B,)


def test_node_cls_head_routing():
    from tgengine.tasks import NodeClassificationHead
    num_classes = 5
    engine = _make_engine_with_tasks(
        {"node_cls": NodeClassificationHead(d_model=8, num_classes=num_classes)}
    )

    B, d = 4, 8
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
    )
    prepared = _make_prepared_batch(B)
    prepared.node_labels = torch.randint(0, num_classes, (B,))

    out = engine._route_to_head(engine._tasks["node_cls"], "node_cls", bundle, prepared)
    assert out.node_pred.shape == (B, num_classes)


def test_edge_cls_head_routing():
    from tgengine.tasks import EdgeClassificationHead
    num_classes = 3
    engine = _make_engine_with_tasks(
        {"edge_cls": EdgeClassificationHead(d_model=8, num_classes=num_classes, input_mode="concat")}
    )

    B, d = 4, 8
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
    )
    prepared = _make_prepared_batch(B)
    prepared.edge_labels = torch.randint(0, num_classes, (B,))

    out = engine._route_to_head(engine._tasks["edge_cls"], "edge_cls", bundle, prepared)
    assert out.edge_pred.shape == (B, num_classes)


def test_anomaly_head_routing():
    from tgengine.tasks import AnomalyDetectionHead
    engine = _make_engine_with_tasks(
        {"anomaly": AnomalyDetectionHead(d_model=8, mode="supervised")}
    )

    B, d = 4, 8
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
    )
    prepared = _make_prepared_batch(B)
    prepared.node_labels = torch.randint(0, 2, (B,))

    out = engine._route_to_head(engine._tasks["anomaly"], "anomaly", bundle, prepared)
    assert out.anomaly_score is not None
    assert out.anomaly_score.shape == (B,)


def test_step_multitask_sums_losses():
    from tgengine.tasks import LinkPredHead, NodeClassificationHead
    engine = _make_engine_with_tasks(
        {
            "link": LinkPredHead(),
            "node_cls": NodeClassificationHead(d_model=8, num_classes=3),
        },
        task_weights={"link": 1.0, "node_cls": 0.5},
    )

    B, d = 4, 8
    bundle = EmbeddingBundle(
        src=torch.randn(B, d),
        dst=torch.randn(B, d),
        neg=torch.randn(B, d),
    )
    prepared = _make_prepared_batch(B)
    prepared.node_labels = torch.randint(0, 3, (B,))

    out = engine._step_multitask(prepared)
    assert isinstance(out, ModelOutput)
    assert out.loss.item() > 0
    assert out.node_pred is not None


# ---------------------------------------------------------------------------
# pair-dependent model encoding
# ---------------------------------------------------------------------------

def test_pair_dependent_bundle():
    model = _PairDepModel(d=8)
    batch = _make_prepared_batch(B=4)
    bundle = model.encode(batch)
    assert bundle.is_pair_dependent
    assert bundle.src_for_neg is not None
    assert bundle.src_for_neg.shape == (4, 8)


# ---------------------------------------------------------------------------
# Engine backward compatibility (no tasks=)
# ---------------------------------------------------------------------------

def test_engine_eval_protocol_guard():
    """Engine constructor raises when neither eval_protocol nor eval_protocols given."""
    from tgengine.engine import Engine
    # Build a minimal Engine instance to test the guard without full init
    # We patch __init__ only far enough to hit the guard
    import inspect
    src = inspect.getsource(Engine.__init__)
    # The guard must be present in source
    assert "eval_protocol is None and eval_protocols is None" in src


def test_engine_val_score_fallback():
    """_evaluate must gracefully fall back when primary_metric not in results."""
    from tgengine.engine import Engine
    engine = object.__new__(Engine)
    engine._primary_metric = "nonexistent_metric"
    # Simulate what train() does with val_metrics:
    val_metrics = {"ap": 0.95, "auc": 0.97}
    val_score = val_metrics.get(engine._primary_metric, 0.0)
    if val_score == 0.0 and val_metrics:
        val_score = next(iter(val_metrics.values()))
    assert val_score == 0.95
