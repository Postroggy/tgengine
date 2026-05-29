"""Tests for tgengine/tasks/ task heads and tgengine/engine/eval.py protocols."""

from __future__ import annotations

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from tgengine.tasks import (
    AnomalyDetectionHead,
    EdgeBinaryClassificationHead,
    EdgeClassificationHead,
    EdgeRegressionHead,
    LinkPredHead,
    MultiTaskHead,
    NodeBinaryClassificationHead,
    NodeClassificationHead,
    NodeRegressionHead,
    TaskHead,
)
from tgengine.models.base import ModelOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, D = 8, 64


def rand(shape):
    return torch.randn(*shape)


def make_labels(n, num_classes=None):
    if num_classes is None:
        return torch.zeros(n)
    return torch.randint(0, num_classes, (n,))


# ---------------------------------------------------------------------------
# TaskHead base
# ---------------------------------------------------------------------------

def test_task_head_is_abstract():
    with pytest.raises(TypeError):
        TaskHead()  # type: ignore


# ---------------------------------------------------------------------------
# LinkPredHead
# ---------------------------------------------------------------------------

def test_link_pred_head_forward():
    head = LinkPredHead()
    src = rand((B, D))
    dst = rand((B, D))
    neg = rand((B, D))
    out = head(rand((B, D)), src_emb=src, dst_emb=dst, neg_emb=neg)
    assert isinstance(out, ModelOutput)
    assert out.loss.shape == ()
    assert out.pos_score.shape == (B,)
    assert out.neg_score.shape == (B,)


def test_link_pred_head_with_decoder():
    class DotDecoder(torch.nn.Module):
        def forward(self, a, b):
            return (a * b).sum(-1)

    head = LinkPredHead(decoder=DotDecoder())
    src = rand((B, D))
    dst = rand((B, D))
    neg = rand((B, D))
    out = head(rand((B, D)), src_emb=src, dst_emb=dst, neg_emb=neg)
    assert out.loss > 0


# ---------------------------------------------------------------------------
# NodeClassificationHead
# ---------------------------------------------------------------------------

def test_node_cls_multiclass():
    head = NodeClassificationHead(d_model=D, num_classes=5)
    emb = rand((B, D))
    labels = make_labels(B, 5)
    out = head(emb, labels=labels)
    assert out.loss.item() >= 0
    assert out.node_pred.shape == (B, 5)
    assert torch.equal(out.node_labels, labels)


def test_node_cls_no_labels():
    head = NodeClassificationHead(d_model=D, num_classes=3)
    out = head(rand((B, D)))
    assert out.loss.item() == 0.0
    assert out.node_pred.shape == (B, 3)


def test_node_binary_cls():
    head = NodeBinaryClassificationHead(d_model=D)
    labels = torch.randint(0, 2, (B,)).float()
    out = head(rand((B, D)), labels=labels)
    assert out.loss.item() >= 0
    assert out.node_pred.shape == (B,)


# ---------------------------------------------------------------------------
# NodeRegressionHead
# ---------------------------------------------------------------------------

def test_node_reg_mse():
    head = NodeRegressionHead(d_model=D, loss="mse")
    labels = torch.randn(B)
    out = head(rand((B, D)), labels=labels)
    assert out.loss.item() >= 0
    assert out.node_pred.shape == (B,)


def test_node_reg_mae():
    head = NodeRegressionHead(d_model=D, loss="mae")
    labels = torch.randn(B)
    out = head(rand((B, D)), labels=labels)
    assert out.loss.item() >= 0


def test_node_reg_multidim():
    head = NodeRegressionHead(d_model=D, output_dim=3)
    out = head(rand((B, D)))
    assert out.node_pred.shape == (B, 3)


# ---------------------------------------------------------------------------
# EdgeClassificationHead
# ---------------------------------------------------------------------------

def test_edge_cls_concat():
    head = EdgeClassificationHead(d_model=D, num_classes=4, input_mode="concat")
    src = rand((B, D))
    dst = rand((B, D))
    labels = make_labels(B, 4)
    out = head(src, labels=labels, dst_emb=dst)
    assert out.loss.item() >= 0
    assert out.edge_pred.shape == (B, 4)


def test_edge_cls_single():
    head = EdgeClassificationHead(d_model=D, num_classes=3, input_mode="single")
    out = head(rand((B, D)))
    assert out.edge_pred.shape == (B, 3)


def test_edge_binary_cls():
    head = EdgeBinaryClassificationHead(d_model=D, input_mode="concat")
    src = rand((B, D))
    dst = rand((B, D))
    labels = torch.randint(0, 2, (B,)).float()
    out = head(src, labels=labels, dst_emb=dst)
    assert out.loss.item() >= 0
    assert out.edge_pred.shape == (B,)


# ---------------------------------------------------------------------------
# EdgeRegressionHead
# ---------------------------------------------------------------------------

def test_edge_reg_mse():
    head = EdgeRegressionHead(d_model=D, input_mode="concat")
    src = rand((B, D))
    dst = rand((B, D))
    labels = torch.randn(B)
    out = head(src, labels=labels, dst_emb=dst)
    assert out.loss.item() >= 0
    assert out.edge_pred.shape == (B,)


def test_edge_reg_mae_single():
    head = EdgeRegressionHead(d_model=D, input_mode="single", loss="mae")
    labels = torch.randn(B)
    out = head(rand((B, D)), labels=labels)
    assert out.loss.item() >= 0


def test_edge_reg_multidim():
    head = EdgeRegressionHead(d_model=D, output_dim=2, input_mode="single")
    out = head(rand((B, D)))
    assert out.edge_pred.shape == (B, 2)


# ---------------------------------------------------------------------------
# AnomalyDetectionHead
# ---------------------------------------------------------------------------

def test_anomaly_supervised():
    head = AnomalyDetectionHead(d_model=D, mode="supervised")
    labels = torch.randint(0, 2, (B,))
    out = head(rand((B, D)), labels=labels)
    assert out.loss.item() >= 0
    assert out.anomaly_score is not None
    assert out.anomaly_score.shape == (B,)
    assert (out.anomaly_score >= 0).all() and (out.anomaly_score <= 1).all()


def test_anomaly_unsupervised():
    head = AnomalyDetectionHead(d_model=D, mode="unsupervised")
    out = head(rand((B, D)))
    assert out.loss.item() >= 0
    assert out.anomaly_score is not None
    assert out.anomaly_score.shape == (B,)
    assert (out.anomaly_score >= 0).all()


# ---------------------------------------------------------------------------
# MultiTaskHead
# ---------------------------------------------------------------------------

def test_multi_task_head_two_heads():
    head = MultiTaskHead(
        heads={
            "node_cls": NodeClassificationHead(D, 3),
            "anomaly": AnomalyDetectionHead(D, mode="supervised"),
        },
        weights={"node_cls": 1.0, "anomaly": 0.5},
    )
    emb = rand((B, D))
    node_labels = make_labels(B, 3)
    anomaly_labels = torch.randint(0, 2, (B,))
    out = head(
        emb,
        labels=None,
        node_cls_labels=node_labels,
        anomaly_labels=anomaly_labels,
    )
    assert out.loss.item() >= 0
    assert out.node_pred is not None
    assert out.anomaly_score is not None


def test_multi_task_shared_label():
    head = MultiTaskHead(
        heads={"reg": NodeRegressionHead(D)},
        weights={"reg": 2.0},
    )
    emb = rand((B, D))
    labels = torch.randn(B)
    out = head(emb, labels=labels)
    assert out.node_pred is not None


# ---------------------------------------------------------------------------
# Eval protocol tests (unit, no GPU needed)
# ---------------------------------------------------------------------------

def _make_mock_model(node_pred=None, edge_pred=None, anomaly_score=None, node_labels=None, edge_labels=None):
    """Return a mock model that produces controlled outputs."""
    model = MagicMock()
    model.eval = MagicMock(return_value=None)

    pos = torch.randn(B)
    neg = torch.randn(B)
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=pos,
        neg_score=neg,
        node_pred=node_pred,
        node_labels=node_labels,
        edge_pred=edge_pred,
        edge_labels=edge_labels,
        anomaly_score=anomaly_score,
    )
    model.return_value = out
    model.__call__ = lambda self, x: out
    return model


def _make_pipeline(model):
    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)
    return pipeline


def _make_batches(n=4):
    from tgengine.core.batch import RawBatch
    return [
        RawBatch(
            src=torch.randint(0, 100, (B,)),
            dst=torch.randint(0, 100, (B,)),
            time=torch.rand(B),
            edge_feat=None,
            neg=torch.randint(0, 100, (B,)),
        )
        for _ in range(n)
    ]


def test_node_cls_eval():
    from tgengine.engine.eval import NodeClsEval

    num_classes = 5
    logits = torch.randn(B, num_classes)
    labels = torch.randint(0, num_classes, (B,))

    model = MagicMock()
    model.eval = MagicMock()
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=torch.zeros(B),
        neg_score=torch.zeros(B),
        node_pred=logits,
        node_labels=labels,
    )
    model.return_value = out

    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)

    evaluator = NodeClsEval(num_classes=num_classes)
    results = evaluator.evaluate(model, pipeline, _make_batches(), graph=None)
    assert "acc" in results
    assert "f1_macro" in results
    assert 0.0 <= results["acc"] <= 1.0


def test_node_reg_eval():
    from tgengine.engine.eval import NodeRegEval

    preds = torch.randn(B)
    labels = torch.randn(B)

    model = MagicMock()
    model.eval = MagicMock()
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=torch.zeros(B),
        neg_score=torch.zeros(B),
        node_pred=preds,
        node_labels=labels,
    )
    model.return_value = out

    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)

    evaluator = NodeRegEval()
    results = evaluator.evaluate(model, pipeline, _make_batches(), graph=None)
    assert "mae" in results
    assert "rmse" in results
    assert results["rmse"] >= results["mae"] >= 0


def test_edge_cls_eval():
    from tgengine.engine.eval import EdgeClsEval

    logits = torch.randn(B, 3)
    labels = torch.randint(0, 3, (B,))

    model = MagicMock()
    model.eval = MagicMock()
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=torch.zeros(B),
        neg_score=torch.zeros(B),
        edge_pred=logits,
        edge_labels=labels,
    )
    model.return_value = out

    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)

    evaluator = EdgeClsEval(num_classes=3)
    results = evaluator.evaluate(model, pipeline, _make_batches(), graph=None)
    assert "acc" in results and "f1_macro" in results


def test_edge_reg_eval():
    from tgengine.engine.eval import EdgeRegEval

    preds = torch.randn(B)
    labels = torch.randn(B)

    model = MagicMock()
    model.eval = MagicMock()
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=torch.zeros(B),
        neg_score=torch.zeros(B),
        edge_pred=preds,
        edge_labels=labels,
    )
    model.return_value = out

    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)

    evaluator = EdgeRegEval()
    results = evaluator.evaluate(model, pipeline, _make_batches(), graph=None)
    assert "mae" in results and "rmse" in results


def test_anomaly_eval():
    from tgengine.engine.eval import AnomalyEval

    # anomaly_score ∈ [0,1], labels 0/1
    scores = torch.rand(B)
    labels = torch.randint(0, 2, (B,))

    model = MagicMock()
    model.eval = MagicMock()
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=torch.zeros(B),
        neg_score=torch.zeros(B),
        anomaly_score=scores,
        node_labels=labels,
    )
    model.return_value = out

    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)

    evaluator = AnomalyEval(label_source="node")
    results = evaluator.evaluate(model, pipeline, _make_batches(), graph=None)
    assert "auroc" in results and "ap" in results


def test_eval_returns_empty_when_no_predictions():
    from tgengine.engine.eval import NodeClsEval

    model = MagicMock()
    model.eval = MagicMock()
    out = ModelOutput(
        loss=torch.tensor(0.0),
        pos_score=torch.zeros(B),
        neg_score=torch.zeros(B),
    )
    model.return_value = out

    pipeline = MagicMock()
    pipeline.prepare = MagicMock(side_effect=lambda x: x)

    evaluator = NodeClsEval(num_classes=3)
    results = evaluator.evaluate(model, pipeline, _make_batches(), graph=None)
    assert results == {}
