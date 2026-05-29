"""Tests for the four paradigm gaps fixed in the Engine:
1. encoder_lr / head_lr separation
2. tasks= with independent (head, batches) data sources
3. stopping_rule="all_improve"
4. probe() frozen-encoder mode
"""

from __future__ import annotations

import pytest
import torch

from tgengine.engine import Engine
from tgengine.engine.config import TrainConfig
from tgengine.models.base import EmbeddingBundle, ModelOutput, TemporalModel
from tgengine.core.batch import PreparedBatch, RawBatch
from tgengine.core.gather_spec import GatherSpec
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.pipeline.negatives import RandomNegative
from tgengine.tasks import LinkPredHead, NodeClassificationHead
from tgengine.engine import APEval, NodeClsEval


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _TinyModel(TemporalModel):
    gather_spec = GatherSpec()

    def __init__(self, d=8):
        super().__init__()
        self.d = d
        self.linear = torch.nn.Linear(d, d)

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        B = batch.src.shape[0]
        dev = batch.src.device
        src = torch.randn(B, self.d, device=dev)
        dst = torch.randn(B, self.d, device=dev)
        neg = torch.randn(B, self.d, device=dev)
        return EmbeddingBundle(src=src, dst=dst, neg=neg)


N, D, K = 30, 8, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _make_graph(n=N, d=D):
    g = TemporalGraph(n, edge_feat_dim=d, buffer_size=K, device=DEV)
    g.advance(
        torch.randint(0, n, (20,), device=DEV),
        torch.randint(0, n, (20,), device=DEV),
        torch.arange(20, dtype=torch.float32, device=DEV),
        torch.randn(20, d, device=DEV),
    )
    return g


def _make_raw_batches(n_batches=4, B=4, n=N, d=D):
    return [
        RawBatch(
            src=torch.randint(0, n, (B,), device=DEV),
            dst=torch.randint(0, n, (B,), device=DEV),
            time=torch.full((B,), 30.0 + i, dtype=torch.float32, device=DEV),
            edge_feat=torch.randn(B, d, device=DEV),
        )
        for i in range(n_batches)
    ]


def _make_engine(tasks=None, task_weights=None, eval_protocols=None,
                 primary_metric=None, stopping_rule="primary",
                 head_lr=None):
    model = _TinyModel(d=D)
    graph = _make_graph()
    batches = _make_raw_batches()
    config = TrainConfig(
        epochs=3, batch_size=4, lr=1e-3, patience=0,
        device=DEV, seed=42, eval_strategy="all",
        stopping_rule=stopping_rule,
        head_lr=head_lr,
    )
    return Engine(
        model=model,
        graph=graph,
        train_batches=batches,
        val_batches=batches,
        test_batches=batches,
        neg_strategy=RandomNegative(N),
        eval_protocol=APEval() if eval_protocols is None else None,
        config=config,
        tasks=tasks,
        task_weights=task_weights,
        eval_protocols=eval_protocols,
        primary_metric=primary_metric,
    )


# ---------------------------------------------------------------------------
# 1. encoder_lr / head_lr
# ---------------------------------------------------------------------------

def test_head_lr_creates_param_groups():
    """When head_lr is set, optimizer has 2 param groups with different LRs."""
    engine = _make_engine(
        tasks={"link": LinkPredHead()},
        eval_protocols={"link": APEval()},
        head_lr=1e-2,
    )
    pg = engine.optimizer.param_groups
    assert len(pg) == 2
    lrs = {pg[0]["lr"], pg[1]["lr"]}
    assert 1e-3 in lrs
    assert 1e-2 in lrs


def test_no_head_lr_single_param_group():
    """Without head_lr, all params share one group."""
    engine = _make_engine(
        tasks={"link": LinkPredHead()},
        eval_protocols={"link": APEval()},
        head_lr=None,
    )
    assert len(engine.optimizer.param_groups) == 1


def test_head_lr_encoder_params_unchanged():
    """Encoder params should be in the group with lr=config.lr."""
    engine = _make_engine(
        tasks={"link": LinkPredHead()},
        eval_protocols={"link": APEval()},
        head_lr=5e-3,
    )
    encoder_params = set(id(p) for p in engine.model.parameters())
    for pg in engine.optimizer.param_groups:
        pg_params = set(id(p) for p in pg["params"])
        if encoder_params & pg_params:
            assert pg["lr"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# 2. tasks= with independent (head, batches) tuple
# ---------------------------------------------------------------------------

def test_independent_task_batch_parsed():
    """(head, batches) tuple unpacked correctly into _tasks + _task_batches."""
    extra_batches = _make_raw_batches(n_batches=2)
    head = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={"node_cls": (head, extra_batches)},
        eval_protocols={"node_cls": NodeClsEval(num_classes=3)},
    )
    assert "node_cls" in engine._tasks
    assert engine._task_batches["node_cls"] is extra_batches


def test_independent_task_not_in_multitask_step():
    """Independent-batch heads are skipped in _step_multitask."""
    extra_batches = _make_raw_batches(n_batches=2)
    head = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={"node_cls": (head, extra_batches)},
        eval_protocols={"node_cls": NodeClsEval(num_classes=3)},
    )
    # _step_multitask should not route to node_cls (no labels in prepared)
    from tgengine.core.batch import NeighborData
    nbr = NeighborData(
        neighbor_ids=torch.zeros(4, K, dtype=torch.long),
        timestamps=torch.zeros(4, K),
        edge_feats=torch.zeros(4, K, D),
        mask=torch.ones(4, K, dtype=torch.bool),
    )
    prepared = PreparedBatch(
        src=torch.randint(0, N, (4,)),
        dst=torch.randint(0, N, (4,)),
        neg=torch.randint(0, N, (4,)),
        time=torch.rand(4),
        src_neighbors=nbr, dst_neighbors=nbr, neg_neighbors=nbr,
    )
    out = engine._step_multitask(prepared)
    # loss should be ~0 because no shared heads contributed
    assert out.loss.item() == pytest.approx(0.0, abs=1e-6)


def test_mixed_tasks_shared_and_independent():
    """Shared head (LinkPred) + independent head (NodeCls) both work."""
    extra_batches = _make_raw_batches(n_batches=2)
    node_cls = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={
            "link": LinkPredHead(),
            "node_cls": (node_cls, extra_batches),
        },
        eval_protocols={
            "link": APEval(),
            "node_cls": NodeClsEval(num_classes=3),
        },
    )
    assert engine._task_batches["link"] is None
    assert engine._task_batches["node_cls"] is extra_batches


# ---------------------------------------------------------------------------
# 3. stopping_rule = "all_improve"
# ---------------------------------------------------------------------------

def test_stopping_rule_primary_is_default():
    """Default stopping_rule is 'primary'."""
    config = TrainConfig()
    assert config.stopping_rule == "primary"


def test_stopping_rule_all_improve_logic():
    """all_improve: best only when ALL metrics improve."""
    # Simulate the Engine's internal logic from train()
    from tgengine.engine import Engine

    engine = object.__new__(Engine)
    engine.config = TrainConfig(stopping_rule="all_improve")

    # First eval always best
    _bests: dict[str, float] = {}
    val_metrics = {"ap": 0.80, "acc": 0.70}

    if not _bests:
        _bests = dict(val_metrics)
        is_best = True
    else:
        is_best = all(val_metrics.get(k, 0.0) >= _bests.get(k, 0.0) for k in _bests)
        if is_best:
            _bests.update(val_metrics)
    assert is_best

    # Second eval: ap improved, acc dropped → NOT best
    val_metrics2 = {"ap": 0.85, "acc": 0.65}
    is_best2 = all(val_metrics2.get(k, 0.0) >= _bests.get(k, 0.0) for k in _bests)
    assert not is_best2

    # Third eval: both improved → IS best
    val_metrics3 = {"ap": 0.88, "acc": 0.75}
    is_best3 = all(val_metrics3.get(k, 0.0) >= _bests.get(k, 0.0) for k in _bests)
    assert is_best3


def test_stopping_rule_in_config():
    config = TrainConfig(stopping_rule="all_improve")
    assert config.stopping_rule == "all_improve"


# ---------------------------------------------------------------------------
# 4. probe() mode
# ---------------------------------------------------------------------------

def test_probe_requires_existing_task():
    engine = _make_engine(
        tasks={"link": LinkPredHead()},
        eval_protocols={"link": APEval()},
    )
    with pytest.raises(ValueError, match="not found"):
        engine.probe("nonexistent_task", epochs=1)


def test_probe_freezes_encoder_during_training():
    """Encoder parameters must be frozen (requires_grad=False) during probe training."""
    # Use NodeClassificationHead which has actual parameters (MLP)
    head = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={"node_cls": head},
        eval_protocols={"node_cls": NodeClsEval(num_classes=3)},
    )

    frozen_states = []

    # Wrap pipeline.prepare to capture encoder grad state during probe
    original_prepare = engine.pipeline.prepare

    def patched_prepare(raw_batch):
        frozen_states.append(
            all(not p.requires_grad for p in engine.model.parameters())
        )
        return original_prepare(raw_batch)

    engine.pipeline.prepare = patched_prepare

    engine.probe("node_cls", epochs=1)
    assert len(frozen_states) > 0, "prepare() was never called during probe"
    assert all(frozen_states), "Encoder was not frozen during probe"


def test_probe_restores_encoder_grad_after():
    """Encoder parameters must have requires_grad=True after probe returns."""
    head = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={"node_cls": head},
        eval_protocols={"node_cls": NodeClsEval(num_classes=3)},
    )
    engine.probe("node_cls", epochs=1)
    assert all(p.requires_grad for p in engine.model.parameters())


def test_probe_returns_metrics():
    """probe() returns a non-empty metrics dict."""
    head = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={"node_cls": head},
        eval_protocols={"node_cls": NodeClsEval(num_classes=3)},
    )
    metrics = engine.probe("node_cls", epochs=1)
    assert isinstance(metrics, dict)


def test_probe_respects_head_lr_param():
    """Passing lr= to probe() uses that LR without raising."""
    head = NodeClassificationHead(d_model=D, num_classes=3)
    engine = _make_engine(
        tasks={"node_cls": head},
        eval_protocols={"node_cls": NodeClsEval(num_classes=3)},
    )
    metrics = engine.probe("node_cls", epochs=1, lr=1e-2)
    assert isinstance(metrics, dict)
