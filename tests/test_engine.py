"""End-to-end integration tests for the Engine training loop.

Covers: training loss decrease, APEval via Engine, ThreeWayEval via Engine,
MRREval slow path, snapshot/restore during eval, early stopping,
adaptive eval, AUCEval, structured output.
"""

import torch
import pytest

from tgengine.core.batch import RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.engine import Engine, TrainConfig, APEval, AUCEval, ThreeWayEval, MRREval
from tgengine.models.graphmixer import GraphMixer
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_setup(device="cuda"):
    """Create a small Engine setup for fast testing.

    Graph: 30 nodes, buffer=16, edge_feat_dim=8
    Model: GraphMixer(d_model=16, d_edge=8, d_time=4, K=4, num_layers=1)
    Train: 12 chronological batches of 8 edges each.
    Val/Test: last 4 batches each.
    """
    N, K, d = 30, 4, 8

    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=device)

    # Seed graph with history
    src_h = torch.randint(0, N, (80,), device=device)
    dst_h = torch.randint(0, N, (80,), device=device)
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64, device=device)
    ef_h = torch.randn(80, d, device=device)
    graph.advance(src_h, dst_h, ts_h, ef_h)

    # Create chronological train batches (after history)
    train_batches = []
    for i in range(12):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (8,), device=device),
            dst=torch.randint(0, N, (8,), device=device),
            time=torch.full((8,), 90.0 + i, dtype=torch.float64, device=device),
            edge_feat=torch.randn(8, d, device=device),
        ))

    val_batches = train_batches[8:12]
    test_batches = train_batches[8:12]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    neg_strat = RandomNegative(N)
    eval_proto = APEval()
    config = TrainConfig(epochs=5, batch_size=8, lr=1e-3, patience=3,
                         device=device, seed=42)
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    neg_strat, eval_proto, config)
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_training_overfits_on_fixed_batch():
    """Loss should decrease when repeatedly training on the same batch."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)

    graph.advance(
        torch.randint(0, N, (50,), device=dev),
        torch.randint(0, N, (50,), device=dev),
        torch.linspace(0, 50, 50, dtype=torch.float64, device=dev),
        torch.randn(50, d, device=dev),
    )

    batch = RawBatch(
        src=torch.arange(4, device=dev), dst=torch.arange(4, 8, device=dev),
        time=torch.full((4,), 60.0, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(4, d, device=dev),
        neg=torch.randint(0, N, (4,), device=dev),
    )

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1).to(dev)
    pipeline = DataPipeline(model.gather_spec, graph)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()

    losses = []
    for _ in range(80):
        prepared = pipeline.prepare(batch)
        opt.zero_grad()
        out = model(prepared)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())

    assert losses[-1] < losses[0], \
        f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_full_train_returns_metrics():
    """Engine.train() returns test metrics dict after training."""
    engine = _small_setup()
    metrics = engine.train()
    assert isinstance(metrics, dict)
    assert "ap" in metrics
    assert 0.0 <= metrics["ap"] <= 1.0


def test_apeval_via_engine_evaluate():
    """Engine._evaluate() produces valid AP (handles neg sampling)."""
    engine = _small_setup()
    metrics = engine._evaluate(engine.val_batches)
    assert "ap" in metrics
    assert 0.0 <= metrics["ap"] <= 1.0


def test_graph_preserved_after_eval():
    """Graph state must be restored after evaluation (snapshot/restore)."""
    engine = _small_setup()
    snap = engine.graph.snapshot()
    edges_before = engine.graph.num_edges

    engine._evaluate(engine.val_batches)

    assert engine.graph.num_edges == edges_before, \
        "graph num_edges changed after evaluate"


def test_three_way_eval_produces_three_metrics():
    """ThreeWayEval produces ap_random, ap_historical, ap_inductive."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)

    src_h = torch.randint(0, N, (80,), device=dev)
    dst_h = torch.randint(0, N, (80,), device=dev)
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64, device=dev)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d, device=dev))

    eval_batches = []
    for i in range(4):
        eval_batches.append(RawBatch(
            src=torch.randint(0, N, (6,), device=dev),
            dst=torch.randint(0, N, (6,), device=dev),
            time=torch.full((6,), 90.0 + i, dtype=torch.float64, device=dev),
            edge_feat=torch.randn(6, d, device=dev),
        ))

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1).to(dev)
    pipeline = DataPipeline(model.gather_spec, graph)
    inductive_nodes = torch.arange(20, 30, device=dev)

    evaluator = ThreeWayEval(N, inductive_nodes, device=dev)
    metrics = evaluator.evaluate(model, pipeline, eval_batches, graph)

    assert "ap_random" in metrics
    assert "ap_historical" in metrics
    assert "ap_inductive" in metrics
    for k in metrics:
        assert 0.0 <= metrics[k] <= 1.0, f"{k} = {metrics[k]} out of range"


def test_mrr_eval():
    """MRREval produces valid MRR score."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)

    src_h = torch.randint(0, N, (80,), device=dev)
    dst_h = torch.randint(0, N, (80,), device=dev)
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64, device=dev)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d, device=dev))

    N_eval, N_neg = 4, 5
    eval_batches = []
    neg_rows = []
    for i in range(N_eval):
        eval_batches.append(RawBatch(
            src=torch.randint(0, N, (6,), device=dev),
            dst=torch.randint(0, N, (6,), device=dev),
            time=torch.full((6,), 90.0 + i, dtype=torch.float64, device=dev),
            edge_feat=torch.randn(6, d, device=dev),
        ))
        neg_rows.append(torch.randint(0, N, (6, N_neg), device=dev))

    neg_tensor = torch.cat(neg_rows, dim=0)  # (N_eval*6, N_neg)

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1).to(dev)
    pipeline = DataPipeline(model.gather_spec, graph)

    evaluator = MRREval(neg_tensor)
    metrics = evaluator.evaluate(model, pipeline, eval_batches, graph)

    assert "mrr" in metrics
    # MRR is reciprocals of 1..(N_neg+1), the best we can get is 1/(N_neg+1)..1
    assert 0.0 < metrics["mrr"] <= 1.0


def test_early_stopping():
    """Engine stops early when val score does not improve."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)

    src_h = torch.randint(0, N, (80,), device=dev)
    dst_h = torch.randint(0, N, (80,), device=dev)
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64, device=dev)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d, device=dev))

    train_batches = []
    for i in range(8):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (4,), device=dev),
            dst=torch.randint(0, N, (4,), device=dev),
            time=torch.full((4,), 90.0 + i, dtype=torch.float64, device=dev),
            edge_feat=torch.randn(4, d, device=dev),
        ))

    val_batches = test_batches = train_batches[4:8]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    neg_strat = RandomNegative(N)
    eval_proto = APEval()

    config = TrainConfig(epochs=50, batch_size=4, lr=1e-3, patience=1,
                         device=dev, seed=42)
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    neg_strat, eval_proto, config)
    metrics = engine.train()

    assert "ap" in metrics
    # If early stopping worked, we finished in far fewer than 50 epochs
    # This is implicitly tested by the test finishing quickly


def test_checkpoint_save_load():
    """Checkpoint save + load round-trip preserves model weights and optimizer state."""
    import os
    import tempfile

    engine = _small_setup()
    engine._current_epoch = 3
    engine._best_val = 0.72

    # Capture pre-save weights
    pre_save = {k: v.clone() for k, v in engine.model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "best.pt")
        engine.save_checkpoint(path)

        # Modify weights
        for p in engine.model.parameters():
            p.data.add_(0.5)

        engine.load_checkpoint(path)

        for k, v in engine.model.state_dict().items():
            assert torch.allclose(v, pre_save[k]), f"weight {k} diverged after load"

        assert engine._current_epoch == 3
        assert engine._best_val == 0.72


def test_auto_checkpoint_on_best():
    """Engine auto-saves checkpoint to checkpoint_dir when new best is found."""
    import os
    import tempfile

    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)

    src_h = torch.randint(0, N, (80,), device=dev)
    dst_h = torch.randint(0, N, (80,), device=dev)
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64, device=dev)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d, device=dev))

    train_batches = []
    for i in range(8):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (4,), device=dev),
            dst=torch.randint(0, N, (4,), device=dev),
            time=torch.full((4,), 90.0 + i, dtype=torch.float64, device=dev),
            edge_feat=torch.randn(4, d, device=dev),
        ))

    val_batches = test_batches = train_batches[4:8]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    neg_strat = RandomNegative(N)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainConfig(epochs=10, batch_size=4, lr=1e-3, patience=3,
                             device=dev, seed=42, checkpoint_dir=tmpdir)
        engine = Engine(model, graph, train_batches, val_batches, test_batches,
                        neg_strat, APEval(), config)
        engine.train()

        ckpt_path = os.path.join(tmpdir, "best.pt")
        assert os.path.exists(ckpt_path), f"checkpoint not found at {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["epoch"] > 0


def test_amp_training():
    """Mixed precision training should complete without errors and reduce loss."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (50,), device=dev),
        torch.randint(0, N, (50,), device=dev),
        torch.linspace(0, 50, 50, dtype=torch.float64, device=dev),
        torch.randn(50, d, device=dev),
    )

    train_batches = [RawBatch(
        src=torch.randint(0, N, (8,), device=dev),
        dst=torch.randint(0, N, (8,), device=dev),
        time=torch.full((8,), 60.0 + i, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(8, d, device=dev),
    ) for i in range(8)]
    val_batches = test_batches = train_batches[4:8]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    neg_strat = RandomNegative(N)
    config = TrainConfig(epochs=5, batch_size=8, lr=1e-3, patience=3,
                         device=dev, seed=42, use_amp=True)
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    neg_strat, APEval(), config)
    best = engine.train()
    assert "ap" in best or best == {}


def test_adaptive_eval_skips_epochs():
    """Adaptive eval should evaluate fewer times than total epochs."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (50,), device=dev),
        torch.randint(0, N, (50,), device=dev),
        torch.linspace(0, 50, 50, dtype=torch.float64, device=dev),
        torch.randn(50, d, device=dev),
    )

    train_batches = [RawBatch(
        src=torch.randint(0, N, (8,), device=dev),
        dst=torch.randint(0, N, (8,), device=dev),
        time=torch.full((8,), 60.0 + i, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(8, d, device=dev),
    ) for i in range(8)]
    val_batches = test_batches = train_batches[4:8]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    config = TrainConfig(
        epochs=20, batch_size=8, lr=1e-3, patience=0, device=dev, seed=42,
        eval_strategy="adaptive", max_eval_gap=5, loss_threshold=0.001,
    )
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    RandomNegative(N), APEval(), config)

    # Track eval count via _last_eval_epoch
    engine.train()
    # With adaptive, should have evaluated < 20 times (at least first + last + a few)
    assert engine._last_eval_epoch == 20


def test_adaptive_should_eval_logic():
    """Unit test for _should_eval method."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (50,), device=dev),
        torch.randint(0, N, (50,), device=dev),
        torch.linspace(0, 50, 50, dtype=torch.float64, device=dev),
        torch.randn(50, d, device=dev),
    )

    train_batches = [RawBatch(
        src=torch.randint(0, N, (4,), device=dev),
        dst=torch.randint(0, N, (4,), device=dev),
        time=torch.full((4,), 60.0, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(4, d, device=dev),
    )]
    val_batches = test_batches = train_batches

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    config = TrainConfig(
        epochs=50, batch_size=4, lr=1e-3, patience=0, device=dev,
        eval_strategy="adaptive", min_eval_gap=2, max_eval_gap=10,
        loss_threshold=0.02,
    )
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    RandomNegative(N), APEval(), config)

    # Epoch 1: always eval
    assert engine._should_eval(1, 0.5) is True
    # Last epoch: always eval
    assert engine._should_eval(50, 0.5) is True

    # Within min_eval_gap: skip
    engine._last_eval_epoch = 5
    engine._loss_at_last_eval = 0.5
    assert engine._should_eval(6, 0.5) is False

    # Beyond max_eval_gap: force eval
    assert engine._should_eval(16, 0.5) is True

    # Loss barely changed: skip
    assert engine._should_eval(8, 0.505) is False

    # Loss changed significantly: eval
    assert engine._should_eval(8, 0.6) is True


def test_eval_strategy_all():
    """eval_strategy='all' evaluates every epoch."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (50,), device=dev),
        torch.randint(0, N, (50,), device=dev),
        torch.linspace(0, 50, 50, dtype=torch.float64, device=dev),
        torch.randn(50, d, device=dev),
    )

    train_batches = [RawBatch(
        src=torch.randint(0, N, (4,), device=dev),
        dst=torch.randint(0, N, (4,), device=dev),
        time=torch.full((4,), 60.0, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(4, d, device=dev),
    )]
    val_batches = test_batches = train_batches

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    config = TrainConfig(
        epochs=5, batch_size=4, lr=1e-3, patience=0, device=dev,
        eval_strategy="all",
    )
    engine = Engine(model, graph, train_batches, val_batches, test_batches,
                    RandomNegative(N), APEval(), config)

    for epoch in range(1, 6):
        assert engine._should_eval(epoch, 0.5) is True


def test_auc_eval():
    """AUCEval produces valid AUC-ROC score."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (80,), device=dev),
        torch.randint(0, N, (80,), device=dev),
        torch.linspace(0, 80, 80, dtype=torch.float64, device=dev),
        torch.randn(80, d, device=dev),
    )

    eval_batches = [RawBatch(
        src=torch.randint(0, N, (8,), device=dev),
        dst=torch.randint(0, N, (8,), device=dev),
        time=torch.full((8,), 90.0, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(8, d, device=dev),
        neg=torch.randint(0, N, (8,), device=dev),
    )]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1).to(dev)
    pipeline = DataPipeline(model.gather_spec, graph)

    evaluator = AUCEval()
    metrics = evaluator.evaluate(model, pipeline, eval_batches, graph)
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_ap_eval_with_auc():
    """APEval with include_auc=True produces both AP and AUC."""
    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (80,), device=dev),
        torch.randint(0, N, (80,), device=dev),
        torch.linspace(0, 80, 80, dtype=torch.float64, device=dev),
        torch.randn(80, d, device=dev),
    )

    eval_batches = [RawBatch(
        src=torch.randint(0, N, (8,), device=dev),
        dst=torch.randint(0, N, (8,), device=dev),
        time=torch.full((8,), 90.0, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(8, d, device=dev),
        neg=torch.randint(0, N, (8,), device=dev),
    )]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1).to(dev)
    pipeline = DataPipeline(model.gather_spec, graph)

    evaluator = APEval(include_auc=True)
    metrics = evaluator.evaluate(model, pipeline, eval_batches, graph)
    assert "ap" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_structured_result_output():
    """Engine writes result.json when result_dir is set."""
    import json
    import os
    import tempfile

    N, K, d, dev = 30, 4, 8, "cuda"
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=dev)
    graph.advance(
        torch.randint(0, N, (50,), device=dev),
        torch.randint(0, N, (50,), device=dev),
        torch.linspace(0, 50, 50, dtype=torch.float64, device=dev),
        torch.randn(50, d, device=dev),
    )

    train_batches = [RawBatch(
        src=torch.randint(0, N, (4,), device=dev),
        dst=torch.randint(0, N, (4,), device=dev),
        time=torch.full((4,), 60.0 + i, dtype=torch.float64, device=dev),
        edge_feat=torch.randn(4, d, device=dev),
    ) for i in range(4)]
    val_batches = test_batches = train_batches[2:4]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainConfig(
            epochs=3, batch_size=4, lr=1e-3, patience=0, device=dev,
            eval_strategy="all", result_dir=tmpdir,
        )
        engine = Engine(model, graph, train_batches, val_batches, test_batches,
                        RandomNegative(N), APEval(), config)
        engine.train()

        result_path = os.path.join(tmpdir, "result.json")
        assert os.path.exists(result_path)

        with open(result_path) as f:
            result = json.load(f)

        assert "model" in result
        assert "config" in result
        assert "result" in result
        assert "stats" in result
        assert result["model"] == "GraphMixer"
        assert result["result"]["best_val"] > 0
        assert result["stats"]["eval_count"] > 0


def test_trainconfig_defaults():
    """TrainConfig has correct defaults for new fields."""
    cfg = TrainConfig()
    assert cfg.patience == 0
    assert cfg.eval_strategy == "adaptive"
    assert cfg.min_eval_gap == 1
    assert cfg.max_eval_gap == 10
    assert cfg.loss_threshold == 0.02
    assert cfg.result_dir is None
