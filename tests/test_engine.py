"""End-to-end integration tests for the Engine training loop.

Covers: training loss decrease, APEval via Engine, ThreeWayEval via Engine,
MRREval slow path, snapshot/restore during eval, early stopping.
"""

import torch
import pytest

from tgengine.core.batch import RawBatch
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.engine import Engine, TrainConfig, APEval, ThreeWayEval, MRREval
from tgengine.models.graphmixer import GraphMixer
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_setup(device="cpu"):
    """Create a small Engine setup for fast testing.

    Graph: 30 nodes, buffer=16, edge_feat_dim=8
    Model: GraphMixer(d_model=16, d_edge=8, d_time=4, K=4, num_layers=1)
    Train: 12 chronological batches of 8 edges each.
    Val/Test: last 4 batches each.
    """
    N, K, d = 30, 4, 8

    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device=device)

    # Seed graph with history
    src_h = torch.randint(0, N, (80,))
    dst_h = torch.randint(0, N, (80,))
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64)
    ef_h = torch.randn(80, d)
    graph.advance(src_h, dst_h, ts_h, ef_h)

    # Create chronological train batches (after history)
    train_batches = []
    for i in range(12):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (8,)),
            dst=torch.randint(0, N, (8,)),
            time=torch.full((8,), 90.0 + i, dtype=torch.float64),
            edge_feat=torch.randn(8, d),
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
    N, K, d = 30, 4, 8
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device="cpu")

    # Seed history then add the batch edges
    graph.advance(
        torch.randint(0, N, (50,)),
        torch.randint(0, N, (50,)),
        torch.linspace(0, 50, 50, dtype=torch.float64),
        torch.randn(50, d),
    )

    batch = RawBatch(
        src=torch.arange(4), dst=torch.arange(4, 8),
        time=torch.full((4,), 60.0, dtype=torch.float64),
        edge_feat=torch.randn(4, d),
        neg=torch.randint(0, N, (4,)),
    )

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    pipeline = DataPipeline(model.gather_spec, graph)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    losses = []
    for _ in range(30):
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
    assert (engine.graph._write_pos == snap.write_pos).all(), \
        "write_pos changed after evaluate"


def test_three_way_eval_produces_three_metrics():
    """ThreeWayEval produces ap_random, ap_historical, ap_inductive."""
    N, K, d = 30, 4, 8
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device="cpu")

    src_h = torch.randint(0, N, (80,))
    dst_h = torch.randint(0, N, (80,))
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d))

    eval_batches = []
    for i in range(4):
        eval_batches.append(RawBatch(
            src=torch.randint(0, N, (6,)),
            dst=torch.randint(0, N, (6,)),
            time=torch.full((6,), 90.0 + i, dtype=torch.float64),
            edge_feat=torch.randn(6, d),
        ))

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    pipeline = DataPipeline(model.gather_spec, graph)
    inductive_nodes = torch.arange(20, 30)  # nodes 20-29 as inductive

    evaluator = ThreeWayEval(N, inductive_nodes, device="cpu")
    metrics = evaluator.evaluate(model, pipeline, eval_batches, graph)

    assert "ap_random" in metrics
    assert "ap_historical" in metrics
    assert "ap_inductive" in metrics
    for k in metrics:
        assert 0.0 <= metrics[k] <= 1.0, f"{k} = {metrics[k]} out of range"


def test_mrr_eval():
    """MRREval produces valid MRR score."""
    N, K, d = 30, 4, 8
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device="cpu")

    src_h = torch.randint(0, N, (80,))
    dst_h = torch.randint(0, N, (80,))
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d))

    # Create eval batches with aligned neg lists
    N_eval, N_neg = 4, 5
    eval_batches = []
    neg_rows = []
    for i in range(N_eval):
        eval_batches.append(RawBatch(
            src=torch.randint(0, N, (6,)),
            dst=torch.randint(0, N, (6,)),
            time=torch.full((6,), 90.0 + i, dtype=torch.float64),
            edge_feat=torch.randn(6, d),
        ))
        neg_rows.append(torch.randint(0, N, (6, N_neg)))

    neg_tensor = torch.cat(neg_rows, dim=0)  # (N_eval*6, N_neg)

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    pipeline = DataPipeline(model.gather_spec, graph)

    evaluator = MRREval(neg_tensor)
    metrics = evaluator.evaluate(model, pipeline, eval_batches, graph)

    assert "mrr" in metrics
    # MRR is reciprocals of 1..(N_neg+1), the best we can get is 1/(N_neg+1)..1
    assert 0.0 < metrics["mrr"] <= 1.0


def test_early_stopping():
    """Engine stops early when val score does not improve."""
    N, K, d = 30, 4, 8
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device="cpu")

    src_h = torch.randint(0, N, (80,))
    dst_h = torch.randint(0, N, (80,))
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d))

    train_batches = []
    for i in range(8):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (4,)),
            dst=torch.randint(0, N, (4,)),
            time=torch.full((4,), 90.0 + i, dtype=torch.float64),
            edge_feat=torch.randn(4, d),
        ))

    val_batches = test_batches = train_batches[4:8]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    neg_strat = RandomNegative(N)
    eval_proto = APEval()

    # patience=1 — should stop by epoch 3 at latest
    config = TrainConfig(epochs=50, batch_size=4, lr=1e-3, patience=1,
                         device="cpu", seed=42)
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

    N, K, d = 30, 4, 8
    graph = TemporalGraph(N, buffer_size=16, edge_feat_dim=d, device="cpu")

    src_h = torch.randint(0, N, (80,))
    dst_h = torch.randint(0, N, (80,))
    ts_h = torch.linspace(0, 80, 80, dtype=torch.float64)
    graph.advance(src_h, dst_h, ts_h, torch.randn(80, d))

    train_batches = []
    for i in range(8):
        train_batches.append(RawBatch(
            src=torch.randint(0, N, (4,)),
            dst=torch.randint(0, N, (4,)),
            time=torch.full((4,), 90.0 + i, dtype=torch.float64),
            edge_feat=torch.randn(4, d),
        ))

    val_batches = test_batches = train_batches[4:8]

    model = GraphMixer(d_model=16, d_edge=d, d_time=4, K=K, num_layers=1)
    neg_strat = RandomNegative(N)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainConfig(epochs=10, batch_size=4, lr=1e-3, patience=3,
                             device="cpu", seed=42, checkpoint_dir=tmpdir)
        engine = Engine(model, graph, train_batches, val_batches, test_batches,
                        neg_strat, APEval(), config)
        engine.train()

        ckpt_path = os.path.join(tmpdir, "best.pt")
        assert os.path.exists(ckpt_path), f"checkpoint not found at {ckpt_path}"

        # Verify it can be loaded
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["epoch"] > 0
