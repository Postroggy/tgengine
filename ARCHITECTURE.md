# TGEngine Architecture

## Overview

TGEngine is a high-performance framework for Continuous-Time Dynamic Graph (CTDG) learning.
It is designed for both rapid model prototyping and production-grade training at scale.

## Design Philosophy

1. **Separation of data access and computation** — Models declare what data they need (GatherSpec); the framework optimizes how to fetch it.
2. **Global optimization of data access** — All graph operations are batched and executed in a single kernel launch.
3. **Minimal concepts** — Four layers, four core types. Nothing more.
4. **Performance by default** — GPU-resident storage, zero-copy batch assembly, async prefetch pipeline.
5. **Innovation at the right layer** — Researchers only write SequenceEncoder subclasses or compose existing components.

## Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: Models (ready-to-use complete models)              │
│  DyGFormer / TGN / GraphMixer / DyGMamba / EdgeBank / ...    │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Components (pre-built building blocks)             │
│  SequenceEncoders / TimeEncoders / Decoders / NegStrategies  │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: DataPipeline (optimized data preparation)          │
│  TemporalGraph + GatherSpec + DataPipeline + PreparedBatch   │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Engine (lifecycle management)                      │
│  TrainLoop + EvalProtocol + Checkpoint + AsyncPrefetch       │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Dataset (csv/npy files on disk)
    │  [load once at startup]
    ▼
TemporalGraph (GPU-resident circular buffer)
    │  [Engine iterates chronologically]
    ▼
RawBatch (src, dst, neg, time — integer tensors)
    │  [DataPipeline.prepare() — single fused kernel]
    ▼
PreparedBatch (neighbors, timestamps, edge_feats, co_occurrence — all on GPU)
    │  [Model.forward() — pure neural network computation]
    ▼
ModelOutput (loss, pos_score, neg_score)
    │  [backward + optimizer step]
    │  [Graph.advance() — append new edges to buffer]
    ▼
[next batch]
```

## Core Types

### 1. TemporalGraph

GPU-resident temporal graph storage. Maintains a circular buffer per node for O(1) recent
neighbor access. All operations are batch-vectorized (no Python loops in hot path).

**Key invariant**: The graph represents "all events observed so far". It grows during training
via `advance()`. For evaluation, `snapshot()`/`restore()` allow temporary rollback.

**Internal storage**:
- `neighbor_ids`: Tensor(num_nodes, buffer_size) — ring buffer of neighbor node IDs
- `neighbor_times`: Tensor(num_nodes, buffer_size) — timestamps of interactions
- `neighbor_feats`: Tensor(num_nodes, buffer_size, d_edge) — edge features
- `write_pos`: Tensor(num_nodes) — current write position per node

**Operations**:
- `recent(nodes, times, k)` → NeighborData — most recent k neighbors before query time
- `co_neighbors(src, dst, times)` → Tensor — co-occurrence counts between src/dst neighbor sets
- `advance(src, dst, time, edge_feat)` — append new edges
- `snapshot()` → Snapshot — lightweight checkpoint (O(1), just copies write_pos)
- `restore(snap)` — restore to snapshot

### 2. GatherSpec

Static declaration of what data a model needs. Defined as a class attribute on the model.
The DataPipeline reads this at initialization time to build an optimized execution plan.

```python
@dataclass
class GatherSpec:
    neighbors: NeighborSpec
    co_occurrence: bool = False
    memory: bool = False

@dataclass
class NeighborSpec:
    k: int = 32
    strategy: str = 'recency'  # recency | uniform | time_weighted
    hops: int = 1
    include_edge_feat: bool = True
    for_nodes: tuple = ('src', 'dst', 'neg')
```

### 3. PreparedBatch

The fully-assembled input to a model. All tensors are on GPU, ready for neural computation.
No further data fetching needed inside forward().

```python
@dataclass
class PreparedBatch:
    src: Tensor              # (B,)
    dst: Tensor              # (B,)
    neg: Tensor              # (B,) or (B, N_neg) for MRR eval
    time: Tensor             # (B,)
    src_neighbors: NeighborData
    dst_neighbors: NeighborData
    neg_neighbors: NeighborData
    co_occurrence: Tensor | None   # (B, k) or None
    memory_emb: Tensor | None      # (B, d) or None — for TGN-style
```

### 4. TemporalModel

Base class for all models. Defines gather_spec + forward().

```python
class TemporalModel(nn.Module):
    gather_spec: GatherSpec

    def forward(self, batch: PreparedBatch) -> ModelOutput: ...

    # Optional: for stateful models (TGN)
    def evolve(self, raw_batch: RawBatch) -> None: ...
    def freeze(self) -> Any: ...
    def thaw(self, state: Any) -> None: ...

    # Optional: for MRR eval (independent encoding)
    def encode_nodes(self, neighbors: NeighborData, times: Tensor) -> Tensor: ...
    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor: ...
```

## Performance Strategy

### V1: Vectorized PyTorch (no custom CUDA)
- Circular buffer operations via `torch.gather` / `torch.scatter`
- All operations batch-vectorized (no Python loops)
- Expected speedup vs DyGLib: 5-8x

### V2: Custom CUDA Kernels
- Fused temporal neighbor sampling kernel
- Fused co-occurrence counting kernel
- Expected additional speedup: 2-3x (total 10-20x vs DyGLib)

### Async Prefetch Pipeline
- While GPU computes forward/backward for batch_i, DataPipeline prepares batch_i+1
  on a separate CUDA stream
- Hides data preparation latency completely for compute-bound models

## Evaluation Protocols

All protocols are pluggable via `EvalProtocol` interface:

| Protocol | Description | Metric |
|----------|-------------|--------|
| `APEval` | 1 pos + 1 random neg per edge | Average Precision |
| `ThreeWayEval` | random + historical + inductive neg | AP per strategy |
| `MRREval` | 1 pos + N fixed neg (TGB) | Mean Reciprocal Rank |
| `HitsEval` | 1 pos + N neg, top-K | Hits@K |

MRR eval optimization: deduplicate negative nodes across the batch, encode unique nodes
only once, then gather back. Typically 5-7x faster than naive per-edge evaluation.

## Stateful Models (TGN-style)

Models with persistent state (e.g., node memory) implement three additional methods:
- `evolve(raw_batch)`: update internal state after each training batch
- `freeze()`: checkpoint state before evaluation
- `thaw(state)`: restore state after evaluation

The Engine calls these automatically at the right lifecycle points. Stateless models
(DyGFormer, Mamba, etc.) leave these as no-ops.

## Directory Structure

```
tgengine/
├── ARCHITECTURE.md          ← this file
├── pyproject.toml           ← project metadata, dependencies
├── tgengine/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── temporal_graph.py    # TemporalGraph (circular buffer)
│   │   ├── batch.py             # RawBatch, PreparedBatch, NeighborData
│   │   ├── gather_spec.py       # GatherSpec, NeighborSpec
│   │   └── dataset.py           # Dataset loading (csv/npy → TemporalGraph)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # DataPipeline (GatherSpec → PreparedBatch)
│   │   ├── negatives.py         # Negative sampling strategies
│   │   └── prefetch.py          # Async prefetch logic
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── time_encoding.py     # Time2Vec, Fourier, Harmonic, etc.
│   │   ├── seq_encoder.py       # SequenceEncoder base + implementations
│   │   ├── decoder.py           # BilinearDecoder, MergeLayer, etc.
│   │   ├── memory.py            # GRUMemory, EMAMemory (for TGN-style)
│   │   └── co_neighbor.py       # CoNeighborEncoder
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # TemporalModel base class
│   │   ├── dygformer.py
│   │   ├── tgn.py
│   │   ├── graphmixer.py
│   │   ├── dygmamba.py
│   │   └── edgebank.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── engine.py            # Main Engine class
│   │   ├── trainer.py           # Training loop logic
│   │   ├── evaluator.py         # EvalProtocol + implementations
│   │   ├── early_stopping.py
│   │   └── metrics.py           # AP, AUC, MRR computation
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── logging.py
│       └── config.py            # Config loading (yaml + dataclass)
├── configs/
│   ├── dygformer/
│   │   ├── wikipedia.yaml
│   │   └── reddit.yaml
│   ├── dygmamba/
│   │   └── wikipedia.yaml
│   └── ...
├── tests/
│   ├── test_temporal_graph.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── ...
└── examples/
    ├── train_dygformer.py       # minimal script: ~20 lines
    ├── train_custom_model.py    # how to define a new model
    └── eval_tgb.py              # TGB MRR evaluation example
```

## Design Decisions & Tradeoffs

| Decision | Tradeoff | Rationale |
|----------|----------|-----------|
| GatherSpec is static | Can't do conditional data fetching in forward | Enables full pipeline optimization |
| PreparedBatch is the only model input | Model can't access TemporalGraph directly | Clean separation, testable models |
| Circular buffer (not CSR) | Fixed max_k per node, can't query arbitrary ranges | O(1) append, O(1) recent-k query, proven by TGM |
| No Hook system | Less composable than TGM for data transforms | Simpler mental model, optimization in Pipeline |
| Single training script pattern | Less flexible than per-model scripts | Reproducibility, fair comparison |

## Comparison to Existing Frameworks

| | DyGLib | TGM | TGEngine |
|---|---|---|---|
| Data access | CPU Python dict | GPU Hook pipeline | GPU Pipeline + GatherSpec |
| Model interface | free-form forward() | forward(DGBatch) | forward(PreparedBatch) |
| Optimization | none | Hook batching | Pipeline global fusion + prefetch |
| Add new model | modify train script | write encoder + hooks + example | write Model class (GatherSpec + forward) |
| Eval protocols | AP only | TGB MRR | AP + 3-way + MRR (pluggable) |
| Performance | 1x | 5-8x | 8-15x (target) |
