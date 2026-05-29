# Getting Started with TGEngine

This guide walks you through installing TGEngine, running your first experiment, and understanding the framework's core concepts.

## Installation

```bash
# Basic install
pip install tgengine

# With all optional dependencies
pip install tgengine[all]

# Development install from source
git clone https://github.com/YOUR_USERNAME/tgengine.git
cd tgengine
pip install -e ".[dev]"
```

## Preparing Data

TGEngine supports two dataset formats:

### DyGLib Format (default)

Expected directory structure:
```
datasets/wikipedia/
├── ml_wikipedia.csv       # columns: u, i, ts, label, idx
├── ml_wikipedia.npy       # edge features (N, d_edge)
└── ml_wikipedia_node.npy  # node features (optional)
```

```python
from tgengine import load_dataset

dataset = load_dataset("wikipedia", dataset_path="datasets")
# dataset.src, dataset.dst, dataset.time — full edge list
# dataset.train_end, dataset.val_end — split boundaries
# dataset.edge_feat — (N, 172) padded edge features
```

### TGB Format

```python
from tgengine.core.dataset import load_tgb_dataset

dataset, val_negs, test_negs = load_tgb_dataset("tgbl-wiki", dataset_path="datasets")
# val_negs: (N_val, N_neg) fixed negative candidate lists
# test_negs: (N_test, N_neg) fixed negative candidate lists
```

## Core Concepts

### 1. TemporalGraph

GPU-resident storage for dynamic edges. Uses a frozen CSR (Compressed Sparse Row) for efficient neighbor lookups and a ring buffer for incremental updates.

```python
from tgengine import TemporalGraph

graph = TemporalGraph(
    num_nodes=9228,
    buffer_size=32,        # ring buffer capacity per node
    edge_feat_dim=172,
    device="cuda",
)

# Add edges (typically done by Engine automatically)
graph.advance(src, dst, timestamps, edge_features)

# Query recent neighbors
neighbors = graph.recent(node_ids, query_times, k=32)
# neighbors.neighbor_ids: (B, K)
# neighbors.timestamps:   (B, K)
# neighbors.edge_feats:   (B, K, d)
# neighbors.mask:         (B, K) — True for valid entries
```

### 2. GatherSpec + PreparedBatch

The key abstraction that separates model logic from data preparation:

```python
from tgengine import GatherSpec, NeighborSpec

# Model declares what it needs (static, at class level)
class MyModel(TemporalModel):
    gather_spec = GatherSpec(
        neighbors=NeighborSpec(k=32),  # 32 most recent neighbors
    )
```

The `DataPipeline` reads this spec and produces a `PreparedBatch`:
```python
# PreparedBatch fields:
batch.src            # (B,) source node IDs
batch.dst            # (B,) destination node IDs  
batch.neg            # (B,) negative node IDs
batch.time           # (B,) timestamps
batch.edge_feat      # (B, d) edge features

batch.src_neighbors  # NeighborData for source nodes
batch.dst_neighbors  # NeighborData for destination nodes
batch.neg_neighbors  # NeighborData for negative nodes
# Each NeighborData has: .neighbor_ids, .timestamps, .edge_feats, .mask
```

### 3. TemporalModel

Base class for all models. You only need to implement:

```python
class MyModel(TemporalModel):
    gather_spec = GatherSpec(...)  # what data you need

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        # compute embeddings and return scores
        ...
        return ModelOutput(
            pos_score=pos_scores,   # (B,) logits for positive edges
            neg_score=neg_scores,   # (B,) logits for negative edges
            loss=loss,              # scalar training loss
        )
```

Optional methods for stateful models (e.g., TGN):
- `evolve(src, dst, time, edge_feat)` — update internal state after each batch
- `freeze() -> state` — snapshot state before eval
- `thaw(state)` — restore state after eval

Optional methods for MRR evaluation:
- `encode_nodes(neighbors, times) -> embeddings` — encode nodes independently
- `score_pairs(src_emb, dst_emb) -> scores` — score pairs from embeddings

### 4. Engine

Manages the complete training lifecycle:

```python
from tgengine import Engine, TrainConfig, APEval, RandomNegative

engine = Engine(
    model=model,
    graph=graph,
    train_batches=dataset.get_batches("train", batch_size=200),
    val_batches=dataset.get_batches("val", batch_size=200),
    test_batches=dataset.get_batches("test", batch_size=200),
    neg_strategy=RandomNegative(dataset.num_nodes),
    eval_protocol=APEval(),
    config=TrainConfig(
        epochs=100,
        lr=1e-4,
        eval_strategy="adaptive",  # smart eval scheduling
        patience=0,                # 0 = run all epochs
        result_dir="results/",     # auto-save result.json
    ),
)

test_metrics = engine.train()
```

## Eval Protocols

### AP / AUC-ROC (Binary Classification)

```python
from tgengine import APEval, AUCEval

# AP only (default)
eval_proto = APEval()

# AP + AUC together
eval_proto = APEval(include_auc=True)

# AUC only
eval_proto = AUCEval()
```

### Three-Way Negative Evaluation

```python
from tgengine import ThreeWayEval

eval_proto = ThreeWayEval(
    num_nodes=dataset.num_nodes,
    inductive_nodes=inductive_node_tensor,
    device="cuda",
)
# Returns: {"ap_random": ..., "ap_historical": ..., "ap_inductive": ...}
```

### MRR / Hits@K (Ranking with Fixed Negatives)

```python
from tgengine import MRREval, HitsEval

# For TGB-style evaluation with fixed negative lists
eval_proto = MRREval(neg_lists=test_neg_tensor)  # (N_test, N_neg)
eval_proto = HitsEval(neg_lists=test_neg_tensor, ks=[1, 3, 10])
```

## Negative Sampling Strategies

```python
from tgengine import (
    RandomNegative,          # uniform random
    HistoricalNegative,      # reservoir-sampled historical edges
    InductiveNegative,       # sample from unseen nodes
    InBatchNegative,         # use other positive dst in batch
    FixedNegative,           # TGB fixed negative lists
)

# Historical with reservoir sampling (GPU, O(B))
neg = HistoricalNegative(num_nodes=9228, pool_size=512, device="cuda")

# In-batch with random mixing
neg = InBatchNegative(num_nodes=9228, mix_random=0.3)
```

## Eval Scheduling Strategies

```python
TrainConfig(
    # Option 1: Adaptive (default, recommended)
    eval_strategy="adaptive",
    loss_threshold=0.02,    # eval when loss changes > 2%
    max_eval_gap=10,        # force eval every 10 epochs max
    min_eval_gap=1,         # allow eval every epoch if loss is volatile

    # Option 2: Fixed interval
    eval_strategy="every_n",
    eval_every=5,           # eval every 5 epochs

    # Option 3: Every epoch (TGM-style)
    eval_strategy="all",
)
```

## Multi-Run Experiments

```python
from tgengine.engine import run_experiment

def build_engine(seed):
    dataset = load_dataset("wikipedia", dataset_path="datasets")
    graph = TemporalGraph(...)
    model = DyGFormer(...)
    return Engine(model, graph, ..., config=TrainConfig(seed=seed))

results = run_experiment(
    build_fn=build_engine,
    seeds=[1, 2, 3, 4, 5],
    result_dir="results/dygformer_wiki/",
)
# Prints: ap: 0.9908 +- 0.0012
# Saves: results/dygformer_wiki/experiment.json
```

## YAML Configuration

```yaml
# configs/my_experiment.yaml
model:
  name: DyGFormer
  d_edge: 172
  d_time: 100
  K: 32
  num_layers: 2
  num_heads: 2

train:
  epochs: 100
  lr: 1e-4
  batch_size: 200
  eval_strategy: adaptive
  patience: 0
  result_dir: results/

dataset:
  name: wikipedia
  path: datasets
```

```bash
python -m tgengine.run --config configs/my_experiment.yaml
```

## Next Steps

- See [Adding a New Model](adding_models.md) for a step-by-step guide
- See [API Reference](api_reference.md) for complete API documentation
- Check `examples/` for runnable scripts
