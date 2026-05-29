<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img alt="TGEngine" src="assets/logo.svg" width="420">
  </picture>
</p>

<p align="center">
  <strong>High-Performance Continuous-Time Dynamic Graph Learning Framework</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.0+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/lang-中文-red" alt="中文"></a>
</p>

<p align="center">
  <a href="#-installation">Installation</a> &bull;
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-architecture">Architecture</a> &bull;
  <a href="#-models">Models</a> &bull;
  <a href="#-benchmarks">Benchmarks</a> &bull;
  <a href="docs/getting_started.md">Docs</a>
</p>

---

## Why TGEngine?

Existing CTDG frameworks force you to rewrite training loops for every new model and leave performance on the table with CPU-bound data pipelines. TGEngine solves both:

- **60 lines to add a new model** &mdash; declare `GatherSpec`, implement `forward()`, done.
- **1.3&ndash;1.6x faster end-to-end** than DyGLib, with **5&ndash;7x faster data pipeline** via GPU-fused neighbor sampling.
- **Pluggable eval** &mdash; AP, AUC-ROC, MRR, Hits@K, three-way neg splits, all out of the box.
- **Adaptive eval scheduling** &mdash; smart loss-gated validation saves up to 70% eval time on large datasets.

## Installation

```bash
pip install tgengine
```

From source (for development):

```bash
git clone https://github.com/YOUR_USERNAME/tgengine.git
cd tgengine
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, PyTorch 2.0+, CUDA-capable GPU recommended.

## Quick Start

### Train DyGFormer on Wikipedia in 10 lines

```python
from tgengine import (
    load_dataset, TemporalGraph, Engine, TrainConfig,
    APEval, RandomNegative, DyGFormer,
)

dataset = load_dataset("wikipedia", dataset_path="datasets")
graph = TemporalGraph(dataset.num_nodes, buffer_size=32,
                      edge_feat_dim=dataset.edge_feat_dim, device="cuda")

model = DyGFormer(d_edge=172, d_time=100, K=32, num_layers=2, num_heads=2)
engine = Engine(
    model, graph,
    train_batches=dataset.get_batches("train", 200),
    val_batches=dataset.get_batches("val", 200),
    test_batches=dataset.get_batches("test", 200),
    neg_strategy=RandomNegative(dataset.num_nodes),
    eval_protocol=APEval(),
    config=TrainConfig(epochs=100, lr=1e-4, device="cuda"),
)
results = engine.train()  # {"ap": 0.990}
```

### Add a New Model (~60 lines)

```python
from tgengine import TemporalModel, GatherSpec, NeighborSpec, PreparedBatch, ModelOutput
from tgengine.nn import Time2Vec, TransformerSeqEncoder, ConcatDecoder

class MyModel(TemporalModel):
    gather_spec = GatherSpec(neighbors=NeighborSpec(k=20))

    def __init__(self):
        super().__init__()
        self.time_enc = Time2Vec(d_model=172)
        self.encoder = TransformerSeqEncoder(d_in=172, n_layers=2, n_heads=2)
        self.decoder = ConcatDecoder(d_in=172)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self.encoder(
            self.time_enc(batch.src_neighbors), batch.src_neighbors.mask
        )
        dst_emb = self.encoder(
            self.time_enc(batch.dst_neighbors), batch.dst_neighbors.mask
        )
        neg_emb = self.encoder(
            self.time_enc(batch.neg_neighbors), batch.neg_neighbors.mask
        )
        return self.decoder(src_emb, dst_emb, neg_emb)
```

That's it. No training script changes, no data pipeline code. The Engine handles everything.

### Multi-Run Experiment

```python
from tgengine.engine import run_experiment

def build(seed):
    # ... create fresh model, graph, engine with this seed ...
    return engine

results = run_experiment(build, seeds=[1, 2, 3, 4, 5], result_dir="results/")
# ap: 0.9908 +- 0.0012
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Code                           │
│   model = MyModel()     # Just define gather_spec + forward │
│   engine = Engine(...)  # One-liner training                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 4: Models                                            │
│  DyGFormer │ TGN │ GraphMixer │ DyGMamba │ FreeDyG │ ...   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Neural Components  (tgengine.nn)                  │
│  SeqEncoder │ TimeEncoder │ Decoder │ Memory │ CoNeighbor   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Data Pipeline                                     │
│  GatherSpec → DataPipeline → PreparedBatch (fused GPU ops)  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Engine + Eval                                     │
│  Train loop │ Adaptive eval │ AP/AUC/MRR/Hits │ JSON out   │
├─────────────────────────────────────────────────────────────┤
│  Storage: TemporalGraph  (GPU-resident CSR + ring buffer)   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design: GatherSpec + PreparedBatch

Models never touch raw graph data. Instead:

1. **Declare** what you need via `GatherSpec` (neighbor count, features, co-occurrence)
2. **Receive** a `PreparedBatch` with everything pre-fetched and GPU-ready
3. **Compute** pure neural network operations in `forward()`

This separation enables the pipeline to fuse all graph operations into a single optimized GPU kernel call, regardless of which model is running.

## Models

| Model | Paper | Lines of Code | Key Innovation |
|-------|-------|:---:|----------------|
| **DyGFormer** | NeurIPS 2023 | ~380 | Patched neighbor attention |
| **FreeDyG** | AAAI 2024 | ~240 | Frequency-domain encoding |
| **GraphMixer** | ICLR 2023 | ~180 | MLP-Mixer on temporal sequences |
| **TGN** | ICML 2020 | ~120 | Memory + GRU message passing |
| **DyGMamba** | 2024 | ~60 | Mamba SSM for temporal encoding |
| **EdgeBank** | &mdash; | ~20 | Heuristic baseline (no learning) |

### Neural Components (`tgengine.nn`)

| Component | Variants |
|-----------|----------|
| **SequenceEncoder** | `TransformerSeqEncoder`, `GRUSeqEncoder`, `MambaSeqEncoder`, `MeanPoolEncoder` |
| **TimeEncoder** | `Time2Vec`, `HarmonicEncoder`, `FixedCosineTimeEncoder` |
| **Decoder** | `ConcatDecoder`, `BilinearDecoder`, `ConcatMLPDecoder`, `MergeDecoder` |
| **Memory** | `NodeMemory` (TGN-style stateful memory) |
| **CoNeighbor** | `CoNeighborEncoder` (co-occurrence patterns) |

## Evaluation

TGEngine provides a comprehensive, pluggable evaluation system:

| Protocol | Metric | Use Case |
|----------|--------|----------|
| `APEval` | Average Precision | Standard 1v1 binary classification |
| `APEval(include_auc=True)` | AP + AUC-ROC | DyGLib-compatible dual metric |
| `AUCEval` | AUC-ROC | Standalone ROC evaluation |
| `ThreeWayEval` | AP (random / historical / inductive) | Fine-grained negative analysis |
| `MRREval` | Mean Reciprocal Rank | TGB-style ranking with fixed negatives |
| `HitsEval` | Hits@1/3/10 | Top-K ranking quality |

### Adaptive Eval Scheduling

Instead of evaluating every epoch (wasteful) or every N epochs (might miss the peak), TGEngine uses **loss-gated adaptive scheduling**:

- Evaluates when train loss changes significantly (> threshold)
- Guarantees eval on first and last epoch
- Forces eval if too many epochs pass without one (`max_eval_gap`)
- Saves up to **70% eval time** on large datasets vs. evaluate-every-epoch

```python
TrainConfig(
    eval_strategy="adaptive",   # "adaptive" | "every_n" | "all"
    loss_threshold=0.02,        # relative loss change to trigger eval
    max_eval_gap=10,            # max epochs between evals
)
```

## Benchmarks

### End-to-End Training Speed (vs. DyGLib)

<table>
<tr>
<th>Dataset</th><th>K</th><th>TGEngine</th><th>DyGLib</th><th>Speedup</th>
</tr>
<tr><td>Wikipedia</td><td>32</td><td>17.3s</td><td>25.4s</td><td><strong>1.47x</strong></td></tr>
<tr><td>Reddit</td><td>64</td><td>86.2s</td><td>134.8s</td><td><strong>1.56x</strong></td></tr>
<tr><td>LastFM</td><td>512</td><td>153.2s</td><td>205.8s</td><td><strong>1.34x</strong></td></tr>
</table>

> Per-epoch times on RTX 4080. DyGFormer architecture, identical hyperparameters.

### Data Pipeline Speedup

| Component | TGEngine | DyGLib | Speedup |
|-----------|----------|--------|---------|
| Neighbor Sampling | GPU `torch.searchsorted` | CPU Python loop | **5&ndash;7x** |
| Data Transfer | Zero-copy (all GPU) | numpy &rarr; torch &rarr; `.to(device)` | **3&ndash;5x** |
| CUDA Kernel (Triton) | Fused temporal sampling | N/A | **3&ndash;28x** |

### Accuracy Alignment

| Model | Dataset | TGEngine AP | DyGLib AP | Gap |
|-------|---------|:-----------:|:---------:|:---:|
| DyGFormer | Wikipedia | 0.9908 | 0.9903 | +0.05% |
| DyGFormer | UCI | 0.9526 | 0.9613 | -0.9% |
| GraphMixer | UCI | 0.9315 | 0.9331 | -0.2% |
| GraphMixer | Wikipedia | 0.9644 | 0.9725 | -0.8% |

## Structured Output

Every training run can produce a standardized `result.json`:

```python
TrainConfig(result_dir="results/dygformer_wiki/")
```

```json
{
  "model": "DyGFormer",
  "config": {"epochs": 100, "lr": 0.0001, "eval_strategy": "adaptive", ...},
  "result": {"best_val": 0.9912, "best_epoch": 87, "test_metrics": {"ap": 0.9908}},
  "stats": {"total_epochs": 100, "eval_count": 15, "elapsed_seconds": 1842.3}
}
```

## Project Structure

```
tgengine/
├── core/            # TemporalGraph, Batch, GatherSpec, Dataset
│   ├── temporal_graph.py   # GPU-resident CSR + ring buffer storage
│   ├── gather_spec.py      # Static data requirement declarations
│   ├── batch.py            # RawBatch → PreparedBatch types
│   └── dataset.py          # DyGLib & TGB format loaders
├── pipeline/        # GatherSpec → PreparedBatch execution
│   ├── __init__.py         # DataPipeline (fused GPU ops)
│   └── negatives.py        # Negative sampling strategies
├── nn/              # Reusable neural components
│   ├── seq_encoder.py      # Transformer, GRU, Mamba, MeanPool
│   ├── time_encoding.py    # Time2Vec, Harmonic, FixedCosine
│   ├── decoder.py          # Concat, Bilinear, MLP decoders
│   ├── memory.py           # TGN-style node memory
│   └── co_neighbor.py      # Co-occurrence encoding
├── models/          # Complete model implementations
│   ├── dygformer.py        # DyGFormer (85 lines)
│   ├── tgn.py              # TGN (90 lines)
│   ├── graphmixer.py       # GraphMixer (70 lines)
│   ├── dygmamba.py         # DyGMamba (65 lines)
│   └── freedyg.py          # FreeDyG (80 lines)
├── engine/          # Training & evaluation lifecycle
│   ├── __init__.py         # Engine (train loop, state management)
│   ├── config.py           # TrainConfig
│   └── eval.py             # AP, AUC, MRR, Hits, ThreeWay protocols
└── utils/           # Logging, config, seeding
```

## Configuration

### YAML-Driven Training

```yaml
# configs/dygformer_wiki.yaml
model:
  name: DyGFormer
  d_edge: 172
  d_time: 100
  K: 32
  num_layers: 2

train:
  epochs: 100
  lr: 1e-4
  batch_size: 200
  eval_strategy: adaptive
  patience: 0          # 0 = run all epochs (TGM-style)

dataset:
  name: wikipedia
  path: datasets
```

```bash
python -m tgengine.run --config configs/dygformer_wiki.yaml
```

## Citation

If you find TGEngine useful, please cite:

```bibtex
@software{tgengine2025,
  title={TGEngine: High-Performance Continuous-Time Dynamic Graph Learning Framework},
  year={2025},
  url={https://github.com/YOUR_USERNAME/tgengine}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
