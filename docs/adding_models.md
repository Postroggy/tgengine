# Adding a New Model

This guide shows how to implement a new CTDG model in TGEngine, from minimal to full-featured.

## Step 1: Minimal Model (3 things to define)

```python
from tgengine import TemporalModel, GatherSpec, NeighborSpec, PreparedBatch, ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalModel(TemporalModel):
    # 1. Declare data requirements
    gather_spec = GatherSpec(neighbors=NeighborSpec(k=20))

    def __init__(self, d_edge: int = 172, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_edge, d_model)
        self.out = nn.Linear(d_model * 2, 1)

    # 2. Implement forward
    def forward(self, batch: PreparedBatch) -> ModelOutput:
        # Mean-pool neighbor edge features as node embedding
        src_emb = self.proj(batch.src_neighbors.edge_feats).mean(dim=1)
        dst_emb = self.proj(batch.dst_neighbors.edge_feats).mean(dim=1)
        neg_emb = self.proj(batch.neg_neighbors.edge_feats).mean(dim=1)

        pos_score = self.out(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)
        neg_score = self.out(torch.cat([src_emb, neg_emb], dim=-1)).squeeze(-1)

        # 3. Return ModelOutput
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )
        return ModelOutput(pos_score=pos_score, neg_score=neg_score, loss=loss)
```

That's it. This model works with Engine, all eval protocols, and all negative strategies.

## Step 2: Use Built-in Components

TGEngine provides reusable building blocks so you don't reimplement common patterns:

```python
from tgengine.nn import Time2Vec, TransformerSeqEncoder, ConcatDecoder

class BetterModel(TemporalModel):
    gather_spec = GatherSpec(neighbors=NeighborSpec(k=32))

    def __init__(self):
        super().__init__()
        self.time_enc = Time2Vec(d_model=172)
        self.encoder = TransformerSeqEncoder(d_in=172, n_layers=2, n_heads=2)
        self.decoder = ConcatDecoder(d_in=172)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        # Time2Vec adds temporal encoding to neighbor features
        src_feat = self.time_enc(batch.src_neighbors)  # (B, K, d)
        dst_feat = self.time_enc(batch.dst_neighbors)
        neg_feat = self.time_enc(batch.neg_neighbors)

        # Encode sequences into fixed-size embeddings
        src_emb = self.encoder(src_feat, batch.src_neighbors.mask)  # (B, d)
        dst_emb = self.encoder(dst_feat, batch.dst_neighbors.mask)
        neg_emb = self.encoder(neg_feat, batch.neg_neighbors.mask)

        # Decoder handles scoring + loss computation
        return self.decoder(src_emb, dst_emb, neg_emb)
```

### Available Components

| Category | Options |
|----------|---------|
| **Time Encoders** | `Time2Vec`, `HarmonicEncoder`, `FixedCosineTimeEncoder` |
| **Sequence Encoders** | `TransformerSeqEncoder`, `GRUSeqEncoder`, `MambaSeqEncoder`, `MeanPoolEncoder` |
| **Decoders** | `ConcatDecoder`, `BilinearDecoder`, `ConcatMLPDecoder`, `MergeDecoder` |

## Step 3: Stateful Models (TGN-style)

If your model maintains per-node state (like TGN's memory), implement three additional methods:

```python
class StatefulModel(TemporalModel):
    gather_spec = GatherSpec(neighbors=NeighborSpec(k=10))

    def __init__(self, num_nodes: int, d_mem: int = 172):
        super().__init__()
        self.memory = NodeMemory(num_nodes, d_mem)
        # ... other layers ...

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        mem = self.memory.get(batch.src)  # read current memory
        # ... use memory in computation ...
        return ModelOutput(...)

    def evolve(self, src, dst, time, edge_feat):
        """Update memory after each training batch."""
        self.memory.update(src, dst, time, edge_feat)

    def freeze(self):
        """Snapshot memory state before evaluation."""
        return self.memory.snapshot()

    def thaw(self, state):
        """Restore memory state after evaluation."""
        self.memory.restore(state)
```

The Engine calls these automatically at the right times.

## Step 4: MRR Evaluation Support

For TGB-style ranking evaluation, implement `encode_nodes` and `score_pairs`:

```python
class MRRCapableModel(TemporalModel):
    supports_independent_encode = True  # enable fast MRR path

    def encode_nodes(self, neighbors: NeighborData, times: Tensor) -> Tensor:
        """Encode nodes independently (no src-dst coupling)."""
        feat = self.time_enc(neighbors)
        return self.encoder(feat, neighbors.mask)  # (N, d)

    def score_pairs(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        """Score pre-encoded (src, dst) pairs."""
        return (src_emb * dst_emb).sum(dim=-1)  # (N,)
```

This enables the fast MRR evaluation path (encode unique nodes once, score all pairs via dot product) instead of the slow path (run full forward for each candidate).

## Step 5: Run It

```python
from tgengine import load_dataset, TemporalGraph, Engine, TrainConfig, APEval, RandomNegative

dataset = load_dataset("wikipedia", dataset_path="datasets")
graph = TemporalGraph(dataset.num_nodes, buffer_size=32,
                      edge_feat_dim=dataset.edge_feat_dim, device="cuda")

model = BetterModel()
engine = Engine(
    model, graph,
    train_batches=dataset.get_batches("train", 200),
    val_batches=dataset.get_batches("val", 200),
    test_batches=dataset.get_batches("test", 200),
    neg_strategy=RandomNegative(dataset.num_nodes),
    eval_protocol=APEval(),
    config=TrainConfig(epochs=50, lr=1e-4),
)
results = engine.train()
print(f"Test AP: {results['ap']:.4f}")
```

## Checklist

- [ ] `gather_spec` is a class attribute (not instance)
- [ ] `forward()` returns `ModelOutput` with `pos_score`, `neg_score`, `loss`
- [ ] Stateful models implement `evolve()`, `freeze()`, `thaw()`
- [ ] MRR-capable models set `supports_independent_encode = True`
- [ ] No direct `TemporalGraph` access in `forward()` (use `PreparedBatch` only)
