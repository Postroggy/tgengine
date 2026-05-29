# API Reference

## Core

### `tgengine.load_dataset(name, dataset_path, val_ratio=0.15, test_ratio=0.15) -> TemporalDataset`

Load a DyGLib-format CTDG dataset. Features are zero-padded to 172 dimensions. Splits use time-quantile thresholds.

### `tgengine.TemporalGraph(num_nodes, buffer_size, edge_feat_dim, device)`

GPU-resident temporal graph with CSR storage + ring buffer.

| Method | Description |
|--------|-------------|
| `advance(src, dst, time, edge_feat)` | Add edges to the graph |
| `recent(nodes, times, k) -> NeighborData` | Query k most recent neighbors before given times |
| `freeze_csr()` | Freeze current edges into sorted CSR for fast queries |
| `snapshot() -> state` | Capture current state |
| `restore(state)` | Restore to a previous state |

### `tgengine.GatherSpec(neighbors=NeighborSpec(k=32))`

Static declaration of a model's data requirements.

### `tgengine.PreparedBatch`

| Field | Type | Description |
|-------|------|-------------|
| `src` | `Tensor (B,)` | Source node IDs |
| `dst` | `Tensor (B,)` | Destination node IDs |
| `neg` | `Tensor (B,)` | Negative node IDs |
| `time` | `Tensor (B,)` | Timestamps |
| `edge_feat` | `Tensor (B, d)` | Edge features |
| `src_neighbors` | `NeighborData` | Source node neighbor data |
| `dst_neighbors` | `NeighborData` | Destination node neighbor data |
| `neg_neighbors` | `NeighborData` | Negative node neighbor data |

### `tgengine.NeighborData`

| Field | Type | Description |
|-------|------|-------------|
| `neighbor_ids` | `Tensor (B, K)` | Neighbor node IDs |
| `timestamps` | `Tensor (B, K)` | Edge timestamps |
| `edge_feats` | `Tensor (B, K, d)` | Edge features |
| `mask` | `Tensor (B, K)` | True for valid entries |

---

## Models

### `tgengine.TemporalModel` (base class)

| Attribute/Method | Description |
|-----------------|-------------|
| `gather_spec` | Class attribute: `GatherSpec` |
| `forward(batch) -> ModelOutput` | Main computation |
| `evolve(src, dst, time, edge_feat)` | Update state (stateful models) |
| `freeze() -> state` | Snapshot state before eval |
| `thaw(state)` | Restore state after eval |
| `supports_independent_encode` | `bool`, enables fast MRR path |
| `encode_nodes(neighbors, times)` | Encode nodes independently |
| `score_pairs(src_emb, dst_emb)` | Score pre-encoded pairs |

### `tgengine.ModelOutput`

| Field | Type | Description |
|-------|------|-------------|
| `pos_score` | `Tensor (B,)` | Positive edge logits |
| `neg_score` | `Tensor (B,)` | Negative edge logits |
| `loss` | `Tensor (scalar)` | Training loss |

### Built-in Models

| Model | Key Parameters |
|-------|---------------|
| `DyGFormer(d_edge, d_time, K, num_layers, num_heads, d_channel)` | Patched neighbor attention |
| `TGN(d_edge, d_time, d_mem, K, num_layers, num_heads)` | Memory-based GNN |
| `GraphMixer(d_model, d_edge, d_time, K, num_layers, dropout)` | MLP-Mixer temporal |
| `DyGMamba(d_edge, d_time, K, d_state, num_layers)` | Mamba SSM encoder |
| `FreeDyG(d_edge, d_time, K, num_layers, num_heads)` | Frequency-domain |

---

## Engine

### `tgengine.Engine`

```python
Engine(model, graph, train_batches, val_batches, test_batches,
       neg_strategy, eval_protocol, config, inductive_edges=None,
       eval_neg_strategy=None)
```

| Method | Description |
|--------|-------------|
| `train(resume=False) -> dict` | Run full training, return best test metrics |
| `save_checkpoint(path)` | Save model + optimizer + scaler state |
| `load_checkpoint(path) -> epoch` | Load checkpoint, return epoch number |

### `tgengine.TrainConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `epochs` | 100 | Total epochs |
| `batch_size` | 200 | Batch size |
| `lr` | 1e-4 | Learning rate |
| `patience` | 0 | Early stop patience (0=disabled) |
| `device` | `"cuda"` | Device |
| `seed` | 42 | Random seed |
| `eval_strategy` | `"adaptive"` | `"adaptive"` / `"every_n"` / `"all"` |
| `eval_every` | 1 | Interval for `"every_n"` strategy |
| `min_eval_gap` | 1 | Min epochs between adaptive evals |
| `max_eval_gap` | 10 | Max epochs between adaptive evals |
| `loss_threshold` | 0.02 | Relative loss change to trigger eval |
| `use_amp` | False | Mixed precision training |
| `grad_clip` | 1.0 | Gradient clipping (0=disabled) |
| `warmup_steps` | 0 | LR warmup steps |
| `result_dir` | None | Auto-save result.json |
| `checkpoint_dir` | None | Auto-save best checkpoint |
| `wandb_project` | None | W&B project name |

### `tgengine.engine.run_experiment(build_fn, seeds=None, n_runs=5, result_dir=None)`

Run multiple training runs and report mean +- std.

---

## Eval Protocols

| Class | Returns |
|-------|---------|
| `APEval(include_auc=False)` | `{"ap": float}` or `{"ap": float, "auc": float}` |
| `AUCEval()` | `{"auc": float}` |
| `ThreeWayEval(num_nodes, inductive_nodes, device)` | `{"ap_random": ..., "ap_historical": ..., "ap_inductive": ...}` |
| `MRREval(neg_lists)` | `{"mrr": float}` |
| `HitsEval(neg_lists, ks=[1,3,10])` | `{"hits@1": ..., "hits@3": ..., "hits@10": ...}` |

---

## Neural Components (`tgengine.nn`)

### Time Encoders

| Class | Input | Output |
|-------|-------|--------|
| `Time2Vec(d_model)` | `NeighborData` | `(B, K, d_model)` with temporal encoding added |
| `HarmonicEncoder(d_model)` | `(B, K)` timestamps | `(B, K, d_model)` time features |
| `FixedCosineTimeEncoder(d_model)` | `(B, K)` timestamps | `(B, K, d_model)` time features |

### Sequence Encoders

All follow the signature: `(B, K, d_in) + mask -> (B, d_out)`

| Class | Description |
|-------|-------------|
| `TransformerSeqEncoder(d_in, n_layers, n_heads)` | Multi-head self-attention |
| `GRUSeqEncoder(d_in, n_layers)` | Bidirectional GRU |
| `MambaSeqEncoder(d_in, n_layers)` | Mamba state-space model |
| `MeanPoolEncoder(d_in)` | Masked mean pooling |

### Decoders

All follow: `(src_emb, dst_emb, neg_emb) -> ModelOutput`

| Class | Description |
|-------|-------------|
| `ConcatDecoder(d_in)` | Concatenate + MLP |
| `BilinearDecoder(d_in)` | Bilinear scoring |
| `ConcatMLPDecoder(d_in, hidden_dim)` | Concat + deeper MLP |
| `MergeDecoder(d_in)` | DyGLib MergeLayer compatible |

---

## Negative Sampling

| Class | Description |
|-------|-------------|
| `RandomNegative(num_nodes, valid_dst_nodes=None)` | Uniform random |
| `HistoricalNegative(num_nodes, pool_size=512, device)` | Reservoir-sampled history |
| `InductiveNegative(inductive_nodes)` | Sample from unseen nodes |
| `InBatchNegative(num_nodes, mix_random=0.0)` | Use other batch destinations |
| `FixedNegative(neg_lists)` | TGB fixed negative lists |
| `CollisionFreeNegative(num_nodes)` | Random with collision avoidance |
