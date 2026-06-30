# Foundation Model 实现与预训练日志

> 目标：实现 d=512/K=64 的跨域 CTDG 基础模型，三任务预训练（MTM+NTP+LP）
> 设计文档：`docs/foundation_model_design.md`

## 2026-06-30 模型实现完成

### 架构（d=512, K=64, ~34M params）

```
InputTokenizer (domain-agnostic, no node IDs)
  ├── edge_feat_proj: 172 → 512
  ├── time_enc (trainable sinusoidal, d_time=64) + time_proj → 512
  ├── pair_proj (pair_count + pair_recency) → 512
  └── ctx_proj (5 node context features) → 512, broadcast to K positions
  → additive assembly → token_norm → (B, K, 512)

Encoder: TimeAwareMambaBlock×8 + GraphCrossAttention×1 (every 4 layers)
  → final LayerNorm → (B, K, 512)

Heads:
  ├── MTMHead: decoder MLP, reconstruct edge_feat+pair_feat at masked positions
  ├── NTPHead: pool → MLP → predict next-event time encoding (d_time)
  └── LPHead: pool → dot-product link prediction
```

### 关键实现细节

1. **Domain-agnostic tokenization**：完全不用 node ID。token = edge_proj + time_proj + pair_proj + ctx_proj（加法组合）。跨域可迁移。

2. **Block-wise MTM masking**：mask 连续 block（TGPM ICML 2026 证明是信息论必要条件），block_size=4, mask_ratio=0.15。EMA encoder 生成稳定重建目标（BYOL-style，防 collapse）。

3. **NTP 目标**：预测下一个交互的时间编码向量（不是原始时间值），对齐 Hawkes 过程。从 pooled src 表示预测 Δt_0 的时间编码。

4. **效率优化**：NTP 和 LP 共用一次 clean src encoding（不过 mask），MTM 单独 encode（要 mask）。从 4 次 src encode 降到 2 次。

5. **CSR 预加载**：训练前把所有训练边 `graph.advance` 进 TemporalGraph 再 `freeze_csr`，否则邻居查询返回全 padding（loss=0 bug）。

### 修复的 bug

1. **EmbeddingBundle 导入位置**：在 models/base.py 不在 core/batch.py
2. **dtype 不匹配**：TemporalGraph 用 float64 存 timestamps，所有 dt 计算需 `.float()` 转 float32
3. **pair_features gather 越界**：padding 位置的 neighbor_ids 是 PADDING_ID，需 `clamp(min=0)` + mask 保护
4. **AMP + Mamba 不兼容**：Mamba 的 Conv1d 在 autocast fp16 下 cuDNN 报 CUDNN_STATUS_NOT_INITIALIZED。禁用 AMP，靠 batch_size 控制显存
5. **CSR 未预加载**：自定义训练循环需手动预加载训练边（Engine 在 init 自动做）

### 验证

- 单元测试 `tests/test_foundation.py`：16 个测试覆盖 InputTokenizer / heads / FoundationModel
- 端到端：d=512/K=64/batch128 在 RTX 4080 (16GB) 用 13.6GB，146s/epoch
- 1 epoch per-domain AP: enron=0.54, BitcoinAlpha=0.59, uci=0.67（基线）

### 参数量核算

实际 34M（比设计文档估算的 19M 多），因为：
- n_mamba_layers=8（不是 10）但 d_state=32（不是 16）
- GCA 的 ff_mult=2 增加 FFN 参数
- EMA encoder 是 online 的深拷贝（不参与梯度但占显存）

## 文件

- `tgengine/nn/input_tokenizer.py` — domain-agnostic tokenization
- `tgengine/nn/pretraining_heads.py` — MTM/NTP/LP heads + EMA + block mask
- `tgengine/models/foundation.py` — FoundationModel
- `examples/train_foundation.py` — 预训练脚本
- `tests/test_foundation.py` — 单元测试（16 个）

## 待验证

- 20 epoch 完整训练结果（进行中）
- per-domain AP 是否随 epoch 提升
- 三任务 loss 是否协调下降
