# Phase 1 预训练实验日志

> 目标：验证多域 CTDG 混合训练（LP BCE）是否比单域训练更有效
> 方案：见 `docs/foundation_model_design.md` Phase 1

## 2026-06-30 实验 1：初始测试（d=64，有 bug）

模型太小（443K params），per-domain eval 有时间戳归一化 bug（用了原始时间戳而非 [0,1] 归一化）。结果无效。

## 2026-06-30 实验 2：完整对比（d=128, L=3, 修正 eval）

### 配置
- 模型：TimeAwareMambaBlock×3, d_model=128, K=32
- 混合：1.1M params（node_emb 5869×128）
- 单域：uci 607K, enron 387K, BitcoinAlpha 848K params
- 训练：LP BCE, 20 epochs, lr=1e-3, batch_size=200
- 混合训练用 balanced round-robin（每域 cap 13943 边）
- 修正 per-domain eval：时间戳归一化到 [0,1]（和 MixedDataset 一致）

### 结果

| 数据集 | 混合训练 per-domain AP | 单域训练 per-domain AP | Δ |
|--------|----------------------|----------------------|---|
| enron (107K 边) | 0.6416 | 0.8764 | **-0.235** ❌ |
| BitcoinAlpha (20K 边) | 0.6652 | 0.6184 | **+0.047** ✅ |
| uci (52K 边) | 0.6790 | 0.6981 | **-0.019** ⚠️ |

### 分析

**Negative transfer**：混合训练对小数据集有帮助，对大数据集有害。

- **BitcoinAlpha 受益（+4.7%）**：只有 20K 边，其他域的数据提供额外学习信号
- **enron 受害（-23.5%）**：balanced 训练 cap 到 13943 边（实际 107K），损失 87% 训练数据
- **uci 中性（-1.9%）**：cap 到 13943 边（实际 37K），损失 62% 但影响小

### 结论

1. **纯混合训练不会自动带来跨域增益**——需要 MoE 或更智能的数据混合
2. **Balanced round-robin 不适合差异大的数据集**——应改用加权混合（按数据集大小比例）
3. **MoE 是必须的**（OOD 综述正确预测）——不同专家处理不同域，保留每个域有效容量
4. **per-domain eval 是正确指标**——merged AP（0.9354）被域捷径虚高，无意义

### 下一步

- Phase 2: 加 MoE 路由（至少 3 expert），验证 MoE 是否解决 negative transfer
- 或先尝试加权混合（不 cap，按比例采样），看是否能改善
- 模型可能需要更大（d=256 或更多层）才能同时学 3 域

## 脚本

- `examples/train_pretrain_phase1.py` — Phase 1 训练 + per-domain eval（已修正时间戳 bug）
