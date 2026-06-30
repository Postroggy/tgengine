# TGEngine 动态图基础模型设计

> Status: 设计阶段（2026-06-28 起草）
> 目标: 跨域动态图基础模型，参数量 100M，4×3090 预训练，下游 zero-shot / few-shot 适配多任务

## 定位

**不是** task-specific 模型（DyGFormer/TGN 那种单图单任务）。
**不是** DyG-Mamba 的复刻（它是单图 task-specific，把图退化为序列，丢拓扑）。

**是**：跨域预训练的动态图基础模型——在多域图上预训练，迁移到未见域做 zero-shot 或少量微调。类比 LLM 之于 NLP，但输入是动态图事件流。

项目两条腿：
1. **高效训练/eval 框架**（TGEngine 本身，GatherSpec + fused pipeline + CUDA kernels）
2. **自研动态图基础模型**（本文档设计）

## 核心设计原则

动态图输入是**两条信息流**，不能简化为单一序列：
- **时序流**：事件序列 + 不规则时间间隔（burst、周期、趋势）
- **图结构流**：邻居拓扑、co-occurrence、社区、中心性

模型要让两条流**深度交互**，不是简单拼接。时序流走 Mamba SSM 主干（线性复杂度适合长序列），图结构流通过 cross-attention 调制时序状态。

## 参数预算（100M）

| 组件 | 参数量 | 占比 | 说明 |
|------|--------|------|------|
| 输入嵌入（edge feat + node ID + time RoPE） | ~10M | 10% | edge proj + node embedding table + rotary time |
| Mamba 主干（12 层 × d=1024） | ~70M | 70% | 事件序列建模，A(Δt) 时间感知 |
| Graph Cross-Attention（3 层，每 4 层 Mamba 插 1 层） | ~12M | 12% | co-occurrence + 邻居拓扑调制 SSM 状态 |
| 输出头（多任务：链接预测 / 节点分类 / 异常） | ~5M | 5% | MoE 多任务路由 |
| 预留（位置编码、LayerNorm 等） | ~3M | 3% | |
| **合计** | **~100M** | | |

## 架构

```
Input: (src, dst, t, edge_feat) + K neighbors per node
  │
  ├── 时序流
  │   edge_feat proj + node_emb → token
  │   time-RoPE(Δt) 注入 token
  │   → Mamba block ×4  (A(Δt) 时间感知 SSM)
  │
  ├── 图结构流
  │   co-occurrence freq (CUDA kernel) → struct token
  │   neighbor ID embedding → topology token
  │
  ├── Graph Cross-Attention
  │   Q = Mamba hidden state (时序)
  │   K,V = struct + topology tokens (图结构)
  │   → 调制后的 hidden state
  │
  ├── Mamba block ×4  (继续时序建模，已融合图信息)
  │
  ├── [重复 GCA + Mamba ×4 共 3 轮]
  │
  └── Multi-task head (MoE)
      链接预测 / 节点分类 / 边分类 / 异常检测
```

## 三个核心技术（落地细节）

### 1. 时间 Rotary Encoding（替换 Time2Vec）

**依据**：Hawkes 过程 log-likelihood 只依赖 Δt（平移不变），rotary 天然编码相对时间差。RoTHP (BDMA 2025) 证明 rotary 是事件流时间编码的理论最优。

**实现**：
- 每个邻居事件，Δt = t_now - t_neighbor，log-scale 后作为旋转角度
- 旋转注入 Mamba input projection（不单独占通道，和 edge feat 融合）
- 邻居 ID embedding 走另一组维度（异构 RoPE，SIREN-RoPE 思路）

**改动**：`nn/time_encoder.py` 新增 `RotaryTimeEncoder`，~80 行。替换 `FixedCosineTimeEncoder`。

### 2. 不规则时间间隔作为 SSM 控制信号

**依据**：DyG-Mamba (NeurIPS 2025) 的核心创新——Ebbinghaus 遗忘曲线启发，长 Δt 加强遗忘。标准 Mamba 的 A 是固定离散步，不适配事件流不规则间隔。

**实现**：
- A 参数化：$A(\Delta t) = \exp\!\big(-\text{softplus}(\Delta t) \cdot A_{\text{base}}\big)$
- mamba-ssm 库的 `dt` 参数已支持 input-dependent，传 Δt 即可
- 额外加 DyG-Mamba 的 "review cycle"（选择性回顾历史关键事件）

**改动**：自定义 MambaBlock，override SSM 的 A 计算，~120 行。

### 3. Next-Neighbor-Patch 预训练目标

**依据**：aLLM4TS (ICML 2024) 证明 next-patch prediction 比 mask-reconstruction 更适合时序。动态图天然有"下一个邻居是谁"信号——链接预测的预训练化，下游 gap 小。

**实现**：
- 邻居序列按时间分 patch（每 4 个邻居一个 patch）
- 因果 Mamba 预测 next patch 的邻居 ID 分布（$\text{softmax}$ over node vocab）
- 预训练 loss：$\mathcal{L} = \mathcal{L}_{\text{next-patch CE}} + \mathcal{L}_{\text{link BCE}}$（多任务）

**改动**：新增 `pretraining objective` 模式，engine 支持，~150 行。

## 图结构与时序的融合（关键设计）

**不做**：简单 concat(time_emb, struct_emb) → MLP。这会让图信息淹没时序记忆。

**做**：Graph Cross-Attention（GCA），图结构"调制"时序状态：
- Q = Mamba hidden state（时序流主导）
- K, V = co-occurrence freq + neighbor topology embedding（图结构流）
- GCA 输出加回 Mamba hidden state（residual）
- 每 4 层 Mamba 插 1 层 GCA，3 轮共 3 层 GCA

这样图结构信息是**辅助信号**，时序 SSM 仍是主干记忆。避免 DyG-Mamba 的痛点（纯序列丢拓扑）同时不让图信息喧宾夺主。

## 训练计划

| 阶段 | 数据 | 目标 | 硬件 |
|------|------|------|------|
| 预训练 | 多域图混合（社交/金融/交通，MixedDataset balanced） | next-neighbor-patch + 链接预测 | 4×3090, fp16+ZeRO-2 |
| 评估 | zero-shot 迁移到未见域（CrossMamba 已验证可行） | AP / MRR / Hits@K | 单卡 |
| 微调 | few-shot 适配目标域 | 下游任务指标 | 单卡 |

## 与框架的协同

基础模型用框架的能力：
- **GatherSpec**：声明需要 neighbors + co_occurrence + edge_feat，pipeline fused 准备
- **CUDA kernels**：`cuda_temporal_recent_k`（邻居采样）+ `cuda_co_occurrence_freq`（共现特征）已就绪
- **MixedDataset**：跨域数据混合（balanced mixing 已实现）
- **多任务 head**：框架已支持 TaskHead 插件

基础模型反哺框架：
- 时间 Rotary encoder 作为 `tgengine.nn` 新组件（其他模型也能用）
- A(Δt) Mamba block 作为新 SequenceEncoder
- 预训练目标作为 Engine 的新训练模式

## 待验证（优先级）

1. **RoTHP 时间编码**（最小改动，先验证收益）—— 替换 Time2Vec，uci/wikipedia 跑通看 AP
2. **A(Δt) SSM**（核心创新）—— 改 MambaBlock，对比固定 A 的 baseline
3. **GCA 融合**（图结构交互）—— 验证 co-occurrence 调制是否提升 zero-shot 迁移
4. **Next-neighbor-patch 预训练**（scaling 基础）—— 多域预训练后看 zero-shot
5. **100M 参数 scaling**—— 验证 loss 随参数下降，zero-shot 提升

## 参考文献

- DyG-Mamba (NeurIPS 2025) — 不规则时间间隔作 SSM 控制信号
- RoTHP (BDMA 2025) — Hawkes 过程的 rotary 时间编码
- aLLM4TS (ICML 2024) — next-patch prediction 预训练
- SIREN-RoPE (2024) — 异构维度 rotary
- Moirai-MoE (Salesforce 2024) — sparse MoE 多模式
- Chronos / TimesFM — 时序基础模型 patching 范式
