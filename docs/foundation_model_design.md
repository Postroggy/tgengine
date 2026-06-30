# TGEngine 动态图基础模型设计

> Status: 设计阶段（2026-06-28 起草，2026-06-30 基于 2026 文献深读修正）
> 目标: 跨域动态图基础模型，参数量 100M，4×3090 预训练，下游 zero-shot / few-shot 适配多任务

## 定位

**不是** task-specific 模型（DyGFormer/TGN 那种单图单任务）。
**不是** DyG-Mamba 的复刻（它是单图 task-specific，把图退化为序列，丢拓扑）。

**是**：跨域预训练的动态图基础模型——在多域图上预训练，迁移到未见域做 zero-shot 或少量微调。类比 LLM 之于 NLP，但输入是动态图事件流。

项目两条腿：
1. **高效训练/eval 框架**（TGEngine 本身，GatherSpec + fused pipeline + CUDA kernels）
2. **自研动态图基础模型**（本文档设计）

## 核心问题诊断（来自 TGPM, ICML 2026）

现有 CTDG 方法有三个根本性假设缺陷，直接阻碍学到可迁移的演化模式。我们的设计必须正面解决：

1. **静态邻居语义假设**：一跳邻居 + 时间编码压缩了非平稳的邻居语义。→ 用 interaction patch / 多跳结构缓解
2. **短期依赖假设**：最近邻居不足以刻画节点状态，限制了长期依赖捕获。→ Mamba SSM 的线性复杂度天然支持长序列
3. **回顾性时间建模假设**：时间只作辅助标注（衰减/attention），不是显式建模目标。→ **NTP 任务**直接预测未来时间

## 核心设计原则

动态图输入是**两条信息流**，不能简化为单一序列：
- **时序流**：事件序列 + 不规则时间间隔（burst、周期、趋势）
- **图结构流**：邻居拓扑、degree/centrality/community（**不用 co-occurrence**——二部图失效，见数据分析）

模型要让两条流**深度交互**，不是简单拼接。时序流走 Mamba SSM 主干（线性复杂度适合长序列），图结构流通过 cross-attention 调制时序状态。

## 参数预算（100M）

| 组件 | 参数量 | 占比 | 说明 |
|------|--------|------|------|
| 输入嵌入（edge feat + 结构特征 + time RoPE） | ~10M | 10% | edge proj + 结构特征 proj + rotary time，**不用 node ID**（domain-specific） |
| Mamba 主干（12 层 × d=1024） | ~70M | 70% | 事件序列建模，A(Δt) 时间感知 |
| Graph Cross-Attention（3 层，每 4 层 Mamba 插 1 层） | ~12M | 12% | 结构特征调制 SSM 状态 |
| 预训练头（MTM decoder + NTP head + LP head） | ~5M | 5% | 多任务预训练头 |
| MoE 路由（跨域专家） | ~3M | 3% | 不同专家处理不同图类型 |
| **合计** | **~100M** | | |

## 架构

```
Input: (src, dst, t, edge_feat) + K neighbors per node
  │
  ├── 时序流（domain-agnostic token）
  │   edge_feat proj + 结构特征(degree/centrality/community) → token
  │   time-RoPE(Δt) 注入 token
  │   → Mamba block ×4  (A(Δt) 时间感知 SSM)
  │
  ├── 图结构流
  │   结构特征: degree rank + centrality + community membership
  │   （不用 co-occurrence——二部图 AUC=0.5）
  │
  ├── Graph Cross-Attention
  │   Q = Mamba hidden state (时序)
  │   K,V = 结构特征 tokens (图结构)
  │   → 调制后的 hidden state
  │
  ├── Mamba block ×4  (继续时序建模，已融合图信息)
  │
  ├── [重复 GCA + Mamba ×4 共 3 轮]
  │
  ├── MoE 路由: router 基于图统计(密度/重复率/二部性)选专家
  │
  └── 预训练头 (多任务)
      Task 1: Link Prediction BCE
      Task 2: Masked Token Modeling (block-wise, EMA target)
      Task 3: Next Time Prediction (预测时间编码向量)
```

## 三个核心技术（落地细节）

### 1. 时间 Rotary Encoding（替换 Time2Vec）

**依据**：Hawkes 过程 log-likelihood 只依赖 Δt（平移不变），rotary 天然编码相对时间差。RoTHP (BDMA 2025) 证明 rotary 是事件流时间编码的理论最优。TGPM (ICML 2026) 也用可训练频率的正弦时间编码，思路一致。

**实现**：
- 每个邻居事件，$\Delta t = t_{\text{now}} - t_{\text{neighbor}}$，log-scale 后作为旋转角度
- 旋转注入 Mamba input projection（不单独占通道，和 edge feat 融合）
- 结构特征（degree/centrality）走另一组维度

**改动**：`nn/time_encoder.py` 新增 `RotaryTimeEncoder`，~80 行。替换 `FixedCosineTimeEncoder`。

### 2. 不规则时间间隔作为 SSM 控制信号

**依据**：DyG-Mamba (NeurIPS 2025) 的核心创新——Ebbinghaus 遗忘曲线启发，长 Δt 加强遗忘。标准 Mamba 的 A 是固定离散步，不适配事件流不规则间隔。

**实现**：
- A 参数化：$A(\Delta t) = \exp\!\big(-\text{softplus}(\Delta t) \cdot A_{\text{base}}\big)$
- mamba-ssm 库的 `dt` 参数已支持 input-dependent，传 Δt 即可
- 额外加 DyG-Mamba 的 "review cycle"（选择性回顾历史关键事件）

**改动**：自定义 MambaBlock，override SSM 的 A 计算，~120 行。

### 3. 三任务生成式预训练（核心创新，来自 TGPM + Chronos + GraphMAE）

**依据**：TGPM (ICML 2026) 证明 MTM + NTP 双任务在 CTDG 跨域迁移上大幅领先（average rank 1.0）。NTP 直接解决"回顾性时间建模"缺陷，对齐 Hawkes 过程。GraphMAE (KDD 2022) 的 scaled cosine error 比 MSE 稳定。

**总损失函数**：

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{link}} + \beta \cdot \mathcal{L}_{\text{MTM}} + \gamma \cdot \mathcal{L}_{\text{NTP}}$$

#### Task 1: Link Prediction BCE（主信号，直接对齐 eval）

$$\mathcal{L}_{\text{link}} = -\frac{1}{B} \sum_{i=1}^{B} \left[ \log \sigma\!\big(f_\theta(\mathbf{s}_i, \mathbf{d}_i)\big) + \log \sigma\!\big(1 - f_\theta(\mathbf{s}_i, \mathbf{n}_i)\big) \right]$$

- $\mathbf{s}_i, \mathbf{d}_i, \mathbf{n}_i$ 分别是 src、dst、neg 的表示
- DDGPrompt (CIKM 2025) 验证标准 LP 预训练是 CTDG 领域默认做法

#### Task 2: Masked Token Modeling — 学"什么演化"（来自 TGPM）

**Block-wise masking**（不是随机单 token）——block size $b$ 控制模型被迫推理的最小时间跨度：

$$\mathcal{L}_{\text{MTM}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \mathbf{r}_i - \hat{\mathbf{p}}_i \right\|^2$$

- $\mathcal{M}$ 为连续 token block 的 mask 集合
- $\hat{\mathbf{p}}_i = f_{\text{EMA}}(\cdot)$ 为 **EMA encoder** 生成的稳定重建目标（防 representation collapse，类似 BYOL/BEiT）
- $\mathbf{r}_i$ 为 SimMIM 式重建输出
- **关键**：block size 是多尺度时间依赖的超参——TGPM 证明这是信息论必要条件

#### Task 3: Next Time Prediction — 学"何时演化"（来自 TGPM，**我们原方案缺失**）

$$\mathcal{L}_{\text{NTP}} = \frac{1}{m-1} \sum_{i=1}^{m-1} \left\| f_{\text{NTP}}(\mathbf{p}'_i) - \mathbf{t}_{i+1} \right\|$$

- $f_{\text{NTP}}$ 为两层 MLP head
- $\mathbf{p}'_i$ 为 Mamba encoder 输出的 contextualized embedding
- $\mathbf{t}_{i+1}$ 为下一个交互的**时间间隔编码向量**（正弦/RoPE 编码，不是原始时间值）
- 自回归分解：$p(\mathbf{t}_1, \dots, \mathbf{t}_m) = \prod_{i=1}^m p(\mathbf{t}_i \mid \mathbf{t}_{<i}, \bar{\mathbf{p}}_{<i})$
- **为什么重要**：迫使模型编码演化时间粒度（典型间隔）+ 频率信号（交互频率），把不同演化模式关联到不同时间节奏。直接对齐 Hawkes 强度函数。

**改动**：新增 `pretraining` 模块，engine 支持三任务联合训练，~200 行。

## 跨域处理：Structure + MoE（来自 OOD 综述 + AnyGraph + Scalable LP）

### Domain-agnostic 输入特征（不用 node ID）

Universal GFM (Stanford, 2026) 验证：用 feature-agnostic 结构属性作 prompts，不同图嵌入共享空间。我们用：

$$\text{token} = \text{edge\_feat\_proj}(\mathbf{e}) + \text{RoPE}(\Delta t) + \text{struct\_proj}(\text{degree}, \text{centrality}, \text{community}) + \text{recency}$$

- **不用 node ID**（domain-specific，跨域不迁移）
- **不用 co-occurrence**（数据分析显示二部图 AUC=0.5，失效）
- 用 degree rank / centrality / community membership（Universal GFM 验证鲁棒）
- recency（上次交互时间差）+ interaction count（重复交互次数）

### MoE 路由（处理域层 negative transfer）

OOD 综述 (2026) 明确推荐 MoE 处理域层 OOD。Scalable LP (KDD 2025) 用"冻结 expert + 只学 router assignment"实现 10000x 高效适配。

$$\mathbf{y} = \sum_{k=1}^{K} g_\phi(\mathbf{x})_k \cdot \text{Expert}_k(\mathbf{x}), \quad g_\phi(\mathbf{x}) = \text{TopK}\!\big(\text{softmax}(W\mathbf{x})\big)$$

- Router 基于图统计特征（密度、重复率、二部性）路由
- Expert 1: 密集图专家（reddit/lastfm/enron）
- Expert 2: 稀疏图专家（BitcoinAlpha/uci）
- Expert 3: 二部图专家（wikipedia/mooc）
- 下游适配：冻结 expert，只学 router assignment

### 训练协议（来自 MiNT, NeurIPS 2025）

- **Order shuffling**：每 epoch shuffle 数据集顺序
- **Context switching**：切图时重置模型状态（Mamba hidden state / graph CSR buffer）
- **State reset**：防止跨域状态泄漏

## 时间突发性警示（来自 TGPM 的重要限制）

TGPM 实验发现：**高突发性数据集预训练收益小甚至有害**。

| 数据集 | Δt median | 突发性 | 预训练预期 |
|--------|-----------|--------|-----------|
| reddit | 3.2 | 高 | ⚠️ 收益有限 |
| mooc | 4.0 | 高 | ⚠️ 收益有限 |
| uci | 31.0 | 中 | ✅ 有效 |
| wikipedia | 16.0 | 中 | ✅ 有效 |
| lastfm | 82.0 | 低 | ✅ 有效 |
| enron | 1080.0 | 低 | ✅ 有效 |
| BitcoinAlpha | 86400.0 | 低 | ✅ 有效 |

**数据混合策略**：
- Phase 1 先在低突发性数据集（enron/BitcoinAlpha/lastfm）上验证预训练有效性
- 高突发性数据集（reddit/mooc）降低预训练权重或作为 hold-out 测试
- 考虑按突发性分组训练不同 MoE expert

## 训练计划

| 阶段 | 数据 | 目标 | 硬件 |
|------|------|------|------|
| 预训练 Phase 1 | 2-3 低突发性数据集混合 | 纯 link prediction BCE + MiNT 协议 | 4×3090 DDP |
| 预训练 Phase 2 | + 加 MTM (block-wise) + NTP | 三任务联合 loss | 4×3090 DDP |
| 预训练 Phase 3 | + MoE 路由 + 全数据集 | 完整方案 | 4×3090 DDP |
| 评估 | zero-shot 迁移到未见域 | AP / MRR / Hits@K | 单卡 |
| 微调 | few-shot 适配目标域（冻结 expert，只学 router） | 下游任务指标 | 单卡 |

## 与框架的协同

基础模型用框架的能力：
- **GatherSpec**：声明需要 neighbors + edge_feat + 结构特征，pipeline fused 准备
- **CUDA kernels**：`cuda_temporal_recent_k`（邻居采样）已就绪；结构特征（degree/centrality）预计算
- **MixedDataset**：跨域数据混合（balanced mixing 已实现）
- **多任务 head**：框架已支持 TaskHead 插件
- **DDP**：rank-0-only eval 修复已完成，支持多卡预训练

基础模型反哺框架：
- 时间 Rotary encoder 作为 `tgengine.nn` 新组件
- A(Δt) Mamba block 作为新 SequenceEncoder
- 三任务预训练（MTM + NTP + LP）作为 Engine 的新训练模式
- 结构特征计算（degree/centrality/community）作为 GatherSpec 新字段

## 待验证（优先级）

1. **RoTHP 时间编码**（最小改动）—— 替换 Time2Vec，uci/wikipedia 跑通看 AP
2. **A(Δt) SSM**（核心创新）—— 改 MambaBlock，对比固定 A 的 baseline
3. **NTP 任务单独验证**—— 加 NTP head，看是否提升时序建模（对比无 NTP）
4. **Phase 1: 多域 LP 预训练**—— 2-3 低突发性数据集混合，验证 zero-shot 迁移
5. **Phase 2: + MTM + NTP**—— 三任务联合，看自监督是否提升迁移
6. **Phase 3: + MoE + 全数据集**—— 完整方案，100M 参数 scaling

## 参考文献

### 2026 年（深读，见 `experiments/pretraining/2026_deep_read.md`）
- **TGPM** (ICML 2026, arXiv 2601.22454) — CTDG 自监督预训练，MTM + NTP 双任务，interaction patch tokenization
- **OOD Generalization in GFMs** (2026, arXiv 2601.21067, 清华) — 四层 OOD 挑战框架，推荐 MoE
- **Universal Graph FM** (2026, arXiv 2604.06391, Stanford) — 结构特征 prompts（degree/centrality/community）
- **CrossHGL** (2026, arXiv 2603.27685) — text-free 跨域异构图 FM，SVD + Tri-Prompt
- **DDGPrompt** (CIKM 2025, arXiv 2601.11954) — CTDG data-centric prompt tuning
- **SDG** (2026, arXiv 2601.23233) — CTDG 序列扩散模型
- **Scalable LP Pretraining** (KDD 2025, Meta, arXiv 2508.04645) — LP pairwise 预训练 + MoE + 冻结 expert

### 基础工作
- DyG-Mamba (NeurIPS 2025) — 不规则时间间隔作 SSM 控制信号
- RoTHP (BDMA 2025) — Hawkes 过程的 rotary 时间编码
- MiNT (NeurIPS 2025) — 多网络训练协议（shuffle + context switch）
- AnyGraph (2024) — MoE + SVD 特征统一化
- GraphMAE (KDD 2022) — scaled cosine error + re-mask decoding
- Chronos (2024) — 时序值量化 + next-token
- Moirai 2.0 (2025) — decoder-only + quantile loss + multi-token
