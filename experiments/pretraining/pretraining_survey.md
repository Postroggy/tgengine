# 预训练目标调研：图/时序/LLM 基础模型对 CTDG 的启示

> 调研日期：2026-06-30
> 目的：为 TGEngine 动态图基础模型的预训练目标设计提供文献依据
> 范围：图基础模型（GFM）、时序基础模型（TSFM）、LLM 三大范式的预训练目标

---

## 一、三大范式的预训练目标分类

所有基础模型的预训练目标都可归为以下四类（来源：Graph Foundation Models: A Comprehensive Survey, arXiv 2505.15116, 2025）：

| 目标类型 | 核心思想 | Loss 形式 | 代表模型 |
|---------|---------|----------|----------|
| **自回归生成** | 给定前缀，预测下一个 token/patch | Cross-entropy over vocab | GPT, Chronos, GraphGPT |
| **掩码重建** | mask 输入部分，预测被 mask 的内容 | Cross-entropy / MSE / Cosine | BERT, GraphMAE, Moirai |
| **对比学习** | 拉近正样本对，推远负样本对 | InfoNCE | GraphCL, DGI, DySubC |
| **直接任务预测** | 直接用下游任务作预训练 | Task-specific (BCE等) | AnyGraph, MiNT |

**2025 综述的核心结论**（§4.3.4 Discussion）：
> "No single method is universally superior—rather, each presents complementary advantages."
> - 监督预训练：任务对齐好，但需大量标注数据
> - 生成式预训练：可大规模无标注训练，但缺乏任务特异性
> - 对比预训练：学到判别性表示，但对数据增强方式敏感

---

## 二、图基础模型（GFM）预训练目标

### 2.1 经典工作（2020-2024）

#### GPT-GNN（KDD 2020）— 图上的自回归生成
- **目标**：自回归式地生成节点属性 + 边
- **方法**：将图转为序列（通过节点排序），逐个生成节点及其连接
- **自回归分解**：

$$p_\theta(\mathbf{S}^\pi) = \prod_{i=1}^{n} p_\theta\!\left(\mathbf{S}_i^\pi \mid \mathbf{S}_{<i}^\pi\right)$$

  其中 $\pi$ 为节点排列，$\mathbf{S}_i^\pi$ 编码节点 $\pi(v_i)$ 与前序节点的连接。
- **Loss**：属性重建 MSE + 边预测 CE
- **启示**：图可以当作序列做自回归，但需要节点排序方案

#### GraphMAE（KDD 2022）— 掩码特征重建
- **目标**：mask 节点特征，重建被 mask 的特征（不重建结构）
- **关键创新**：
  1. **Scaled Cosine Error**（不用 MSE）：解决图特征向量范数差异大 + 维度灾难问题

$$\mathcal{L}_{\text{SCE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left(1 - \frac{\hat{\mathbf{x}}_i^\top \mathbf{x}_i}{\|\hat{\mathbf{x}}_i\| \cdot \|\mathbf{x}_i\|}\right)^\gamma, \quad \gamma \geq 1$$

  其中 $\mathcal{M}$ 为 mask 集合，$\mathbf{x}_i$ 为真值特征，$\hat{\mathbf{x}}_i$ 为重建特征，$\gamma$ 控制难易样本权重。

  2. **Re-mask decoding**：decoder 前再次 mask encoder 输出，防止信息泄漏
  3. **GNN decoder**（不用 MLP）：更强的 decoder 弥补 encoder-目标 gap
- **启示**：图特征重建用 cosine error 比 MSE 稳定；mask 策略需要防止信息泄漏

#### GraphCL（NeurIPS 2020）— 对比学习
- **目标**：图增强 → InfoNCE 对比不同 view
- **增强方式**：节点丢弃、边扰动、子图采样
- **Loss**：InfoNCE

$$\mathcal{L}_{\text{InfoNCE}} = -\sum_{i=1}^{N} \log \frac{\exp\!\left(\text{sim}(f(x_i),\, f(x_i^+)) / \tau\right)}{\sum_{j=1}^{N} \exp\!\left(\text{sim}(f(x_i),\, f(x_j^-)) / \tau\right)}$$

  其中 $x_i^+$ 为正样本（同一实例的增强 view），$x_j^-$ 为负样本，$\tau$ 为温度系数。
- **启示**：对比学习对增强方式敏感，图增强的"label-invariance"难以保证

#### AnyGraph（2024）— MoE + 链接预测
- **目标**：链接预测（softmax over all nodes + negative sampling）

$$\mathcal{L}_{\text{AnyGraph}} = \sum_{b \in B} -\frac{1}{B} \log \frac{\exp(\hat{y}_{c_b, p_b} - \hat{y}_{\max})}{\sum_{v_n \in \mathcal{V}} \exp(\hat{y}_{c_b, n} - \hat{y}_{\max})}$$

  其中 $(v_{c_b}, v_{p_b})$ 为正样本边，$\hat{y}_{\max}$ 为 batch 内最大预测分（数值稳定）。
- **跨域处理**：
  1. **特征统一化**：SVD + simplified GCN 将不同图的特征映射到统一 embedding 空间
  2. **MoE 路由**：不同专家处理不同域的子图，基于自监督 loss 自动路由
  3. **训练频率正则化**：防止 winner-takes-all
- **启示**：跨域 GFM 的核心难题是特征/结构异质性，MoE + 特征统一化是有效方案

#### MiNT（NeurIPS 2025）— 多网络训练
- **目标**：标准链接预测 / 图属性预测（无特殊 SSL 目标）
- **核心创新在训练协议**：
  1. **Order Shuffling**：每 epoch shuffle 数据集顺序
  2. **Context Switching**：切图时重置历史状态（防止跨域状态泄漏）
- **结果**：64 个网络预训练 → 20 个未见网络 zero-shot，正向 scaling
- **启示**：多网络预训练用标准 loss 就能 work，关键在训练协议

### 2.2 最新工作（2025-2026）

#### GraphGPT（ICML 2025）— 图欧拉路径自回归
- **目标**：自回归 next-token prediction
- **创新**：**Graph Eulerian Transformer** — 用欧拉路径将图转为节点/边/属性的 token 序列（可逆）
- **优势**：保留了图结构信息，比随机游走更完整
- **启示**：图可以通过欧拉路径序列化为 token 序列，然后用 LLM 范式训练

#### Graph Foundation Models 综述（arXiv 2505.15116, 2025）
- **统一框架**：将 GFM 分为 backbone + pretraining + adaptation 三层
- **预训练策略分类**：
  - 监督预训练：直接用下游任务标签
  - 生成预训练：自回归（GraphGPT）+ 自编码（GraphMAE 式掩码重建）
  - 对比预训练：instance-instance（GraphCL）+ instance-context（DGI）
- **趋势**：生成式预训练（特别是自回归）在 2025 年成为主流方向

#### Relational Transformer（ICLR 2026）— 关系数据基础模型
- **目标**：**Masked token prediction**（BERT 式）
- **创新**：
  1. **Task table prompting**：通过 task table 指定任务
  2. **Relational Attention**：over columns + rows + PK-FK links
  3. **Schema-invariant**：cell tokenization 带 table/column metadata
- **结果**：22M 参数 zero-shot 达到全监督 93% AUROC（27B LLM 只有 84%）
- **启示**：小模型 + 正确的 tokenization + masked prediction 可以超越大 LLM

#### GFM-RAG（NeurIPS 2025）/ GFM-UAD
- 领域特化的图基础模型（RAG 检索 / 异常检测）
- 趋势：从通用 GFM 向领域特化发展

---

## 三、时序基础模型（TSFM）预训练目标

### 3.1 经典工作（2023-2024）

#### Chronos（Amazon, 2024）— 时序值量化 + next-token
- **Tokenization**：连续值 → mean scaling → uniform quantization → 离散 token（分 bin）
- **目标**：Next-token prediction（和 GPT 完全一样）
- **Loss**：Cross-entropy over quantized bins

$$\mathcal{L}_{\text{Chronos}} = -\sum_{t=1}^{T} \log p_\theta\!\left(q(x_t) \mid q(x_{<t})\right)$$

  其中 $q: \mathbb{R} \to \{1, 2, \dots, B\}$ 是量化函数，$B$ 为 bin 数。
- **核心洞察**："回归通过分类实现" — 把连续值预测转为分类问题
- **数据增强**：TSMix（多时序凸组合）+ KernelSynth（高斯过程生成合成时序）
- **启示**：连续值可以量化为离散 token，直接复用 LLM 架构和训练范式

#### Moirai（Salesforce, 2024）— 掩码 patch 预测
- **Tokenization**：Patching（多频率 patch size projection）
- **目标**：Masked patch prediction（mask forecast horizon 的 patch）
- **Loss**：Mixture distribution NLL（Negative Log-Likelihood）
- **创新**：
  1. **Multi patch size projection**：不同频率用不同 patch size
  2. **Any-variate Attention**：RoPE（时间轴）+ learned bias（变量轴），处理任意维度
  3. **Mixture distribution**：灵活的概率预测分布
- **启示**：patching 是时序 tokenization 的核心；多频率需要多 projection

#### TimesFM（Google, 2024）— 自回归 patch 预测
- **目标**：Next-patch autoregressive prediction
- **架构**：Decoder-only transformer
- **启示**：时序可以用 decoder-only 自回归，和 LLM 完全对齐

#### PatchTST（ICLR 2023）— patching + 掩码自编码
- **目标**：Masked autoencoding on patches
- **Loss**：MSE
- **启示**：patching 思想的起源之一

### 3.2 最新工作（2025-2026）

#### Moirai 2.0（Salesforce, 2025, arXiv 2511.11698）— 重大架构转向
- **从 Moirai 1.0 的改变**：
  - ❌ Masked encoder → ✅ **Decoder-only autoregressive**
  - ❌ Multi-patch input → ✅ **Single patch size**
  - ❌ Mixture distribution → ✅ **Quantile loss (pinball loss)**
  - ✅ 新增 **Multi-token prediction**（每个输出 token 预测多个未来 patch）
- **Loss**：Quantile loss（9 个分位数 $q \in \{0.1, 0.2, \dots, 0.9\}$，即 pinball loss）

$$\mathcal{L}_{\text{Moirai-2}} = \frac{1}{H \cdot |Q|} \sum_{t=1}^{H} \sum_{q \in Q} \left[ q \cdot \max(y_t - \hat{y}_t^{(q)},\, 0) + (1-q) \cdot \max(\hat{y}_t^{(q)} - y_t,\, 0) \right]$$

  其中 $H = K \cdot p$ 为预测长度（$K$ 个 patch，每个 patch size $p$），$y_t$ 为真值，$\hat{y}_t^{(q)}$ 为第 $q$ 分位数预测。
- **结果**：比 Moirai 1.0-Large **快 2x、小 30x、性能更好**
- **关键洞察**：
  > "decoder-only backbone along with recursive multi-quantile decoding contribute most to the gains"
  - Decoder-only > Masked encoder（和 LLM 趋势一致）
  - Quantile loss > Mixture distribution（更简单且直接对齐 CRPS 指标）
  - Multi-token prediction 减少长程预测的误差累积
- **启示**：时序 FM 正在向 LLM 范式靠拢（decoder-only + autoregressive），但用 quantile loss 替代 cross-entropy

#### Chronos 2 / Chronos-Bolt（Amazon, 2025）
- 从 encoder-only 转向 hybrid encoder-decoder
- 直接输出 quantile（不用 mixture distribution）
- 更快的推理（Bolt 版本）

#### 其他 2025 TSFM
- **Sundial**：Flow-based losses 学习连续分布
- **TiRex**：基于 xLSTM（非 transformer）
- **YingLong**（Alibaba）：Output-feedback 架构
- **TabPFN-TS**：Prior-Data Fitted Network
- **趋势**：架构多元化，但 **decoder-only + autoregressive + quantile loss** 是主流方向

#### In-Context Fine-Tuning for TSFM（ICML 2025）
- **创新**：continued pre-training 教模型在推理时适应 in-context examples
- **意义**：从 zero-shot 向 few-shot in-context adaptation 发展

---

## 四、LLM 预训练目标（基线参考）

| 模型 | 架构 | 预训练目标 | Loss |
|------|------|-----------|------|
| **GPT 系列** | Decoder-only | Next-token prediction（自回归） | Cross-entropy |
| **BERT** | Encoder-only | Masked Language Model（mask 15%） | Cross-entropy |
| **T5/BART** | Encoder-Decoder | Denoising seq2seq（corrupt→reconstruct） | Cross-entropy |

**GPT next-token prediction**：

$$\mathcal{L}_{\text{GPT}} = -\sum_{t=1}^{T} \log p_\theta(w_t \mid w_{<t})$$

**BERT Masked Language Model**：

$$\mathcal{L}_{\text{BERT}} = -\sum_{t \in \mathcal{M}} \log p_\theta(w_t \mid w_{\setminus \mathcal{M}})$$

其中 $\mathcal{M}$ 为 mask 集合，$w_{\setminus \mathcal{M}}$ 为未 mask 的上下文。

**LLM 的核心教训**：
1. **目标越简单越好**：next-token prediction 一个目标就够 scaling
2. **力量来自数据多样性**：不是目标复杂，是数据多
3. **Decoder-only 自回归成为主流**：GPT 系列证明了这条路线
4. **Tokenization 是关键**：BPE 将异构语言统一为共享词汇表

---

## 五、时序图（CTDG）专项工作

### 5.1 现有 CTDG 预训练尝试

#### MiNT（NeurIPS 2025）— 最直接的 CTDG 预训练
- 见 §2.1，DTDG 多网络训练，标准链接预测 loss
- **局限**：DTDG（非 CTDG），同域（加密货币）

#### DySubC — 时序子图对比学习
- **目标**：Temporal subgraph contrastive learning
- **方法**：不同时间窗口的子图作为正样本对
- **启示**：时序图的对比学习可以用时间窗口切分

#### PT-DGNN（Neurocomputing 2022）— 动态图生成预训练
- **目标**：Dynamic graph generation task
- **方法**：同时预测未来时刻的邻接矩阵 + 节点特征
- **启示**：动态图可以做"生成未来"的预训练

#### MaskDGNN（IJCAI 2025）— 活跃度感知时序掩码
- **目标**：Self-supervised temporal masking
- **创新**：
  1. **Activeness-aware masking**：高活跃度节点的边保留，低活跃度的 mask（减少冗余）
  2. **Adaptive frequency enhancing**：频域特征增强，应对 distribution shifting
- **结果**：链接预测 accuracy +7.07%，MRR +13.87%
- **启示**：动态图掩码不应随机，应基于节点活跃度；频域特征对 distribution shift 鲁棒

### 5.2 CTDG 跨域迁移的挑战

#### Transfer Learning for Temporal Link Prediction（arXiv 2504.10925, 2025）
- **核心挑战**：memory-laden 模型（TGN 等）的 memory 只存训练时见过的节点，无法直接迁移到新图
- **解决方案**：**Structural mapping module** — 从图结构拓扑特征映射到 memory embedding
- **意义**：为 "memory-free foundation model for TLP" 铺路
- **启示**：CTDG 跨域迁移需要**结构特征到表示的映射**，而非依赖 node ID 或 memory

---

## 六、跨范式洞察与 CTDG 迁移方案

### 6.1 三大范式的收敛趋势

| 趋势 | LLM | 时序 FM | 图 FM |
|------|-----|---------|-------|
| **架构** | Decoder-only | Decoder-only（Moirai 2.0）| 混合（transformer + GNN）|
| **目标** | Next-token | Next-patch / Quantile | Masked / Link pred |
| **Tokenization** | BPE | Quantization / Patching | Eulerian path / Node ordering |
| **数据** | 多域文本混合 | 多域时序混合 | 多域图混合 |

**核心收敛点**：
1. **Decoder-only 自回归**正在成为主流（LLM 一直是，时序 FM 2025 转向，图 FM 在探索）
2. **目标简单化**：从复杂的多任务/对比学习 → 简单的 next-token/masked prediction
3. **Tokenization 是核心创新点**：怎么把领域数据转为统一 token 序列

### 6.2 CTDG 的特殊性

CTDG 和上述三大范式的关键区别：
1. **不是单一序列**：每个节点有自己的邻居序列，是多序列交织
2. **时间不规则**：事件间隔不均匀（burst、周期、趋势）
3. **图结构动态演化**：邻居集随时间变化
4. **跨域异质性大**：二部图 vs 同构图，稀疏 vs 密集，重复交互率 0%-93%

### 6.3 从各范式可迁移的要素

| 来源范式 | 可迁移要素 | CTDG 应用 |
|---------|-----------|----------|
| **Chronos** | 连续值量化 → next-token | Δt 量化为 token → 预测下一个 Δt |
| **Moirai 2.0** | Quantile loss + multi-token | Δt 预测用 quantile loss，多步预测 |
| **GraphMAE** | Scaled cosine error + re-mask | edge_feat 重建用 cosine error |
| **GraphGPT** | Eulerian path 序列化 | 邻居序列天然有序（按时间），已是序列 |
| **GPT-GNN** | 多模态自回归生成 | 时间 + 结构 + 属性多任务生成 |
| **AnyGraph** | MoE + 特征统一化 | 跨域结构特征 + MoE 路由 |
| **MiNT** | 训练协议（shuffle + context switch）| 多数据集训练协议 |
| **MaskDGNN** | 活跃度感知掩码 | 邻居序列掩码基于节点活跃度 |
| **Transfer TLP** | 结构特征→表示映射 | 跨域用结构特征，不用 node ID |
| **Relational Transformer** | Masked token + schema metadata | task table prompting + 结构化 attention |

### 6.4 推荐的 CTDG 预训练目标方案

基于调研，推荐**多任务生成式预训练**（融合 GPT-GNN + Chronos + GraphMAE）：

**总损失函数**：

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{temporal}} + \beta \cdot \mathcal{L}_{\text{link}} + \gamma \cdot \mathcal{L}_{\text{edge\_feat}}$$

**Task 1: Masked Temporal Reconstruction**（from Chronos / Moirai 2.0）

mask 邻居序列中 15% 的 $\Delta t$，预测被 mask 的值。两种 loss 选择：

- Chronos 式（量化为 $B$ 个 bin，cross-entropy）：

$$\mathcal{L}_{\text{temporal}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta\left(q(\Delta t_i) \mid \Delta t_{\setminus \mathcal{M}}\right)$$

  其中 $q(\cdot)$ 是量化函数，$\mathcal{M}$ 是 mask 集合。

- Moirai 2.0 式（quantile / pinball loss，9 个分位数 $q \in \{0.1, \dots, 0.9\}$）：

$$\mathcal{L}_{\text{temporal}} = \frac{1}{|\mathcal{M}| \cdot |Q|} \sum_{i \in \mathcal{M}} \sum_{q \in Q} \left[ q \cdot \max(\Delta t_i - \hat{\Delta t}_i^{(q)},\, 0) + (1-q) \cdot \max(\hat{\Delta t}_i^{(q)} - \Delta t_i,\, 0) \right]$$

→ 迫使模型理解时序因果结构

**Task 2: Link Prediction**（from AnyGraph / MiNT）

$$\mathcal{L}_{\text{link}} = -\frac{1}{B} \sum_{i=1}^{B} \left[ \log \sigma\left(f_\theta(\mathbf{s}_i, \mathbf{d}_i)\right) + \log \sigma\left(1 - f_\theta(\mathbf{s}_i, \mathbf{n}_i)\right) \right]$$

其中 $\mathbf{s}_i, \mathbf{d}_i, \mathbf{n}_i$ 分别是 src、dst、neg 的表示，$f_\theta$ 是打分函数。→ 主信号，直接对齐 eval，保证 pretrain→eval gap 小

**Task 3: Edge Feature Reconstruction**（from GraphMAE）

mask edge_feat $\mathbf{e}_i$，用 **scaled cosine error** 重建（不用 MSE）：

$$\mathcal{L}_{\text{edge\_feat}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left(1 - \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_i}{\|\hat{\mathbf{e}}_i\| \cdot \|\mathbf{e}_i\|}\right)^\gamma, \quad \gamma \geq 1$$

→ 迫使模型理解交互属性；cosine error 对特征范数差异鲁棒

**跨域处理：Structure + MoE**（from AnyGraph + Transfer TLP）

- domain-agnostic 特征：$\text{edge\_feat\_proj}(\mathbf{e}) + \text{RoPE}(\Delta t) + \text{recency} + \text{interaction\_count}$
- 不用 node ID（domain-specific）
- 不用 co-occurrence（二部图失效，见 `data_analysis.md`）
- MoE 路由：router $g_\phi$ 基于图统计特征（密度、重复率、二部性）选择专家：

$$\mathbf{y} = \sum_{k=1}^{K} g_\phi(\mathbf{x})_k \cdot \text{Expert}_k(\mathbf{x}), \quad g_\phi(\mathbf{x}) = \text{TopK}\left(\text{softmax}(W \mathbf{x})\right)$$

**训练协议：MiNT-style**

- Order shuffling：每 epoch shuffle 数据集顺序
- Context switching：切图时重置模型状态（memory / CSR buffer）
- State reset：防止跨域状态泄漏

### 6.5 实现优先级

| Phase | 目标 | 验证问题 |
|-------|------|---------|
| **Phase 1** | 纯 link prediction BCE + 结构特征 + MiNT 协议 | 多域 CTDG 混合训练 > 单域训练？ |
| **Phase 2** | + masked Δt reconstruction（Chronos 量化）| 自监督时序信号提升迁移？ |
| **Phase 3** | + edge_feat reconstruction（cosine error）+ MoE | 完整方案效果如何？ |

---

## 七、参考文献

### 图基础模型
- **GPT-GNN** (Hu et al., KDD 2020) — 图自回归生成预训练
- **GraphMAE** (Hou et al., KDD 2022) — 掩码图自编码器，scaled cosine error
- **GraphCL** (You et al., NeurIPS 2020) — 图对比学习
- **AnyGraph** (Xia & Huang, 2024, arXiv 2408.10700) — MoE 跨域 GFM
- **MiNT** (Ngo et al., NeurIPS 2025, arXiv 2406.10426) — 多网络时序图训练
- **GraphGPT** (ICML 2025, arXiv 2401.00529) — 欧拉路径自回归
- **Graph Foundation Models Survey** (2025, arXiv 2505.15116) — GFM 综述
- **Relational Transformer** (ICLR 2026, arXiv 2510.06377) — 关系数据基础模型
- **GFM-RAG** (NeurIPS 2025) — 图基础模型 for RAG
- **GFM-UAD** (2025) — 图基础模型 for 异常检测

### 时序基础模型
- **Chronos** (Ansari et al., 2024, arXiv 2403.07815) — 时序量化 + next-token
- **Moirai** (Woo et al., ICML 2024, arXiv 2402.02592) — 掩码 patch + mixture distribution
- **TimesFM** (Das et al., 2024) — 自回归 patch 预测
- **PatchTST** (Nie et al., ICLR 2023) — patching + 掩码自编码
- **Moirai 2.0** (2025, arXiv 2511.11698) — decoder-only + quantile loss + multi-token
- **Moirai-MoE** (Liu et al., 2024, arXiv 2410.10469) — MoE 时序基础模型
- **In-Context Fine-Tuning for TSFM** (ICML 2025) — few-shot in-context adaptation

### 时序图专项
- **DySubC** — 时序子图对比学习
- **PT-DGNN** (Neurocomputing 2022) — 动态图生成预训练
- **MaskDGNN** (IJCAI 2025) — 活跃度感知时序掩码
- **Transfer Learning for TLP** (2025, arXiv 2504.10925) — 结构特征映射实现跨域迁移

### LLM
- **GPT** (Radford et al.) — Next-token prediction, decoder-only
- **BERT** (Devlin et al.) — Masked Language Model
- **T5** (Raffel et al.) — Denoising seq2seq

### 综述
- **Foundation Models for Structured Data** (2026 preprint) — 表格/时序/图统一综述
- **Unraveling Spatio-Temporal Foundation Models** (TKDE 2026) — 时空基础模型综述
