# 2026 年相关文献深读：对 CTDG 预训练的 insight

> 深读日期：2026-06-30
> 目标：逐篇深读 2026 年发表的相关文献，提取对 CTDG 预训练任务有用的 insight
> 原则：详细提取方法细节，不做摘要式罗列

---

## 文献 1：TGPM — Temporal Graph Pattern Machine（ICML 2026, arXiv 2601.22454）

**这是与我们任务最直接相关的工作——CTDG 上的自监督预训练基础模型。**

### 1.1 核心问题诊断：三个错误假设

TGPM 的出发点是诊断现有 CTDG 方法的三个根本性假设缺陷，这直接阻碍了模型学到可迁移的演化模式：

1. **静态邻居语义假设**（Static neighborhood semantics）
   - 现有方法（DyGFormer/TGN/GraphMixer）用一跳邻居动态 + 时间编码建模演化
   - **问题**：邻居语义是非平稳的——同一个邻居的功能角色和行为模式会随图结构演化而改变
   - 把这种语义动态压缩成固定时间 embedding + 采样过程，限制了模型学习跨图实例/时间域的演化模式

2. **短期依赖假设**（Short-term dependency）
   - 现有方法假设最近邻居交互足以刻画节点当前状态，限制在局部时间窗口
   - **问题**：系统性偏向短期信号，无法捕获控制真实系统演化的长期依赖

3. **回顾性时间建模假设**（Retrospective temporal modeling）
   - 时间信息只通过"事后条件化"（post-hoc conditioning）历史事件来引入——时间衰减 + attention 权重
   - **问题**：时间只是历史事件的辅助标注，不是显式建模目标。模型能识别"哪些历史事件相关"，但无法准确刻画"未来事件何时发生"

**对我们的启示**：我们的 Mamba backbone + K 邻居序列设计也受这三个假设影响。特别是第三点——我们目前的时间编码（RoPE Δt）也是"回顾性"的，没有把"预测未来时间"作为目标。TGPM 的 NTP 任务直接解决这个问题。

### 1.2 Tokenization：Interaction Patch（交互块）

TGPM 的 tokenization 机制值得深入理解——这是把 CTDG 转为序列的核心设计：

**核心思想**：每个交互 $(v, v', t')$ 不只由节点 ID + 时间戳表示，而是用一个 **interaction patch** 表示——通过聚合 $k$ 条"时间偏置随机游走"（temporally biased random walks）构建。

**时间偏置随机游走的设计**：
- 以 $v'$ 为根，采样长度 $L$ 的游走 $w = (v_0, v_1, \dots, v_L)$，$v_0 = v'$
- 转移概率：$\eta(u,v) = \exp(t' - \mathcal{T}(u,v))$（边越接近 $t'$，采样优先级越高）
- **关键创新**：不要求时间单调递减（relaxes strict causal ordering）——允许非单调遍历历史交互
- 证明：非单调时间游走严格比因果单调游走更有表达力（Proposition 2.1）

**Patch embedding 构建**：
- 每条游走 $w$ 提取节点特征 $\mathbf{X}_{v'}^{t'}$ + 边特征 $\mathbf{E}_{v'}^{t'}$ + 时间间隔编码 $\mathbf{T}_{inter}$
- 时间编码用正弦函数：$T_{enc}(\Delta t) = \sqrt{1/d_t}[\cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots]$，$\{\omega_i\}$ 可训练
- 拼接后线性投影 + Transformer encoder 编码每条游走
- $k$ 条游走的 embedding 做 mean pooling 得到 patch embedding $\mathbf{p} = \frac{1}{k}\sum_w \mathbf{h}_w$

**对我们的启示**：
- 我们的 K 邻居序列是"一跳邻居按时间排序"，本质上是 TGPM 说的"静态邻居语义"假设
- TGPM 的 interaction patch 通过随机游走捕获多跳结构 + 非单调时间依赖，比一跳邻居丰富
- 但随机游走采样比我们的 `torch.gather` 邻居查询慢得多——这是性能 vs 表达力的 trade-off
- **可借鉴**：时间编码用可训练频率的正弦函数（我们用 RoPE，类似但 TGPM 更明确）

### 1.3 两个预训练任务（核心创新）

TGPM 设计了两个互补的自监督预训练任务，这是对我们方案设计最有价值的参考：

#### Task 1: Masked Token Modeling (MTM) — 学"什么演化"

- mask 连续 token 块（block-wise masking，不是随机单 token）
- 用 EMA encoder 生成稳定重建目标 $\hat{\mathbf{p}}_i = f_{EMA}(\{w \mid v_0 = v'_i\})$（类似 BYOL/BEiT 的 momentum encoder）
- SimMIM 式重建：visible tokens + learnable mask tokens → TGPM encoder + Transformer decoder → 重建 token
- **关键理论**：block size $b$ 直接控制模型被迫推理的最小时间跨度——block 越大，模型必须跨越越长的依赖来重建
- 证明（Proposition 2.2）：block-wise masking 不是启发式设计，而是强制多尺度时间依赖学习的信息论必要条件

**对我们的启示**：
- 我们的"masked Δt reconstruction"应该用 **block-wise masking** 而非随机 mask
- block size 控制时间跨度——这是多尺度时间依赖的关键超参
- EMA encoder 做重建目标——比直接用真值更稳定（防止 representation collapse）

#### Task 2: Next Time Prediction (NTP) — 学"何时演化"

这是 TGPM 最独特的贡献，直接解决"回顾性时间建模"假设：

- TGPM encoder 输出 $\mathbf{P}' = [\mathbf{p}'_1, \dots, \mathbf{p}'_m]$
- 两层 MLP head $f_{NTP}$ 预测下一个交互的时间间隔编码
- 自回归分解：$p(\mathbf{t}_1, \dots, \mathbf{t}_m) = \prod_{i=1}^m p(\mathbf{t}_i \mid \mathbf{t}_{<i}, \bar{\mathbf{p}}_{<i})$
- **Loss**：$\mathcal{L}_{NTP} = \frac{1}{m-1}\sum_{i=1}^{m-1} \|f_{NTP}(\mathbf{p}'_i) - \mathbf{t}_{i+1}\|$
- 注意：预测的是**时间间隔编码** $\mathbf{t}_{i+1}$（正弦编码向量），不是原始时间值

**为什么这个任务重要**：
- 迫使模型编码演化时间粒度（连续交互模式的典型间隔）
- 编码频率相关信号（某些演化模式的交互频率）
- 让模型把不同演化模式关联到不同时间节奏

**对我们的启示**：
- **这是我们方案里缺失的任务**。我们原计划只有 link prediction BCE + masked Δt reconstruction，没有"显式预测未来时间"
- NTP 直接对齐 Hawkes 过程的强度函数建模——CTDG 的理论根基
- 实现成本低：一个 MLP head 预测时间编码向量，L2 loss
- **应该加入我们的方案**：$\mathcal{L} = \alpha \mathcal{L}_{link} + \beta \mathcal{L}_{MTM} + \gamma \mathcal{L}_{NTP}$

### 1.4 联合训练与下游使用

- 总 loss：$\mathcal{L} = \mathcal{L}_{MTM} + \mathcal{L}_{NTP}$
- 预训练后丢弃 decoder，只保留 TGPM encoder
- 下游任务：对 $\mathbf{P}'$ 做 mean pooling → task-specific head

### 1.5 实验关键发现

**跨域迁移结果**（Table 2，这是最 relevant 的数据）：

| 训练图 → 测试图 | TGPM | PT-DGNN | DDGCL | CPDG |
|----------------|------|---------|-------|------|
| Enron → Googlemap CT | **87.06** | 65.79 | 66.66 | 75.74 |
| Enron → ICEWS1819 | **88.92** | 66.99 | 56.71 | 66.12 |
| ICEWS1819 → Googlemap CT | **56.21** | 53.79 | 52.42 | 51.96 |
| Googlemap CT → Enron | **92.51** | 90.83 | 91.40 | 89.18 |

- TGPM 跨域迁移大幅领先（average rank 1.0）
- **关键洞察**：时间移位和结构邻近性的自监督方法（PT-DGNN/DDGCL/CPDG）对时间动态的细微 shift 敏感，而 TGPM 学到的是"可迁移的演化模式"

**时间突发性（temporal burstiness）的限制**——重要警示：

> "on datasets with significant temporal burstiness, pre-training strategies are much less beneficial and may even lead to trivial solution"

- 在时间突发性强的数据集上，预训练收益小甚至有害
- TGPM 在处理"并发边"（concurrent edges）场景时 scaling 困难
- **对我们的启示**：我们的数据集里 reddit（Δt median 3.2）和 mooc（Δt median 4.0）是高突发性，预训练可能效果有限；而 enron（Δt 1080）、BitcoinAlpha（Δt 86400）是低突发性，更适合预训练

**Scaling**：增加参数持续提升 transductive 性能，但在高并发边数据集上 scaling 困难。

---

## 文献 2：CrossHGL — 跨域异构图基础模型（arXiv 2603.27685, 2026.03）

**静态异构图，不是时序的。但跨域处理方法有参考价值。**

### 2.1 核心问题：text-free 跨域异构

- 现有图 FM 要么聚焦同构图，要么依赖域特定 schema，要么依赖文本属性做语义对齐
- **text-free 场景**（金融交易/网络异常/分子图）被忽视——特征是数值/类别/匿名化的
- **对我们的启示**：我们的 CTDG 数据集 edge_feat 是 172 维 LIAR 编码，但语义不同（文本 vs 邮件 vs 交互）——本质也是 text-free 跨域

### 2.2 三阶段方案

**阶段 1：语义保持图变换**（Semantic-preserving graph transformation）
- **SVD 特征对齐**：不同维度 $d_a$ 的节点特征用 SVD 降到统一维度 $d$
- **自动 meta-pattern 挖掘**：异构拓扑同质化，把多关系上下文压缩到"语义增强边"里
- 关键：不直接扁平化拓扑（会丢语义），而是把异构语义编码进边特征

**阶段 2：Tri-Prompt 多域预训练**
- 三个 prompt 矩阵：**feature prompt** + **edge prompt** + **structure prompt**
- 共享 GNN backbone + 自监督图对比学习
- Tri-Prompt 解耦多维图语义

**阶段 3：参数高效微调**
- 冻结预训练 backbone
- Attention-based prompt composition 适配目标域
- 非参数原型网络做 few-shot 分类

**对我们的启示**：
- SVD 特征对齐是处理不同维度 edge_feat 的简单有效方法（我们的数据集都是 172 维，暂不需要，但如果加入 TGB 数据集可能需要）
- Tri-Prompt（feature/edge/structure 三个维度）比单一 prompt 更全面——DDGPrompt 也有类似设计
- 冻结 backbone + prompt 微调是 few-shot 适配的标准做法

---

## 文献 3：OOD Generalization in GFMs 综述（arXiv 2601.21067, 2026.01, 清华）

**第一个从 OOD 视角系统综述 GFM 的工作。**

### 3.1 四层 OOD 挑战框架

这是对我们跨域 CTDG 问题最有用的分析框架：

| 层级 | 挑战 | 现有解法 |
|------|------|---------|
| **结构层** | 拓扑/属性/关系模式差异大；spurious correlation | alignment + invariance 目标 |
| **域层** | 多域预训练导致 negative transfer；共享模型偏向主导域 | **MoE + adaptive routing** |
| **模态层** | 辅助模态（文本/分子特征）可用性/质量变化；过度依赖某模态 | alignment + gating |
| **任务层** | 不同输出空间/监督粒度；task-specific 微调导致遗忘 | prompting + instruction |

**对我们的启示**：
- 我们的跨域 CTDG 问题同时面临**结构层**（二部图 vs 同构图，密度差 20x）+ **域层**（不同数据集语义不同）挑战
- 域层挑战的解法明确：**MoE + adaptive routing**（AnyGraph 和这个综述都推荐）
- 这验证了我们选择 "Structure + MoE" 方向的正确性

### 3.2 Homogeneous-task vs Heterogeneous-task GFM 分类

**Homogeneous-task GFMs**（固定任务，如只做链接预测）：
- GraphFM：Perceiver-style encoder，100+ 图预训练，node classification
- AnyGraph：MoE + link prediction + SVD 特征统一
- MDGPT：domain tokens + universal link prediction
- **PatchNet**：可学习 graph patches 处理特征异质性——把节点属性展开为固定大小 token channels
- GraphAny：fully inductive，LinearGNN 闭式参数
- GraphLoRA：low-rank adaptation + 结构感知 MMD 对齐
- **MDGFM**：拓扑对齐 + 对比互信息最大化
- SAMGPT：text-free，structure tokens + 对比学习
- **GOODFormer (2026)**：invariant graph transformer + entropy-guided subgraph disentangler——分离 invariant vs variant 子结构
- GraphPFN：prior-data fitted network，合成图预训练

**Heterogeneous-task GFMs**（支持多任务）：
- OFA：text-attributed，nodes of interest 统一任务
- **OpenGraph**：universal tokenization + masked autoencoding + 合成图增强
- **GOFA**：generative graph completion（next-token prediction 推广到节点）
- **GFT**：transferable tree vocabulary——向量量化学习离散结构词汇表
- GIT：task-trees 统一抽象
- UniGraph/UniGraph2：masked graph modeling + cascaded backbone

**对我们的启示**：
- 我们属于 **homogeneous-task GFM**（只做链接预测），重点在结构层 + 域层 OOD
- **GOODFormer 的 invariant subgraph disentangler** 值得借鉴——分离跨域不变子结构 vs 域特异子结构
- **GFT 的 tree vocabulary**（向量量化结构模式）是另一种 tokenization 思路——学习离散结构词汇表
- **OpenGraph 的 universal tokenization + masked autoencoding** 和 TGPM 的 MTM 类似，验证了这条路线

---

## 文献 4：Scalable Pretraining for Link Prediction（KDD 2025, Meta/MSU, arXiv 2508.04645）

**第一个专门针对链接预测预训练的系统研究。虽然静态图，但 LP 预训练的 insight 直接适用。**

### 4.1 核心洞察：LP 是 pairwise 任务

- 现有 GFM 聚焦 node-level 表示，对 LP 次优
- LP 需要两类信号：
  - **Feature Proximity (FP)**：节点特征相似性（homophily）→ NodeEncoder 捕获
  - **Structure Proximity (SP)**：邻居重叠/路径信息 → EdgeEncoder 捕获
- MPNN 无法数三角形 → 无法计算 common neighbors / Adamic-Adar 等关键 LP 启发式

**双模块设计**：
$$H = \text{NodeEncoder}(A, X), \quad p_{ij} = \text{ScoreFunction}(H_i \odot H_j)$$
$$e_{ij} = \text{EdgeEncoder}(A, i, j), \quad p_{ij} = \text{ScoreFunction}(e_{ij})$$

**对我们的启示**：
- 我们的 Mamba backbone 是 NodeEncoder（编码 src/dst 邻居序列）
- **我们缺 EdgeEncoder**——co-occurrence 本来可以充当 SP 信号，但数据分析显示它在二部图上失效
- 需要考虑：是否加一个显式的 pairwise/structural encoder 来捕获 SP

### 4.2 MoE 框架处理多样化预训练数据

- 不同 expert 捕获不同模式，避免 negative transfer
- **参数高效适配**：只学习 expert assignment（每个下游数据集的专家选择），保持 expert 参数不变
- 10,000x 低计算开销
- 实验在 16 个数据集跨 2 个域验证

**对我们的启示**：
- MoE + 冻结 expert + 只学 router assignment——这是高效适配的实用方案
- 比直接微调整个 backbone 高效得多，适合我们 4 卡资源约束

### 4.3 Late Fusion 策略

- 发现 node module 和 edge module 训练不平衡
- Late fusion（分别训练后融合）比 early fusion（输入层融合）更稳

---

## 文献 5：Universal Graph FM（Stanford, arXiv 2604.06391, 2026.04）

**生物医学图，但"结构特征统一化"方法直接验证我们的方向。**

### 5.1 Feature-agnostic 结构 prompts

核心思想：不用 node identity 或 feature scheme，只用**图结构属性**：

- **Degree statistics**（度统计）
- **Centrality measures**（中心性度量）
- **Community structure indicators**（社区结构指标）
- **Diffusion-based signatures**（基于扩散的签名）

这些结构属性编码为 "structural prompts"，与 message-passing backbone 集成，把不同图嵌入共享表示空间。

**对我们的启示**：
- **这直接验证了我们 "结构特征统一化" 方向**——用 domain-agnostic 结构特征而非 node ID
- 具体可用的结构特征：degree rank、centrality、community membership、diffusion fingerprint
- 我们的数据分析已显示 co-occurrence 在二部图失效，但这些更基础的结构特征（degree/centrality）应该更鲁棒
- **应该把这些结构特征加入我们的 input token**：edge_feat_proj + RoPE(Δt) + degree/centrality/community + recency

---

## 文献 6：DDGPrompt — Data-centric Prompt Tuning for Dynamic Graphs（CIKM 2025, arXiv 2601.11954）

**CTDG 上的 prompt tuning，验证标准链接预测预训练 + prompt 适配路线。**

### 6.1 预训练 + 适配范式

- **预训练任务**：动态链接预测（标准 BCE）——这是 CTDG 领域的默认做法
- **问题**：pretrain（链接预测）→ downstream（节点分类/异常检测等）task gap 大，few-shot 下性能退化
- 现有 prompt 方法（TIGPrompt/DyGPrompt）只改 node/time 特征，忽略空间结构

### 6.2 Node Expression Feature Matrix

定义统一的节点表达特征矩阵：
- 编码节点最近一阶邻居信息（node features + edge features + time features）
- 兼容各种 backbone（TGN/GraphMixer/DyGFormer）
- 作为模型输入生成可迁移的 node temporal embedding

### 6.3 三个互补 Prompt 矩阵

1. **Temporal bias prompt**：动态调整每个邻居的时间特征
2. **Edge weight prompt**：给每个邻居分配可学习重要性分数（捕获空间结构相关性）
3. **Feature mask prompt**：task-aware 增强网络，选择性调制特征维度

**对我们的启示**：
- **标准链接预测 BCE 预训练是 CTDG 领域共识**——TGPM 用 MTM+NTP 是创新，但 DDGPrompt 验证了纯 LP 预训练也是可行 baseline
- 三维 prompt（temporal/edge_weight/feature_mask）比单一 prompt 全面——edge weight prompt 特别值得借鉴（给邻居加权而非简单聚合）
- 我们的 Engine 已支持多任务 head，可以加 prompt tuning 作为 few-shot 适配手段

---

## 文献 7：SDG — Sequence Diffusion Model for CTDG LP（arXiv 2601.23233, 2026.01）

**把扩散模型引入 CTDG 链接预测。**

### 7.1 核心创新

- 现有 CTDG 模型是判别式的（点估计），缺乏不确定性 + 序列结构建模
- SDG 把噪声注入**整个历史交互序列**，通过条件去噪过程联合重建所有交互 embedding
- Cross-attention denoising decoder 指导目标序列重建
- 端到端优化

**对我们的启示**：
- 扩散模型作为预训练目标是另一种思路——不是 mask+predict，而是 noise+denoise
- 序列级去噪比单点去噪捕获更全面的交互分布
- 但实现复杂度高，且与我们的 Mamba backbone 集成不直观——作为 future direction 考虑

---

## 综合洞察：对我们预训练方案的修正

### 必须加入的改进

1. **NTP 任务（来自 TGPM）**——这是最大的遗漏
   - 我们原方案：link BCE + masked Δt reconstruction
   - 修正：$\mathcal{L} = \alpha \mathcal{L}_{link} + \beta \mathcal{L}_{MTM} + \gamma \mathcal{L}_{NTP}$
   - NTP 预测下一个交互的时间编码向量（不是原始时间值）
   - 直接解决"回顾性时间建模"缺陷，对齐 Hawkes 过程

2. **Block-wise masking（来自 TGPM）**——而非随机 mask
   - block size 控制时间跨度，是多尺度依赖的关键
   - 用 EMA encoder 生成稳定重建目标（防 collapse）

3. **更丰富的结构特征（来自 Universal GFM + 数据分析）**
   - 不只 edge_feat + RoPE(Δt) + recency
   - 加入 degree/centrality/community 结构属性
   - 这些比 co-occurrence 更鲁棒（二部图不失效）

### 需要验证的设计选择

4. **Tokenization：K 邻居序列 vs Interaction Patch（TGPM）**
   - 我们的 K 邻居序列是"一跳 + 时间排序"——TGPM 说的"静态邻居语义"假设
   - TGPM 的 interaction patch（时间偏置随机游走）捕获多跳 + 非单调时间
   - **决策点**：保留 K 邻居序列（性能优先）还是换 interaction patch（表达力优先）？
   - 建议：先保留 K 邻居序列验证预训练有效性，再考虑 interaction patch 作为 ablation

5. **EdgeEncoder 缺失（来自 Scalable LP）**
   - LP 是 pairwise，需要 NodeEncoder + EdgeEncoder
   - 我们只有 NodeEncoder（Mamba）
   - co-occurrence 本可作 SP 信号但二部图失效
   - **决策点**：是否加一个 pairwise structural encoder？

### 已验证正确的方向

6. **Structure + MoE（来自 OOD 综述 + AnyGraph + Scalable LP）**
   - OOD 综述明确推荐 MoE 处理域层 negative transfer
   - Scalable LP 用 MoE + 冻结 expert + 只学 router assignment
   - 我们的 "Structure + MoE" 方向完全正确

7. **标准链接预测 BCE 作为 baseline（来自 DDGPrompt）**
   - CTDG 领域默认预训练任务就是 LP
   - 即使 TGPM 用 MTM+NTP，也和 LP 互补不冲突
   - Phase 1 先做纯 BCE 验证多域训练有效性的策略合理

### 时间突发性警示（来自 TGPM）

8. **不是所有数据集都适合预训练**
   - 高突发性数据集（reddit Δt=3.2, mooc Δt=4.0）预训练收益小甚至有害
   - 低突发性数据集（enron Δt=1080, BitcoinAlpha Δt=86400）更适合
   - **预训练数据混合策略**：可能需要按突发性分组，或在高突发性数据集上降低预训练权重
