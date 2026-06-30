# 预训练方案决策日志

> 记录预训练目标设计过程中的关键决策、备选方案、依据
> 每个决策记录：背景 → 考量的备选 → 最终选择 → 理由 → 支撑文献

---

## 决策 1：预训练目标从 Next-Neighbor-Patch 改为三任务生成式预训练

**日期**：2026-06-30

### 背景

原设计文档（2026-06-28 起草）的预训练目标是 **Next-Neighbor-Patch Prediction**：
- 邻居序列按时间分 patch（每 4 个邻居一个 patch）
- 因果 Mamba 预测 next patch 的邻居 ID 分布（softmax over node vocab）
- loss = next-neighbor-patch CE + 链接预测 BCE

### 问题

数据分析（`data_analysis.md`）+ 文献深读（`2026_deep_read.md`）发现两个根本缺陷：

1. **node ID 跨域不迁移**：Reddit 11M 节点 vs Wikipedia 9K 节点，softmax over node vocab 的输出层无法跨域。Next-Neighbor-Patch 本质是预测"下一个邻居是谁"，但"谁"是 domain-specific 的。

2. **TGPM (ICML 2026) 已证明更好的方案**：MTM + NTP 双任务在 CTDG 跨域迁移上 average rank 1.0，大幅领先 PT-DGNN/DDGCL/CPDG。且 NTP（预测下一个事件的时间编码向量）直接解决"回顾性时间建模"缺陷——这是现有 CTDG 方法的共性问题。

### 备选方案

| 方案 | 来源 | 优势 | 劣势 |
|------|------|------|------|
| A. Next-Neighbor-Patch（原方案）| aLLM4TS | next-patch 思路有时序依据 | node ID 跨域不迁移 |
| B. 纯 LP BCE | MiNT/DDGPrompt | 简单，对齐 eval | 无自监督，可能过拟合短期 |
| C. MTM + NTP（纯自监督）| TGPM | 跨域迁移最好 | 无 BCE 强信号 |
| D. LP BCE + MTM + NTP（三任务混合）| 综合 | 信号丰富 | BCE 可能主导表示学习 |
| E. **分阶段：先 MTM+NTP，再加 BCE** | LLM pretrain→finetune | 兼顾通用模式与 eval 对齐 | 需调两阶段超参 |

### 最终选择：E（分阶段）

- Phase 2a：纯自监督 $\mathcal{L} = \beta \mathcal{L}_{MTM} + \gamma \mathcal{L}_{NTP}$（学通用模式）
- Phase 2b：Phase 2a checkpoint + $\alpha \mathcal{L}_{link}$（对齐 eval）

### 理由

1. TGPM 证明纯自监督跨域迁移最好（Table 2），但其 NTP 任务对齐 Hawkes 过程，是时序建模的理论最优
2. BCE 是强二元监督信号，直接对齐 eval（AP/MRR），但 TGPM 警示它会"过拟合短期相关性"
3. LLM 的成功范式是 pretrain（next-token，通用）→ finetune（任务对齐），而非从一开始就混合任务 loss
4. 分阶段还能对比验证 TGPM 核心论断：Phase 1（纯 BCE）vs Phase 2a（纯自监督）哪个跨域更好

### 支撑文献

- TGPM (ICML 2026, arXiv 2601.22454) — MTM + NTP，跨域迁移 average rank 1.0
- DDGPrompt (CIKM 2025, arXiv 2601.11954) — 验证标准 LP BCE 是 CTDG 默认预训练
- MiNT (NeurIPS 2025) — 标准 LP 多网络预训练成功迁移

---

## 决策 2：跨域结构特征用时序局部特征，不用静态全局 centrality/community

**日期**：2026-06-30

### 背景

用户选择"结构特征统一化"方向后，初版方案直接搬 Universal GFM (Stanford, 2026) 的 degree/centrality/community 作为 domain-agnostic 结构特征。

### 问题

用户指出：Universal GFM 是**静态图**，但我们是 **CTDG（时序动态图）**，且我们的框架只有 **K 邻居 ring buffer**，没有全图拓扑。具体：

1. **静态 vs 时序**：centrality/community 是全局属性，在 CTDG 里随时间变化。静态版的 degree/centrality 不能直接用。
2. **K 邻居约束**：我们的 `NeighborData` 只有 `(neighbor_ids, timestamps, edge_feats, mask)`——最近 K 个交互。**全局 centrality/community 从 K 邻居 buffer 根本算不出来**。
3. **co-occurrence 已排除**：数据分析显示二部图（wikipedia/reddit/lastfm/mooc）上 co-occurrence AUC=0.5（src 和 dst 在不同分区，共同邻居恒为 0）。

### 备选方案

| 方案 | 可算性（K 邻居）| 二部图有效 | 跨域 |
|------|----------------|-----------|------|
| A. 静态 degree/centrality/community（Universal GFM）| ❌ 算不出 | 部分 | ✅ |
| B. co-occurrence | ✅ 已有 kernel | ❌ AUC=0.5 | ✅ |
| C. 时序局部特征（recent_degree/Δt-stats/novelty）| ✅ 现算 | ✅ | ✅ |
| D. preferential_attachment (PA) | ✅ mask.sum | ✅ | ✅ |
| E. **C + D 组合** | ✅ | ✅ | ✅ |

### 最终选择：E（时序局部特征 + PA）

**单节点时序特征**（从 K 邻居 buffer 在 query time $t$ 现算）：
- `recent_degree` = `mask.sum(dim=1)`（时序度，≤K）
- `activity_rate` = `recent_degree / (t - t_oldest)`
- `Δt_mean / Δt_var`（时序规律性 / 突发性）
- `novelty_ratio` = `unique(neighbor_ids) / recent_degree`
- `recency_last` = `t - timestamps[newest]`

**配对时序特征**：
- `pair_history_count`（dst 在 src 邻居 buffer 出现次数）
- `pair_recency`（距上次 src-dst 交互时间）
- **`preferential_attachment`** = `recent_degree(src) × recent_degree(dst)`——$O(1)$，二部图同构图都有效

### 理由

1. **框架约束硬限制**：K 邻居 buffer 是 GatherSpec 的核心设计，改它等于重构框架。时序局部特征是这个约束下唯一可行的跨域结构信号。
2. **PA 被遗漏但关键**：数据分析里 degree 单独 AUC=0.5（随机负采样下无法区分），但 PA（度乘积）是经典 LP 启发式，对二部图和同构图都有效，是 co-occurrence 的正确替代。
3. **时序性**：recent_degree 是"最近 K 个交互的度"，天然时序——比 Universal GFM 的静态度更符合 CTDG。
4. **突发性特征**（Δt_var）额外价值：TGPM 警示高突发性数据集预训练收益小，Δt_var 既是输入特征也能用于 MoE 路由（按突发性分专家）。

### 支撑文献

- Universal GFM (Stanford, 2026, arXiv 2604.06391) — 验证结构特征统一化方向（但静态，需时序化）
- Scalable LP Pretraining (KDD 2025, Meta) — LP 是 pairwise，需 node + edge 双模块信号
- 数据分析 `data_analysis.md` — co-occurrence 二部图失效，degree 单独无信号

---

## 决策 3：跨域处理用 Structure + MoE

**日期**：2026-06-30

### 背景

预训练需要在多个异质数据集上混合训练（规模差 60x，同构 vs 二部图，重复率 0%~93%）。如何处理跨域异质性是核心难题。

### 备选方案

| 方案 | 来源 | 复杂度 | 跨域效果 |
|------|------|--------|---------|
| A. 共享 edge feat 空间 | — | 低 | 丢 node 信息 |
| B. Domain token | — | 中 | zero-shot 时新域 token 无定义 |
| C. Dataset-specific projection | — | 低 | 跨域交互有限 |
| D. **Structure + MoE** | AnyGraph + OOD 综述 | 高 | 最好 |

### 最终选择：D（Structure + MoE）

- **Structure**：domain-agnostic 时序结构特征（决策 2），不用 node ID
- **MoE 路由**：router 基于图统计（密度/重复率/二部性）选专家
  - Expert 1: 密集图（reddit/lastfm/enron）
  - Expert 2: 稀疏图（BitcoinAlpha/uci）
  - Expert 3: 二部图（wikipedia/mooc）
- **高效适配**：冻结 expert，只学 router assignment（Scalable LP 的 10000x 方案）

### 理由

1. **OOD 综述 (2026) 明确推荐**：四层 OOD 挑战中，域层 negative transfer 的解法就是 MoE + adaptive routing
2. **AnyGraph (2024) 验证**：MoE + SVD 特征统一化在静态图跨域成功，zero-shot 随 scaling 持续提升
3. **Scalable LP (KDD 2025) 验证**：冻结 expert + 只学 router assignment 实现 10000x 低开销适配——适合我们 4 卡资源约束
4. **数据分析支撑分组**：7 个数据集按密度/重复率/二部性自然分 3 组，MoE 专家划分有数据依据

### 支撑文献

- OOD Generalization in GFMs (2026, arXiv 2601.21067, 清华) — 四层 OOD 框架，推荐 MoE
- AnyGraph (2024, arXiv 2408.10700) — MoE + SVD 特征统一化
- Scalable LP Pretraining (KDD 2025, Meta, arXiv 2508.04645) — 冻结 expert + router-only 适配

---

## 决策 4：训练协议用 MiNT-style（shuffle + context switch）

**日期**：2026-06-30

### 背景

多数据集混合训练时，如何避免跨域状态泄漏和顺序偏差。

### 最终选择：MiNT 训练协议

- **Order shuffling**：每 epoch shuffle 数据集顺序（防止固定顺序的 spurious correlation）
- **Context switching**：切图时重置模型状态（Mamba hidden state / graph CSR buffer）
- **State reset**：防止跨域状态泄漏

### 理由

1. MiNT (NeurIPS 2025) 是第一个多网络 CTDG 预训练工作，64 网络训练 → 20 网络零样本迁移成功
2. Context switching 等价于 RNN 每 sequence 重置初始状态——是多序列训练的标准做法
3. 我们的框架已支持 graph snapshot/restore（eval 用），context switch 可复用

### 支撑文献

- MiNT (NeurIPS 2025, arXiv 2406.10426) — 多网络训练协议

---

## 决策 5：预训练数据按时间突发性分组

**日期**：2026-06-30

### 背景

TGPM 实验发现：高突发性数据集（concurrent edges 多）预训练收益小甚至有害。

### 数据分析依据

| 数据集 | Δt median | 突发性 | 预训练预期 |
|--------|-----------|--------|-----------|
| reddit | 3.2 | 高 | ⚠️ 收益有限 |
| mooc | 4.0 | 高 | ⚠️ 收益有限 |
| uci | 31.0 | 中 | ✅ |
| wikipedia | 16.0 | 中 | ✅ |
| lastfm | 82.0 | 低 | ✅ |
| enron | 1080.0 | 低 | ✅ |
| BitcoinAlpha | 86400.0 | 低 | ✅ |

### 最终选择

- **Phase 1/2**：先用低突发性数据集（enron/BitcoinAlpha/lastfm）验证预训练有效性
- **高突发性数据集**（reddit/mooc）：降低预训练权重或作为 hold-out 测试
- **MoE 路由**：考虑按突发性分组训练不同专家（Δt_var 作为 router 输入）

### 理由

1. TGPM 明确警示：高突发性下"pre-training strategies are much less beneficial and may even lead to trivial solution"
2. 我们的数据分析显示 reddit/mooc 的 Δt median（3-4）比 enron（1080）小 300x，突发性差异巨大
3. 保守策略：先在预训练大概率有效的数据集上验证，再扩展到高突发性

### 支撑文献

- TGPM (ICML 2026) — 时间突发性限制的实验发现

---

## 决策 6：保留 Mamba backbone，不用 TGPM 的 interaction patch tokenization

**日期**：2026-06-30

### 背景

TGPM 的 tokenization 是 interaction patch（时间偏置随机游走聚合），比我们的 K 邻居序列（一跳 + 时间排序）表达力更强——能捕获多跳结构 + 非单调时间依赖。TGPM 诊断这是"静态邻居语义假设"的解法。

### 备选方案

| 方案 | 表达力 | 速度 | 框架改动 |
|------|--------|------|---------|
| A. 保留 K 邻居序列（现有）| 中（一跳）| 快（torch.gather）| 无 |
| B. 换 interaction patch（TGPM）| 高（多跳+非单调）| 慢（随机游走采样）| 大（重构 GatherSpec）|

### 最终选择：A（保留 K 邻居序列）

### 理由

1. **性能优先**：我们的框架优势是 fused pipeline + CUDA kernel 的高效数据获取。随机游走采样破坏这个优势。
2. **渐进验证**：先在 K 邻居序列上验证预训练有效性（NTP/MTM 不依赖 tokenization 方式），再考虑 interaction patch 作为 ablation。
3. **部分缓解静态假设**：Mamba SSM 的线性复杂度天然支持长序列（缓解"短期依赖假设"），NTP 任务解决"回顾性时间建模"——TGPM 三假设里我们已解决两个，剩"静态邻居语义"留作 future work。
4. **K 可调**：K=512 时邻居序列已较长，部分弥补一跳限制。

### 风险

- 若 K 邻居序列在跨域迁移上明显不如 interaction patch，需在 Phase 3 后重新评估 tokenization 方案

### 支撑文献

- TGPM (ICML 2026) — interaction patch tokenization，但随机游走采样开销大
- 项目 memory — TGEngine 的 fused pipeline + CUDA kernel 是核心性能优势

---

## 决策演进时间线

| 日期 | 决策 | 触发 |
|------|------|------|
| 2026-06-28 | 初版：Next-Neighbor-Patch + GCA + co-occurrence | 设计起草 |
| 2026-06-30 | 数据分析：co-occurrence 二部图失效，degree 无信号 | 7 数据集实证 |
| 2026-06-30 | 文献深读：TGPM 的 MTM+NTP 优于 next-patch | 2026 文献深读 |
| 2026-06-30 | 决策 1：改三任务预训练（MTM+NTP+BCE）| TGPM insight |
| 2026-06-30 | 决策 1 修正：BCE 分阶段（pretrain→finetune）| 用户指出 BCE 是强信号 |
| 2026-06-30 | 决策 2：时序局部结构特征 + PA，去静态 centrality | 用户指出 CTDG 时序约束 |
| 2026-06-30 | 决策 3-6：MoE + MiNT 协议 + 突发性分组 + 保留 K 邻居 | 综合文献与框架约束 |
