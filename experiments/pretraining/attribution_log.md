# 归因验证实验日志

> 目标：验证 Foundation 模型预训练效果差的原因归因
> 4 个控制变量实验（小模型 d=128, K=32, 10 epoch 快速验证）

## 基准（baseline）：balanced mix + 三任务 + 无 node emb

| 数据集 | per-domain AP |
|--------|-------------|
| enron | 0.7038 |
| BitcoinAlpha | 0.7808 |
| uci | 0.8128 |

注意：小模型 baseline 比大模型（d=512）还好——enron 0.704 vs 0.711，BA 0.781 vs 0.732，uci 0.813 vs 0.777。大模型有容量浪费或过拟合。

## 实验 A：proportional mixing（按比例采样，不 cap）

**假设**：balanced round-robin 把 enron 从 75K cap 到 14K，数据不足导致差。

**结果**：

| 数据集 | baseline (balanced cap) | A_prop (按比例) | Δ |
|--------|------------------------|----------------|---|
| enron | 0.7038 | 0.7018 | -0.002 |
| BitcoinAlpha | 0.7808 | 0.6800 | -0.10 |
| uci | 0.8128 | 0.8379 | +0.025 |

**结论：假设被推翻！** 按比例采样（enron 保留全部 75K 边）反而让 enron 和 BA 变差，只有 uci 提升。

**新 insight**：enron 效果差**不是数据 cap 问题**。enron 有 629 batches（3x 数据）但 AP 没提升。可能原因：
- enron 只有 185 节点，avg degree 1157（极稠密图）——模型架构不适合极稠密图
- K=32 邻居序列对 degree 1157 的图严重不足（只能看到 32 个邻居）
- enron 的 edge_feat 语义和模型特征不匹配

## 实验 B：pure self-supervised（MTM+NTP，无 LP）

**假设**：LP dominate 自监督信号，导致模型没学到可迁移模式。去掉 LP 应该让自监督发挥作用。

**结果**：

| 数据集 | baseline (三任务) | B_noss (纯自监督) | Δ |
|--------|------------------|------------------|---|
| enron | 0.7038 | 0.5059 | -0.20 ❌ |
| BitcoinAlpha | 0.7808 | 0.5194 | -0.26 ❌ |
| uci | 0.8128 | 0.5306 | -0.28 ❌ |

**结论：假设被推翻！** 纯自监督（MTM+NTP，无 LP）**全面崩盘**，AP 降到 ~0.52（接近随机）。

**关键 insight**：LP 不是 dominate 自监督——反而是**唯一有效的学习信号**。去掉 LP 后 MTM+NTP 完全学不到链接预测能力。

**深层问题**：我们的 MTM/NTP 任务设计有缺陷：
- MTM 重建 edge_feat+pair_feat，但重建这些特征 ≠ 学到链接预测相关表示
- NTP 预测时间编码，时间预测能力 ≠ 链接预测能力
- TGPM 用纯自监督能 work，说明他们的 MTM/NTP 设计更好（interaction patch tokenization + 更强的 EMA target）
- 我们的简化版 MTM/NTP 没有学到对 LP 有用的表示

**这解释了为什么三任务混合时 LP loss 大幅下降而 MTM/NTP 几乎不动**——不是 LP 挤压自监督，而是自监督本身没学到东西，模型只能靠 LP。

## 实验 E：纯 LP（无 MTM/NTP），无 node emb

**假设**：MTM/NTP 是噪声，去掉应该提升（LP 是唯一有效信号）。

**结果**：

| 数据集 | baseline (三任务) | E_lponly (纯LP) | Δ |
|--------|------------------|----------------|---|
| enron | 0.7038 | 0.6567 | -0.05 ❌ |
| BitcoinAlpha | 0.7808 | 0.8081 | +0.03 ✅ |
| uci | 0.8128 | 0.7988 | -0.01 |

**结论**：MTM/NTP 对 enron/uci 有帮助，对 BA 是噪声。自监督任务的作用因域而异。

## 实验 D：纯 LP + node emb

**假设**：Foundation + node emb + 纯 LP 应该接近 Phase 1 的 0.876。

**结果**：enron=0.787（vs Phase 1 的 0.876，差 0.09）

## 最终归因（enron 0.876 → 0.704）

| 因素 | enron AP | 损失 |
|------|---------|------|
| Phase 1 _MiniMamba (node emb + 纯LP) | 0.876 | 基准 |
| D_lponly_nodeemb (Foundation + node emb + 纯LP) | 0.787 | -0.09（架构差异）|
| baseline (Foundation 无 node emb + 三任务) | 0.704 | -0.17（去 node emb）|

**enron 差的真正原因**：
1. **去 node embedding 损失 0.08-0.09**（主因）——Foundation 的 domain-agnostic 特征无法替代 node emb 对 enron 的作用
2. **Foundation 架构 vs _MiniMamba 差 0.09**——InputTokenizer 的结构特征（recent_degree/PA/Δt-stats）不如 _MiniMamba 的简单 node emb 对 enron 有效
3. **MTM/NTP 自监督对 enron 几乎无影响**（0.787 vs 0.783）

## 完整对比表

| 实验 | 配置 | enron | BA | uci |
|------|------|-------|-----|-----|
| baseline | 三任务, 无 node emb | 0.704 | 0.781 | 0.813 |
| A_prop | 按比例采样 | 0.702 | 0.680 | 0.838 |
| B_noss | 纯自监督(无LP) | 0.506 | 0.519 | 0.531 |
| C_nodeemb | 三任务+node emb | 0.783 | 0.651 | 0.728 |
| E_lponly | 纯LP, 无node emb | 0.657 | 0.808 | 0.799 |
| D_lponly_nodeemb | 纯LP+node emb | 0.787 | 0.683 | 0.672 |
| (Phase 1 ref) | _MiniMamba, node emb, 纯LP | 0.876 | 0.618 | 0.698 |

## 核心结论

### 1. 原假设全部推翻，真正原因是 node emb 缺失 + 架构差异

- ❌ 数据 cap（A_prop 推翻）
- ❌ LP dominate（B_noss 推翻）
- ❌ K 太小（K=32/64/128/256 enron 都 0.69-0.70）
- ✅ **去 node emb 是主因**（损失 0.08-0.09）
- ✅ **Foundation 架构不如 _MiniMamba**（差 0.09）

### 2. 没有单一配置对所有域最优——验证 MoE 必要性

- enron：需要 node emb（D=0.787 > baseline=0.704）
- BA：不需要 node emb，纯 LP 最佳（E=0.808）
- uci：三任务无 node emb 最佳（baseline=0.813）

这正好说明 OOD 综述推荐的 **MoE**（不同域用不同专家/特征）是正确方向。

### 3. 自监督任务设计需改进

- B_noss 证明去掉 LP 后 MTM/NTP 完全失效
- 但 E_lponly 证明去掉 MTM/NTP 后 enron 也降
- 说明 MTM/NTP 有微弱正面作用，但远不如 LP
- 需要更强的自监督目标（参考 TGPM interaction patch）
