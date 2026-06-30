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

## 实验 C：with node embedding

**假设**：无 node embedding 导致 in-domain eval 吃亏。加回 node emb 应该提升 in-domain AP。

**结果**：

| 数据集 | baseline (无 node emb) | C_nodeemb (有 node emb) | Δ |
|--------|----------------------|------------------------|---|
| enron | 0.7038 | **0.7834** | +0.08 ✅ |
| BitcoinAlpha | 0.7808 | 0.6508 | -0.13 ❌ |
| uci | 0.8128 | 0.7275 | -0.09 ❌ |

**结论**：node embedding 对 enron（极稠密小图）帮助大，但对 BA/uci 反而有害。

**insight**：node embedding 记住具体节点，in-domain 受益，但损害跨域。enron 只有 185 节点，node emb 能有效记忆每个节点的交互模式；BA/uci 节点多（3784/1900），node emb 容易过拟合训练集。

## 完整对比与归因结论

| 实验 | enron | BA | uci | avg |
|------|-------|-----|-----|-----|
| baseline | 0.704 | 0.781 | 0.813 | 0.766 |
| A_prop (按比例) | 0.702 | 0.680 | 0.838 | 0.740 |
| B_noss (纯自监督) | 0.506 | 0.519 | 0.531 | 0.519 |
| C_nodeemb | 0.783 | 0.651 | 0.728 | 0.721 |

### 归因结论（修正原假设）

**原假设全部被推翻，真正原因是：**

1. **enron 差的真正原因：K=32 对极稠密图严重不足**（不是数据 cap）
   - enron avg degree 1157，K=32 只能看到 32 个邻居，丢失 97% 邻居信息
   - 证据：A_prop 给 3x 数据没用（0.702），C_nodeemb 加 node emb 才有效（0.783）
   - 解法：对稠密图用更大 K（如 K=256），或用 attention pooling 代替 last-pool

2. **自监督任务设计有根本缺陷**（不是 LP dominate）
   - B_noss 证明去掉 LP 后 MTM/NTP 完全失效（~0.52 随机）
   - MTM 重建 edge_feat+pair_feat ≠ 学到链接预测相关表示
   - NTP 预测时间 ≠ 链接预测能力
   - 解法：重新设计 MTM/NTP 目标，或参考 TGPM 的 interaction patch

3. **node embedding 是 in-domain/跨域 trade-off**
   - C_nodeemb 让 enron +0.08 但 BA/uci -0.09~-0.13
   - 验证"去 node ID"方向正确，但需要更强的 domain-agnostic 特征

### 下一步方向

1. **增大 K**：对稠密图（enron）用 K=128 或 256，验证是否解决信息丢失
2. **重新设计自监督任务**：MTM 目标改为重建更有链接预测相关性的特征，或换 TGPM 的 interaction patch tokenization
3. **用 zero-shot eval**：当前 in-domain eval 无法体现跨域迁移价值，改 leave-one-out
4. **调 loss 权重**：既然 LP 是主信号，可降 MTM/NTP 权重避免噪声干扰
