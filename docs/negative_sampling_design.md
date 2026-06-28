# 负采样策略设计与语义对齐决策

> 决策日期: 2026-06-29
> 状态: 已决策

## 背景

DyGLib 的 Historical/Inductive 负采样语义要求"从 src 的**全部历史**交互中随机采样"。
TGEngine 的负采样策略需在**正确性对齐**与**大规模预训练可行性**之间权衡。

## 现有策略语义对照

| 策略 | 语义 | 内存 | 与 DyGLib 对齐 | 适用场景 |
|------|------|------|---------------|---------|
| `HistoricalNegative` | reservoir sampling (pool=512)，每条历史边等概率入池 | O(N×512) | **对齐**（无偏） | 小图、正确性基准 |
| `RandomNegative` | 全图随机采 | O(1) | 不对齐（无历史性） | 快速基线、预训练 |
| `InBatchNegative` | batch 内其他正样本 dst | O(B) | 不对齐（in-batch 偏置） | 大图预训练、hard negative |
| `InductiveNegative` | 从未见节点采 | O(N) | 对齐（DyGLib inductive） | inductive 评估 |
| `DyGLibHistoricalNegative` | CPU 全历史 set 扫描 | O(E) | **精确对齐** | 验证用（慢） |

## 决策

**1. 保留 `HistoricalNegative`（reservoir 512）作为 DyGLib 对齐的正确性基准。**
- reservoir sampling 保证无偏：每条历史边等概率进入 pool，从 pool 采样等价于从全历史采样。
- pool_size=512 对 degree ≤ 512 的节点是精确的；对 degree > 512 的节点是均匀抽样（统计等价，非精确）。
- **这不是 ring-buffer 近似**（旧版从最近 K 个采，有偏）。reservoir 是无偏的。

**2. 大规模预训练不强制对齐 HistoricalNegative。**
- pool `(num_nodes, 512)` int32：1M 节点 = 2GB，10M 节点 = 20GB，100M 节点 = 200GB——预训练扛不住。
- 预训练用 `InBatchNegative`（mix_random=0.5）或 `RandomNegative`，**明确标注语义不同**。
- 命名已区分（HistoricalNegative vs InBatchNegative vs RandomNegative），符合 CLAUDE.md 规则。

**3. Engine 必须调 `neg_strategy.update(src, dst)` 维护 pool。**
- 之前 Engine 未集成 update——HistoricalNegative 配置后 pool 一直空，采到 PADDING（bug）。
- 修复：train loop 每 batch 后 `if hasattr(neg_strategy, 'update'): neg_strategy.update(src, dst)`。

## 不做精确全历史对齐的理由

精确对齐（DyGLibHistoricalNegative，CPU O(E) set 扫描）每 batch 扫全历史边，本质无法 GPU 加速。
对预训练（10M-100M 边）完全不可行。reservoir 512 已是无偏近似的最优解，再追求精确对齐
违背框架高性能定位。预训练阶段放松语义是合理的工程 tradeoff，下游评估时可切换回
HistoricalNegative（小图）或 FixedNegative（TGB 固定负样本）保证指标可比。

## 相关

- CLAUDE.md「性能优化的正确性原则」——近似必须命名区分
- memory `project_tgengine_status.md`——Historical/Inductive 负采样决策记录
- TGM issue #413——ring buffer 有界丢邻居的同类问题（reservoir 避免了此问题）
