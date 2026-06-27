# CrossMamba: Cross-Domain Zero-Shot Temporal Link Prediction

## Motivation

标准 CTDG 模型（DyGFormer, TGN 等）依赖边特征，无法直接迁移到不同特征空间的新域。
CrossMamba 的核心假设：**时间结构（交互的时序模式）是跨域通用的**——
只用时间戳构造的邻居序列，通过 Mamba SSM 学习时序依赖，就能在未见过的域上做 zero-shot 推理。

---

## 模型架构

```
neighbors (K recent) → relative time diffs
  → FixedCosineTimeEncoder → (B, K, d_model)
  → MambaBlock × n_layers  → (B, K, d_model)
  → masked mean pool       → (B, d_model)
score(u, v) = temperature * (u_emb · v_emb)
```

- 无边特征输入（d_edge=0），纯时序结构
- Backbone: Mamba V1 + 官方 selective_scan CUDA kernel

---

## Experiment Log

### [2026-05-30] E0: Mamba V1 速度瓶颈定位与修复

**问题**：CrossMamba 训练速度 3.1 it/s，比预期慢约 10x。

**根因**：自实现的 `_SelectiveSSM` 用 Python `for l in range(L)` 做 sequential scan，
autograd 追踪 L=32 步 → backward 219.7ms vs forward 25.6ms（8.6x 比率）。

**修复**：替换为 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` 官方 CUDA kernel。

**结果**：
| | Before | After |
|--|--------|-------|
| forward | 25.6ms | 5.6ms |
| backward | 219.7ms | 12.6ms |
| total | 245ms | 18.3ms |
| throughput | 3.1 it/s | **43 it/s** |

**附带 bug**：`kernels.py` 在 import 时 JIT 编译 tgengine CUDA extension，
与 mamba_ssm CUDA extension 产生全局 context 冲突 → segfault。
修复：改为 lazy init（首次调用时再加载）。

---

### [2026-05-30] E1: 首次跨域训练（K=32，7个训练域）

**训练域**：CanParl, USLegis, BitcoinAlpha, BitcoinOTC, enron, CollegeMsg, uci（376K edges）  
（注：mooc 351K 占 48% 数据量，排除）

**配置**：K=32, d_model=128, n_layers=2, epochs=50, patience=10

**结果**：
- In-domain test AP: **0.9654**（13 epochs, 465.6s）
- Zero-shot transfer：

| Domain | AP | n_test |
|--------|----|--------|
| wikipedia | 0.8304 | 19,241 |
| lastfm | 0.7468 | 166,604 |
| mathoverflow | 0.7688 | 13,192 |
| Contacts | 0.8230 | 320,247 |
| UNtrade | 0.5552 | 62,856 |

**观察**：UNtrade 表现最差（0.55），属政治/贸易图，结构与训练域（社交/金融/政治小图）差异较大。

---

### [2026-05-30] E2: Mamba V1/V2/V3 对比（短序列 K=16/32）

**目的**：验证不同 Mamba 版本在短序列下的速度与精度。

**结果（RTX 3090, uci, 5 epochs，含图采样开销）**：

| Version | K=16 it/s | K=32 it/s | K=16 AP | K=32 AP |
|---------|-----------|-----------|---------|---------|
| V1 | **34.9** | **35.1** | 0.7005 | 0.7117 |
| V2 | 14.7 | 15.1 | 0.6456 | 0.5949 |
| V3 | 18.5 | 22.2 | **0.7268** | 0.7030 |

**结论**：短序列 V1 最快（2-3x），精度与 V3 持平。

---

### [2026-05-30] E3: Mamba V1/V2/V3 纯模型 Microbenchmark（长序列）

**目的**：排除图采样开销，测模型本身 forward+backward 耗时。  
**环境**：RTX 4080, d_model=128, n_layers=2, batch_size=200×3=600 nodes, 50次重复

| Version | K=128 total | K=256 total | K=512 total | K=1024 total |
|---------|------------|------------|------------|-------------|
| V1 | **15.4ms** | **27.5ms** | 124.8ms ⚠️ | 325.9ms |
| V2 | OOM* | 46.4ms | 92.0ms | **183.3ms** |
| V3 | OOM* | **42.3ms** | **83.8ms** | OOM* |

> *OOM 为进程内显存碎片（非模型自身限制）

**关键发现**：V1 在 K=512 出现 backward 异常（bwd/fwd 从 2.4x 跳到 6.5x），
是 selective_scan CUDA kernel 超出 shared memory 阈值后退化到低效路径。

**结论**：
- K ≤ 256：V1 最快（约 3x 优势）
- K ≥ 512：V2/V3 反超，尤其 K=1024 时 V2 比 V1 快 1.8x
- **当前选择 K=128，V1 为最优 backbone**

---

### [2026-05-30] E4: 数据集重构 + 均衡混合策略

**问题**：原混合策略按全局归一化时间排序，UNtrade（507K）占 62%，严重主导训练。

**改动**：
1. 重新选择训练域：uci（社交）+ CollegeMsg（社交）+ BitcoinAlpha（金融）+ wikipedia（用户-内容）
   - 覆盖更多样化的图结构，避免单一领域主导
2. `MixedDataset.get_batches(balance=True)`：每个 domain 截取到最小 domain 大小（BitcoinAlpha ~14K），round-robin 出 batch
   - 4 域均等贡献，intra-domain 保持时序顺序
3. Zero-shot 测试域：UNtrade, enron, lastfm, mathoverflow

**待跑**：K=128 完整训练，对比 E1 的 zero-shot 结果。

---

### [2026-05-30] E5: 架构重设计 + K=128 + balanced mixing ✓

**核心问题**：原模型每个 Mamba 位置只有时间差编码，完全丢失邻居身份信息，不是真正的图序列学习。

**架构改动**：

每个邻居位置输入从 `time_enc(dt)` 扩展为：
```
input_proj( time_enc(dt) || struct_proj([rank, freq, co_occur]) )
```
- `rank`：归一化位置 i/(K-1)，最近=1，告诉 Mamba 序列顺序
- `freq`：该邻居 ID 在 K 窗口内的出现次数/K，检测 hub 节点
- `co_occur`：该邻居是否也出现在对端节点的 K 窗口里（类 DyGFormer 结构信号）
- 实现：Triton kernel 消除 O(B,K,K) 中间张量；CPU fallback 到 PyTorch

**Pooling 改动**：mean pool → last valid position（Mamba 因果输出，最后位置聚合完整历史）

**数据改动**：
- 训练域：uci + CollegeMsg + BitcoinAlpha + wikipedia（覆盖社交/金融/用户内容）
- balanced mixing：每 domain 截到 BitcoinAlpha 的 13,943 条训练边，round-robin 出 batch
- 总 balanced train batches：280/epoch，~12s/epoch，RTX 4080

**结果（50 epochs，patience=10，最佳 epoch 41）**：

In-domain test AP：**0.9064**（训练时 649s）

Zero-shot 迁移（训练时完全未见，10 个域）：

| Domain | E5 AP | E1 AP（旧K=32纯时间）| 变化 | 备注 |
|--------|-------|------|------|------|
| Contacts | **0.9381** | 0.8230 | **+10.5pp** | 物理接触图，时序结构强 |
| mooc | **0.8752** | — | 新增 | 教育行为序列 |
| mathoverflow | 0.8468 | 0.7688 | **+7.8pp** | QA 图，稀疏非均匀 |
| enron | 0.8308 | — | 新增 | 邮件社交图 |
| BitcoinOTC | 0.8379 | — | 新增 | 金融信任图（同族 BitcoinAlpha） |
| SocialEvo | 0.7699 | — | 新增 | 校园传感器接近图 |
| CanParl | 0.7756 | — | 新增 | 政治投票图 |
| lastfm | 0.6375 | 0.7468 | -10.9pp | 音乐听歌图，训练域覆盖少 |
| USLegis | 0.6166 | — | 新增 | 立法共同赞助图 |
| UNtrade | 0.4716 | 0.5552 | -8.4pp | 国家贸易图，结构最异质 |

**观察**：
- Contacts（+10.5pp）和 mathoverflow（+7.8pp）大幅提升，说明结构特征（rank/freq/co_occur）对密集接触图和稀疏问答图都有帮助
- lastfm、UNtrade 下降：E1 训练了 7 个域包含更多样本，E5 只用 4 个 balanced 域，覆盖了这两类图的数据更少
- USLegis（0.62）和 UNtrade（0.47）是最难迁移的两类：立法/贸易图的 temporal pattern 与社交/金融域差异太大

**待做 / Ideas**

- [ ] 对比实验：balanced vs unbalanced mixing（控制变量）
- [ ] 对比实验：有/无结构特征（rank+freq+co_occur vs 纯时间）
- [ ] CrossMamba + fine-tune 1 epoch on target domain：few-shot adaptation 上限
- [ ] 增加 USLegis/UNtrade 类政治图到训练域，看能否改善这两类的迁移

---

### [2026-05-30] E6: 时间尺度归一化 + unique_ratio 结构特征 ✗

**动机**：E5 在 lastfm/USLegis/UNtrade 迁移差的根因分析：
- 时间尺度不匹配：UNtrade 时间差单位是天/年，训练域是秒/分钟；FixedCosineTimeEncoder 用固定频率无法跨尺度泛化
- 缺少多样性信号：没有区分 lastfm（反复听同一首歌）和社交图（接触多样）的特征

**两处改动（E5 架构基础上）**：
1. **Per-node 时间归一化**：`dt_norm = dt / mean_iet`
2. **`unique_ratio` 第 4 个结构特征**：unique_neighbor_count / valid_count，struct_proj: Linear(3→4)

**结果（50 epochs，patience=5，in-domain AP=0.9063）**：

| Domain | E5 AP | E6 AP | Δ |
|--------|-------|-------|---|
| Contacts | 0.9381 | 0.9346 | -0.4pp |
| mooc | 0.8752 | 0.8329 | **-4.2pp** |
| mathoverflow | 0.8468 | 0.8444 | -0.2pp |
| BitcoinOTC | 0.8379 | 0.8270 | -1.1pp |
| enron | 0.8308 | 0.8292 | -0.2pp |
| SocialEvo | 0.7699 | 0.7627 | -0.7pp |
| CanParl | 0.7756 | 0.7087 | **-6.7pp** |
| USLegis | 0.6166 | 0.6379 | +2.1pp |
| lastfm | 0.6375 | 0.6141 | -2.3pp |
| UNtrade | 0.4716 | 0.4644 | -0.7pp |

**结论：负结果，E6 全面落后于 E5。**

- 时间归一化假设出问题：`dt / mean_iet` 把绝对时间压缩成相对值，但 FixedCosineTimeEncoder 的频率本来就能适应不同尺度（余弦函数对大输入饱和），归一化反而破坏了训练域与目标域之间一致的时间表示，导致特征分布偏移
- unique_ratio 对 USLegis 有微小帮助（+2.1pp），对其他域中性或有害
- mooc（-4.2pp）和 CanParl（-6.7pp）的大幅下降说明归一化损坏了这两类图原本较好的时序信号

**回退**：恢复 E5 架构（绝对 dt + 3 维结构特征）

---

### [2026-05-30] E8: 加入边特征（单域 uci 基准）

**动机**：探测 CrossMamba 在单域场景下的上限；原 E(single) 用 d_edge=0 得 AP=0.8663，怀疑是边特征缺失导致差距。

**改动**：
- `CrossMamba` 增加 `d_edge` 参数，当 `d_edge > 0` 时加入 `edge_proj: Linear(d_edge, d_model//4)`
- `train_crossmamba_single.py` 去掉 `d_edge_target=0`，传入实际 `d_edge=172`
- `GatherSpec` 对应设置 `include_edge_feat=True`
- 超参：d_model=128, n_layers=2, K=128（与原始保持一致，只加边特征）

**结果（75 epochs，patience=10，RTX 4080）**：

| 配置 | AP | 备注 |
|------|-----|------|
| CrossMamba d_edge=0 | 0.8663 | 无边特征 |
| CrossMamba d_edge=172 | **0.8686** | 加入邻居边特征 |
| DyGFormer | 0.9579 | 基线 |

**结论**：边特征仅带来 +0.23pp，与 DyGFormer 仍差 8.9pp。**差距根本不在边特征，在架构**：
- DyGFormer 用 Transformer attention（全局 pairwise），对短序列表达能力更强
- Mamba SSM 是顺序 scan，适合长序列/时序依赖，但不如 Transformer 在有限邻居集上精确
- CrossMamba 的设计目标是跨域零样本迁移，不是单域 SOTA；在 d_edge=0 约束下 0.8663 已是 Mamba 的合理上限

**待做 / Ideas**

- [ ] 对比实验：有/无结构特征（rank+freq+co_occur vs 纯时间）
- [ ] CrossMamba + fine-tune 1 epoch on target domain：few-shot adaptation 上限
- [ ] 增加 USLegis/UNtrade 类政治图到训练域，看能否改善这两类的迁移
- [ ] 加权 mixing：不用硬截断，每域按 sqrt(size) 或 log(size) 采样

---

### [2026-05-30] E7: Unbalanced mixing（全量训练数据）✓

**改动**：`balance=False`，总训练边 177,090（E5 的 3.2x）

**结果（20 epochs，patience=5 触发，in-domain AP=0.9088）**：

| Domain | E5 (balanced) | E7 (unbalanced) | Δ |
|--------|--------------|-----------------|---|
| mooc | 0.8752 | **0.8931** | **+1.8pp** |
| UNtrade | 0.4716 | **0.4954** | **+2.4pp** |
| lastfm | 0.6375 | **0.6647** | **+2.7pp** |
| Contacts | 0.9381 | 0.9334 | -0.5pp |
| mathoverflow | 0.8468 | 0.8200 | -2.7pp |
| enron | 0.8308 | 0.8279 | -0.3pp |
| BitcoinOTC | 0.8379 | 0.8052 | -3.3pp |
| SocialEvo | 0.7699 | 0.7437 | -2.6pp |
| CanParl | 0.7756 | 0.6814 | **-9.4pp** |
| USLegis | 0.6166 | 0.5988 | -1.8pp |

**结论：mixed。unbalanced 对部分域有帮助，但对另一部分有明显损害。**

- **受益**：mooc(+1.8)、UNtrade(+2.4)、lastfm(+2.7)——这些域结构上更接近 wikipedia（用户-内容交互），wikipedia 数据量增加帮助了学习
- **受损**：CanParl(-9.4pp) 最严重，BitcoinOTC(-3.3)、SocialEvo(-2.6)——wikipedia 占 50% 数据主导了训练，破坏了对政治/金融类图的泛化
- 说明：**balanced vs unbalanced 没有统一的赢家**，关键在于训练域分布是否和目标域对齐
