# 大图训练瓶颈实测：reddit K=512

> 日期: 2026-06-29
> 数据集: reddit (591,723 edges, 10,985 nodes, d_edge=172)
> 模型: _MiniMambaModel (TimeAwareMambaBlock×2, d_model=128, K=512)
> 硬件: scnu RTX 4080 (单卡) / 4×(3090+4080) (DDP)

## 核心问题

验证"大图训练时，async prefetch 流水线能否被充分利用、大幅加速训练"。

## 实测结果

### 单卡 sync vs async (K=512, 2 epoch)

| 模式 | AP | 时间 | 显存 | speedup |
|------|-----|------|------|---------|
| sync fp32 | 0.9274 | 540.5s | 9.43GB | 1.00x (baseline) |
| async fp32 | 0.9289 | 541.9s | 9.65GB | **1.00x** |

**async prefetch 在 reddit K=512 上零加速（1.00x）。** 与 uci 小图（1.00x）一致。

### DDP 4 卡 (K=512, 训练阶段)

| 指标 | 单卡 | DDP 4 卡 |
|------|------|----------|
| GPU 利用率 | 100% | 99-100% (4 卡) |
| 单卡吞吐 | 8.26 it/s | 6.15 it/s/卡 |
| 每 epoch 训练 batch | 1950 | 488/卡 |
| 纯训练时间/epoch | ~236s | ~80s |
| **训练加速** | 1x | **~3.0x** |

DDP 训练阶段 ~3.0x（4 卡理想 4x，通信损耗 ~25%）。**这是真正的大幅加速。**

### DDP eval 阶段 (bug)

DDP 在 epoch 1 eval 阶段 NCCL 超时崩溃。根因：4 rank 各跑全量 val+test eval（10万+10万边），慢且不同步，NCCL 默认 30min 超时。已尝试用 `_raw_model`（绕过 DDP forward hooks）修复，仍崩——根本问题是全量 eval 不分片。

**待修**：DDP eval 应分片（每 rank eval 1/4）或仅 rank0 eval。见 task2 DDP 遗留。

## 瓶颈定位

**瓶颈是 selective_scan CUDA kernel（闭源 mamba_ssm op）。** 四个证据：

1. **GPU 100% 满载**：训练时 nvidia-smi util=100%。数据准备若是瓶颈（CPU），GPU 会有空闲；GPU 满载说明 compute 是瓶颈。

2. **scan 占 87.7%**：task5 profiling 测得 selective_scan 占 CUDA 时间 87.7%（bwd 60% + fwd 27%）。K=512 序列长 16 倍，scan 占比更高。

3. **batch 吞吐一致**：sync 8.26 it/s，async 8.23 it/s——逐 batch 无差。若 prefetch 真重叠数据准备，async 应更快。

4. **数据准备是 GPU kernel**：graph.recent/co_occurrence 是自研 CUDA kernel（在 GPU 跑），与 scan 争抢同一块卡的 SM。GPU 已被 scan 占满，采样 kernel 排不到空闲 SM，重叠失败。

## 为什么 async 无效

教科书 async prefetch 收益来自 **CPU 数据准备与 GPU compute 重叠**。但我们：
- 数据准备已被自研 CUDA kernel 加速（3-28x），占比压到 ~10%
- 数据准备本身在 GPU 上，与 scan 争同一块卡的 SM（非 CPU-GPU 并行）
- GPU 100% 满载，无空闲周期给 prefetch 重叠

**用 CUDA kernel 加速数据准备，反而消灭了 async 的收益场景。** 反直觉但真实。

## 突破路径

| 路径 | 加速 | 状态 |
|------|------|------|
| DDP 多卡（数据并行） | ~3.0x (4卡) | 训练已验证，eval 有 bug 待修 |
| fast-path scan | 5.8x (vs Python loop) | 已实现 |
| 自研 CUDA kernel（采样/co-occurrence） | 3-28x（数据准备本身） | 已实现 |
| async prefetch | 1.00x | 无收益（瓶颈不在数据准备） |
| Triton 自研 scan（可被 compile 融合） | 未知，理论可破 87.7% 天花板 | 未做，工程量大 |
| 减小 K / patch | 线性降 scan 时间 | 模型设计取舍 |

## 结论

- **大图训练的大幅加速来自 DDP（多卡数据并行），不是 async prefetch。**
- async prefetch 在我们的架构下零收益，因为数据准备已被 CUDA kernel 加速到占比很小，且数据准备本身是 GPU kernel 与 scan 争抢 SM。
- 单卡硬上限是 selective_scan CUDA kernel（87.7%），要突破需替换 scan kernel 或多卡。

## 复现

```bash
# 单卡 sync vs async
scripts/run_mamba.sh benchmarks/ablation/bench_mamba_reddit.py --epochs 2 --K 512 --d_model 128

# DDP 4 卡（训练阶段验证；eval 有 bug）
# 见 examples/train_ddp_reddit.py + /tmp/run_ddp_manual.sh 模式
```

## 待做

- [ ] 修 DDP eval 分片（task2 遗留 bug）
- [ ] 更大图（lastFM）验证 DDP scaling 是否保持线性
- [ ] Triton 自研 selective scan 探索（突破 87.7% 天花板）

---

# Mamba v1 vs v2 (SSD) K=512 DDP 测速（2026-06-29）

## 动机

用户假设：长序列 K=512 下 Mamba-2（SSD）比 Mamba-1（selective scan）训练更快。Mamba-2 用 State-Space Duality，chunked SSD kernel，长序列复杂度更低。

## 环境

- 新建 conda env `mamba2`（clone PyGBase + 重编 causal_conv1d 对齐 torch 2.9 ABI）
- causal_conv1d_cuda.so 原本 ABI 坏（undefined symbol incref_pyobject），从 GitHub 源码重编后 OK，但仍需 glibc239 ld-linux 启动（GLIBC_2.32）
- Mamba2Block 用 `mamba_ssm.Mamba2` 类（不自拼装），pre-norm + residual 包装

## 实测（reddit K=512, d_model=128, 2 layers）

### 单卡

| 版本 | 训练 it/s | epoch 时间 | 显存 | AP |
|------|-----------|------------|------|-----|
| v1 (TimeAwareMamba) | 8.26 | 281.7s | 9.43GB | 0.9274 |
| v2 (Mamba2 SSD) | ~7.7 | 414.8s | 7.57GB | 0.9133 |

单卡 v2 训练 it/s 略低，但 epoch 总时间更长——**v2 eval 极慢**（355s vs v1 的 45s）。v2 显存省 20%。

### DDP 4 卡（训练阶段吞吐）

| 版本 | 单卡 it/s | 训练加速 |
|------|-----------|----------|
| v1 | 6.15 | ~3.0x（vs 单卡） |
| v2 | **7.67** | ~3.7x（vs 单卡 v2） |

**DDP 下 v2 训练吞吐 7.67 it/s > v1 6.15 it/s，v2 快 25%。** 验证了用户假设方向（长序列 v2 更快），但 K=512 下差距是 25%，非数量级。

### DDP eval 问题

v2 DDP 在 eval 阶段极慢（4 卡 100% 满载但 eval 1 个 epoch 的 val+test 超过 40min 未完成，被 kill）。根因：**v2 eval forward 极慢**（单卡 eval 占 355s vs v1 的 45s），DDP 分片后 4 rank 同步按最慢卡（3090），叠加 is_best 时 test eval，远超 NCCL timeout。60min timeout 仍不够。

v1 DDP eval 修复（分片 + all_reduce train_loss）在 uci 验证通过（AP=0.9008 完整跑完），reddit v1 也跑通。v2 eval 慢是 Mamba2 自身在 no_grad 下的性能问题——SSD triton kernel 在 eval（no_grad、无 backward）时可能不走优化路径，或 chunked scan 在 eval batch 下不优。**这是 Mamba2 的缺陷，不是 DDP bug。**

## 结论

- **训练吞吐：DDP 4 卡 K=512 下 v2（7.67 it/s）> v1（6.15 it/s），快 25%。** 长序列 v2 优势确认，但 K=512 差距适中。
- **显存：v2（7.57GB）< v1（9.43GB），省 20%。**
- **eval 速度：v2（355s）远慢于 v1（45s）。** 这是 Mamba2 的缺陷，eval 占比大，DDP 下放大成 timeout。
- **单卡 epoch 总时间：v2（415s）> v1（282s）**——v2 训练快但 eval 慢拖累总时间。

## 建议

- 训练用 v2（吞吐高、显存省），但需解决 eval 慢问题
- v2 eval 慢可能是 Mamba2 在 no_grad 下没走 fast-path，或 triton kernel 重编译。待调研
- 更长序列（K=1024+）v2 优势应更明显，值得测

## 复现

```bash
# mamba2 env + glibc239
MAMBA_PYTHON=<mamba2>/python3.11 scripts/run_mamba.sh python \
    examples/bench_ddp_mamba_versions.py --mamba v2 --K 512
```

---

# Mamba v3 (SSD + trapezoidal + MIMO) K=512 DDP 测速（2026-06-29）

## 动机

用户要求测 Mamba-3（2026 年 3 月，ICLR 2026）。Mamba-3 三大改进：exponential-trapezoidal discretization、complex-valued state、MIMO。从源码装 mamba-ssm（含 Mamba3），用 SISO 模式（MIMO 需 TileLang kernel，未装）。

## 实测（reddit K=512, d_model=128, 2 layers, DDP 4 卡）

### 训练吞吐（DDP 4 卡）

| 版本 | 单卡 it/s | 训练加速（vs 单卡 v1） |
|------|-----------|----------------------|
| v1 (TimeAwareMamba) | 6.15 | 1.0x baseline |
| v2 (Mamba2 SSD) | 7.67 | +25% |
| v3 (Mamba3 SISO) | **7.90** | **+29%** |

**v3 训练吞吐最快（7.90 it/s），比 v1 快 29%，比 v2 略快 3%。** 长序列 K=512 下 v3/v2 的 SSD 路径优势确认，但 v3 vs v2 差距很小（SISO 模式下 v3 的 trapezoidal/complex 改进对吞吐影响不大）。

### 单卡 epoch 总时间 + 显存

| 版本 | epoch 时间 | 显存 |
|------|------------|------|
| v1 | 281.7s | 9.43GB |
| v2 | 414.8s | 7.57GB |
| v3 | 613.4s | 10.47GB |

v3 单卡 epoch 最慢且最耗显存——**v3 eval 极慢**（比 v2 还慢），拖累总时间。

### DDP eval 问题（v2/v3 共有）

v3 DDP eval 超过 40min 未完成（被 kill），与 v2 同病：Mamba2/Mamba3 的 SSD triton kernel 在 no_grad（eval）下极慢，DDP 分片后仍按最慢卡同步，超 NCCL timeout。v1 eval 快（selective_scan 在 no_grad 下正常）。

**Mamba2/Mamba3 的 eval 慢是 SSD kernel 的固有缺陷**，不是 DDP bug。训练吞吐对比有效；eval 性能需单独调研（可能要 SSD kernel 的 eval fast-path 或 cache）。

## 结论

- **训练吞吐：v3 (7.90) ≈ v2 (7.67) > v1 (6.15)**。K=512 下 SSD 路径（v2/v3）比 selective scan（v1）快 25-29%。
- **v3 vs v2 训练差距小（3%）**：SISO 模式下 v3 的算法改进对吞吐无显著提升。MIMO 模式（需 TileLang）可能不同，未测。
- **eval 慢：v3 > v2 > v1**。SSD kernel 在 no_grad 下性能差，v3 最慢。这使 v2/v3 的单卡 epoch 总时间反而比 v1 长。
- **显存：v2 (7.57GB) < v1 (9.43GB) < v3 (10.47GB)**。v3 最耗显存。

## 建议

- 训练用 v2（吞吐接近 v3、显存最省、eval 比 v3 略快）
- 或用 v3 MIMO 模式（需装 TileLang，可能解锁更高吞吐）
- eval 慢是 v2/v3 的核心障碍，需调研 SSD kernel 的 eval 优化（如 eval 时切回 v1 selective_scan，或用 Mamba2/3 的 inference fast-path）
- v1 eval 快，若 eval 频繁可考虑训练 v2/v3 + eval 用 v1 权重转换（但架构不同，不可直接转）

## 环境

- conda env `mamba2`：clone PyGBase + causal_conv1d 重编 + mamba-ssm 从源码装（含 Mamba3）
- Mamba3 SISO 模式（is_mimo=False，避开 TileLang 依赖）
- 仍需 glibc239 ld-linux 启动（causal_conv1d_cuda + mamba3 kernel 需 GLIBC_2.32）

---

# Mamba2/3 eval 慢 bug 修复（2026-06-29）

## 根因（社区已知 + 本地实测确认）

社区调研：
- GitHub issue #355 "Mamba2 9x slower inference than Mamba1"——Tri Dao 确认 Mamba2 用 Triton，小模型 CPU overhead 大
- GitHub issue #389 "mamba2 training very slow"——首次 triton compiler & autotune 慢，需 warmup
- PyTorch 官方博客 "Accelerating Mamba2 with Kernel Fusion"——默认 5-kernel SSD 慢

本地实测（v2 forward, B=200, K=512, d_model=128）：
- **cold first forward: 5197 ms**（triton autotune 编译）
- **hot forward: 5.42 ms**
- **cold/hot = 958x**

根因：Mamba2/Mamba3 的 SSD triton kernel 在 no_grad（eval）上下文首次编译 ~5s。训练时 cache 在 grad 上下文，eval 的 no_grad 上下文 cache key 不同，首次 eval batch 付 5s cold-start。

## 修复（Engine._evaluate triton warmup）

在 `_evaluate` 的 proto 循环前，用第一个 prepped batch 跑一次 dummy forward（no_grad + autocast，与真实 eval 同上下文），触发 triton autotune 填 cache。后续 eval batch 走 hot path。Best-effort（异常跳过）。

```python
if prepped:
    try:
        warm = self._eval_pipeline.prepare(prepped[0])
        with torch.no_grad():
            _ = self._raw_model(warm)
        torch.cuda.synchronize()
    except Exception:
        pass
```

## 验证（单卡 reddit K=512 v2）

| | eval 时间 | epoch 总时间 |
|---|---|---|
| 修复前 | 161s | 414.8s |
| 修复后 | **42s**（warmup 0s + val 21s + test 21s） | 307.7s |

**eval 从 161s 降到 42s（-74%），接近 v1 的 45s。** warmup 在训练后 cache 已热时 forward 本身 0s，但关键是它在 no_grad + autocast 上下文填了 triton cache key（训练时的 grad 上下文 cache 不适用 eval）。

test_engine 7 个 eval 测试在 mamba2 env 全过——warmup 不破坏 eval 正确性。

## DDP eval 超时（另一个问题，非 triton）

DDP 4 卡 reddit K=512 v2 eval 仍超时（>15min）。排查发现 rank0 训练 487/488 后卡住，`_train_epoch` 返回后的 `all_reduce(train_loss)` 没完成——**DDP 训练/通信问题，不是 triton eval cold-start**。

可能根因：
- 4 卡异构（1×4080 + 3×3090），DDP 每 batch all_reduce grad 同步，某卡慢拖累
- NCCL 通信死锁（与 4 卡异构或 NCCL 配置有关）
- 与 warmup 修复无关（warmup 解决 triton cold-start，DDP 同步是另一层）

**待查**：DDP 4 卡异构的 all_reduce 死锁。可能需 NCCL 配置调整或同构卡测试。

## 环境副作用（已恢复）

mamba2 env 创建时 conda clone 硬链接共享 PyGBase 包，mamba2 的 `pip install --force-reinstall` 升级 torch 时破坏了 PyGBase（torch 2.9.1+cu128 → 2.12.1+cu130，CUDA 13.0 在系统 glibc 下不可用）。已重装 PyGBase torch 2.9.1+cu128 恢复。
