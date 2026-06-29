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
