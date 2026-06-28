# Task 5: Mamba fwd/bwd 优化调研

> 调研日期: 2026-06-29
> 状态: 已完成调研 + 验证

## 背景

任务5 要求"先调研再开始做"。backbone 大概率用 Mamba。任务4 已把
`MambaBlock` / `TimeAwareMambaBlock` 模块化进 `tgengine.nn`，走 mamba_ssm
的 `selective_scan_fn` CUDA fast-path（memory `project_mamba_fastpath_glibc.md`
记录了 fast-path 修复：77s/epoch → 13.2s/epoch）。

本调研回答：**fast-path 之上，fwd/bwd 还有多少优化空间，在哪？**

## Profiling 方法

`benchmarks/ablation/bench_mamba_fwdbwd.py`，配置 d_model=256, K=32, B=200,
n_layers=3（接近 DyGFormerMamba uci 实配），用 PyTorch profiler 测 per-op
CUDA 时间 + 显存。scnu RTX 4080, glibc 2.39 fast-path。

## 核心发现：scan kernel 占 87.7%

| op | CUDA 占比 |
|---|---|
| SelectiveScanFnBackward | **60.2%** |
| SelectiveScanFn (fwd) | **27.5%** |
| aten::mm (projections) | 3.3% |
| aten::copy_ (contiguous 转置) | 3.0% |
| conv1d | 2.3% |
| elementwise | 2.1% |

**fwd+bwd 总时间 0.040s（AMP）/ 0.046s（fp32），其中 scan kernel 独占 87.7%。**
projection、copy、conv 加起来不到 11%。

结论：**fwd/bwd 已被 selective_scan CUDA kernel 主导。** 这是 mamba_ssm 库
的内部 kernel，我们无法修改。projection 层的 micro-optimization（去 contiguous、
合并 linear+transpose）理论上限是 ~11%，实际收益 < 3%。

## 各优化方向可行性

### 1. AMP（已验证，部分有效）
- 整体 fwd+bwd：fp32 0.0462s → AMP 0.0402s（**13% 加速**）
- 显存：0.66GB → 0.45GB（**32% 节省**）
- 但 scan kernel 内部仍是 fp32（selective_scan_fn 的数值稳定性要求），
  AMP 只加速了 projection 的 mm。所以 13% 是上限附近。
- **Engine `use_amp` 默认 False**（保守，数值稳定）。Mamba 训练建议显式开。

### 2. torch.compile（不可行 ❌）
- inductor 后端依赖 triton JIT，运行时 shell 调 `/usr/bin/gcc` 编译 cuda_utils。
- glibc239 启动方案下，系统 gcc 无法加载 glibc239 的 libc.so.6（需 GLIBC_2.35），
  inductor 运行时编译崩溃。
- `tests/conftest.py` 的 env 清理只在 collection 阶段生效，inductor 在 forward
  时重新 JIT，再次撞 gcc 崩溃。
- **结论：fast-path 进程下 torch.compile 路线关闭。** 要用 compile 需解决
  glibc239 + 系统 gcc 的根本冲突（换 conda gcc 或静态链接 inductor runtime），
  ROI 低，不做。

### 3. 减少 contiguous 转置（收益 < 3%）
- `_SelectiveSSM.forward` 有 4 处 `.contiguous()`，是 selective_scan_fn 对
  (B, d, L) 布局的硬性要求，无法绕过。
- `aten::copy_` 仅占 3% CUDA，优化空间可忽略。

### 4. scan kernel 本身（不可改）
- selective_scan_cuda.so 是 mamba_ssm 预编译的闭源 CUDA kernel。
- 替换为自研 kernel（如 Triton scan）是独立大工程，超出 task5 范围。

### 5. async prefetch 与 Mamba 的组合（任务1×任务5 交汇，待验证）
- 任务1 把 prefetch 重排到 compute 之前，理论上让数据准备与 Mamba fwd/bwd 重叠。
- profiling 显示 scan kernel 是 fwd/bwd 主体，数据准备（graph.recent）与之重叠
  能隐藏采样延迟。需 e2e 验证 TimeAwareMamba + async + AMP 组合正确且提速。

## 决策

**fwd/bwd 在 fast-path 之上已无显著优化空间**（scan kernel 87.7% 是硬上限）。
task5 的交付为：

1. 本调研文档（记录瓶颈与各方向结论，避免重复踩坑）。
2. AMP 默认建议：Mamba 训练脚本显式 `use_amp=True`（13% 速度 + 32% 显存）。
3. e2e 验证：TimeAwareMamba + AMP + async prefetch 组合正确提速（task1×task5 交汇）。
4. profiling benchmark `bench_mamba_fwdbwd.py` 留作回归工具。

**不做**：torch.compile（glibc239 冲突）、自研 scan kernel、projection micro-opt（< 3%）。
这些要么不可行，要么 ROI 不足以匹配工程成本。

## 相关

- memory `project_mamba_fastpath_glibc.md` — fast-path + glibc 2.39 修复
- `tgengine/nn/mamba_block.py` — MambaBlock / TimeAwareMambaBlock
- `benchmarks/ablation/bench_mamba_fwdbwd.py` — 本调研的 profiling 脚本
