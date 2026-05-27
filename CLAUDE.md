# TGEngine — High-Performance CTDG Learning Framework

## 项目定位

为连续时间动态图（CTDG）链接预测设计的高性能学习框架。兼顾研究者快速试新模型和 production 级训练性能。

## 架构核心：四层

```
Layer 4: Models      — 开箱即用完整模型 (DyGFormer, TGN, DyGMamba, ...)
Layer 3: Components  — 预置积木 (SequenceEncoder, TimeEncoder, Decoder, NegStrategy)
Layer 2: DataPipeline— 全局优化数据准备 (GatherSpec → fused PreparedBatch)
Layer 1: Engine      — 训练/评估生命周期管理 (train loop, eval protocol, snapshot/restore)
```

## 核心设计决策

**GatherSpec + PreparedBatch 模式**：
- 模型通过 `gather_spec`（类属性，静态）声明需要什么数据
- DataPipeline 读取 spec，将所有 graph 操作合并为**单次 fused kernel call**
- 模型 forward() 只接收 `PreparedBatch`，只做神经网络计算
- 模型**不能**在 forward 中直接访问 TemporalGraph

**取舍**：牺牲 forward 中的条件数据获取灵活性，换取全局 pipeline 优化（batch fusion + async prefetch）。95% 模型的需求（邻居序列 + 时间 + 共现）完全覆盖。

## 核心类型

| 类型 | 位置 | 职责 |
|------|------|------|
| `TemporalGraph` | core/temporal_graph.py | GPU-resident circular buffer，存储 + 快速邻居查询 |
| `GatherSpec` | core/gather_spec.py | 模型的数据需求静态声明 |
| `PreparedBatch` | core/batch.py | 全部数据已就绪的 model input |
| `TemporalModel` | models/base.py | 模型基类：gather_spec + forward() |
| `SequenceEncoder` | nn/seq_encoder.py | (B,K,d)→(B,d) 序列编码器，**创新热点** |
| `DataPipeline` | pipeline/__init__.py | GatherSpec → PreparedBatch 的优化执行器 |
| `Engine` | engine/__init__.py | 训练循环 + 评估 + 状态管理 |

## 数据流

```
Dataset (csv/npy) → TemporalGraph (GPU circular buffer)
  → Engine 按时间迭代 → RawBatch
    → DataPipeline.prepare() (1 kernel, all nodes fused)
      → PreparedBatch → Model.forward() → ModelOutput
        → backward → Graph.advance() → next batch
```

## 性能策略

- **V1（当前）**: Vectorized PyTorch（torch.gather/scatter），目标 5-8x vs DyGLib
- **V2（未来）**: Custom CUDA kernels for neighbor sampling，目标 10-20x
- **Async prefetch**: 计算 batch_i 的同时准备 batch_i+1 数据

## 评估支持

- `APEval`: 标准 Average Precision (1 pos + 1 neg)
- `ThreeWayEval`: random + historical + inductive 三路
- `MRREval`: TGB 固定负样本列表 + ranking
- MRR 优化：批内负样本去重，只 encode unique nodes

## 有状态模型 (TGN-style)

实现 `evolve()` / `freeze()` / `thaw()` 三个方法。Engine 自动在正确时机调用。无状态模型（DyGFormer, Mamba）留空即可。

## MRR 评估兼容

模型可选实现 `supports_independent_encode`、`encode_nodes()`、`score_pairs()`。对于 co-neighbor 类模型（src 表示依赖 dst），MRR eval 退化为逐 pair 计算。

## 目录结构

```
tgengine/
├── core/           — TemporalGraph, Batch, GatherSpec, Dataset
├── pipeline/       — DataPipeline, NegativeStrategies
├── nn/             — TimeEncoder, SeqEncoder, Decoder, Memory, CoNeighbor
├── models/         — TemporalModel base + complete models
├── engine/         — Engine, EvalProtocol, Trainer
└── utils/          — seed, logging, config
```

## 加新模型的标准流程

1. 继承 `TemporalModel`
2. 设置 `gather_spec = GatherSpec(neighbors=NeighborSpec(k=32, ...))`
3. 实现 `forward(self, batch: PreparedBatch) -> ModelOutput`
4. （可选）实现 `encode_nodes` + `score_pairs` 支持 MRR eval

## 与竞品对比

| | DyGLib | TGM | TGEngine |
|---|---|---|---|
| 数据优化 | 无(CPU Python) | Hook pipeline 合并 | GatherSpec + fused pipeline |
| 加新模型 | 改 500 行训练脚本 | 写 encoder+hook+example | 写 1 个 Model 类 (~60行) |
| 评估 | AP only | TGB MRR | AP + 3way + MRR (pluggable) |
| 目标性能 | 1x | 5-8x | 8-15x |

## 相关资源

- 参考 TGM 代码：`/Users/xg/Coding/PersonalFile/Claude_DyG/code-refs/tgm/`
- 参考 DyGLib 代码：`/Users/xg/Coding/PersonalFile/Claude_DyG/code-refs/exp_sourcecode/`
- 研究项目上下文：`/Users/xg/Coding/PersonalFile/Claude_DyG/CLAUDE.md`

## GPU 使用约束（重要）

服务器 `scnu` 有 4 张 GPU，编号如下：

```
nvidia-smi GPU 0: RTX 3090 (25.3 GB) ← 同学使用，禁止占用
nvidia-smi GPU 1: RTX 4080 (16.7 GB) ← 我们专用
nvidia-smi GPU 2: RTX 3090 (25.3 GB) ← 同学使用，禁止占用
nvidia-smi GPU 3: RTX 3090 (25.3 GB) ← 同学使用，禁止占用
```

⚠️ **注意：nvidia-smi 编号 ≠ CUDA device 编号（取决于 CUDA_DEVICE_ORDER）**

**不设置 `CUDA_DEVICE_ORDER`（默认，当前服务器状态）：**
- CUDA device 0 = nvidia-smi GPU 1 = RTX 4080 ← **唯一可用**
- CUDA device 1/2/3 = nvidia-smi GPU 0/2/3 = RTX 3090 ← **禁止使用**

**若设置了 `CUDA_DEVICE_ORDER=PCI_BUS_ID`（匹配 nvidia-smi）：**
- CUDA device 0 = nvidia-smi GPU 0 = RTX 3090 ← **禁止！**
- CUDA device 1 = nvidia-smi GPU 1 = RTX 4080 ← 此时需改用 `CUDA_VISIBLE_DEVICES=1`

**所有在 scnu 上运行的脚本，不得设置 `CUDA_DEVICE_ORDER`，且必须设置 `CUDA_VISIBLE_DEVICES=0`**：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/xxx.py
CUDA_VISIBLE_DEVICES=0 conda run -n PyGBase python -m pytest tests/ -v
```

RTX 4080 显存 16.7 GB；Reddit ring buffer 在 K=32 时约 14.8 GB，需自动降到 K≤17。

## 开发规范

- Commit message: English
- 对话/注释: 中文解释可以，代码注释用英文
- 优先 vectorized PyTorch 操作，禁止 hot path 里出现 Python for loop
- 新增模块必须有对应 test
- 不引入 Lightning / Hydra 等重依赖

## 性能优化的正确性原则

**做性能优化时，必须保证 input→output 语义不变，否则不是优化，是换了个算法。**

教训来源（2026-05）：实现 `HistoricalNegative` 时，为了利用已有的 k=32 ring buffer
直接从最近 32 个邻居里采负样本，比 DyGLib/TGM 快数倍。但 DyGLib/TGM 的语义是
"从 src **所有历史**交互中随机采样"，我们的语义是"从 src **最近 k 个**中采"。
两者 given same input，output distribution 根本不同。benchmark 因此无效——
我们做了更少的工作，拿更快的速度对比做更多工作的基线，结论没有意义。

**规则：**
1. 优化前，先写出目标算法的精确 input/output 规范（用例子说明）。
2. 优化后，验证给定相同 input，优化版与参考版产出相同 output（允许随机性则验证分布等价）。
3. 若做了近似（bounded pool 代替 full history），必须明确命名区分（如 `RecentHistoricalNegative`），
   并在 benchmark 注释里说明与基线的语义差异。
4. 不允许以"近似更快"为由在 benchmark 中直接对比语义不同的算法。

## 测试环境

代码测试在远程服务器 `scnu` 上执行：

```bash
# 同步本地改动到远程
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='*.egg-info' \
    /Users/xg/Coding/PersonalFile/tgengine/ scnu:~/CodeBase/Graph/tgengine/

# 在远程运行测试（conda 环境 PyGBase，PyTorch 2.10+cu128，有 GPU）
ssh scnu 'bash -l -c "cd ~/CodeBase/Graph/tgengine && conda run -n PyGBase python -m pytest tests/ -v 2>&1"'
```

- 远程项目路径：`~/CodeBase/Graph/tgengine/`
- 数据集路径：`/mnt/home/gyq/CodeBase/Graph/DG_Data`
- conda 环境：`PyGBase`（PyTorch 2.10.0+cu128，CUDA 可用）
- 首次部署需 `conda run -n PyGBase pip install -e .`
