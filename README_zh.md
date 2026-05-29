<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img alt="TGEngine" src="assets/logo.svg" width="420">
  </picture>
</p>

<p align="center">
  <strong>高性能连续时间动态图学习框架</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.0+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue" alt="English"></a>
</p>

<p align="center">
  <a href="#-安装">安装</a> &bull;
  <a href="#-快速开始">快速开始</a> &bull;
  <a href="#-架构">架构</a> &bull;
  <a href="#-内置模型">模型</a> &bull;
  <a href="#-性能基准">基准</a> &bull;
  <a href="docs/getting_started.md">文档</a>
</p>

---

## 为什么选择 TGEngine？

现有的 CTDG 框架要求你为每个新模型重写训练脚本，且数据管线受限于 CPU 瓶颈。TGEngine 同时解决这两个问题：

- **60 行添加新模型** &mdash; 声明 `GatherSpec`，实现 `forward()`，完成。
- **端到端训练快 1.3&ndash;1.6x**，数据管线快 **5&ndash;7x**（GPU 融合邻居采样）。
- **可插拔评估** &mdash; AP、AUC-ROC、MRR、Hits@K、三路负采样，开箱即用。
- **智能评估调度** &mdash; 基于 loss 变化自适应触发验证，大数据集节省高达 70% 评估时间。

## 安装

```bash
pip install tgengine
```

从源码安装（开发模式）：

```bash
git clone https://github.com/YOUR_USERNAME/tgengine.git
cd tgengine
pip install -e ".[dev]"
```

**依赖**: Python 3.10+，PyTorch 2.0+，推荐 CUDA GPU。

## 快速开始

### 10 行训练 DyGFormer

```python
from tgengine import (
    load_dataset, TemporalGraph, Engine, TrainConfig,
    APEval, RandomNegative, DyGFormer,
)

dataset = load_dataset("wikipedia", dataset_path="datasets")
graph = TemporalGraph(dataset.num_nodes, buffer_size=32,
                      edge_feat_dim=dataset.edge_feat_dim, device="cuda")

model = DyGFormer(d_edge=172, d_time=100, K=32, num_layers=2, num_heads=2)
engine = Engine(
    model, graph,
    train_batches=dataset.get_batches("train", 200),
    val_batches=dataset.get_batches("val", 200),
    test_batches=dataset.get_batches("test", 200),
    neg_strategy=RandomNegative(dataset.num_nodes),
    eval_protocol=APEval(),
    config=TrainConfig(epochs=100, lr=1e-4, device="cuda"),
)
results = engine.train()  # {"ap": 0.990}
```

### 添加新模型（约 60 行）

```python
from tgengine import TemporalModel, GatherSpec, NeighborSpec, PreparedBatch, ModelOutput
from tgengine.nn import Time2Vec, TransformerSeqEncoder, ConcatDecoder

class MyModel(TemporalModel):
    gather_spec = GatherSpec(neighbors=NeighborSpec(k=20))

    def __init__(self):
        super().__init__()
        self.time_enc = Time2Vec(d_model=172)
        self.encoder = TransformerSeqEncoder(d_in=172, n_layers=2, n_heads=2)
        self.decoder = ConcatDecoder(d_in=172)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self.encoder(
            self.time_enc(batch.src_neighbors), batch.src_neighbors.mask
        )
        dst_emb = self.encoder(
            self.time_enc(batch.dst_neighbors), batch.dst_neighbors.mask
        )
        neg_emb = self.encoder(
            self.time_enc(batch.neg_neighbors), batch.neg_neighbors.mask
        )
        return self.decoder(src_emb, dst_emb, neg_emb)
```

无需修改训练脚本、数据管线代码。Engine 自动处理一切。

### 多次实验（报告 mean +- std）

```python
from tgengine.engine import run_experiment

def build(seed):
    # ... 使用该 seed 创建模型、图、引擎 ...
    return engine

results = run_experiment(build, seeds=[1, 2, 3, 4, 5], result_dir="results/")
# ap: 0.9908 +- 0.0012
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户代码                              │
│   model = MyModel()     # 只需定义 gather_spec + forward    │
│   engine = Engine(...)  # 一行启动训练                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  第4层：模型                                                 │
│  DyGFormer │ TGN │ GraphMixer │ DyGMamba │ FreeDyG │ ...   │
├─────────────────────────────────────────────────────────────┤
│  第3层：神经网络组件  (tgengine.nn)                           │
│  序列编码器 │ 时间编码器 │ 解码器 │ 节点记忆 │ 共现编码器      │
├─────────────────────────────────────────────────────────────┤
│  第2层：数据管线                                              │
│  GatherSpec → DataPipeline → PreparedBatch (GPU 融合操作)    │
├─────────────────────────────────────────────────────────────┤
│  第1层：训练引擎 + 评估                                       │
│  训练循环 │ 自适应评估 │ AP/AUC/MRR/Hits │ JSON 结果输出     │
├─────────────────────────────────────────────────────────────┤
│  底层存储：TemporalGraph (GPU 常驻 CSR + 环形缓冲区)         │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计：GatherSpec + PreparedBatch

模型永远不直接操作图数据，而是：

1. **声明**所需数据（`GatherSpec`：邻居数量、特征维度、共现信息）
2. **接收**预处理好的 `PreparedBatch`（GPU 就绪）
3. **计算**纯神经网络前向传播

这种分离使得管线可以将所有图操作融合为一次 GPU kernel 调用，与具体模型无关。

## 内置模型

| 模型 | 论文 | 代码行数 | 核心创新 |
|------|------|:---:|----------|
| **DyGFormer** | NeurIPS 2023 | ~380 | 分段邻居注意力 |
| **FreeDyG** | AAAI 2024 | ~240 | 频域编码 |
| **GraphMixer** | ICLR 2023 | ~180 | MLP-Mixer 处理时序序列 |
| **TGN** | ICML 2020 | ~120 | 记忆模块 + GRU 消息传递 |
| **DyGMamba** | 2024 | ~60 | Mamba SSM 时序编码 |
| **EdgeBank** | &mdash; | ~20 | 启发式基线（无需学习） |

## 评估体系

| 协议 | 指标 | 适用场景 |
|------|------|---------|
| `APEval` | Average Precision | 标准 1v1 二分类 |
| `APEval(include_auc=True)` | AP + AUC-ROC | DyGLib 兼容双指标 |
| `AUCEval` | AUC-ROC | 独立 ROC 评估 |
| `ThreeWayEval` | AP (随机 / 历史 / 归纳) | 细粒度负采样分析 |
| `MRREval` | Mean Reciprocal Rank | TGB 风格排序评估 |
| `HitsEval` | Hits@1/3/10 | Top-K 排序质量 |

## 性能基准

### 端到端训练速度（对比 DyGLib）

| 数据集 | K | TGEngine | DyGLib | 加速比 |
|--------|---|----------|--------|--------|
| Wikipedia | 32 | 17.3s | 25.4s | **1.47x** |
| Reddit | 64 | 86.2s | 134.8s | **1.56x** |
| LastFM | 512 | 153.2s | 205.8s | **1.34x** |

> 每 epoch 时间，RTX 4080，DyGFormer 架构，完全相同超参数。

### 精度对齐

| 模型 | 数据集 | TGEngine AP | DyGLib AP | 差距 |
|------|--------|:-----------:|:---------:|:----:|
| DyGFormer | Wikipedia | 0.9908 | 0.9903 | +0.05% |
| DyGFormer | UCI | 0.9526 | 0.9613 | -0.9% |
| GraphMixer | UCI | 0.9315 | 0.9331 | -0.2% |

## 许可证

MIT License. 详见 [LICENSE](LICENSE)。
