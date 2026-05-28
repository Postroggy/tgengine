# DyGFormer 实现对比分析

## 目标

对照 DyGLib 原始 `models/DyGFormer.py`，逐条核查 TGEngine 实现的正确性。

---

## 意外发现 (LastFM 高 val AP)

**现象**：在某次实验（DyGLib-style 修改但尚不完整的版本）中，LastFM 在 epoch 4 取得了
`best val AP = 0.9437`，远高于 DyGLib 参考值 0.773。

**背景**：
- LastFM 的静态节点特征 `ml_lastfm.npy` 全为零 → 加 4th channel / self-token 无效
- 这个高分来自三个频道（edge / time / co-occurrence）本身就能很好地区分 LastFM 的链接
- 也可能是评估方式的差异（AP 计算方式、split 方式）

**意义**：说明对于边特征丰富、节点特征全零的数据集，TGEngine 3-channel 版本已能超越 DyGLib 参考。
节点特征仅对 LIWC 特征非零的数据集（如 Wikipedia）有用。

---

## DyGLib 模型结构精读

### 1. 序列构建 (`pad_sequences`)

DyGLib 构建序列的方式：

```
position 0: node_id = query_node,  edge_id = 0,  time = query_time  ← SELF token
position 1..K: recent K neighbors (oldest to newest? or newest to oldest?)
```

关键：**self 永远在 position 0**，且包含在 co-occurrence 计算中。

序列长度 = `min(num_neighbors, max_input_seq_len - 1) + 1`（自适应，逐 batch 取最大值），
再 round up 为 patch_size 的倍数。

### 2. Co-occurrence 含义

DyGLib `count_nodes_appearances` 对 src 序列中的每个 position j 计算：
```
[count(src_list[j] in src_list), count(src_list[j] in dst_list)]
```

DyGLib 对 dst 序列中的每个 position j 计算：
```
[count(dst_list[j] in src_list), count(dst_list[j] in dst_list)]
```

注意：self token 也在序列里，所以 `src_list[0] = src_id`，
自身计数 ≥ 1（如果 src 在自己历史里出现过多次则 > 1）。

### 3. TGEngine Co-occurrence Bug（已发现）

我们的 `_CoOccurrenceEncoder` 里：

```python
# BUGGY
a_freq = torch.stack([a_self.sum(1), cross.sum(1)], dim=2)
b_freq = torch.stack([b_self.sum(1), cross.sum(2)], dim=2)
```

矩阵定义：
- `a_self[b,i,j] = (a[b,i] == a[b,j])`  
- `cross[b,i,j] = (a[b,i] == b[b,j])`
- `b_self[b,i,j] = (b[b,i] == b[b,j])`

正确含义：
- `a_self.sum(1)[b,j]` = count of a[b,j] in a[b] = count(src[j] in src) ✓
- `cross.sum(2)[b,i]` = count of a[b,i] in b[b] = count(src[i] in dst) ✓  ← 用 `i` 不是 `j`
- `cross.sum(1)[b,j]` = count of b[b,j] in a[b] = count(dst[j] in src)
- `b_self.sum(1)[b,j]` = count of b[b,j] in b[b] = count(dst[j] in dst) ✓

因此正确实现：
```python
# CORRECT
a_freq = torch.stack([a_self.sum(1), cross.sum(2)], dim=2)   # [count(src[j] in src), count(src[j] in dst)]
b_freq = torch.stack([cross.sum(1), b_self.sum(1)], dim=2)   # [count(dst[j] in src), count(dst[j] in dst)]
```

当前代码 `a_freq` 的第二列用了 `cross.sum(1)[b,j]` = count(dst[j] in src) ——
这是 dst 的第一列，而不是 src 的第二列，语义完全错误。

### 4. Self Token 在 DyGLib 中的处理

DyGLib 的 self token（position 0）特征：
- `node_feat = node_raw_features[node_id]`（自己的 LIWC 特征）
- `edge_feat = edge_raw_features[0]`（edge 0 = padding row，全零）
- `time_feat = time_enc(0)`（时间差为 0）
- `co_occurrence`：正常计算（self 包含在序列里）

TGEngine 当前实现（最新改动后）：self token 独立于邻居序列，co-occurrence 设为零。
这与 DyGLib 有差异：DyGLib self 的 co-occurrence 计数是有意义的（≥ 1）。

### 5. Pooling 差异

DyGLib 使用 `torch.mean` 对所有 patches 做 unmasked mean pool（包括全零 padding patch）。
TGEngine 使用 masked mean（只对 non-padding patches）。

TGEngine 的做法更合理，但这是一个语义差异，影响 loss landscape。

### 6. TransformerEncoder 差异

DyGLib：`MultiheadAttention(batch_first=False)`，代码内部 transpose。Pre-LN。
TGEngine：`MultiheadAttention(batch_first=True)`。Pre-LN。

等价，无问题。

### 7. 输出维度

DyGLib：`output_layer` 的 `out_features = node_feat_dim`（Wikipedia=172）
TGEngine：`out_proj` 的 `out_features = d_model`（设 172 即可对齐）

---

## 修复优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| 🔴 P0 | Co-occurrence sum 维度错误 | co-occurrence 语义完全错误，影响所有数据集 |
| 🟡 P1 | Self-token co-occurrence 为零 | DyGLib self-token 有非零 co-occurrence；对稀疏 Wikipedia 节点有影响 |
| 🟢 P2 | Pooling: masked vs unmasked | 实现更合理但非完全对齐 DyGLib |

---

## 下一步

1. Fix co-occurrence bug (P0)
2. 重新跑 Wikipedia benchmark 验证 AP 是否提升
3. 如果仍低，考虑对齐 self-token 的 co-occurrence 计算
