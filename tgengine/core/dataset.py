"""Dataset loading utilities.

Supports loading CTDG datasets from csv/npy files (DyGLib format)
and TGB format, converting them into RawBatch streams.

DyGLib-compatible data loading:
  - Time-quantile splits (not index-based)
  - Feature padding to 172 dimensions
  - Transductive / inductive eval split
  - Keeps zero-padding row at index 0 of feature matrices
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .batch import RawBatch

PAD_FEAT_DIM = 172


@dataclass
class TemporalDataset:
    """A loaded temporal graph dataset.

    Edges are sorted chronologically within the training set, then
    validation set, then test set (matching DyGLib's reordering).
    """

    src: Tensor  # (N,) all source nodes, chronologically sorted within each split
    dst: Tensor  # (N,) all destination nodes
    time: Tensor  # (N,) all timestamps
    edge_feat: Optional[Tensor]  # (N, d_edge) or None
    node_feat: Optional[Tensor]  # (num_nodes, d_node) or None

    num_nodes: int
    num_edges: int

    # Split boundaries: edges are ordered [train | val | test]
    train_end: int
    val_end: int

    # Inductive eval masks (same length as full dataset, aligned to reordered edges)
    new_node_val_mask: Optional[Tensor] = None   # bool (N,)
    new_node_test_mask: Optional[Tensor] = None  # bool (N,)

    @property
    def edge_feat_dim(self) -> int:
        return self.edge_feat.shape[1] if self.edge_feat is not None else 0

    @property
    def node_feat_dim(self) -> int:
        return self.node_feat.shape[1] if self.node_feat is not None else 0

    def get_batches(self, split: str, batch_size: int, device: str = "cuda",
                    inductive_only: bool = False) -> list[RawBatch]:
        """Get chronological batches for a split.

        Args:
            split: "train", "val", or "test".
            batch_size: number of edges per batch.
            device: target device.
            inductive_only: if True, only return edges with new nodes (for inductive eval).
        """
        if split == "train":
            start, end = 0, self.train_end
        elif split == "val":
            start, end = self.train_end, self.val_end
        elif split == "test":
            start, end = self.val_end, self.num_edges
        else:
            raise ValueError(f"Unknown split: {split}")

        # Build index list, optionally filtered for inductive eval
        if inductive_only and split == "val" and self.new_node_val_mask is not None:
            idx = torch.where(self.new_node_val_mask[start:end])[0] + start
        elif inductive_only and split == "test" and self.new_node_test_mask is not None:
            idx = torch.where(self.new_node_test_mask[start:end])[0] + start
        else:
            idx = torch.arange(start, end)

        batches = []
        for i in range(0, len(idx), batch_size):
            j = min(i + batch_size, len(idx))
            batch_idx = idx[i:j]
            feat = self.edge_feat[batch_idx].to(device) if self.edge_feat is not None else None
            batches.append(RawBatch(
                src=self.src[batch_idx].to(device),
                dst=self.dst[batch_idx].to(device),
                time=self.time[batch_idx].to(device),
                edge_feat=feat,
            ))
        return batches


def load_dataset(
    dataset_name: str,
    dataset_path: str = "datasets",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TemporalDataset:
    """Load a CTDG dataset with DyGLib-compatible preprocessing.

    Features are zero-padded to 172 dimensions. Splits use time-quantile
    thresholds. Random 10% of test-time nodes are held out for inductive eval.

    Expected files:
        {dataset_path}/{dataset_name}/ml_{dataset_name}.csv
        {dataset_path}/{dataset_name}/ml_{dataset_name}.npy (edge features)
        {dataset_path}/{dataset_name}/ml_{dataset_name}_node.npy (node features, optional)
    """
    import pandas as pd

    base = Path(dataset_path) / dataset_name

    csv_path = base / f"ml_{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(str(csv_path))
    src = df["u"].values.astype(np.int64)
    dst = df["i"].values.astype(np.int64)
    time_vals = df["ts"].values.astype(np.float64)
    labels = df["label"].values

    # ---- feature loading & padding to 172 ----
    node_feat = None
    node_path = base / f"ml_{dataset_name}_node.npy"
    if node_path.exists():
        node_feat = torch.from_numpy(np.load(str(node_path))).float()
        if node_feat.shape[1] < PAD_FEAT_DIM:
            pad = torch.zeros(node_feat.shape[0], PAD_FEAT_DIM - node_feat.shape[1])
            node_feat = torch.cat([node_feat, pad], dim=1)

    edge_feat = None
    feat_path = base / f"ml_{dataset_name}.npy"
    if feat_path.exists():
        edge_feat = torch.from_numpy(np.load(str(feat_path))).float()
        if edge_feat.shape[1] < PAD_FEAT_DIM:
            pad = torch.zeros(edge_feat.shape[0], PAD_FEAT_DIM - edge_feat.shape[1])
            edge_feat = torch.cat([edge_feat, pad], dim=1)
        # Keep zero-row at index 0 (DyGLib compatibility: 0 = padding sentinel)

    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)

    # ---- time-quantile splits ----
    val_time, test_time = list(
        np.quantile(time_vals, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )

    train_mask = time_vals <= val_time
    val_mask = (time_vals > val_time) & (time_vals <= test_time)
    test_mask = time_vals > test_time

    # ---- inductive node sets (matching DyGLib) ----
    random.seed(2020)
    node_set = set(src) | set(dst)
    num_total_unique = len(node_set)

    # Nodes appearing at test time
    test_node_set = set(src[time_vals > val_time]) | set(dst[time_vals > val_time])
    # 10% of all unique nodes, sampled from test-time nodes, held out
    new_test_node_set = set(
        random.sample(list(test_node_set), int(0.1 * num_total_unique))
    )

    # Remove edges involving new_test_nodes from training
    new_test_src = np.isin(src, list(new_test_node_set))
    new_test_dst = np.isin(dst, list(new_test_node_set))
    observed_mask = ~(new_test_src | new_test_dst)
    train_mask = train_mask & observed_mask

    # New nodes = nodes never seen in training
    train_node_set = set(src[train_mask]) | set(dst[train_mask])
    new_node_set = node_set - train_node_set

    # Inductive eval masks: edges with at least one node from new_node_set
    edge_contains_new = np.array([
        (s in new_node_set or d in new_node_set)
        for s, d in zip(src, dst)
    ])
    new_node_val_mask = val_mask & edge_contains_new
    new_node_test_mask = test_mask & edge_contains_new

    # ---- reorder: train first, then val, then test ----
    order = np.concatenate([
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    ])

    src_t = torch.from_numpy(src[order]).long()
    dst_t = torch.from_numpy(dst[order]).long()
    time_t = torch.from_numpy(time_vals[order]).double()
    edge_feat_t = edge_feat[order] if edge_feat is not None else None
    new_node_val_t = torch.from_numpy(new_node_val_mask[order])
    new_node_test_t = torch.from_numpy(new_node_test_mask[order])

    train_end = int(train_mask.sum())
    val_end = train_end + int(val_mask.sum())
    num_edges = val_end + int(test_mask.sum())  # excludes train-time new-node edges

    print(
        f"The dataset has {num_edges} interactions, involving {num_total_unique} different nodes"
    )
    print(
        f"The training dataset has {train_end} interactions, "
        f"involving {len(train_node_set)} different nodes"
    )
    print(
        f"The validation dataset has {val_mask.sum()} interactions, "
        f"involving {len(set(src[val_mask]) | set(dst[val_mask]))} different nodes"
    )
    print(
        f"The test dataset has {test_mask.sum()} interactions, "
        f"involving {len(set(src[test_mask]) | set(dst[test_mask]))} different nodes"
    )
    print(
        f"The new node validation dataset has {new_node_val_mask.sum()} interactions"
    )
    print(
        f"The new node test dataset has {new_node_test_mask.sum()} interactions"
    )
    print(
        f"{len(new_test_node_set)} nodes were used for the inductive testing, "
        f"i.e. are never seen during training"
    )

    return TemporalDataset(
        src=src_t,
        dst=dst_t,
        time=time_t,
        edge_feat=edge_feat_t,
        node_feat=node_feat,
        num_nodes=num_nodes,
        num_edges=num_edges,
        train_end=train_end,
        val_end=val_end,
        new_node_val_mask=new_node_val_t,
        new_node_test_mask=new_node_test_t,
    )


def load_tgb_dataset(
    dataset_name: str,
    dataset_path: str = "datasets",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[TemporalDataset, Optional[Tensor], Optional[Tensor]]:
    """Load a TGB-format dataset with fixed negative candidate lists.

    Expected directory layout::

        {dataset_path}/{dataset_name}/
            ml_{dataset_name}.pkl          # DataFrame: u, i, ts, idx, [w]
            ml_{dataset_name}_edge.pkl     # edge features (N, d_edge) ndarray
            *val_ns*.pkl                   # val fixed negative dict
            *test_ns*.pkl                  # test fixed negative dict

    Edges are sorted chronologically and split into train/val/test.

    Returns:
        dataset: TemporalDataset with chronological splits.
        val_neg_lists: (N_val, N_neg) tensor or None if not found.
        test_neg_lists: (N_test, N_neg) tensor or None if not found.
    """
    import pandas as pd

    base = Path(dataset_path) / dataset_name
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    # ---- locate DataFrame pickle -------------------------------------
    df_candidates = sorted(base.glob("ml_*.pkl"))
    if not df_candidates:
        raise FileNotFoundError(f"No ml_*.pkl DataFrame found in {base}")
    df_path = df_candidates[0]

    df = pd.read_pickle(str(df_path))
    # Standardize column names
    col_map = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("u", "src", "user_id", "source"):
            col_map[c] = "src"
        elif low in ("i", "dst", "item_id", "destination"):
            col_map[c] = "dst"
        elif low in ("ts", "timestamp", "time"):
            col_map[c] = "time"
        elif low in ("idx", "edge_idx", "index"):
            col_map[c] = "idx"
    df = df.rename(columns=col_map)

    # ---- sort chronologically ---------------------------------------
    df = df.sort_values("time").reset_index(drop=True)

    src_t = torch.from_numpy(df["src"].values.astype(np.int64)).long()
    dst_t = torch.from_numpy(df["dst"].values.astype(np.int64)).long()
    time_t = torch.from_numpy(df["time"].values.astype(np.float64)).double()
    edge_indices = torch.from_numpy(df.get("idx", df.index).values.astype(np.int64)).long()

    # ---- edge features -----------------------------------------------
    edge_feat_t: Optional[Tensor] = None
    feat_candidates = sorted(base.glob("ml_*_edge*"))
    if feat_candidates:
        edge_feat = np.load(str(feat_candidates[0]), allow_pickle=True)
        if edge_feat.dtype == np.float64:
            edge_feat = edge_feat.astype(np.float32)
        edge_feat_t = torch.from_numpy(edge_feat).float()
        # Edge feature rows correspond to original df order; reorder to match sort
        if "idx" in df.columns:
            idx_after_sort = df["idx"].values.astype(np.int64)
            edge_feat_t = edge_feat_t[idx_after_sort]

    num_nodes = max(src_t.max(), dst_t.max()).item() + 1
    num_edges = len(src_t)

    # ---- load negative dicts & determine splits ----------------------
    val_keys = _load_tgb_key_set(base, "val")
    test_keys = _load_tgb_key_set(base, "test")

    # Classify each edge as train/val/test based on neg dict membership
    edge_keys = set()
    val_mask = np.zeros(num_edges, dtype=bool)
    test_mask = np.zeros(num_edges, dtype=bool)

    for i in range(num_edges):
        k = (int(src_t[i].item()), int(dst_t[i].item()), int(time_t[i].item()))
        edge_keys.add(k)
        if val_keys is not None and k in val_keys:
            val_mask[i] = True
        elif test_keys is not None and k in test_keys:
            test_mask[i] = True

    train_mask = ~(val_mask | test_mask)

    # Reorder: train first, then val, then test
    order = np.concatenate([
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    ])

    src_t = src_t[order]
    dst_t = dst_t[order]
    time_t = time_t[order]
    if edge_feat_t is not None:
        edge_feat_t = edge_feat_t[order]

    train_end = int(train_mask.sum())
    val_end = train_end + int(val_mask.sum())

    dataset = TemporalDataset(
        src=src_t, dst=dst_t, time=time_t, edge_feat=edge_feat_t,
        node_feat=None, num_nodes=num_nodes, num_edges=num_edges,
        train_end=train_end, val_end=val_end,
    )

    # ---- extract aligned negative lists ------------------------------
    val_neg = _build_tgb_neg_tensor(dataset, train_end, val_end, val_keys, "val") \
        if val_keys is not None else None
    test_neg = _build_tgb_neg_tensor(dataset, val_end, num_edges, test_keys, "test") \
        if test_keys is not None else None

    return dataset, val_neg, test_neg


def _load_tgb_key_set(base: Path, split: str) -> Optional[dict]:
    """Load TGB negative pickle and return {(src,dst,time) -> neg_array} dict."""
    candidates = sorted(base.glob(f"*{split}_ns*.pkl"))
    if not candidates:
        return None

    with open(candidates[0], "rb") as f:
        neg_dict = pickle.load(f)

    return {(int(k[0]), int(k[1]), int(k[2])): v for k, v in neg_dict.items()}


def _build_tgb_neg_tensor(
    dataset: TemporalDataset, start: int, end: int,
    lookup: dict, split: str,
) -> Tensor:
    """Build (N_eval, n_neg) tensor from pre-loaded neg dict."""
    neg_list: list[Tensor] = []
    for i in range(start, end):
        key = (int(dataset.src[i].item()), int(dataset.dst[i].item()),
               int(dataset.time[i].item()))
        arr = lookup.get(key)
        if arr is None:
            raise KeyError(
                f"Negative list not found for edge {key} in {split} "
                f"(edge {i - start}/{end - start})"
            )
        neg_list.append(torch.from_numpy(arr.astype(np.int64)).long())

    if not neg_list:
        raise ValueError(f"No eval edges found for {split}")

    n_neg = max(n.shape[0] for n in neg_list)
    padded = []
    for n in neg_list:
        if n.shape[0] < n_neg:
            n = torch.cat([n, torch.full((n_neg - n.shape[0],), -1, dtype=torch.long)])
        padded.append(n)
    return torch.stack(padded)  # (N_eval, n_neg)


def _load_tgb_negatives(
    base: Path, dataset: TemporalDataset, start: int, end: int, split: str,
) -> Optional[Tensor]:
    """Load TGB fixed-negative pickle and align with eval edges.

    TGB negative dicts are keyed by (src, dst, time) with np.int64 values.
    We convert dataset timestamps to int for matching.
    """
    candidates = sorted(base.glob(f"*{split}_ns*.pkl"))
    if not candidates:
        return None

    with open(candidates[0], "rb") as f:
        neg_dict = pickle.load(f)

    # Build lookup from (src, dst, int_time) → neg array
    lookup: dict[tuple[int, int, int], np.ndarray] = {}
    for k, v in neg_dict.items():
        lookup[(int(k[0]), int(k[1]), int(k[2]))] = v

    neg_list: list[Tensor] = []
    for i in range(start, end):
        key = (int(dataset.src[i].item()), int(dataset.dst[i].item()),
               int(dataset.time[i].item()))
        arr = lookup.get(key)
        if arr is None:
            raise KeyError(
                f"Negative list not found for edge {key} in {split} "
                f"(edge {i - start}/{end - start})"
            )
        neg_list.append(torch.from_numpy(arr.astype(np.int64)).long())

    if not neg_list:
        return None

    n_neg = max(n.shape[0] for n in neg_list)
    padded = []
    for n in neg_list:
        if n.shape[0] < n_neg:
            n = torch.cat([n, torch.full((n_neg - n.shape[0],), -1, dtype=torch.long)])
        padded.append(n)
    return torch.stack(padded)  # (N_eval, n_neg)
