"""Dataset loading utilities.

Supports loading CTDG datasets from csv/npy files (DyGLib format)
and TGB format, converting them into RawBatch streams.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .batch import RawBatch


@dataclass
class TemporalDataset:
    """A loaded temporal graph dataset."""

    src: Tensor  # (N,) all source nodes, chronologically sorted
    dst: Tensor  # (N,) all destination nodes
    time: Tensor  # (N,) all timestamps
    edge_feat: Optional[Tensor]  # (N, d_edge) or None
    node_feat: Optional[Tensor]  # (num_nodes, d_node) or None

    num_nodes: int
    num_edges: int

    # Split indices
    train_end: int
    val_end: int

    @property
    def edge_feat_dim(self) -> int:
        return self.edge_feat.shape[1] if self.edge_feat is not None else 0

    @property
    def node_feat_dim(self) -> int:
        return self.node_feat.shape[1] if self.node_feat is not None else 0

    def get_batches(self, split: str, batch_size: int, device: str = "cuda") -> list[RawBatch]:
        """Get chronological batches for a split."""
        if split == "train":
            start, end = 0, self.train_end
        elif split == "val":
            start, end = self.train_end, self.val_end
        elif split == "test":
            start, end = self.val_end, self.num_edges
        else:
            raise ValueError(f"Unknown split: {split}")

        batches = []
        for i in range(start, end, batch_size):
            j = min(i + batch_size, end)
            feat = self.edge_feat[i:j].to(device) if self.edge_feat is not None else None
            batches.append(RawBatch(
                src=self.src[i:j].to(device),
                dst=self.dst[i:j].to(device),
                time=self.time[i:j].to(device),
                edge_feat=feat,
            ))
        return batches


def load_dataset(
    dataset_name: str,
    dataset_path: str = "datasets",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TemporalDataset:
    """Load a CTDG dataset from DyGLib-format files.

    Expected files:
        {dataset_path}/{dataset_name}/ml_{dataset_name}.csv
        {dataset_path}/{dataset_name}/ml_{dataset_name}.npy (edge features)
        {dataset_path}/{dataset_name}/ml_{dataset_name}_node.npy (node features, optional)

    Args:
        dataset_name: Name of the dataset (e.g., "wikipedia", "reddit").
        dataset_path: Root directory containing dataset folders.
        val_ratio: Fraction of edges for validation.
        test_ratio: Fraction of edges for test.

    Returns:
        TemporalDataset with all data loaded and splits computed.
    """
    base = Path(dataset_path) / dataset_name

    # Load edge list
    csv_path = base / f"ml_{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    # CSV format: u, i, ts, label, idx
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    src = torch.from_numpy(data[:, 0]).long()
    dst = torch.from_numpy(data[:, 1]).long()
    time = torch.from_numpy(data[:, 2]).double()

    # Edge features
    feat_path = base / f"ml_{dataset_name}.npy"
    edge_feat = None
    if feat_path.exists():
        edge_feat = torch.from_numpy(np.load(str(feat_path))).float()
        # First row is often padding (index 0), skip it
        if edge_feat.shape[0] == len(src) + 1:
            edge_feat = edge_feat[1:]

    # Node features
    node_path = base / f"ml_{dataset_name}_node.npy"
    node_feat = None
    if node_path.exists():
        node_feat = torch.from_numpy(np.load(str(node_path))).float()

    num_nodes = max(src.max(), dst.max()).item() + 1
    num_edges = len(src)

    # Chronological split
    val_start = int(num_edges * (1 - val_ratio - test_ratio))
    test_start = int(num_edges * (1 - test_ratio))

    return TemporalDataset(
        src=src,
        dst=dst,
        time=time,
        edge_feat=edge_feat,
        node_feat=node_feat,
        num_nodes=num_nodes,
        num_edges=num_edges,
        train_end=val_start,
        val_end=test_start,
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
