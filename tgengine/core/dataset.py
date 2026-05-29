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

    # Edges filtered from training (new_test_node edges in train time window)
    # These must be loaded into graph during eval to match DyGLib's full_neighbor_sampler
    inductive_edges: Optional[dict] = None  # {src, dst, time, edge_feat} Tensors

    # TGB/TGB-Seq fixed negative candidates for ranking eval (MRR)
    val_neg_candidates: Optional[Tensor] = None   # (N_val, N_neg) node IDs
    test_neg_candidates: Optional[Tensor] = None  # (N_test, N_neg) node IDs

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
    auto_download: bool = True,
) -> TemporalDataset:
    """Load a CTDG dataset. Unified entry point for all dataset families.

    Automatically detects dataset type by name prefix:
      - tgbl-* → TGB link prediction (with fixed neg samples for MRR)
      - tgbseq-* → TGB-Seq (with fixed neg samples for MRR)
      - others → DyGLib format (time-quantile split + inductive eval)

    If the dataset is not found locally and auto_download is True, it will be
    downloaded automatically from public sources.
    """
    base = Path(dataset_path) / dataset_name
    # Try underscore variant (tgbl_uci vs tgbl-uci)
    if not base.exists():
        alt = Path(dataset_path) / dataset_name.replace("-", "_")
        if alt.exists():
            base = alt

    # Auto-download if needed
    if not base.exists():
        if auto_download:
            from tgengine.utils.download import download_dataset, ALL_DATASETS
            if dataset_name in ALL_DATASETS:
                download_dataset(dataset_name, dest_dir=dataset_path)

    # Dispatch by dataset family
    if dataset_name.startswith("tgbl-"):
        return _load_tgb_csv(dataset_name, base, val_ratio, test_ratio)
    elif dataset_name.startswith("tgbseq-"):
        return _load_tgbseq_csv(dataset_name, base, val_ratio, test_ratio)
    else:
        return _load_dyglib(dataset_name, base, val_ratio, test_ratio)

def _load_dyglib(
    dataset_name: str, base: Path, val_ratio: float, test_ratio: float
) -> TemporalDataset:
    """Load DyGLib-format dataset with time-quantile splits + inductive eval."""
    import pandas as pd

    csv_path = base / f"ml_{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(str(csv_path))
    src = df["u"].values.astype(np.int64)
    dst = df["i"].values.astype(np.int64)
    time_vals = df["ts"].values.astype(np.float64)

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

    test_node_set = set(src[time_vals > val_time]) | set(dst[time_vals > val_time])
    new_test_node_set = set(
        random.sample(list(test_node_set), int(0.1 * num_total_unique))
    )

    new_test_src = np.isin(src, list(new_test_node_set))
    new_test_dst = np.isin(dst, list(new_test_node_set))
    observed_mask = ~(new_test_src | new_test_dst)
    train_mask = train_mask & observed_mask

    train_node_set = set(src[train_mask]) | set(dst[train_mask])
    new_node_set = node_set - train_node_set

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

    inductive_train_mask = (time_vals <= val_time) & ~observed_mask
    inductive_idx = np.where(inductive_train_mask)[0]
    if len(inductive_idx) > 0:
        ind_src = torch.from_numpy(src[inductive_idx]).long()
        ind_dst = torch.from_numpy(dst[inductive_idx]).long()
        ind_time = torch.from_numpy(time_vals[inductive_idx]).double()
        ind_ef = edge_feat[inductive_idx] if edge_feat is not None else None
        inductive_edges = {"src": ind_src, "dst": ind_dst, "time": ind_time, "edge_feat": ind_ef}
    else:
        inductive_edges = None

    src_t = torch.from_numpy(src[order]).long()
    dst_t = torch.from_numpy(dst[order]).long()
    time_t = torch.from_numpy(time_vals[order]).double()
    edge_feat_t = edge_feat[order] if edge_feat is not None else None
    new_node_val_t = torch.from_numpy(new_node_val_mask[order])
    new_node_test_t = torch.from_numpy(new_node_test_mask[order])

    train_end = int(train_mask.sum())
    val_end = train_end + int(val_mask.sum())
    num_edges = val_end + int(test_mask.sum())

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
        inductive_edges=inductive_edges,
    )


def _load_tgb_csv(
    dataset_name: str, base: Path, val_ratio: float, test_ratio: float
) -> TemporalDataset:
    """Load TGB dataset from original edgelist CSV + fixed negative samples.

    Reads the TGB edgelist CSV directly (no reindexing during download).
    Uses the same time-quantile split as TGB (0.70/0.85 by default) to
    preserve alignment with pre-computed negative sample pkl files.
    """
    import pandas as pd

    # Find edgelist CSV
    edgelist = None
    for pattern in ["*edgelist*.csv"]:
        candidates = sorted(base.glob(pattern))
        if candidates:
            edgelist = candidates[0]
            break

    if edgelist is None:
        raise FileNotFoundError(f"No edgelist CSV found in {base}")

    # TGB CSVs like tgbl-wiki have a "comma_separated_list_of_features" header
    # that expands to 172 columns. Detect header/data column count mismatch.
    with open(edgelist, "r") as f:
        header_line = f.readline().strip()
        data_line = f.readline().strip()
    n_header = len(header_line.split(","))
    n_data = len(data_line.split(","))

    if n_data > n_header:
        # Re-read with explicit column names: first (n_header-1) named + rest as feat_*
        header_cols = header_line.split(",")
        all_cols = header_cols[:-1] + [f"feat_{i}" for i in range(n_data - n_header + 1)]
        df = pd.read_csv(str(edgelist), names=all_cols, skiprows=1)
    else:
        df = pd.read_csv(str(edgelist))

    # Standardize column names
    col_map = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("source", "src", "u", "head", "user_id"):
            col_map[c] = "u"
        elif low in ("destination", "dst", "i", "tail", "item_id"):
            col_map[c] = "i"
        elif low in ("timestamp", "ts", "time", "day"):
            col_map[c] = "ts"
    df = df.rename(columns=col_map)

    if "u" not in df.columns or "i" not in df.columns or "ts" not in df.columns:
        raise ValueError(f"Cannot identify u/i/ts columns in {edgelist}. Columns: {list(df.columns)}")

    # Sort chronologically
    df = df.sort_values("ts").reset_index(drop=True)

    src = df["u"].values.astype(np.int64)
    dst = df["i"].values.astype(np.int64)
    time_vals = df["ts"].values.astype(np.float64)

    # Extract edge features (numeric columns besides u, i, ts)
    meta_cols = {"u", "i", "ts"}
    feat_cols = [c for c in df.columns if c not in meta_cols
                 and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]

    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)

    # Edge features: try pre-processed pkl/npy first, then CSV columns
    edge_feat_t = _load_tgb_edge_features(base, df, meta_cols, num_edges)

    # Time-quantile split — same as TGB's generate_splits()
    val_time, test_time = list(
        np.quantile(time_vals, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )
    train_mask = time_vals <= val_time
    val_mask = (time_vals > val_time) & (time_vals <= test_time)
    test_mask = time_vals > test_time

    # Reorder: train | val | test (already sorted by time within each split)
    order = np.concatenate([
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    ])

    src_t = torch.from_numpy(src[order]).long()
    dst_t = torch.from_numpy(dst[order]).long()
    time_t = torch.from_numpy(time_vals[order]).double()
    edge_feat_ordered = edge_feat_t[order] if edge_feat_t is not None else None

    train_end = int(train_mask.sum())
    val_end = train_end + int(val_mask.sum())

    # Load fixed negative samples (keyed by original node IDs — no reindex needed)
    val_neg = _load_neg_pkl(base, "val")
    test_neg = _load_neg_pkl(base, "test")

    # Node features (from node_feat CSV if present)
    node_feat = _load_tgb_node_features(base, num_nodes)

    print(f"TGB dataset '{dataset_name}': {num_edges} edges, {num_nodes} nodes")
    print(f"  train: {train_end}, val: {val_end - train_end}, test: {num_edges - val_end}")
    if val_neg is not None:
        print(f"  val neg candidates: {val_neg.shape}")
    if test_neg is not None:
        print(f"  test neg candidates: {test_neg.shape}")

    return TemporalDataset(
        src=src_t, dst=dst_t, time=time_t,
        edge_feat=edge_feat_ordered, node_feat=node_feat,
        num_nodes=num_nodes, num_edges=num_edges,
        train_end=train_end, val_end=val_end,
        val_neg_candidates=val_neg,
        test_neg_candidates=test_neg,
    )


def _load_tgbseq_csv(
    dataset_name: str, base: Path, val_ratio: float, test_ratio: float
) -> TemporalDataset:
    """Load TGB-Seq dataset from original CSV, preserving the split column.

    TGB-Seq CSVs contain a 'split' column (0=train, 1=val, 2=test) with
    pre-computed splits that include node degree filtering. We use these
    splits directly instead of re-computing time-quantile.
    """
    import pandas as pd

    # Find the dataset CSV (named after HF repo name)
    from tgengine.utils.download import _TGBSEQ_DATASETS
    info = _TGBSEQ_DATASETS.get(dataset_name, {})
    hf_name = info.get("hf_name", dataset_name)

    csv_path = base / f"{hf_name}.csv"
    if not csv_path.exists():
        # Try any CSV
        candidates = sorted(base.glob("*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"No CSV found in {base}")

    df = pd.read_csv(str(csv_path))

    # Standardize column names
    col_map = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("source", "src", "u", "head", "user_id", "user"):
            col_map[c] = "u"
        elif low in ("destination", "dst", "i", "tail", "item_id", "item"):
            col_map[c] = "i"
        elif low in ("timestamp", "ts", "time"):
            col_map[c] = "ts"
    df = df.rename(columns=col_map)

    if "u" not in df.columns or "i" not in df.columns or "ts" not in df.columns:
        raise ValueError(f"Cannot identify u/i/ts columns. Columns: {list(df.columns)}")

    src = df["u"].values.astype(np.int64)
    dst = df["i"].values.astype(np.int64)
    time_vals = df["ts"].values.astype(np.float64)

    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)

    # Use pre-computed split column if available
    if "split" in df.columns:
        split_col = df["split"].values
        train_mask = split_col == 0
        val_mask = split_col == 1
        test_mask = split_col == 2
    else:
        # Fallback to time-quantile
        val_time, test_time = list(
            np.quantile(time_vals, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
        )
        train_mask = time_vals <= val_time
        val_mask = (time_vals > val_time) & (time_vals <= test_time)
        test_mask = time_vals > test_time

    # Edge features (numeric columns besides u, i, ts, split)
    meta_cols = {"u", "i", "ts", "split", "label", "idx"}
    feat_cols = [c for c in df.columns if c not in meta_cols
                 and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]

    edge_feat_t: Optional[Tensor] = None
    if feat_cols:
        feats = df[feat_cols].values.astype(np.float32)
        edge_feat_t = torch.from_numpy(feats).float()
        if edge_feat_t.shape[1] < PAD_FEAT_DIM:
            pad = torch.zeros(num_edges, PAD_FEAT_DIM - edge_feat_t.shape[1])
            edge_feat_t = torch.cat([edge_feat_t, pad], dim=1)

    # Reorder: train | val | test
    order = np.concatenate([
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    ])

    src_t = torch.from_numpy(src[order]).long()
    dst_t = torch.from_numpy(dst[order]).long()
    time_t = torch.from_numpy(time_vals[order]).double()
    edge_feat_ordered = edge_feat_t[order] if edge_feat_t is not None else None

    train_end = int(train_mask.sum())
    val_end = train_end + int(val_mask.sum())

    # Load fixed negative samples (npy)
    test_neg = _load_neg_npy(base)

    print(f"TGB-Seq dataset '{dataset_name}': {num_edges} edges, {num_nodes} nodes")
    print(f"  train: {train_end}, val: {val_end - train_end}, test: {num_edges - val_end}")
    if test_neg is not None:
        print(f"  test neg candidates: {test_neg.shape}")

    return TemporalDataset(
        src=src_t, dst=dst_t, time=time_t,
        edge_feat=edge_feat_ordered, node_feat=None,
        num_nodes=num_nodes, num_edges=num_edges,
        train_end=train_end, val_end=val_end,
        test_neg_candidates=test_neg,
    )


def _load_tgb_edge_features(
    base: Path, df, meta_cols: set, num_edges: int
) -> Optional[Tensor]:
    """Load TGB edge features from pkl/npy file, or extract from CSV columns.

    TGB datasets often have a pre-processed edge feature file (ml_*_edge.pkl
    or similar). If not found, falls back to numeric CSV columns.
    """
    import pickle

    # Try pkl edge feature file
    for pattern in ["*_edge*.pkl", "*_edge*.npy"]:
        candidates = sorted(base.glob(pattern))
        if candidates:
            path = candidates[0]
            if path.suffix == ".pkl":
                with open(path, "rb") as f:
                    arr = pickle.load(f)
            else:
                arr = np.load(str(path), allow_pickle=True)
            if isinstance(arr, np.ndarray):
                feat = torch.from_numpy(arr.astype(np.float32)).float()
                if feat.shape[0] == num_edges + 1:
                    feat = feat[1:]  # strip zero-row if present
                if feat.shape[0] == num_edges:
                    if feat.shape[1] < PAD_FEAT_DIM:
                        pad = torch.zeros(num_edges, PAD_FEAT_DIM - feat.shape[1])
                        feat = torch.cat([feat, pad], dim=1)
                    return feat

    # Fallback: numeric columns from CSV
    feat_cols = [c for c in df.columns if c not in meta_cols
                 and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    if feat_cols:
        feats = df[feat_cols].values.astype(np.float32)
        feat = torch.from_numpy(feats).float()
        if feat.shape[1] < PAD_FEAT_DIM:
            pad = torch.zeros(num_edges, PAD_FEAT_DIM - feat.shape[1])
            feat = torch.cat([feat, pad], dim=1)
        return feat

    return None


def _load_tgb_node_features(base: Path, num_nodes: int) -> Optional[Tensor]:
    """Load TGB node features from CSV if present."""
    import pandas as pd

    candidates = sorted(base.glob("*node_feat*"))
    if not candidates:
        return None
    path = candidates[0]
    if path.suffix == ".csv":
        nf = pd.read_csv(str(path))
        arr = nf.iloc[:, 1:].values.astype(np.float32)
    elif path.suffix == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    else:
        return None
    node_feat = torch.from_numpy(arr).float()
    if node_feat.shape[1] < PAD_FEAT_DIM:
        pad = torch.zeros(node_feat.shape[0], PAD_FEAT_DIM - node_feat.shape[1])
        node_feat = torch.cat([node_feat, pad], dim=1)
    return node_feat


def _load_neg_pkl(base: Path, split: str) -> Optional[Tensor]:
    """Load TGB negative sample pkl file → (N, N_neg) tensor."""
    import pickle

    candidates = sorted(base.glob(f"*{split}_ns*.pkl"))
    if not candidates:
        return None

    print(f"  Loading {candidates[0].name}...")
    with open(candidates[0], "rb") as f:
        neg_dict = pickle.load(f)

    if isinstance(neg_dict, dict):
        neg_arrays = list(neg_dict.values())
        if not neg_arrays:
            return None
        # All arrays usually same length; use np.stack for speed
        n_neg = neg_arrays[0].shape[0]
        uniform = all(a.shape[0] == n_neg for a in neg_arrays)
        if uniform:
            stacked = np.stack(neg_arrays)
        else:
            n_neg = max(a.shape[0] for a in neg_arrays)
            stacked = np.full((len(neg_arrays), n_neg), -1, dtype=np.float64)
            for i, a in enumerate(neg_arrays):
                stacked[i, :a.shape[0]] = a
        return torch.from_numpy(stacked.astype(np.int64)).long()
    elif isinstance(neg_dict, np.ndarray):
        return torch.from_numpy(neg_dict.astype(np.int64)).long()
    return None


def _load_neg_npy(base: Path) -> Optional[Tensor]:
    """Load TGB-Seq negative sample npy file → (N, N_neg) tensor."""
    candidates = sorted(base.glob("*_test_ns*.npy"))
    if not candidates:
        return None
    arr = np.load(str(candidates[0]))
    return torch.from_numpy(arr.astype(np.int64)).long()


