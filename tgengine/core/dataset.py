"""Dataset loading utilities.

Supports loading CTDG datasets from csv/npy files (DyGLib format)
and converting them into RawBatch streams.
"""

from __future__ import annotations

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
