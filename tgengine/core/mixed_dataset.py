"""MixedDataset: merge multiple TemporalDataset instances into one unified dataset.

Node IDs are remapped with per-dataset offsets to avoid collisions.
Timestamps are per-dataset normalized to [0, 1] so different time scales
don't dominate the TimeEncoder.
Edge features are aligned by padding to the maximum d_edge across all datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .dataset import TemporalDataset


@dataclass
class DatasetInfo:
    name: str
    node_offset: int       # added to all node IDs of this dataset
    num_nodes: int
    d_edge: int
    n_edges: int


class MixedDataset:
    """Merge multiple TemporalDataset instances for cross-domain training.

    Example::

        from tgengine import load_dataset
        from tgengine.core.mixed_dataset import MixedDataset

        ds_list = [load_dataset(n, dataset_path=...) for n in ["CanParl", "USLegis", "BitcoinAlpha"]]
        mixed = MixedDataset(ds_list, names=["CanParl", "USLegis", "BitcoinAlpha"])

        # Use like a TemporalDataset
        train_batches = mixed.get_batches("train", batch_size=200, device="cuda")
        graph = mixed.make_graph(device="cuda")
    """

    def __init__(
        self,
        datasets: List[TemporalDataset],
        names: Optional[List[str]] = None,
        d_edge_target: Optional[int] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        if names is None:
            names = [f"ds{i}" for i in range(len(datasets))]
        assert len(names) == len(datasets)

        # Determine common d_edge (pad to max)
        d_edges = [ds.edge_feat_dim for ds in datasets]
        self.d_edge = d_edge_target if d_edge_target is not None else max(d_edges)

        # Assign node ID offsets
        self._infos: List[DatasetInfo] = []
        offset = 0
        for name, ds in zip(names, datasets):
            info = DatasetInfo(
                name=name,
                node_offset=offset,
                num_nodes=ds.num_nodes,
                d_edge=ds.edge_feat_dim,
                n_edges=ds.num_edges,
            )
            self._infos.append(info)
            offset += ds.num_nodes

        self.num_nodes: int = offset
        self.dataset_names = names

        # Build merged arrays by processing each dataset
        all_src, all_dst, all_time, all_feat = [], [], [], []

        for info, ds in zip(self._infos, datasets):
            src = ds.src.cpu().long() + info.node_offset
            dst = ds.dst.cpu().long() + info.node_offset

            # Normalize timestamps to [0, 1] per dataset
            t = ds.time.cpu().float()
            t_min, t_max = float(t.min()), float(t.max())
            if t_max > t_min:
                t = (t - t_min) / (t_max - t_min)
            else:
                t = torch.zeros_like(t)

            # Align edge features
            feat = ds.edge_feat  # (N, d_edge_i)
            if feat is None or ds.edge_feat_dim == 0:
                feat = torch.zeros(len(src), self.d_edge)
            elif ds.edge_feat_dim < self.d_edge:
                pad = torch.zeros(len(src), self.d_edge - ds.edge_feat_dim)
                feat = torch.cat([feat.cpu(), pad], dim=1)
            else:
                feat = feat.cpu()[:, :self.d_edge]

            all_src.append(src)
            all_dst.append(dst)
            all_time.append(t)
            all_feat.append(feat)

        # Concatenate: keep each dataset's internal chronological order,
        # but interleave across datasets by sorting on normalized time.
        # To preserve intra-dataset order we use a stable sort.
        self.src: Tensor = torch.cat(all_src)
        self.dst: Tensor = torch.cat(all_dst)
        self.time: Tensor = torch.cat(all_time)
        self.edge_feat: Tensor = torch.cat(all_feat)

        order = torch.argsort(self.time, stable=True)
        self.src = self.src[order]
        self.dst = self.dst[order]
        self.time = self.time[order]
        self.edge_feat = self.edge_feat[order]

        self.num_edges: int = len(self.src)
        self.edge_feat_dim: int = self.d_edge

        # Compute split boundaries (event-based)
        n = self.num_edges
        self.train_end: int = int(n * (1 - val_ratio - test_ratio))
        self.val_end: int = int(n * (1 - test_ratio))

    # ------------------------------------------------------------------
    # API compatible with TemporalDataset
    # ------------------------------------------------------------------

    @property
    def train_size(self) -> int:
        return self.train_end

    @property
    def val_size(self) -> int:
        return self.val_end - self.train_end

    @property
    def test_size(self) -> int:
        return self.num_edges - self.val_end

    def summary(self) -> str:
        lines = [
            f"MixedDataset: {len(self._infos)} domains  |  {self.num_nodes} nodes  |  {self.num_edges} edges  |  d_edge={self.d_edge}",
        ]
        for info in self._infos:
            lines.append(f"  [{info.name:15s}]  nodes={info.num_nodes:6d} (offset +{info.node_offset})  edges={info.n_edges:7d}  d_edge={info.d_edge}")
        lines.append(f"  Split  train={self.train_size}  val={self.val_size}  test={self.test_size}")
        return "\n".join(lines)

    def get_batches(
        self,
        split: str,
        batch_size: int,
        device: str = "cpu",
    ):
        """Return list of RawBatch for the given split."""
        from tgengine.core.batch import RawBatch

        if split == "train":
            start, end = 0, self.train_end
        elif split == "val":
            start, end = self.train_end, self.val_end
        elif split == "test":
            start, end = self.val_end, self.num_edges
        else:
            raise ValueError(f"Unknown split: {split}")

        src = self.src[start:end].to(device)
        dst = self.dst[start:end].to(device)
        time = self.time[start:end].to(device)
        feat = self.edge_feat[start:end].to(device)

        batches = []
        for i in range(0, end - start, batch_size):
            s = slice(i, i + batch_size)
            batches.append(RawBatch(
                src=src[s], dst=dst[s], time=time[s], edge_feat=feat[s],
            ))
        return batches

    def make_graph(self, buffer_size: int = 32, device: str = "cpu"):
        """Create a TemporalGraph sized for this mixed dataset."""
        from tgengine.core.temporal_graph import TemporalGraph
        return TemporalGraph(
            self.num_nodes,
            buffer_size=buffer_size,
            edge_feat_dim=self.d_edge,
            device=device,
        )
