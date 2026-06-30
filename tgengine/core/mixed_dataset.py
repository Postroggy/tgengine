"""MixedDataset: merge multiple TemporalDataset instances into one unified dataset.

Node IDs are remapped with per-dataset offsets to avoid collisions.
Timestamps are per-dataset normalized to [0, 1] so different time scales
don't dominate the TimeEncoder.
Edge features are aligned by padding to the maximum d_edge across all datasets.

Mixing strategy (controlled by get_batches balance parameter):
  - balance=False (default): global time-sorted interleave — used for val/test
    so all edges are seen in chronological order.
  - balance=True: round-robin across domains, each capped at `per_domain_cap`
    edges (default: min domain size). Keeps intra-domain chronological order
    within each domain's slice. Use for training to prevent large domains from
    dominating and ensure the model sees each domain equally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    # per-split edge ranges in the merged (time-sorted) array
    train_end: int = 0
    val_end: int = 0
    total: int = 0


class MixedDataset:
    """Merge multiple TemporalDataset instances for cross-domain training.

    Example::

        from tgengine import load_dataset
        from tgengine.core.mixed_dataset import MixedDataset

        ds_list = [load_dataset(n, dataset_path=...) for n in ["uci", "BitcoinAlpha"]]
        mixed = MixedDataset(ds_list, names=["uci", "BitcoinAlpha"])

        # Balanced training (equal samples per domain, round-robin batches)
        train_batches = mixed.get_batches("train", batch_size=200, balance=True, device="cuda")

        # Full val/test (unbalanced, all edges in chronological order)
        val_batches = mixed.get_batches("val", batch_size=200, device="cuda")
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

        self.d_edge = d_edge_target if d_edge_target is not None else max(
            ds.edge_feat_dim for ds in datasets
        )

        # Assign node ID offsets and prepare per-dataset tensors
        self._infos: List[DatasetInfo] = []
        self._per_ds_src: List[Tensor] = []
        self._per_ds_dst: List[Tensor] = []
        self._per_ds_time: List[Tensor] = []
        self._per_ds_feat: List[Tensor] = []

        offset = 0
        all_src, all_dst, all_time, all_feat = [], [], [], []

        for name, ds in zip(names, datasets):
            n = ds.num_edges
            src = ds.src.cpu().long() + offset
            dst = ds.dst.cpu().long() + offset

            # Normalize timestamps to [0, 1] per dataset
            t = ds.time.cpu().float()
            t_min, t_max = float(t.min()), float(t.max())
            t = (t - t_min) / (t_max - t_min) if t_max > t_min else torch.zeros_like(t)

            # Align edge features
            feat = ds.edge_feat
            if feat is None or ds.edge_feat_dim == 0:
                feat = torch.zeros(n, self.d_edge)
            elif ds.edge_feat_dim < self.d_edge:
                feat = torch.cat([feat.cpu(), torch.zeros(n, self.d_edge - ds.edge_feat_dim)], dim=1)
            else:
                feat = feat.cpu()[:, :self.d_edge]

            # per-dataset split boundaries (chronological)
            train_end = int(n * (1 - val_ratio - test_ratio))
            val_end = int(n * (1 - test_ratio))

            info = DatasetInfo(
                name=name,
                node_offset=offset,
                num_nodes=ds.num_nodes,
                d_edge=ds.edge_feat_dim,
                n_edges=n,
                train_end=train_end,
                val_end=val_end,
                total=n,
            )
            self._infos.append(info)
            self._per_ds_src.append(src)
            self._per_ds_dst.append(dst)
            self._per_ds_time.append(t)
            self._per_ds_feat.append(feat)

            all_src.append(src)
            all_dst.append(dst)
            all_time.append(t)
            all_feat.append(feat)

            offset += ds.num_nodes

        self.num_nodes: int = offset
        self.dataset_names = names

        # Global time-sorted merged arrays (used for val/test and make_graph)
        src_cat = torch.cat(all_src)
        dst_cat = torch.cat(all_dst)
        time_cat = torch.cat(all_time)
        feat_cat = torch.cat(all_feat)

        order = torch.argsort(time_cat, stable=True)
        self.src: Tensor = src_cat[order]
        self.dst: Tensor = dst_cat[order]
        self.time: Tensor = time_cat[order]
        self.edge_feat: Tensor = feat_cat[order]

        self.num_edges: int = len(self.src)
        self.edge_feat_dim: int = self.d_edge

        n = self.num_edges
        self.train_end: int = int(n * (1 - val_ratio - test_ratio))
        self.val_end: int = int(n * (1 - test_ratio))

    # ------------------------------------------------------------------
    # Properties
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
            f"MixedDataset: {len(self._infos)} domains  |  {self.num_nodes} nodes  "
            f"|  {self.num_edges} edges  |  d_edge={self.d_edge}",
        ]
        for info in self._infos:
            lines.append(
                f"  [{info.name:15s}]  nodes={info.num_nodes:6d} (offset +{info.node_offset})"
                f"  edges={info.n_edges:7d}  train={info.train_end:6d}  d_edge={info.d_edge}"
            )
        lines.append(f"  Global split  train={self.train_size}  val={self.val_size}  test={self.test_size}")
        min_train = min(i.train_end for i in self._infos)
        lines.append(f"  Balanced cap  {min_train} edges/domain  "
                     f"({min_train * len(self._infos)} total balanced train edges)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def get_batches(
        self,
        split: str,
        batch_size: int,
        device: str = "cpu",
        balance: bool = False,
        per_domain_cap: Optional[int] = None,
        mode: str = "balanced",
    ):
        """Return list of RawBatch for the given split.

        Args:
            split: "train", "val", or "test"
            batch_size: edges per batch
            device: target device
            balance: if True (train only), use domain-aware mixing.
                Ignored for val/test.
            per_domain_cap: max edges per domain for balanced mode.
                Defaults to min(train_size across domains).
            mode: mixing strategy when balance=True:
                - "balanced": round-robin, each domain capped at
                  per_domain_cap (equal samples per domain). Simple but
                  wastes large datasets' data.
                - "proportional": each domain contributes edges
                  proportional to its train size (no cap). Domains
                  stay in chronological order within their slices;
                  slices are interleaved proportionally. Preserves all
                  training data.
        """
        from tgengine.core.batch import RawBatch

        if split != "train":
            balance = False

        if not balance:
            # Global chronological slice (original behavior)
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
                batches.append(RawBatch(src=src[s], dst=dst[s], time=time[s], edge_feat=feat[s]))
            return batches

        # --- Balanced or proportional mixing ---
        if mode == "proportional":
            # Each domain contributes ALL its training edges, interleaved
            # proportionally. No cap — preserves all data.
            domain_batches: List[List] = []
            for info, src_ds, dst_ds, time_ds, feat_ds in zip(
                self._infos,
                self._per_ds_src, self._per_ds_dst,
                self._per_ds_time, self._per_ds_feat,
            ):
                end = info.train_end
                src = src_ds[:end].to(device)
                dst = dst_ds[:end].to(device)
                t = time_ds[:end].to(device)
                feat = feat_ds[:end].to(device)

                ds_batches = []
                for i in range(0, end, batch_size):
                    s = slice(i, i + batch_size)
                    ds_batches.append(RawBatch(src=src[s], dst=dst[s], time=t[s], edge_feat=feat[s]))
                domain_batches.append(ds_batches)

            # Interleave proportionally: step through each domain at a rate
            # proportional to its size. E.g. if enron has 3x more edges than
            # BA, enron contributes 3 batches for every 1 from BA.
            total = sum(info.train_end for info in self._infos)
            batches = []
            # Use a ratio-based round-robin: track progress per domain
            progress = [0.0] * len(domain_batches)
            sizes = [info.train_end for info in self._infos]
            idx = [0] * len(domain_batches)
            steps = total // batch_size + 1
            for _ in range(steps):
                for d, db in enumerate(domain_batches):
                    # Advance domain d if its proportional progress allows
                    target = (sum(progress) + 1) * sizes[d] / total
                    if progress[d] < target and idx[d] < len(db):
                        batches.append(db[idx[d]])
                        progress[d] += 1
                        idx[d] += 1
            return batches

        # --- Balanced round-robin (original, with cap) ---
        # Each domain contributes exactly `cap` edges (first cap chronologically)
        cap = per_domain_cap or min(info.train_end for info in self._infos)

        # Build one batch list per domain, each batch entirely within one domain
        domain_batches: List[List] = []
        for info, src_ds, dst_ds, time_ds, feat_ds in zip(
            self._infos,
            self._per_ds_src, self._per_ds_dst,
            self._per_ds_time, self._per_ds_feat,
        ):
            # Take first `cap` training edges (chronological order preserved)
            end = min(cap, info.train_end)
            src  = src_ds[:end].to(device)
            dst  = dst_ds[:end].to(device)
            t    = time_ds[:end].to(device)
            feat = feat_ds[:end].to(device)

            ds_batches = []
            for i in range(0, end, batch_size):
                s = slice(i, i + batch_size)
                ds_batches.append(RawBatch(src=src[s], dst=dst[s], time=t[s], edge_feat=feat[s]))
            domain_batches.append(ds_batches)

        # Round-robin interleave: batch from domain 0, then 1, then 2, ...
        # Each domain's batches stay in chronological order within that domain.
        batches = []
        max_len = max(len(db) for db in domain_batches)
        for i in range(max_len):
            for db in domain_batches:
                if i < len(db):
                    batches.append(db[i])
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
