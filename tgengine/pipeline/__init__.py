from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from tgengine.core.batch import NeighborData, PreparedBatch, RawBatch
from tgengine.core.gather_spec import GatherSpec
from tgengine.core.temporal_graph import TemporalGraph


class DataPipeline:
    """Optimized data preparation pipeline.

    Reads a model's GatherSpec at init time and builds a fused execution plan.
    The prepare() method executes all graph operations in a single optimized pass.
    """

    def __init__(self, spec: GatherSpec, graph: TemporalGraph):
        self.spec = spec
        self.graph = graph

    def prepare(self, raw_batch: RawBatch) -> PreparedBatch:
        """Prepare a full batch for model consumption.

        All graph operations are fused: neighbor queries for src/dst/neg are
        concatenated into a single kernel call, then split back.
        """
        k = self.spec.neighbors.k
        device = raw_batch.device

        # Fused neighbor sampling: merge all node groups into one call
        node_groups = []
        group_sizes = []

        if "src" in self.spec.neighbors.for_nodes:
            node_groups.append(raw_batch.src)
            group_sizes.append(raw_batch.src.shape[0])
        if "dst" in self.spec.neighbors.for_nodes:
            node_groups.append(raw_batch.dst)
            group_sizes.append(raw_batch.dst.shape[0])
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg is not None:
            neg_flat = raw_batch.neg.reshape(-1) if raw_batch.neg.ndim > 1 else raw_batch.neg
            node_groups.append(neg_flat)
            group_sizes.append(neg_flat.shape[0])

        all_nodes = torch.cat(node_groups)
        # Expand times to match all nodes
        times_list = []
        if "src" in self.spec.neighbors.for_nodes:
            times_list.append(raw_batch.time)
        if "dst" in self.spec.neighbors.for_nodes:
            times_list.append(raw_batch.time)
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg is not None:
            if raw_batch.neg.ndim > 1:
                times_list.append(raw_batch.time.unsqueeze(1).expand_as(raw_batch.neg).reshape(-1))
            else:
                times_list.append(raw_batch.time)
        all_times = torch.cat(times_list)

        # Single fused kernel call
        all_neighbors = self.graph.recent(all_nodes, all_times, k)

        # Split results back into groups
        splits = torch.split_with_sizes(all_neighbors.neighbor_ids, group_sizes, dim=0)
        time_splits = torch.split_with_sizes(all_neighbors.timestamps, group_sizes, dim=0)
        feat_splits = torch.split_with_sizes(all_neighbors.edge_feats, group_sizes, dim=0)
        mask_splits = torch.split_with_sizes(all_neighbors.mask, group_sizes, dim=0)

        idx = 0
        src_nbrs = dst_nbrs = neg_nbrs = None

        if "src" in self.spec.neighbors.for_nodes:
            src_nbrs = NeighborData(splits[idx], time_splits[idx], feat_splits[idx], mask_splits[idx])
            idx += 1
        if "dst" in self.spec.neighbors.for_nodes:
            dst_nbrs = NeighborData(splits[idx], time_splits[idx], feat_splits[idx], mask_splits[idx])
            idx += 1
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg is not None:
            neg_nbrs = NeighborData(splits[idx], time_splits[idx], feat_splits[idx], mask_splits[idx])
            idx += 1

        # Co-occurrence (optional)
        co_occur = None
        if self.spec.co_occurrence:
            co_occur = self.graph.co_neighbors(raw_batch.src, raw_batch.dst, raw_batch.time)

        return PreparedBatch(
            src=raw_batch.src,
            dst=raw_batch.dst,
            neg=raw_batch.neg if raw_batch.neg is not None else torch.empty(0, device=device),
            time=raw_batch.time,
            src_neighbors=src_nbrs,
            dst_neighbors=dst_nbrs,
            neg_neighbors=neg_nbrs,
            co_occurrence=co_occur,
        )
