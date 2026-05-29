from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from tgengine.core.batch import NeighborData, PreparedBatch, RawBatch
from tgengine.core.gather_spec import GatherSpec
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.pipeline.negatives import HistoricalNegative, HistoricalNegPool


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

        Requires raw_batch.neg to be pre-filled.  All neighbor queries for
        src/dst/neg are fused into a single graph.recent() call.
        Co-occurrence is computed from the already-queried src/dst neighbors
        — no redundant graph.recent() call.

        When spec.neighbors.k2 > 0, 2-hop neighbors are also queried in a
        second fused call over all valid 1-hop neighbor nodes.
        """
        k = self.spec.neighbors.k
        k2 = self.spec.neighbors.k2
        device = raw_batch.device

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
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg_src is not None:
            node_groups.append(raw_batch.neg_src)
            group_sizes.append(raw_batch.neg_src.shape[0])

        all_nodes = torch.cat(node_groups)
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
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg_src is not None:
            times_list.append(raw_batch.time)
        all_times = torch.cat(times_list)

        # ---- 1-hop -------------------------------------------------------
        if k2 > 0:
            all_neighbors = self.graph.recent_2hop(all_nodes, all_times, k, k2)
        else:
            all_neighbors = self.graph.recent(all_nodes, all_times, k)

        splits = torch.split_with_sizes(all_neighbors.neighbor_ids, group_sizes, dim=0)
        time_splits = torch.split_with_sizes(all_neighbors.timestamps, group_sizes, dim=0)
        feat_splits = torch.split_with_sizes(all_neighbors.edge_feats, group_sizes, dim=0)
        mask_splits = torch.split_with_sizes(all_neighbors.mask, group_sizes, dim=0)

        hop2_id_splits = hop2_time_splits = hop2_feat_splits = hop2_mask_splits = None
        if k2 > 0:
            hop2_id_splits = torch.split_with_sizes(all_neighbors.hop2_ids, group_sizes, dim=0)
            hop2_time_splits = torch.split_with_sizes(all_neighbors.hop2_times, group_sizes, dim=0)
            hop2_feat_splits = torch.split_with_sizes(all_neighbors.hop2_feats, group_sizes, dim=0)
            hop2_mask_splits = torch.split_with_sizes(all_neighbors.hop2_mask, group_sizes, dim=0)

        def _build_nbr(idx: int) -> NeighborData:
            nd = NeighborData(splits[idx], time_splits[idx], feat_splits[idx], mask_splits[idx])
            if k2 > 0:
                nd.hop2_ids = hop2_id_splits[idx]
                nd.hop2_times = hop2_time_splits[idx]
                nd.hop2_feats = hop2_feat_splits[idx]
                nd.hop2_mask = hop2_mask_splits[idx]
            return nd

        idx = 0
        src_nbrs = dst_nbrs = neg_nbrs = neg_src_nbrs = None

        if "src" in self.spec.neighbors.for_nodes:
            src_nbrs = _build_nbr(idx)
            idx += 1
        if "dst" in self.spec.neighbors.for_nodes:
            dst_nbrs = _build_nbr(idx)
            idx += 1
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg is not None:
            neg_nbrs = _build_nbr(idx)
            idx += 1
        if "neg" in self.spec.neighbors.for_nodes and raw_batch.neg_src is not None:
            neg_src_nbrs = _build_nbr(idx)
            idx += 1

        co_occur = None
        if self.spec.co_occurrence:
            if src_nbrs is not None and dst_nbrs is not None:
                co_occur = DataPipeline._co_occur_from_neighbors(src_nbrs, dst_nbrs)
            else:
                co_occur = self.graph.co_neighbors(raw_batch.src, raw_batch.dst, raw_batch.time, k=k)

        return PreparedBatch(
            src=raw_batch.src,
            dst=raw_batch.dst,
            neg=raw_batch.neg if raw_batch.neg is not None else torch.empty(0, device=device),
            time=raw_batch.time,
            src_neighbors=src_nbrs,
            dst_neighbors=dst_nbrs,
            neg_neighbors=neg_nbrs,
            neg_src_neighbors=neg_src_nbrs,
            neg_src=raw_batch.neg_src,
            co_occurrence=co_occur,
            node_labels=raw_batch.node_labels,
            edge_labels=raw_batch.edge_labels,
            node_feat=raw_batch.node_feat,
        )

    def prepare_with_hist_neg(self, raw_batch: RawBatch, num_nodes: int) -> PreparedBatch:
        """Fused prepare + historical negative sampling (k=32 ring buffer approximation).

        src neighbors are queried ONCE and reused for both the model input
        and historical negative selection, eliminating the redundant
        graph.recent(src) call that HistoricalNegative.sample() would make.

        NOTE: Uses k=32 most-recent neighbors as negative candidates — this is
        a fast approximation, not semantically equivalent to full-history sampling.
        Use DataPipeline.prepare() with a HistoricalNegPool for correct semantics.

        Total kernel calls: 2
          Phase 1 — graph.recent([src, dst], ...)  →  src_nbrs, dst_nbrs
          Phase 2 — graph.recent([neg], ...)        →  neg_nbrs
        """
        k = self.spec.neighbors.k
        B = raw_batch.src.shape[0]
        t = raw_batch.time

        # Phase 1: fused src + dst query
        sd_all = self.graph.recent(
            torch.cat([raw_batch.src, raw_batch.dst]),
            torch.cat([t, t]),
            k,
        )
        src_nbrs = NeighborData(
            sd_all.neighbor_ids[:B],
            sd_all.timestamps[:B],
            sd_all.edge_feats[:B],
            sd_all.mask[:B],
        )
        dst_nbrs = NeighborData(
            sd_all.neighbor_ids[B:],
            sd_all.timestamps[B:],
            sd_all.edge_feats[B:],
            sd_all.mask[B:],
        )

        # Approximate historical neg from src_nbrs — no extra kernel call
        neg = HistoricalNegative.sample_from_neighbors(src_nbrs, num_nodes)

        # Phase 2: neg neighbor query
        neg_nbrs_raw = self.graph.recent(neg, t, k)
        neg_nbrs = NeighborData(
            neg_nbrs_raw.neighbor_ids,
            neg_nbrs_raw.timestamps,
            neg_nbrs_raw.edge_feats,
            neg_nbrs_raw.mask,
        )

        co_occur = None
        if self.spec.co_occurrence:
            co_occur = DataPipeline._co_occur_from_neighbors(src_nbrs, dst_nbrs)

        return PreparedBatch(
            src=raw_batch.src,
            dst=raw_batch.dst,
            neg=neg,
            time=t,
            src_neighbors=src_nbrs,
            dst_neighbors=dst_nbrs,
            neg_neighbors=neg_nbrs,
            co_occurrence=co_occur,
            node_labels=raw_batch.node_labels,
            edge_labels=raw_batch.edge_labels,
            node_feat=raw_batch.node_feat,
        )

    @staticmethod
    def _co_occur_from_neighbors(src_nbrs: NeighborData, dst_nbrs: NeighborData) -> Tensor:
        """Compute co-occurrence counts from already-queried neighbor sets.

        Avoids a redundant graph.recent() call — reuses src/dst NeighborData
        that the main fused query already produced.

        Returns (B,) float tensor of shared-neighbor counts.
        """
        src_ids = src_nbrs.neighbor_ids   # (B, k) int32
        dst_ids = dst_nbrs.neighbor_ids   # (B, k) int32
        eq = src_ids.unsqueeze(2) == dst_ids.unsqueeze(1)           # (B, k, k)
        valid = src_nbrs.mask.unsqueeze(2) & dst_nbrs.mask.unsqueeze(1)
        return (eq & valid).any(dim=2).float().sum(dim=1)            # (B,)


# Import after DataPipeline is defined to avoid circular import
from tgengine.pipeline.async_pipeline import AsyncDataPipeline  # noqa: E402
