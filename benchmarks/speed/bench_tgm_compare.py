"""Benchmark: TGEngine vs TGM neighbor sampling speed.

Both use GPU-vectorized approaches:
- TGEngine: T-CSR + searchsorted + gather (full history, O(log N) per node)
- TGM: Ring buffer + unroll + time_mask + gather (bounded buffer, O(buffer_size) per node)

We isolate the core `_get_recency_neighbors` / `graph.recent()` operation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_tgm_compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
DATASETS = ["uci", "wikipedia", "lastfm"]
BATCH_SIZES = [200, 600, 1000]
K_VALUES = [32, 64, 256]
N_WARMUP = 10
N_ITER = 100

PADDED_NODE_ID = -1


# ============================================================================
# TGM RecencyNeighborHook — core query logic extracted (GPU vectorized)
# ============================================================================

class TGMRecencyBuffer:
    """Minimal reproduction of TGM's RecencyNeighborHook buffer + query."""

    def __init__(self, num_nodes: int, max_nbrs: int, edge_feat_dim: int, device: str):
        self.num_nodes = num_nodes
        self.max_nbrs = max_nbrs
        self.edge_feat_dim = edge_feat_dim
        self.device = device

        self._nbr_ids = torch.full((num_nodes, max_nbrs), PADDED_NODE_ID,
                                   dtype=torch.int32, device=device)
        self._nbr_times = torch.zeros((num_nodes, max_nbrs), dtype=torch.float64, device=device)
        self._nbr_feats = torch.zeros((num_nodes, max_nbrs, edge_feat_dim),
                                      dtype=torch.float32, device=device)
        self._write_pos = torch.zeros(num_nodes, dtype=torch.int32, device=device)

    def update(self, src: Tensor, dst: Tensor, times: Tensor, feats: Tensor):
        """Batch update (bidirectional), matching TGM's _update."""
        node_ids = torch.cat([src, dst])
        nbr_nids = torch.cat([dst, src])
        seed_times = torch.cat([times, times])
        edge_feats = torch.cat([feats, feats])

        max_time = seed_times.max() + 1
        composite_key = node_ids.long() * int(max_time.item()) + seed_times.long()
        perm = torch.argsort(composite_key, stable=True)

        sorted_nodes = node_ids[perm]
        sorted_nbr_ids = nbr_nids[perm]
        sorted_times = seed_times[perm]
        sorted_feats = edge_feats[perm]

        B = self.max_nbrs
        _, inv, cnts = torch.unique_consecutive(sorted_nodes, return_inverse=True, return_counts=True)
        cumcnts = torch.cat([torch.tensor([0], device=self.device), cnts.cumsum(0)[:-1]])
        pos_in_group = torch.arange(len(sorted_nodes), device=self.device) - cumcnts[inv]
        mask = pos_in_group >= (cnts[inv] - B)

        sorted_nodes = sorted_nodes[mask]
        sorted_nbr_ids = sorted_nbr_ids[mask]
        sorted_times = sorted_times[mask]
        sorted_feats = sorted_feats[mask]

        _, inv, cnts = torch.unique_consecutive(sorted_nodes, return_inverse=True, return_counts=True)
        cum_cnts = torch.cat([torch.tensor([0], device=self.device), cnts[:-1]]).cumsum(dim=0)
        offsets = torch.arange(len(sorted_nodes), device=self.device) - cum_cnts[inv]

        write_idx = (self._write_pos[sorted_nodes].long() + offsets) % self.max_nbrs
        self._nbr_ids[sorted_nodes.long(), write_idx] = sorted_nbr_ids.int()
        self._nbr_times[sorted_nodes.long(), write_idx] = sorted_times.double()
        self._nbr_feats[sorted_nodes.long(), write_idx, :] = sorted_feats

        num_writes = torch.ones_like(sorted_nodes, dtype=torch.int32, device=self.device)
        self._write_pos.scatter_add_(0, sorted_nodes.long(), num_writes)

    def get_recency_neighbors(self, node_ids: Tensor, query_times: Tensor, k: int):
        """Core query — exact TGM logic."""
        B = self.max_nbrs
        N = len(node_ids)

        nbr_nids = self._nbr_ids[node_ids.long()]      # (N, B)
        nbr_edge_time = self._nbr_times[node_ids.long()]  # (N, B)
        nbr_edge_x = self._nbr_feats[node_ids.long()]    # (N, B, d)
        write_pos = self._write_pos[node_ids.long()]      # (N,)

        # Unroll: oldest ... newest
        candidate_idx = write_pos[:, None].long() - torch.arange(B, 0, -1, device=self.device)
        candidate_idx %= B

        candidate_times = torch.gather(nbr_edge_time, 1, candidate_idx)
        time_mask = candidate_times < query_times[:, None].double()
        time_mask[torch.gather(nbr_nids, 1, candidate_idx) == PADDED_NODE_ID] = False

        pos = torch.arange(B, device=self.device)
        last_valid_pos = torch.where(
            time_mask.any(dim=1),
            (time_mask * pos).amax(dim=1),
            torch.full((N,), -1, device=self.device),
        )

        offset = torch.arange(k - 1, -1, -1, device=self.device)
        gather_pos = last_valid_pos[:, None] - offset[None, :]
        gather_pos = torch.clamp(gather_pos, min=-1)

        out_idx = torch.where(
            gather_pos >= 0,
            torch.gather(candidate_idx, 1, gather_pos.clamp(min=0)),
            torch.full_like(gather_pos, -1),
        )

        valid_mask = out_idx >= 0
        safe_idx = out_idx.clamp(min=0)

        out_nbrs = torch.gather(nbr_nids, 1, safe_idx)
        out_times = torch.gather(nbr_edge_time, 1, safe_idx)
        out_feats = torch.gather(nbr_edge_x, 1, safe_idx.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim))

        out_nbrs[~valid_mask] = PADDED_NODE_ID
        out_times[~valid_mask] = 0
        out_feats[~valid_mask] = 0.0

        return out_nbrs, out_times, out_feats


# ============================================================================
# Data loading (same as bench_sampling_speed.py)
# ============================================================================

def load_data(dataset_name):
    import pandas as pd
    data_path = Path(DATA_ROOT) / dataset_name / f"ml_{dataset_name}.csv"
    edge_feat_path = Path(DATA_ROOT) / dataset_name / f"ml_{dataset_name}.npy"

    df = pd.read_csv(data_path)
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    edge_feats = np.load(edge_feat_path)
    num_nodes = max(src.max(), dst.max()) + 1
    d_edge = edge_feats.shape[1]
    print(f"  {dataset_name}: {len(src):,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    return src, dst, timestamps, edge_feats, num_nodes, d_edge


# ============================================================================
# Benchmarks
# ============================================================================

def bench_tgm_query(tgm_buf, src_t, dst_t, time_t, num_edges, batch_size, K, device):
    """Benchmark TGM ring buffer query."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        idx = torch.randint(0, num_edges, (batch_size,), device=device)
        batch_src = src_t[idx]
        batch_dst = dst_t[idx]
        batch_time = time_t[idx]
        all_nodes = torch.cat([batch_src, batch_dst])
        all_times = torch.cat([batch_time, batch_time])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tgm_buf.get_recency_neighbors(all_nodes, all_times, K)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_tgengine_query(graph, src_t, dst_t, time_t, num_edges, batch_size, K, device):
    """Benchmark TGEngine T-CSR query."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        idx = torch.randint(0, num_edges, (batch_size,), device=device)
        batch_src = src_t[idx]
        batch_dst = dst_t[idx]
        batch_time = time_t[idx]
        all_nodes = torch.cat([batch_src, batch_dst])
        all_times = torch.cat([batch_time, batch_time])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.recent(all_nodes, all_times, K)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_tgm_update(tgm_buf, src_t, dst_t, time_t, feat_t, num_edges, batch_size, device):
    """Benchmark TGM buffer update (advance)."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        idx = torch.randint(0, num_edges, (batch_size,), device=device)
        bs = src_t[idx]
        bd = dst_t[idx]
        bt = time_t[idx]
        bf = feat_t[idx]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tgm_buf.update(bs, bd, bt, bf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_tgengine_advance(graph, src_t, dst_t, time_t, feat_t, num_edges, batch_size, device):
    """Benchmark TGEngine graph.advance() — only meaningful post-freeze (overflow buffer)."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        idx = torch.randint(0, num_edges, (batch_size,), device=device)
        bs = src_t[idx]
        bd = dst_t[idx]
        bt = time_t[idx]
        bf = feat_t[idx]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.advance(bs, bd, bt, bf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


# ============================================================================
# Main
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name() if device == 'cuda' else 'CPU'})")
    print(f"PyTorch {torch.__version__}")
    print()

    for dataset_name in DATASETS:
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        src, dst, timestamps, edge_feats, num_nodes, d_edge = load_data(dataset_name)
        num_edges = len(src)

        src_t = torch.from_numpy(src).long().to(device)
        dst_t = torch.from_numpy(dst).long().to(device)
        time_t = torch.from_numpy(timestamps).float().to(device)
        feat_t = torch.from_numpy(edge_feats).float().to(device)

        for K in K_VALUES:
            print(f"\n  --- K={K} ---")

            # Build TGM buffer (ring buffer size = K, matching use case)
            # TGM uses max_nbrs = K as the ring buffer size
            tgm_buf = TGMRecencyBuffer(num_nodes, max_nbrs=K, edge_feat_dim=d_edge, device=device)
            # Load all edges into TGM buffer
            CHUNK = 10000
            for i in range(0, num_edges, CHUNK):
                j = min(i + CHUNK, num_edges)
                tgm_buf.update(src_t[i:j], dst_t[i:j], time_t[i:j], feat_t[i:j])

            # Build TGEngine graph (T-CSR)
            from tgengine.core.temporal_graph import TemporalGraph
            graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
            graph.advance(src_t, dst_t, time_t, feat_t)
            graph.freeze_csr()

            # ---- Query benchmark ----
            print(f"  {'BS':>6} | {'TGM (ms)':>12} {'TGEngine (ms)':>14} {'Speedup':>8} | Notes")
            print(f"  {'-'*6} | {'-'*12} {'-'*14} {'-'*8} | -----")

            for bs in BATCH_SIZES:
                tgm_ms, tgm_std = bench_tgm_query(
                    tgm_buf, src_t, dst_t, time_t, num_edges, bs, K, device
                )
                tge_ms, tge_std = bench_tgengine_query(
                    graph, src_t, dst_t, time_t, num_edges, bs, K, device
                )
                speedup = tgm_ms / tge_ms if tge_ms > 0 else float("inf")
                label = "TGEngine faster" if speedup > 1 else "TGM faster"
                print(f"  {bs:>6} | {tgm_ms:>7.2f}±{tgm_std:.2f} {tge_ms:>8.2f}±{tge_std:.2f} {speedup:>7.2f}x | {label}")

            del tgm_buf, graph
            torch.cuda.empty_cache()

        # ---- Update/Advance benchmark (K=32 only) ----
        print(f"\n  --- Advance/Update speed (K=32 buffer) ---")
        K = 32
        tgm_buf = TGMRecencyBuffer(num_nodes, max_nbrs=K, edge_feat_dim=d_edge, device=device)
        from tgengine.core.temporal_graph import TemporalGraph
        graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
        graph.advance(src_t, dst_t, time_t, feat_t)
        graph.freeze_csr()

        print(f"  {'BS':>6} | {'TGM update (ms)':>16} {'TGEngine advance (ms)':>22} {'Speedup':>8}")
        print(f"  {'-'*6} | {'-'*16} {'-'*22} {'-'*8}")
        for bs in BATCH_SIZES:
            tgm_ms, tgm_std = bench_tgm_update(tgm_buf, src_t, dst_t, time_t, feat_t, num_edges, bs, device)
            tge_ms, tge_std = bench_tgengine_advance(graph, src_t, dst_t, time_t, feat_t, num_edges, bs, device)
            speedup = tgm_ms / tge_ms if tge_ms > 0 else float("inf")
            print(f"  {bs:>6} | {tgm_ms:>10.2f}±{tgm_std:.2f} {tge_ms:>14.2f}±{tge_std:.2f} {speedup:>7.2f}x")

        del tgm_buf, graph
        torch.cuda.empty_cache()
        print()

    print("\n" + "="*70)
    print("DESIGN COMPARISON")
    print("="*70)
    print("""
  TGM (Ring Buffer):
    Storage: Fixed-size ring buffer per node (num_nodes × max_nbrs)
    Query:   Unroll buffer → time_mask → find rightmost valid → gather last K
    Pros:    Bounded memory, O(max_nbrs) query
    Cons:    Loses history beyond buffer size, complex unroll logic

  TGEngine (T-CSR + Overflow):
    Storage: Sorted CSR (full history) + overflow ring buffer for post-freeze edges
    Query:   searchsorted to find time boundary → slice last K from CSR
    Pros:    Full history access, O(log degree) query, simpler logic
    Cons:    More memory (stores all edges), CSR is immutable after freeze

  Key insight: TGEngine trades memory for simplicity and correctness.
  Full history means K can be changed at query time without rebuilding.
""")


if __name__ == "__main__":
    main()
