"""Benchmark: TGEngine vs DyGLib neighbor sampling and negative sampling speed.

Compares:
1. Neighbor sampling: TGEngine (GPU T-CSR searchsorted) vs DyGLib (CPU Python loop + np.searchsorted)
2. Negative sampling: TGEngine (GPU randint) vs DyGLib (CPU random.choice)

Both use the same data (UCI, Wikipedia) loaded from DyGLib format.
We measure the core operations in isolation for a fair comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_sampling_speed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
DATASETS = ["uci", "wikipedia", "lastfm"]
BATCH_SIZES = [200, 600, 1000]
K_VALUES = [32, 64, 256]
N_WARMUP = 5
N_ITER = 50


# ============================================================================
# DyGLib NeighborSampler (pure Python/NumPy, per-node loop)
# ============================================================================

class DyGLibNeighborSampler:
    """Faithful reproduction of DyGLib's NeighborSampler for benchmarking."""

    def __init__(self, adj_list):
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        for node_idx, per_node_neighbors in enumerate(adj_list):
            sorted_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(
                np.array([x[0] for x in sorted_neighbors])
            )
            self.nodes_edge_ids.append(
                np.array([x[1] for x in sorted_neighbors])
            )
            self.nodes_neighbor_times.append(
                np.array([x[2] for x in sorted_neighbors])
            )

    def get_historical_neighbors(self, node_ids, node_interact_times, n_neighbors):
        """Get most recent K neighbors for a batch of nodes (DyGLib semantics)."""
        nodes_neighbor_ids_list = []
        nodes_edge_ids_list = []
        nodes_neighbor_times_list = []

        for idx, (node_id, interact_time) in enumerate(
            zip(node_ids, node_interact_times)
        ):
            node_id = int(node_id)
            # Binary search for time cutoff
            i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)
            # Take most recent K
            neighbor_ids = self.nodes_neighbor_ids[node_id][:i][-n_neighbors:]
            edge_ids = self.nodes_edge_ids[node_id][:i][-n_neighbors:]
            times = self.nodes_neighbor_times[node_id][:i][-n_neighbors:]

            # Pad to n_neighbors
            if len(neighbor_ids) < n_neighbors:
                pad_len = n_neighbors - len(neighbor_ids)
                neighbor_ids = np.concatenate([np.zeros(pad_len), neighbor_ids])
                edge_ids = np.concatenate([np.zeros(pad_len), edge_ids])
                times = np.concatenate([np.zeros(pad_len), times])

            nodes_neighbor_ids_list.append(neighbor_ids)
            nodes_edge_ids_list.append(edge_ids)
            nodes_neighbor_times_list.append(times)

        return (
            np.stack(nodes_neighbor_ids_list),
            np.stack(nodes_edge_ids_list),
            np.stack(nodes_neighbor_times_list),
        )


# ============================================================================
# Data loading
# ============================================================================

def load_data(dataset_name):
    """Load dataset in DyGLib format."""
    data_path = Path(DATA_ROOT) / dataset_name / f"ml_{dataset_name}.csv"
    edge_feat_path = Path(DATA_ROOT) / dataset_name / f"ml_{dataset_name}.npy"

    import pandas as pd
    df = pd.read_csv(data_path)
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    edge_feats = np.load(edge_feat_path)

    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = edge_feats.shape[1]

    print(f"  {dataset_name}: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    return src, dst, timestamps, edge_feats, num_nodes, d_edge


def build_dyglib_adj_list(src, dst, timestamps, num_nodes):
    """Build adjacency list as DyGLib does."""
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(len(src)):
        # Bidirectional
        adj_list[src[i]].append((dst[i], i, timestamps[i]))
        adj_list[dst[i]].append((src[i], i, timestamps[i]))
    return adj_list


# ============================================================================
# Benchmarks
# ============================================================================

def bench_dyglib_neighbor_sampling(sampler, src, dst, timestamps, num_edges, batch_size, K):
    """Benchmark DyGLib-style neighbor sampling (CPU, Python loop)."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        # Random batch
        idx = np.random.randint(0, num_edges, batch_size)
        batch_src = src[idx]
        batch_dst = dst[idx]
        batch_time = timestamps[idx]
        # Query for both src and dst (like DyGFormer)
        all_nodes = np.concatenate([batch_src, batch_dst])
        all_times = np.concatenate([batch_time, batch_time])

        t0 = time.perf_counter()
        sampler.get_historical_neighbors(all_nodes, all_times, K)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def bench_tgengine_neighbor_sampling(graph, src, dst, timestamps, num_edges, batch_size, K, device):
    """Benchmark TGEngine neighbor sampling (GPU T-CSR searchsorted)."""
    times = []
    src_t = torch.from_numpy(src).to(device)
    dst_t = torch.from_numpy(dst).to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)

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
    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def bench_dyglib_neg_sampling(num_nodes, batch_size):
    """DyGLib random negative sampling (CPU numpy)."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        t0 = time.perf_counter()
        np.random.randint(0, num_nodes, size=batch_size)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_tgengine_neg_sampling(num_nodes, batch_size, device):
    """TGEngine random negative sampling (GPU randint)."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.randint(0, num_nodes, (batch_size,), device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_dyglib_historical_neg(adj_list, src, dst, timestamps, num_edges, batch_size, num_nodes):
    """DyGLib historical negative: for each src, pick a random historical neighbor."""
    times = []
    for _ in range(N_WARMUP + N_ITER):
        idx = np.random.randint(0, num_edges, batch_size)
        batch_src = src[idx]
        batch_dst = dst[idx]

        t0 = time.perf_counter()
        negs = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            s = batch_src[i]
            nbrs = adj_list[s]
            if len(nbrs) > 0:
                choice_idx = np.random.randint(0, len(nbrs))
                negs[i] = nbrs[choice_idx][0]
            else:
                negs[i] = np.random.randint(0, num_nodes)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_tgengine_historical_neg(graph, src_t, dst_t, timestamps_t, num_edges, batch_size, device):
    """TGEngine historical negative via HistoricalNegPool (GPU)."""
    from tgengine.pipeline.negatives import HistoricalNegative

    hist_neg = HistoricalNegative(graph.num_nodes, device=device)
    # Populate pool with some history
    n_warmup_edges = min(num_edges, 5000)
    hist_neg.update(src_t[:n_warmup_edges], dst_t[:n_warmup_edges])

    times = []
    for _ in range(N_WARMUP + N_ITER):
        idx = torch.randint(0, num_edges, (batch_size,), device=device)
        batch_src = src_t[idx]
        batch_dst = src_t[idx]  # dummy
        batch_time = timestamps_t[idx]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hist_neg.sample(batch_src, batch_dst, batch_time, graph)
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

    results = []

    for dataset_name in DATASETS:
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        src, dst, timestamps, edge_feats, num_nodes, d_edge = load_data(dataset_name)
        num_edges = len(src)

        # Build DyGLib sampler
        print("  Building DyGLib adjacency list...")
        adj_list = build_dyglib_adj_list(src, dst, timestamps, num_nodes)
        dyglib_sampler = DyGLibNeighborSampler(adj_list)

        # Build TGEngine graph
        print("  Building TGEngine T-CSR graph...")
        from tgengine.core.temporal_graph import TemporalGraph
        graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
        # Load all edges (both directions, matching DyGLib)
        src_t = torch.from_numpy(src).long().to(device)
        dst_t = torch.from_numpy(dst).long().to(device)
        time_t = torch.from_numpy(timestamps).float().to(device)
        feat_t = torch.from_numpy(edge_feats).float().to(device)
        graph.advance(src_t, dst_t, time_t, feat_t)
        graph.freeze_csr()
        print(f"  Graph ready. CSR frozen: {graph._offsets is not None}")
        print()

        # ---- Neighbor Sampling ----
        print("  --- Neighbor Sampling (src+dst, 2*B queries) ---")
        print(f"  {'BS':>6} {'K':>4} | {'DyGLib (ms)':>14} {'TGEngine (ms)':>14} {'Speedup':>8}")
        print(f"  {'-'*6} {'-'*4} | {'-'*14} {'-'*14} {'-'*8}")

        for bs in BATCH_SIZES:
            for K in K_VALUES:
                dyglib_ms, dyglib_std = bench_dyglib_neighbor_sampling(
                    dyglib_sampler, src, dst, timestamps, num_edges, bs, K
                )
                tge_ms, tge_std = bench_tgengine_neighbor_sampling(
                    graph, src, dst, timestamps, num_edges, bs, K, device
                )
                speedup = dyglib_ms / tge_ms if tge_ms > 0 else float("inf")
                print(f"  {bs:>6} {K:>4} | {dyglib_ms:>8.2f}±{dyglib_std:.2f} {tge_ms:>8.2f}±{tge_std:.2f} {speedup:>7.1f}x")
                results.append({
                    "dataset": dataset_name, "op": "neighbor_sampling",
                    "batch_size": bs, "K": K,
                    "dyglib_ms": dyglib_ms, "tgengine_ms": tge_ms, "speedup": speedup,
                })

        # ---- Random Negative Sampling ----
        print()
        print("  --- Random Negative Sampling ---")
        print(f"  {'BS':>6} | {'DyGLib (ms)':>14} {'TGEngine (ms)':>14} {'Speedup':>8}")
        print(f"  {'-'*6} | {'-'*14} {'-'*14} {'-'*8}")

        for bs in BATCH_SIZES:
            dyglib_ms, dyglib_std = bench_dyglib_neg_sampling(num_nodes, bs)
            tge_ms, tge_std = bench_tgengine_neg_sampling(num_nodes, bs, device)
            speedup = dyglib_ms / tge_ms if tge_ms > 0 else float("inf")
            print(f"  {bs:>6} | {dyglib_ms:>8.4f}±{dyglib_std:.4f} {tge_ms:>8.4f}±{tge_std:.4f} {speedup:>7.1f}x")
            results.append({
                "dataset": dataset_name, "op": "random_neg",
                "batch_size": bs, "K": 0,
                "dyglib_ms": dyglib_ms, "tgengine_ms": tge_ms, "speedup": speedup,
            })

        # ---- Historical Negative Sampling ----
        print()
        print("  --- Historical Negative Sampling ---")
        print(f"  {'BS':>6} | {'DyGLib (ms)':>14} {'TGEngine (ms)':>14} {'Speedup':>8}")
        print(f"  {'-'*6} | {'-'*14} {'-'*14} {'-'*8}")

        for bs in BATCH_SIZES:
            dyglib_ms, dyglib_std = bench_dyglib_historical_neg(
                adj_list, src, dst, timestamps, num_edges, bs, num_nodes
            )
            tge_ms, tge_std = bench_tgengine_historical_neg(
                graph, src_t, dst_t, time_t, num_edges, bs, device
            )
            speedup = dyglib_ms / tge_ms if tge_ms > 0 else float("inf")
            print(f"  {bs:>6} | {dyglib_ms:>8.3f}±{dyglib_std:.3f} {tge_ms:>8.3f}±{tge_std:.3f} {speedup:>7.1f}x")
            results.append({
                "dataset": dataset_name, "op": "historical_neg",
                "batch_size": bs, "K": 0,
                "dyglib_ms": dyglib_ms, "tgengine_ms": tge_ms, "speedup": speedup,
            })

        print()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<12} {'Operation':<20} {'Avg Speedup':>12}")
    print(f"{'-'*12} {'-'*20} {'-'*12}")

    for dataset_name in DATASETS:
        for op in ["neighbor_sampling", "random_neg", "historical_neg"]:
            op_results = [r for r in results if r["dataset"] == dataset_name and r["op"] == op]
            if op_results:
                avg_speedup = np.mean([r["speedup"] for r in op_results])
                print(f"{dataset_name:<12} {op:<20} {avg_speedup:>10.1f}x")


if __name__ == "__main__":
    main()
