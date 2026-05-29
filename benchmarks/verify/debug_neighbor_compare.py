"""Compare neighbor sampling between DyGLib and TGEngine.

Loads UCI dataset, builds both systems' neighbor structures from the SAME
training data, then queries the same nodes at the same times and compares
the returned neighbor IDs, timestamps, and edge features.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/debug_neighbor_compare.py
"""
import sys
sys.path.insert(0, "/mnt/home/gyq/CodeBase/Graph/DG_Data/../exp_sourcecode")
sys.path.insert(0, ".")

import numpy as np
import torch
from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
DATASET = "uci"
K = 31  # DyGLib max_input_sequence_length - 1 for UCI


def build_dyglib_sampler(ds):
    """Build DyGLib-style NeighborSampler from training data."""
    src = ds.src[:ds.train_end].numpy()
    dst = ds.dst[:ds.train_end].numpy()
    times = ds.time[:ds.train_end].numpy()
    num_nodes = ds.num_nodes

    # adj_list[node] = list of (neighbor_id, edge_id, timestamp), sorted by time
    adj_list = [[] for _ in range(num_nodes)]
    for idx in range(len(src)):
        adj_list[src[idx]].append((dst[idx], idx, times[idx]))
        adj_list[dst[idx]].append((src[idx], idx, times[idx]))

    # Sort by timestamp (DyGLib does this)
    nodes_neighbor_ids = []
    nodes_edge_ids = []
    nodes_neighbor_times = []
    for node_idx in range(num_nodes):
        sorted_nbrs = sorted(adj_list[node_idx], key=lambda x: x[2])
        nodes_neighbor_ids.append(np.array([x[0] for x in sorted_nbrs]))
        nodes_edge_ids.append(np.array([x[1] for x in sorted_nbrs]))
        nodes_neighbor_times.append(np.array([x[2] for x in sorted_nbrs]))

    return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times


def dyglib_query(nodes_neighbor_ids, nodes_neighbor_times, node_id, query_time, k):
    """DyGLib-style neighbor query: all time < query_time, take most recent k."""
    times = nodes_neighbor_times[node_id]
    i = np.searchsorted(times, query_time)  # side='left': returns first index where times[i] >= query_time
    # So times[:i] are all < query_time
    nbr_ids = nodes_neighbor_ids[node_id][:i]
    nbr_times = nodes_neighbor_times[node_id][:i]

    if len(nbr_ids) > k:
        nbr_ids = nbr_ids[-k:]
        nbr_times = nbr_times[-k:]

    return nbr_ids, nbr_times


def main():
    print("Loading UCI dataset...")
    ds = load_dataset(DATASET, DATA_ROOT)

    # Build DyGLib sampler from training data
    print("Building DyGLib neighbor sampler...")
    dyglib_nbr_ids, dyglib_edge_ids, dyglib_nbr_times = build_dyglib_sampler(ds)

    # Build TGEngine graph by replaying training edges (same as Engine does)
    print("Building TGEngine TemporalGraph...")
    device = "cuda"

    graph = TemporalGraph(ds.num_nodes,
                          edge_feat_dim=ds.edge_feat_dim, device=device)

    # Advance ALL training edges at once (same as preload)
    chunk = 10000
    for start in range(0, ds.train_end, chunk):
        end = min(start + chunk, ds.train_end)
        s = ds.src[start:end].to(device)
        d = ds.dst[start:end].to(device)
        t = ds.time[start:end].to(device)
        f = ds.edge_feat[start:end].to(device) if ds.edge_feat is not None else None
        graph.advance(s, d, t, f)
    graph.freeze_csr()

    # Now compare queries for multiple nodes at various times
    print(f"\nComparing neighbor queries (K={K})...")
    print("=" * 70)

    # Pick some active nodes and query times from the latter part of training
    test_queries = []
    # Use edges from the last 10% of training as query points
    late_start = int(ds.train_end * 0.9)
    for idx in range(late_start, min(late_start + 20, ds.train_end)):
        src_id = ds.src[idx].item()
        query_t = ds.time[idx].item()
        test_queries.append((src_id, query_t))

    mismatches = 0
    total = 0
    for node_id, query_time in test_queries:
        total += 1
        # DyGLib query
        dyglib_ids, dyglib_times = dyglib_query(
            dyglib_nbr_ids, dyglib_nbr_times, node_id, query_time, K
        )

        # TGEngine query
        nodes_tensor = torch.tensor([node_id], dtype=torch.long, device=device)
        times_tensor = torch.tensor([query_time], dtype=torch.float64, device=device)
        tge_result = graph.recent(nodes_tensor, times_tensor, K)
        tge_ids = tge_result.neighbor_ids[0].cpu().numpy()
        tge_times = tge_result.timestamps[0].cpu().numpy()
        tge_mask = tge_result.mask[0].cpu().numpy()

        # Filter valid TGEngine results
        valid_tge_ids = tge_ids[tge_mask]
        valid_tge_times = tge_times[tge_mask]

        # Compare
        match = (len(dyglib_ids) == len(valid_tge_ids) and
                 np.array_equal(dyglib_ids, valid_tge_ids) and
                 np.allclose(dyglib_times, valid_tge_times, atol=1e-6))

        if not match:
            mismatches += 1
            if mismatches <= 5:
                print(f"\nMISMATCH for node={node_id}, query_time={query_time:.4f}")
                print(f"  DyGLib:   {len(dyglib_ids)} neighbors")
                print(f"    IDs:    {dyglib_ids[:10]}...")
                print(f"    Times:  {dyglib_times[:10]}...")
                print(f"  TGEngine: {len(valid_tge_ids)} neighbors")
                print(f"    IDs:    {valid_tge_ids[:10]}...")
                print(f"    Times:  {valid_tge_times[:10]}...")
                if len(dyglib_ids) > 0 and len(valid_tge_ids) > 0:
                    # Check ordering
                    print(f"  DyGLib last 5 IDs: {dyglib_ids[-5:]}")
                    print(f"  TGEngine last 5 IDs: {valid_tge_ids[-5:]}")

    print(f"\n{'=' * 70}")
    print(f"Results: {mismatches}/{total} mismatches")
    if mismatches == 0:
        print("PASS: TGEngine and DyGLib return identical neighbors!")
    else:
        print("FAIL: Neighbor sampling differs between implementations")


if __name__ == "__main__":
    main()
