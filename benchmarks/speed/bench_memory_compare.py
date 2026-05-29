"""Memory comparison: TGEngine T-CSR vs TGM Ring Buffer on Reddit.

Measures peak GPU memory after loading all edges, for both approaches.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_memory_compare.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
K = 256
PADDED_NODE_ID = -1


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reddit")
    args, _ = parser.parse_known_args()
    dataset_name = args.dataset

    df = pd.read_csv(Path(DATA_ROOT) / dataset_name / f"ml_{dataset_name}.csv")
    feats = np.load(Path(DATA_ROOT) / dataset_name / f"ml_{dataset_name}.npy")
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = feats.shape[1]

    avg_degree = 2 * num_edges / num_nodes
    print(f"{dataset_name}: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    print(f"Average degree (bidirectional): {avg_degree:.1f}")
    print(f"K={K}")
    print()

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    # Baseline: raw data on GPU
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated() / 1e6
    print(f"Baseline (raw tensors on GPU): {baseline_mem:.1f} MB")
    print()

    # =========================================================================
    # TGM Ring Buffer
    # =========================================================================
    print("=" * 60)
    print("TGM Ring Buffer (num_nodes × K)")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    # Allocate TGM-style ring buffer
    tgm_nbr_ids = torch.full((num_nodes, K), PADDED_NODE_ID, dtype=torch.int32, device=device)
    tgm_nbr_times = torch.zeros((num_nodes, K), dtype=torch.float64, device=device)
    tgm_nbr_feats = torch.zeros((num_nodes, K, d_edge), dtype=torch.float32, device=device)
    tgm_write_pos = torch.zeros(num_nodes, dtype=torch.int32, device=device)

    mem_after = torch.cuda.memory_allocated()
    tgm_buffer_mem = (mem_after - mem_before) / 1e6

    # Theoretical calculation
    tgm_theory = num_nodes * K * (4 + 8 + d_edge * 4 + 0) / 1e6  # int32 + float64 + float32*d + write_pos amortized
    print(f"  Buffer allocation: {tgm_buffer_mem:.1f} MB")
    print(f"  Theoretical: num_nodes({num_nodes:,}) × K({K}) × (4+8+{d_edge}×4) = {tgm_theory:.1f} MB")
    print(f"  Breakdown:")
    print(f"    nbr_ids:   {num_nodes * K * 4 / 1e6:.1f} MB (int32)")
    print(f"    nbr_times: {num_nodes * K * 8 / 1e6:.1f} MB (float64)")
    print(f"    nbr_feats: {num_nodes * K * d_edge * 4 / 1e6:.1f} MB (float32×{d_edge})")

    # Cleanup
    del tgm_nbr_ids, tgm_nbr_times, tgm_nbr_feats, tgm_write_pos
    torch.cuda.empty_cache()

    # =========================================================================
    # TGEngine T-CSR
    # =========================================================================
    print()
    print("=" * 60)
    print("TGEngine T-CSR (full history)")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    from tgengine.core.temporal_graph import TemporalGraph
    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t, dst_t, time_t, feat_t)
    graph.freeze_csr()

    mem_after = torch.cuda.memory_allocated()
    tge_csr_mem = (mem_after - mem_before) / 1e6

    # Count actual CSR edges (bidirectional)
    total_csr_edges = graph._offsets[-1].item() if graph._offsets is not None else 0
    tge_theory = total_csr_edges * (4 + 8 + d_edge * 4) / 1e6 + (num_nodes + 1) * 8 / 1e6
    print(f"  CSR allocation: {tge_csr_mem:.1f} MB")
    print(f"  CSR edges (bidirectional): {total_csr_edges:,}")
    print(f"  Theoretical: edges({total_csr_edges:,}) × (4+8+{d_edge}×4) + offsets = {tge_theory:.1f} MB")
    print(f"  Breakdown:")
    print(f"    offsets:   {(num_nodes + 1) * 8 / 1e6:.1f} MB (int64)")
    print(f"    nbr_ids:   {total_csr_edges * 4 / 1e6:.1f} MB (int32)")
    print(f"    nbr_times: {total_csr_edges * 8 / 1e6:.1f} MB (float64)")
    print(f"    nbr_feats: {total_csr_edges * d_edge * 4 / 1e6:.1f} MB (float32×{d_edge})")

    del graph
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  TGM Ring Buffer:  {tgm_buffer_mem:>8.1f} MB")
    print(f"  TGEngine T-CSR:   {tge_csr_mem:>8.1f} MB")
    ratio = tge_csr_mem / tgm_buffer_mem if tgm_buffer_mem > 0 else float("inf")
    print(f"  Ratio (TGEngine/TGM): {ratio:.2f}x")
    print()
    print(f"  Available on RTX 4080: 16,384 MB")
    print(f"  Remaining after TGM:  {16384 - tgm_buffer_mem:.0f} MB")
    print(f"  Remaining after TGE:  {16384 - tge_csr_mem:.0f} MB")
    print()

    # What if K was smaller?
    print("  --- Sensitivity to K (TGM only, TGEngine is K-independent) ---")
    for k in [32, 64, 128, 256, 512]:
        mem = num_nodes * k * (4 + 8 + d_edge * 4) / 1e6
        print(f"    K={k:>3}: TGM buffer = {mem:.1f} MB")


if __name__ == "__main__":
    main()
