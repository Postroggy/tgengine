"""Benchmark: O(K^2) broadcast vs O(K log K) sort-based co-occurrence counting.

Tests correctness and speed of a sort-based implementation that avoids
the (B, K, K) broadcast tensor.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

PADDING_ID = -1


def co_occurrence_broadcast(a_ids: Tensor, b_ids: Tensor):
    """Current O(K²) implementation — creates (B, K, K) tensors."""
    a_pad = a_ids == PADDING_ID
    b_pad = b_ids == PADDING_ID

    a_self = (a_ids.unsqueeze(1) == a_ids.unsqueeze(2)).float()
    b_self = (b_ids.unsqueeze(1) == b_ids.unsqueeze(2)).float()
    cross = (a_ids.unsqueeze(1) == b_ids.unsqueeze(2)).float()

    a_self = a_self.masked_fill(a_pad.unsqueeze(2) | a_pad.unsqueeze(1), 0.0)
    b_self = b_self.masked_fill(b_pad.unsqueeze(2) | b_pad.unsqueeze(1), 0.0)
    cross = cross.masked_fill(a_pad.unsqueeze(2) | b_pad.unsqueeze(1), 0.0)

    a_freq = torch.stack([a_self.sum(1), cross.sum(2)], dim=2)
    b_freq = torch.stack([cross.sum(1), b_self.sum(1)], dim=2)
    return a_freq, b_freq


def co_occurrence_scatter(a_ids: Tensor, b_ids: Tensor, num_nodes: int):
    """O(K) scatter-based co-occurrence counting.

    Matches broadcast semantics exactly:
    - a_freq[p, 0] = count of non-pad a-positions with same value as a[p]
    - a_freq[p, 1] = count of b-non-padded positions j where a[j]==b[p]
    - b_freq[p, 0] = count of a-non-padded positions i where b[i]==a[p]
    - b_freq[p, 1] = count of non-pad b-positions with same value as b[p]
    """
    B, K = a_ids.shape
    device = a_ids.device

    a_pad = a_ids == PADDING_ID
    b_pad = b_ids == PADDING_ID

    # Shift IDs: padding (-1) → 0, real → 1..num_nodes
    N = num_nodes + 1
    a_sh = (a_ids + 1).long()
    b_sh = (b_ids + 1).long()
    a_sh[a_pad] = 0
    b_sh[b_pad] = 0

    # Self counts: weight = own non-pad mask
    a_self_tab = torch.zeros(B, N, device=device)
    a_self_tab.scatter_add_(1, a_sh, (~a_pad).float())
    a_self = a_self_tab.gather(1, a_sh)
    a_self[a_pad] = 0

    b_self_tab = torch.zeros(B, N, device=device)
    b_self_tab.scatter_add_(1, b_sh, (~b_pad).float())
    b_self = b_self_tab.gather(1, b_sh)
    b_self[b_pad] = 0

    # Cross counts: scatter a-values weighted by (~b_pad), gather at b-values
    # a_freq[p,1] = count of j where ~b_pad[j] AND a[j]==b[p]
    a_cross_tab = torch.zeros(B, N, device=device)
    a_cross_tab.scatter_add_(1, a_sh, (~b_pad).float())  # weight by b's non-pad at same position
    a_cross = a_cross_tab.gather(1, b_sh)  # look up b[p]'s value
    a_cross[a_pad] = 0  # zero if a_pad[p] (which here means position p in a is pad)

    # b_freq[p,0] = count of i where ~a_pad[i] AND b[i]==a[p]
    b_cross_tab = torch.zeros(B, N, device=device)
    b_cross_tab.scatter_add_(1, b_sh, (~a_pad).float())  # weight by a's non-pad at same position
    b_cross = b_cross_tab.gather(1, a_sh)  # look up a[p]'s value
    b_cross[b_pad] = 0

    a_freq = torch.stack([a_self, a_cross], dim=2)  # (B, K, 2)
    b_freq = torch.stack([b_cross, b_self], dim=2)  # (B, K, 2)
    return a_freq, b_freq


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Test correctness
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)
    torch.manual_seed(42)
    B, K = 8, 32
    a = torch.randint(0, 10, (B, K), device=device)
    b = torch.randint(0, 10, (B, K), device=device)
    # Add some padding
    a[0, 25:] = PADDING_ID
    b[0, 20:] = PADDING_ID

    ref_a, ref_b = co_occurrence_broadcast(a, b)
    new_a, new_b = co_occurrence_scatter(a, b, num_nodes=2000)

    a_match = torch.allclose(ref_a, new_a, atol=1e-5)
    b_match = torch.allclose(ref_b, new_b, atol=1e-5)
    print(f"  a_freq match: {a_match}")
    print(f"  b_freq match: {b_match}")
    if not a_match:
        diff = (ref_a - new_a).abs().max()
        print(f"  Max diff a: {diff.item()}")
        idx = (ref_a - new_a).abs().argmax()
        print(f"  First mismatch at flat idx {idx}: ref={ref_a.flatten()[idx]}, new={new_a.flatten()[idx]}")
    if not b_match:
        diff = (ref_b - new_b).abs().max()
        print(f"  Max diff b: {diff.item()}")

    # Larger correctness test
    B, K = 200, 512
    a = torch.randint(0, 2000, (B, K), device=device)
    b = torch.randint(0, 2000, (B, K), device=device)
    a[:, -50:] = PADDING_ID
    b[:, -30:] = PADDING_ID

    ref_a, ref_b = co_occurrence_broadcast(a, b)
    new_a, new_b = co_occurrence_scatter(a, b, num_nodes=2000)
    print(f"\n  Large test (B={B}, K={K}):")
    print(f"  a_freq match: {torch.allclose(ref_a, new_a, atol=1e-5)}")
    print(f"  b_freq match: {torch.allclose(ref_b, new_b, atol=1e-5)}")

    # Speed comparison
    print()
    print("=" * 60)
    print("SPEED COMPARISON")
    print("=" * 60)

    configs = [(200, 32), (200, 64), (200, 256), (200, 512)]
    N_WARMUP = 10
    N_ITER = 50

    for B, K in configs:
        a = torch.randint(0, 2000, (B, K), device=device)
        b = torch.randint(0, 2000, (B, K), device=device)
        a[:, -10:] = PADDING_ID
        b[:, -10:] = PADDING_ID

        # Warmup + bench broadcast
        for _ in range(N_WARMUP):
            co_occurrence_broadcast(a, b)
        torch.cuda.synchronize()

        times_bc = []
        for _ in range(N_ITER):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            co_occurrence_broadcast(a, b)
            torch.cuda.synchronize()
            times_bc.append(time.perf_counter() - t0)

        # Warmup + bench sort
        for _ in range(N_WARMUP):
            co_occurrence_scatter(a, b, num_nodes=2000)
        torch.cuda.synchronize()

        times_sort = []
        for _ in range(N_ITER):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            co_occurrence_scatter(a, b, num_nodes=2000)
            torch.cuda.synchronize()
            times_sort.append(time.perf_counter() - t0)

        bc_ms = np.mean(times_bc) * 1000
        sort_ms = np.mean(times_sort) * 1000
        speedup = bc_ms / sort_ms

        # Memory comparison
        torch.cuda.reset_peak_memory_stats()
        co_occurrence_broadcast(a, b)
        bc_mem = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()
        co_occurrence_scatter(a, b, num_nodes=2000)
        sort_mem = torch.cuda.max_memory_allocated() / 1e6

        print(f"\n  B={B}, K={K}:")
        print(f"    Broadcast (O(K²)):  {bc_ms:>7.2f} ms, peak ~{B*K*K*4*3/1e6:.0f} MB temp")
        print(f"    Sort (O(K log K)):  {sort_ms:>7.2f} ms, peak ~{B*K*2*4*3/1e6:.0f} MB temp")
        print(f"    Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
