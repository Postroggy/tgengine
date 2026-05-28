"""Kernels for temporal neighbor sampling: CUDA > Triton > PyTorch fallback."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

# --- Try loading CUDA extension (JIT compiled) ---
HAS_CUDA_EXT = False
_cuda_ext = None

try:
    from torch.utils.cpp_extension import load as _load_ext

    _csrc_dir = Path(__file__).parent / "csrc"
    if (_csrc_dir / "temporal_sample.cu").exists() and torch.cuda.is_available():
        _cuda_ext = _load_ext(
            name="tgengine_cuda",
            sources=[str(_csrc_dir / "temporal_sample.cu")],
            verbose=False,
        )
        HAS_CUDA_EXT = True
except Exception:
    pass

# --- Triton fallback ---
HAS_TRITON = False

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    pass


if HAS_TRITON:

    @triton.jit
    def _temporal_recent_k_kernel(
        offsets_ptr,
        nbr_times_ptr,
        query_nodes_ptr,
        query_times_ptr,
        out_csr_idx_ptr,
        out_mask_ptr,
        K: tl.constexpr,
        MAX_SEARCH_STEPS: tl.constexpr,
    ):
        query_idx = tl.program_id(0)

        node = tl.load(query_nodes_ptr + query_idx).to(tl.int64)
        query_time = tl.load(query_times_ptr + query_idx)

        start = tl.load(offsets_ptr + node)
        end = tl.load(offsets_ptr + node + 1)
        seg_len = end - start

        lo = tl.zeros([], dtype=tl.int64)
        hi = seg_len
        for _ in range(MAX_SEARCH_STEPS):
            active = lo < hi
            mid = tl.where(active, (lo + hi) // 2, lo)
            t = tl.load(nbr_times_ptr + start + mid)
            lo = tl.where(active & (t < query_time), mid + 1, lo)
            hi = tl.where(active & (t >= query_time), mid, hi)

        valid_count = lo
        take = tl.minimum(valid_count, K)

        slot_offsets = tl.arange(0, K)
        pos_in_window = slot_offsets - (K - take)
        is_valid = (pos_in_window >= 0) & (pos_in_window < take)

        csr_idx = start + valid_count - take + pos_in_window
        safe_csr_idx = tl.maximum(csr_idx, tl.zeros([K], dtype=tl.int64))

        out_base = query_idx * K
        tl.store(out_csr_idx_ptr + out_base + slot_offsets, safe_csr_idx)
        tl.store(out_mask_ptr + out_base + slot_offsets, is_valid.to(tl.int32))


# ============================================================================
# Python wrappers
# ============================================================================

def cuda_temporal_recent_k(
    offsets: Tensor,
    nbr_times: Tensor,
    nbr_ids: Tensor,
    nbr_feats: Tensor,
    query_nodes: Tensor,
    query_times: Tensor,
    k: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused CUDA kernel: binary search + feature copy in one launch."""
    query_nodes_i64 = query_nodes.to(torch.int64).contiguous()
    query_times_f64 = query_times.to(torch.float64).contiguous()

    results = _cuda_ext.temporal_recent_k(
        offsets, nbr_times, nbr_ids, nbr_feats,
        query_nodes_i64, query_times_f64, k,
    )
    out_ids, out_times, out_feats, out_mask = results
    return out_ids, out_times, out_feats, out_mask.bool()


def cuda_temporal_recent_2hop(
    offsets: Tensor,
    nbr_times: Tensor,
    nbr_ids: Tensor,
    nbr_feats: Tensor,
    query_nodes: Tensor,
    query_times: Tensor,
    k1: int,
    k2: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused CUDA 2-hop: hop1 + hop2 in two kernel launches (no Python overhead)."""
    query_nodes_i64 = query_nodes.to(torch.int64).contiguous()
    query_times_f64 = query_times.to(torch.float64).contiguous()

    results = _cuda_ext.temporal_recent_2hop(
        offsets, nbr_times, nbr_ids, nbr_feats,
        query_nodes_i64, query_times_f64, k1, k2,
    )
    # hop1: ids, times, feats, mask; hop2: ids, times, feats, mask
    h1_ids, h1_times, h1_feats, h1_mask = results[0], results[1], results[2], results[3]
    h2_ids, h2_times, h2_feats, h2_mask = results[4], results[5], results[6], results[7]
    return h1_ids, h1_times, h1_feats, h1_mask.bool(), h2_ids, h2_times, h2_feats, h2_mask.bool()


def cuda_co_neighbor_count(
    offsets: Tensor,
    nbr_times: Tensor,
    nbr_ids: Tensor,
    src_nodes: Tensor,
    dst_nodes: Tensor,
    query_times: Tensor,
    k: int,
) -> Tensor:
    """CUDA co-neighbor counting using shared memory."""
    src_i64 = src_nodes.to(torch.int64).contiguous()
    dst_i64 = dst_nodes.to(torch.int64).contiguous()
    times_f64 = query_times.to(torch.float64).contiguous()

    return _cuda_ext.co_neighbor_count(
        offsets, nbr_times, nbr_ids,
        src_i64, dst_i64, times_f64, k,
    )


def cuda_co_occurrence_freq(
    a_ids: Tensor,
    b_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """CUDA co-occurrence frequency for DyGFormer.

    Args:
        a_ids: (B, K) int32, -1 for padding
        b_ids: (B, K) int32, -1 for padding

    Returns:
        a_freq: (B, K, 2) — [self_count, cross_count] per position
        b_freq: (B, K, 2) — [cross_count, self_count] per position
    """
    a_ids_i32 = a_ids.to(torch.int32).contiguous()
    b_ids_i32 = b_ids.to(torch.int32).contiguous()
    results = _cuda_ext.co_occurrence_freq(a_ids_i32, b_ids_i32)
    return results[0], results[1]


def triton_temporal_recent_k(
    offsets: Tensor,
    nbr_times: Tensor,
    nbr_ids: Tensor,
    nbr_feats: Tensor,
    query_nodes: Tensor,
    query_times: Tensor,
    k: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Triton binary search + PyTorch gather."""
    N = query_nodes.shape[0]
    d_edge = nbr_feats.shape[1] if nbr_feats.dim() == 2 else 0
    device = query_nodes.device

    query_nodes_i64 = query_nodes.to(torch.int64).contiguous()
    query_times_f64 = query_times.to(torch.float64).contiguous()

    if N == 0:
        out_ids = torch.full((N, k), -1, dtype=torch.int32, device=device)
        out_times = torch.zeros(N, k, dtype=torch.float64, device=device)
        out_feats = torch.zeros(N, k, d_edge, dtype=torch.float32, device=device)
        out_mask = torch.zeros(N, k, dtype=torch.bool, device=device)
        return out_ids, out_times, out_feats, out_mask

    k_padded = 1 << int(math.ceil(math.log2(max(k, 1))))

    csr_idx = torch.zeros(N, k_padded, dtype=torch.int64, device=device)
    mask_i32 = torch.zeros(N, k_padded, dtype=torch.int32, device=device)

    max_deg = int((offsets[1:] - offsets[:-1]).max().item())
    max_search_steps = max(int(math.ceil(math.log2(max(max_deg, 1)))) + 1, 1)

    grid = (N,)
    _temporal_recent_k_kernel[grid](
        offsets, nbr_times, query_nodes_i64, query_times_f64,
        csr_idx, mask_i32,
        K=k_padded, MAX_SEARCH_STEPS=max_search_steps,
    )

    if k_padded != k:
        offset = k_padded - k
        csr_idx = csr_idx[:, offset:].contiguous()
        mask_i32 = mask_i32[:, offset:].contiguous()

    mask = mask_i32.bool()
    flat_idx = csr_idx.reshape(-1)
    out_ids = nbr_ids[flat_idx].reshape(N, k)
    out_times = nbr_times[flat_idx].reshape(N, k)
    out_feats = nbr_feats[flat_idx].reshape(N, k, d_edge)

    out_ids = out_ids.masked_fill(~mask, -1)
    out_times = out_times.masked_fill(~mask, 0.0)
    out_feats = out_feats.masked_fill(~mask.unsqueeze(-1), 0.0)

    return out_ids, out_times, out_feats, mask
