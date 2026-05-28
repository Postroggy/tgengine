#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// Kernel 1: Fused temporal neighbor sampling (1-hop)
// ============================================================================

__global__ void temporal_recent_k_kernel(
    const int64_t* __restrict__ offsets,
    const double*  __restrict__ nbr_times,
    const int32_t* __restrict__ nbr_ids,
    const float*   __restrict__ nbr_feats,
    const int64_t* __restrict__ query_nodes,
    const double*  __restrict__ query_times,
    int32_t* __restrict__ out_ids,
    double*  __restrict__ out_times,
    float*   __restrict__ out_feats,
    int8_t*  __restrict__ out_mask,
    const int K,
    const int d_edge
) {
    const int query_idx = blockIdx.x;

    __shared__ int64_t sh_start;
    __shared__ int64_t sh_take;
    __shared__ int64_t sh_window_start;  // CSR index where valid window begins

    const int64_t node = query_nodes[query_idx];
    const double qtime = query_times[query_idx];

    if (threadIdx.x == 0) {
        const int64_t start = offsets[node];
        const int64_t end = offsets[node + 1];
        sh_start = start;

        int64_t lo = 0, hi = end - start;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (nbr_times[start + mid] < qtime)
                lo = mid + 1;
            else
                hi = mid;
        }
        int64_t valid_count = lo;
        int64_t take = valid_count < (int64_t)K ? valid_count : (int64_t)K;
        sh_take = take;
        sh_window_start = start + valid_count - take;
    }
    __syncthreads();

    const int64_t take = sh_take;
    const int64_t window_start = sh_window_start;

    for (int slot = threadIdx.x; slot < K; slot += blockDim.x) {
        const int64_t pos = slot - (K - (int)take);
        const int64_t out_base = (int64_t)query_idx * K + slot;

        if (pos >= 0 && pos < take) {
            const int64_t csr_idx = window_start + pos;
            out_ids[out_base] = nbr_ids[csr_idx];
            out_times[out_base] = nbr_times[csr_idx];
            out_mask[out_base] = 1;

            const float* src_f = nbr_feats + csr_idx * d_edge;
            float* dst_f = out_feats + out_base * d_edge;
            for (int d = 0; d < d_edge; d++) {
                dst_f[d] = src_f[d];
            }
        } else {
            out_ids[out_base] = -1;
            out_times[out_base] = 0.0;
            out_mask[out_base] = 0;

            float* dst_f = out_feats + out_base * d_edge;
            for (int d = 0; d < d_edge; d++) {
                dst_f[d] = 0.f;
            }
        }
    }
}

// ============================================================================
// Kernel 2: Fused 2-hop temporal neighbor sampling
// Each block handles one (query_node, hop1_slot) pair for the 2nd hop.
// The 1st hop is done by temporal_recent_k_kernel, then this kernel takes
// the hop1 results and gathers hop2 for each hop1 neighbor.
// ============================================================================

__global__ void temporal_recent_2hop_kernel(
    const int64_t* __restrict__ offsets,
    const double*  __restrict__ nbr_times,
    const int32_t* __restrict__ nbr_ids,
    const float*   __restrict__ nbr_feats,
    // Hop1 results (input)
    const int32_t* __restrict__ hop1_ids,    // [N, K1]
    const double*  __restrict__ hop1_times,  // [N, K1]
    const int8_t*  __restrict__ hop1_mask,   // [N, K1]
    // Hop2 output
    int32_t* __restrict__ hop2_ids,          // [N, K1, K2]
    double*  __restrict__ hop2_times,        // [N, K1, K2]
    float*   __restrict__ hop2_feats,        // [N, K1, K2, d_edge]
    int8_t*  __restrict__ hop2_mask,         // [N, K1, K2]
    const int K1,
    const int K2,
    const int d_edge
) {
    // blockIdx.x = query_idx * K1 + hop1_slot
    const int flat_idx = blockIdx.x;
    const int query_idx = flat_idx / K1;
    const int hop1_slot = flat_idx % K1;

    const int64_t hop1_base = (int64_t)query_idx * K1 + hop1_slot;

    // Check if hop1 neighbor is valid
    if (hop1_mask[hop1_base] == 0) {
        // Fill hop2 with padding
        for (int slot = threadIdx.x; slot < K2; slot += blockDim.x) {
            const int64_t out_base = ((int64_t)query_idx * K1 + hop1_slot) * K2 + slot;
            hop2_ids[out_base] = -1;
            hop2_times[out_base] = 0.0;
            hop2_mask[out_base] = 0;
            float* dst_f = hop2_feats + out_base * d_edge;
            for (int d = 0; d < d_edge; d++) dst_f[d] = 0.f;
        }
        return;
    }

    __shared__ int64_t sh_take;
    __shared__ int64_t sh_window_start;

    const int64_t node = (int64_t)hop1_ids[hop1_base];
    const double qtime = hop1_times[hop1_base];

    if (threadIdx.x == 0) {
        const int64_t start = offsets[node];
        const int64_t end = offsets[node + 1];

        int64_t lo = 0, hi = end - start;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (nbr_times[start + mid] < qtime)
                lo = mid + 1;
            else
                hi = mid;
        }
        int64_t valid_count = lo;
        int64_t take = valid_count < (int64_t)K2 ? valid_count : (int64_t)K2;
        sh_take = take;
        sh_window_start = start + valid_count - take;
    }
    __syncthreads();

    const int64_t take = sh_take;
    const int64_t window_start = sh_window_start;

    for (int slot = threadIdx.x; slot < K2; slot += blockDim.x) {
        const int64_t pos = slot - (K2 - (int)take);
        const int64_t out_base = ((int64_t)query_idx * K1 + hop1_slot) * K2 + slot;

        if (pos >= 0 && pos < take) {
            const int64_t csr_idx = window_start + pos;
            hop2_ids[out_base] = nbr_ids[csr_idx];
            hop2_times[out_base] = nbr_times[csr_idx];
            hop2_mask[out_base] = 1;

            const float* src_f = nbr_feats + csr_idx * d_edge;
            float* dst_f = hop2_feats + out_base * d_edge;
            for (int d = 0; d < d_edge; d++) dst_f[d] = src_f[d];
        } else {
            hop2_ids[out_base] = -1;
            hop2_times[out_base] = 0.0;
            hop2_mask[out_base] = 0;

            float* dst_f = hop2_feats + out_base * d_edge;
            for (int d = 0; d < d_edge; d++) dst_f[d] = 0.f;
        }
    }
}

// ============================================================================
// Kernel 3: Co-neighbor counting
// For each (src, dst) pair, count how many of src's neighbors also appear
// in dst's neighbor set. Uses shared memory for dst neighbor set.
// ============================================================================

__global__ void co_neighbor_count_kernel(
    const int64_t* __restrict__ offsets,
    const double*  __restrict__ nbr_times,
    const int32_t* __restrict__ nbr_ids,
    const int64_t* __restrict__ src_nodes,   // [B]
    const int64_t* __restrict__ dst_nodes,   // [B]
    const double*  __restrict__ query_times, // [B]
    float*  __restrict__ out_counts,         // [B]
    const int K
) {
    extern __shared__ int32_t sh_dst_nbrs[];  // K int32s

    const int batch_idx = blockIdx.x;
    const int64_t src_node = src_nodes[batch_idx];
    const int64_t dst_node = dst_nodes[batch_idx];
    const double qtime = query_times[batch_idx];

    // --- Load dst neighbors into shared memory (thread 0 does binary search) ---
    __shared__ int64_t sh_dst_take;

    if (threadIdx.x == 0) {
        const int64_t start = offsets[dst_node];
        const int64_t end = offsets[dst_node + 1];

        int64_t lo = 0, hi = end - start;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (nbr_times[start + mid] < qtime)
                lo = mid + 1;
            else
                hi = mid;
        }
        int64_t valid_count = lo;
        int64_t take = valid_count < (int64_t)K ? valid_count : (int64_t)K;
        sh_dst_take = take;

        int64_t window_start = start + valid_count - take;
        for (int i = 0; i < take; i++) {
            sh_dst_nbrs[i] = nbr_ids[window_start + i];
        }
        for (int i = take; i < K; i++) {
            sh_dst_nbrs[i] = -1;
        }
    }
    __syncthreads();

    const int dst_take = (int)sh_dst_take;

    // --- Src side: each thread handles one src neighbor slot ---
    // Binary search for src node
    __shared__ int64_t sh_src_take;
    __shared__ int64_t sh_src_window_start;

    if (threadIdx.x == 0) {
        const int64_t start = offsets[src_node];
        const int64_t end = offsets[src_node + 1];

        int64_t lo = 0, hi = end - start;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (nbr_times[start + mid] < qtime)
                lo = mid + 1;
            else
                hi = mid;
        }
        int64_t valid_count = lo;
        int64_t take = valid_count < (int64_t)K ? valid_count : (int64_t)K;
        sh_src_take = take;
        sh_src_window_start = start + valid_count - take;
    }
    __syncthreads();

    const int src_take = (int)sh_src_take;
    const int64_t src_window_start = sh_src_window_start;

    // Each thread checks one src neighbor against all dst neighbors
    int local_count = 0;
    for (int i = threadIdx.x; i < src_take; i += blockDim.x) {
        int32_t src_nbr = nbr_ids[src_window_start + i];
        for (int j = 0; j < dst_take; j++) {
            if (src_nbr == sh_dst_nbrs[j]) {
                local_count++;
                break;  // count unique matches
            }
        }
    }

    // Block-level reduction
    __shared__ int sh_total;
    if (threadIdx.x == 0) sh_total = 0;
    __syncthreads();
    atomicAdd(&sh_total, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        out_counts[batch_idx] = (float)sh_total;
    }
}


// ============================================================================
// C++ entry points
// ============================================================================

std::vector<torch::Tensor> temporal_recent_k_cuda(
    torch::Tensor offsets,
    torch::Tensor nbr_times,
    torch::Tensor nbr_ids,
    torch::Tensor nbr_feats,
    torch::Tensor query_nodes,
    torch::Tensor query_times,
    int k
) {
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA tensor");
    TORCH_CHECK(query_nodes.is_cuda(), "query_nodes must be CUDA tensor");

    const int N = query_nodes.size(0);
    const int d_edge = nbr_feats.size(1);
    auto device = query_nodes.device();

    auto out_ids = torch::empty({N, k}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto out_times = torch::empty({N, k}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    auto out_feats = torch::empty({N, k, d_edge}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto out_mask = torch::empty({N, k}, torch::TensorOptions().dtype(torch::kInt8).device(device));

    if (N == 0) return {out_ids, out_times, out_feats, out_mask};

    const int block_size = k < 256 ? k : 256;
    temporal_recent_k_kernel<<<N, block_size>>>(
        offsets.data_ptr<int64_t>(),
        nbr_times.data_ptr<double>(),
        nbr_ids.data_ptr<int32_t>(),
        nbr_feats.data_ptr<float>(),
        query_nodes.data_ptr<int64_t>(),
        query_times.data_ptr<double>(),
        out_ids.data_ptr<int32_t>(),
        out_times.data_ptr<double>(),
        out_feats.data_ptr<float>(),
        out_mask.data_ptr<int8_t>(),
        k, d_edge
    );
    return {out_ids, out_times, out_feats, out_mask};
}


std::vector<torch::Tensor> temporal_recent_2hop_cuda(
    torch::Tensor offsets,
    torch::Tensor nbr_times,
    torch::Tensor nbr_ids,
    torch::Tensor nbr_feats,
    torch::Tensor query_nodes,
    torch::Tensor query_times,
    int k1, int k2
) {
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA tensor");
    TORCH_CHECK(query_nodes.is_cuda(), "query_nodes must be CUDA tensor");

    const int N = query_nodes.size(0);
    const int d_edge = nbr_feats.size(1);
    auto device = query_nodes.device();

    // Step 1: 1-hop sampling
    auto hop1_ids = torch::empty({N, k1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto hop1_times = torch::empty({N, k1}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    auto hop1_feats = torch::empty({N, k1, d_edge}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto hop1_mask = torch::empty({N, k1}, torch::TensorOptions().dtype(torch::kInt8).device(device));

    if (N == 0) {
        auto hop2_ids = torch::empty({N, k1, k2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto hop2_times = torch::empty({N, k1, k2}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
        auto hop2_feats = torch::empty({N, k1, k2, d_edge}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto hop2_mask = torch::empty({N, k1, k2}, torch::TensorOptions().dtype(torch::kInt8).device(device));
        return {hop1_ids, hop1_times, hop1_feats, hop1_mask,
                hop2_ids, hop2_times, hop2_feats, hop2_mask};
    }

    int block1 = k1 < 256 ? k1 : 256;
    temporal_recent_k_kernel<<<N, block1>>>(
        offsets.data_ptr<int64_t>(),
        nbr_times.data_ptr<double>(),
        nbr_ids.data_ptr<int32_t>(),
        nbr_feats.data_ptr<float>(),
        query_nodes.data_ptr<int64_t>(),
        query_times.data_ptr<double>(),
        hop1_ids.data_ptr<int32_t>(),
        hop1_times.data_ptr<double>(),
        hop1_feats.data_ptr<float>(),
        hop1_mask.data_ptr<int8_t>(),
        k1, d_edge
    );

    // Step 2: 2-hop sampling
    auto hop2_ids = torch::empty({N, k1, k2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto hop2_times = torch::empty({N, k1, k2}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    auto hop2_feats = torch::empty({N, k1, k2, d_edge}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto hop2_mask = torch::empty({N, k1, k2}, torch::TensorOptions().dtype(torch::kInt8).device(device));

    int grid2 = N * k1;
    int block2 = k2 < 256 ? k2 : 256;
    temporal_recent_2hop_kernel<<<grid2, block2>>>(
        offsets.data_ptr<int64_t>(),
        nbr_times.data_ptr<double>(),
        nbr_ids.data_ptr<int32_t>(),
        nbr_feats.data_ptr<float>(),
        hop1_ids.data_ptr<int32_t>(),
        hop1_times.data_ptr<double>(),
        hop1_mask.data_ptr<int8_t>(),
        hop2_ids.data_ptr<int32_t>(),
        hop2_times.data_ptr<double>(),
        hop2_feats.data_ptr<float>(),
        hop2_mask.data_ptr<int8_t>(),
        k1, k2, d_edge
    );

    return {hop1_ids, hop1_times, hop1_feats, hop1_mask,
            hop2_ids, hop2_times, hop2_feats, hop2_mask};
}


torch::Tensor co_neighbor_count_cuda(
    torch::Tensor offsets,
    torch::Tensor nbr_times,
    torch::Tensor nbr_ids,
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor query_times,
    int k
) {
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA tensor");
    TORCH_CHECK(src_nodes.is_cuda(), "src_nodes must be CUDA tensor");

    const int B = src_nodes.size(0);
    auto device = src_nodes.device();

    auto out_counts = torch::zeros({B}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    if (B == 0) return out_counts;

    // Shared memory: K int32 for dst neighbors
    int shared_mem = k * sizeof(int32_t);
    int block_size = k < 256 ? k : 256;

    co_neighbor_count_kernel<<<B, block_size, shared_mem>>>(
        offsets.data_ptr<int64_t>(),
        nbr_times.data_ptr<double>(),
        nbr_ids.data_ptr<int32_t>(),
        src_nodes.data_ptr<int64_t>(),
        dst_nodes.data_ptr<int64_t>(),
        query_times.data_ptr<double>(),
        out_counts.data_ptr<float>(),
        k
    );

    return out_counts;
}


// ============================================================================
// Kernel 4: Co-occurrence frequency vectors for DyGFormer
// Input: a_ids (B, K), b_ids (B, K) — neighbor ID arrays (int32, -1 = padding)
// Output: a_freq (B, K, 2), b_freq (B, K, 2)
//   a_freq[b][i] = [count of a_ids[b][i] in a_ids[b], count of a_ids[b][i] in b_ids[b]]
//   b_freq[b][j] = [count of b_ids[b][j] in a_ids[b], count of b_ids[b][j] in b_ids[b]]
// Replaces three (B,K,K) broadcast comparisons with shared-memory approach.
// ============================================================================

__global__ void co_occurrence_freq_kernel(
    const int32_t* __restrict__ a_ids,   // [B, K]
    const int32_t* __restrict__ b_ids,   // [B, K]
    float* __restrict__ a_freq,          // [B, K, 2]
    float* __restrict__ b_freq,          // [B, K, 2]
    const int K
) {
    extern __shared__ int32_t shared[];
    // Layout: shared[0..K-1] = a_ids for this batch, shared[K..2K-1] = b_ids
    int32_t* sh_a = shared;
    int32_t* sh_b = shared + K;

    const int batch_idx = blockIdx.x;
    const int64_t base = (int64_t)batch_idx * K;

    // Load a_ids and b_ids into shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sh_a[i] = a_ids[base + i];
        sh_b[i] = b_ids[base + i];
    }
    __syncthreads();

    // Each thread handles one slot position across both a and b
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        int32_t aid = sh_a[i];
        float a_self_count = 0.f;
        float a_cross_count = 0.f;

        if (aid != -1) {
            // Count aid in a_ids (self)
            for (int j = 0; j < K; j++) {
                if (sh_a[j] == aid) a_self_count += 1.f;
            }
            // Count aid in b_ids (cross)
            for (int j = 0; j < K; j++) {
                if (sh_b[j] == aid) a_cross_count += 1.f;
            }
        }
        a_freq[(base + i) * 2 + 0] = a_self_count;
        a_freq[(base + i) * 2 + 1] = a_cross_count;

        int32_t bid = sh_b[i];
        float b_cross_count = 0.f;
        float b_self_count = 0.f;

        if (bid != -1) {
            // Count bid in a_ids (cross for b)
            for (int j = 0; j < K; j++) {
                if (sh_a[j] == bid) b_cross_count += 1.f;
            }
            // Count bid in b_ids (self)
            for (int j = 0; j < K; j++) {
                if (sh_b[j] == bid) b_self_count += 1.f;
            }
        }
        b_freq[(base + i) * 2 + 0] = b_cross_count;
        b_freq[(base + i) * 2 + 1] = b_self_count;
    }
}


std::vector<torch::Tensor> co_occurrence_freq_cuda(
    torch::Tensor a_ids,
    torch::Tensor b_ids
) {
    TORCH_CHECK(a_ids.is_cuda(), "a_ids must be CUDA tensor");
    TORCH_CHECK(a_ids.dtype() == torch::kInt32, "a_ids must be int32");
    TORCH_CHECK(b_ids.dtype() == torch::kInt32, "b_ids must be int32");

    const int B = a_ids.size(0);
    const int K = a_ids.size(1);
    auto device = a_ids.device();

    auto a_freq = torch::zeros({B, K, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto b_freq = torch::zeros({B, K, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    if (B == 0 || K == 0) return {a_freq, b_freq};

    int block_size = K < 256 ? K : 256;
    int shared_mem = 2 * K * sizeof(int32_t);

    co_occurrence_freq_kernel<<<B, block_size, shared_mem>>>(
        a_ids.data_ptr<int32_t>(),
        b_ids.data_ptr<int32_t>(),
        a_freq.data_ptr<float>(),
        b_freq.data_ptr<float>(),
        K
    );

    return {a_freq, b_freq};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("temporal_recent_k", &temporal_recent_k_cuda,
          "Fused 1-hop temporal neighbor sampling (CUDA)");
    m.def("temporal_recent_2hop", &temporal_recent_2hop_cuda,
          "Fused 2-hop temporal neighbor sampling (CUDA)");
    m.def("co_neighbor_count", &co_neighbor_count_cuda,
          "Co-neighbor counting between src/dst pairs (CUDA)");
    m.def("co_occurrence_freq", &co_occurrence_freq_cuda,
          "Co-occurrence frequency vectors for DyGFormer (CUDA)");
}
