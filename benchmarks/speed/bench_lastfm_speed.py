"""Quick benchmark: LastFM only — TGEngine vs DyGLib vs TGM neighbor/neg sampling.

LastFM is a dense graph (avg degree 1305, 1.29M edges, 1981 nodes, d_edge=2).

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_lastfm_speed.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
BATCH_SIZES = [200, 600, 1000]
K_VALUES = [32, 64, 256]
N_WARMUP = 10
N_ITER = 100
PADDED_NODE_ID = -1


# ---- DyGLib NeighborSampler (CPU Python loop) ----

class DyGLibNeighborSampler:
    def __init__(self, adj_list):
        self.nodes_neighbor_ids = []
        self.nodes_neighbor_times = []
        for per_node_neighbors in adj_list:
            sorted_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_neighbors]))

    def get_historical_neighbors(self, node_ids, node_interact_times, n_neighbors):
        results = []
        for node_id, interact_time in zip(node_ids, node_interact_times):
            i = np.searchsorted(self.nodes_neighbor_times[int(node_id)], interact_time)
            nbrs = self.nodes_neighbor_ids[int(node_id)][:i][-n_neighbors:]
            if len(nbrs) < n_neighbors:
                nbrs = np.concatenate([np.zeros(n_neighbors - len(nbrs)), nbrs])
            results.append(nbrs)
        return np.stack(results)


# ---- TGM Ring Buffer (GPU vectorized) ----

class TGMRecencyBuffer:
    def __init__(self, num_nodes, max_nbrs, edge_feat_dim, device):
        self.num_nodes = num_nodes
        self.max_nbrs = max_nbrs
        self.edge_feat_dim = edge_feat_dim
        self.device = device
        self._nbr_ids = torch.full((num_nodes, max_nbrs), PADDED_NODE_ID, dtype=torch.int32, device=device)
        self._nbr_times = torch.zeros((num_nodes, max_nbrs), dtype=torch.float64, device=device)
        self._nbr_feats = torch.zeros((num_nodes, max_nbrs, edge_feat_dim), dtype=torch.float32, device=device)
        self._write_pos = torch.zeros(num_nodes, dtype=torch.int32, device=device)

    def update(self, src, dst, times, feats):
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

    def get_recency_neighbors(self, node_ids, query_times, k):
        B = self.max_nbrs
        N = len(node_ids)
        nbr_nids = self._nbr_ids[node_ids.long()]
        nbr_edge_time = self._nbr_times[node_ids.long()]
        nbr_edge_x = self._nbr_feats[node_ids.long()]
        write_pos = self._write_pos[node_ids.long()]

        candidate_idx = write_pos[:, None].long() - torch.arange(B, 0, -1, device=self.device)
        candidate_idx %= B
        candidate_times = torch.gather(nbr_edge_time, 1, candidate_idx)
        time_mask = candidate_times < query_times[:, None].double()
        time_mask[torch.gather(nbr_nids, 1, candidate_idx) == PADDED_NODE_ID] = False

        pos = torch.arange(B, device=self.device)
        last_valid_pos = torch.where(
            time_mask.any(dim=1), (time_mask * pos).amax(dim=1),
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


# ---- Benchmarks ----

def bench_query(fn, N_WARMUP=N_WARMUP, N_ITER=N_ITER):
    times = []
    for _ in range(N_WARMUP + N_ITER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[N_WARMUP:]
    return np.mean(times) * 1000, np.std(times) * 1000


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    import pandas as pd
    df = pd.read_csv(Path(DATA_ROOT) / "lastfm" / "ml_lastfm.csv")
    feats = np.load(Path(DATA_ROOT) / "lastfm" / "ml_lastfm.npy")
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = feats.shape[1]
    avg_deg = 2 * num_edges / num_nodes

    print(f"LastFM: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}, avg_degree={avg_deg:.0f}")
    print()

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    # Build DyGLib sampler
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        adj_list[src[i]].append((dst[i], i, timestamps[i]))
        adj_list[dst[i]].append((src[i], i, timestamps[i]))
    dyglib_sampler = DyGLibNeighborSampler(adj_list)

    # Build TGEngine T-CSR
    from tgengine.core.temporal_graph import TemporalGraph
    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t, dst_t, time_t, feat_t)
    graph.freeze_csr()

    # ======================================================================
    # Neighbor Sampling
    # ======================================================================
    print("=" * 70)
    print("NEIGHBOR SAMPLING (src+dst, 2*BS queries)")
    print("=" * 70)
    print(f"{'BS':>6} {'K':>4} | {'DyGLib(ms)':>11} {'TGM(ms)':>9} {'TGEngine(ms)':>13} | {'vs DyGLib':>9} {'vs TGM':>7}")
    print(f"{'-'*6} {'-'*4} | {'-'*11} {'-'*9} {'-'*13} | {'-'*9} {'-'*7}")

    for K in K_VALUES:
        # Build TGM buffer for this K
        tgm_buf = TGMRecencyBuffer(num_nodes, max_nbrs=K, edge_feat_dim=d_edge, device=device)
        CHUNK = 10000
        for i in range(0, num_edges, CHUNK):
            j = min(i + CHUNK, num_edges)
            tgm_buf.update(src_t[i:j], dst_t[i:j], time_t[i:j], feat_t[i:j])

        for bs in BATCH_SIZES:
            def make_batch():
                idx = torch.randint(0, num_edges, (bs,), device=device)
                return torch.cat([src_t[idx], dst_t[idx]]), torch.cat([time_t[idx], time_t[idx]])

            # DyGLib
            def dyglib_fn():
                idx = np.random.randint(0, num_edges, bs)
                all_nodes = np.concatenate([src[idx], dst[idx]])
                all_times = np.concatenate([timestamps[idx], timestamps[idx]])
                dyglib_sampler.get_historical_neighbors(all_nodes, all_times, K)

            # TGM
            def tgm_fn():
                nodes, times = make_batch()
                tgm_buf.get_recency_neighbors(nodes, times, K)

            # TGEngine
            def tge_fn():
                nodes, times = make_batch()
                graph.recent(nodes, times, K)

            dyglib_ms, _ = bench_query(dyglib_fn)
            tgm_ms, _ = bench_query(tgm_fn)
            tge_ms, _ = bench_query(tge_fn)

            vs_dyglib = dyglib_ms / tge_ms
            vs_tgm = tgm_ms / tge_ms
            print(f"{bs:>6} {K:>4} | {dyglib_ms:>8.2f}   {tgm_ms:>7.2f}   {tge_ms:>8.2f}     | {vs_dyglib:>7.1f}x  {vs_tgm:>5.2f}x")

        del tgm_buf
        torch.cuda.empty_cache()

    # ======================================================================
    # Historical Negative Sampling
    # ======================================================================
    print()
    print("=" * 70)
    print("HISTORICAL NEGATIVE SAMPLING")
    print("=" * 70)
    print(f"{'BS':>6} | {'DyGLib(ms)':>11} {'TGEngine(ms)':>13} | {'Speedup':>8}")
    print(f"{'-'*6} | {'-'*11} {'-'*13} | {'-'*8}")

    from tgengine.pipeline.negatives import HistoricalNegative
    hist_neg = HistoricalNegative(num_nodes, device=device)
    hist_neg.update(src_t[:5000], dst_t[:5000])

    for bs in BATCH_SIZES:
        # DyGLib historical neg
        def dyglib_hist():
            idx = np.random.randint(0, num_edges, bs)
            batch_src = src[idx]
            negs = np.empty(bs, dtype=np.int64)
            for i in range(bs):
                s = batch_src[i]
                nbrs = adj_list[s]
                if len(nbrs) > 0:
                    negs[i] = nbrs[np.random.randint(0, len(nbrs))][0]
                else:
                    negs[i] = np.random.randint(0, num_nodes)
            return negs

        # TGEngine historical neg
        def tge_hist():
            idx = torch.randint(0, num_edges, (bs,), device=device)
            batch_src = src_t[idx]
            batch_dst = dst_t[idx]
            batch_time = time_t[idx]
            hist_neg.sample(batch_src, batch_dst, batch_time, graph)

        dyglib_ms, _ = bench_query(dyglib_hist)
        tge_ms, _ = bench_query(tge_hist)
        print(f"{bs:>6} | {dyglib_ms:>8.3f}   {tge_ms:>8.3f}     | {dyglib_ms/tge_ms:>6.1f}x")

    del graph
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
