"""End-to-end training speed: TGEngine vs TGM pipeline on LastFM (3 epochs).

Both use the same DyGFormer model — only the data pipeline differs:
- TGEngine: T-CSR + searchsorted neighbor sampling
- TGM-style: Ring buffer + vectorized gather

We simulate TGM's pipeline by using TGMRecencyBuffer for neighbor gathering
and feed the same model architecture.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
BS = 200
K = 512
PATCH_SIZE = 16
N_EPOCHS = 1
PADDED_NODE_ID = -1


class TGMRecencyBuffer:
    """Minimal TGM ring buffer for pipeline comparison."""
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
        B = self.max_nbrs

        max_time = seed_times.max() + 1
        composite_key = node_ids.long() * int(max_time.item()) + seed_times.long()
        perm = torch.argsort(composite_key, stable=True)
        sorted_nodes = node_ids[perm]
        sorted_nbr_ids = nbr_nids[perm]
        sorted_times = seed_times[perm]
        sorted_feats = edge_feats[perm]

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


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Config: BS={BS}, K={K}, epochs={N_EPOCHS}")
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
    train_end = int(num_edges * 0.7)

    print(f"LastFM: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    print(f"Train edges: {train_end:,}, Steps/epoch: {train_end // BS}")
    print()

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    # =========================================================================
    # TGEngine training
    # =========================================================================
    print("=" * 60)
    print("TGEngine (T-CSR pipeline)")
    print("=" * 60)

    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline import DataPipeline

    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t[:train_end], dst_t[:train_end], time_t[:train_end], feat_t[:train_end])
    graph.freeze_csr()

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2, patch_size=PATCH_SIZE, K=K, num_nodes=num_nodes).to(device)
    pipeline = DataPipeline(model.gather_spec, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    torch.cuda.synchronize()
    tge_start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()
        for step_start in range(0, train_end, BS):
            step_end = min(step_start + BS, train_end)
            idx = slice(step_start, step_end)
            b = step_end - step_start

            neg = torch.randint(0, num_nodes, (b,), device=device)
            raw = RawBatch(src=src_t[idx], dst=dst_t[idx], time=time_t[idx],
                           edge_feat=feat_t[idx], neg=neg)
            prepared = pipeline.prepare(raw)
            out = model(prepared)

            loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
                   criterion(out.neg_score, torch.zeros(b, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        print(f"  Epoch {epoch+1}: {epoch_time:.1f}s (loss={loss.item():.4f})")

    torch.cuda.synchronize()
    tge_total = time.perf_counter() - tge_start
    print(f"  Total: {tge_total:.1f}s ({tge_total/N_EPOCHS:.1f}s/epoch)")

    del model, pipeline, graph, optimizer
    torch.cuda.empty_cache()

    # =========================================================================
    # TGM-style training (ring buffer pipeline, same model)
    # =========================================================================
    print()
    print("=" * 60)
    print("TGM-style (Ring Buffer pipeline)")
    print("=" * 60)

    from tgengine.core.batch import PreparedBatch, NeighborData

    # Rebuild model fresh
    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2, patch_size=PATCH_SIZE, K=K, num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Build TGM buffer and preload train edges
    tgm_buf = TGMRecencyBuffer(num_nodes, max_nbrs=K, edge_feat_dim=d_edge, device=device)
    CHUNK = 10000
    for i in range(0, train_end, CHUNK):
        j = min(i + CHUNK, train_end)
        tgm_buf.update(src_t[i:j], dst_t[i:j], time_t[i:j], feat_t[i:j])

    model.train()
    torch.cuda.synchronize()
    tgm_start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()
        for step_start in range(0, train_end, BS):
            step_end = min(step_start + BS, train_end)
            idx = slice(step_start, step_end)
            b = step_end - step_start

            batch_src = src_t[idx]
            batch_dst = dst_t[idx]
            batch_time = time_t[idx]
            batch_feat = feat_t[idx]
            neg = torch.randint(0, num_nodes, (b,), device=device)

            # TGM-style: query ring buffer for src, dst, neg
            all_nodes = torch.cat([batch_src, batch_dst, neg])
            all_times = torch.cat([batch_time, batch_time, batch_time])
            nbr_ids, nbr_times, nbr_feats = tgm_buf.get_recency_neighbors(all_nodes, all_times, K)

            # Split into src/dst/neg neighbor data
            src_nbr_ids = nbr_ids[:b]
            dst_nbr_ids = nbr_ids[b:2*b]
            neg_nbr_ids = nbr_ids[2*b:]
            src_nbr_times = nbr_times[:b].float()
            dst_nbr_times = nbr_times[b:2*b].float()
            neg_nbr_times = nbr_times[2*b:].float()
            src_nbr_feats = nbr_feats[:b]
            dst_nbr_feats = nbr_feats[b:2*b]
            neg_nbr_feats = nbr_feats[2*b:]

            src_mask = src_nbr_ids != PADDED_NODE_ID
            dst_mask = dst_nbr_ids != PADDED_NODE_ID
            neg_mask = neg_nbr_ids != PADDED_NODE_ID

            prepared = PreparedBatch(
                src=batch_src, dst=batch_dst, neg=neg, time=batch_time,
                src_neighbors=NeighborData(
                    neighbor_ids=src_nbr_ids.long(), timestamps=src_nbr_times,
                    edge_feats=src_nbr_feats, mask=src_mask),
                dst_neighbors=NeighborData(
                    neighbor_ids=dst_nbr_ids.long(), timestamps=dst_nbr_times,
                    edge_feats=dst_nbr_feats, mask=dst_mask),
                neg_neighbors=NeighborData(
                    neighbor_ids=neg_nbr_ids.long(), timestamps=neg_nbr_times,
                    edge_feats=neg_nbr_feats, mask=neg_mask),
            )
            out = model(prepared)

            loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
                   criterion(out.neg_score, torch.zeros(b, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        print(f"  Epoch {epoch+1}: {epoch_time:.1f}s (loss={loss.item():.4f})")

    torch.cuda.synchronize()
    tgm_total = time.perf_counter() - tgm_start
    print(f"  Total: {tgm_total:.1f}s ({tgm_total/N_EPOCHS:.1f}s/epoch)")

    del model, optimizer, tgm_buf
    torch.cuda.empty_cache()

    # =========================================================================
    # DyGLib-style training (CPU neighbor sampling)
    # =========================================================================
    print()
    print("=" * 60)
    print("DyGLib-style (CPU Python-loop neighbor sampling)")
    print("=" * 60)

    # Build DyGLib adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(train_end):
        adj_list[src[i]].append((dst[i], i, timestamps[i]))
        adj_list[dst[i]].append((src[i], i, timestamps[i]))

    # DyGLib NeighborSampler
    nodes_neighbor_ids = []
    nodes_neighbor_times = []
    nodes_edge_feats = []
    for per_node in adj_list:
        sorted_n = sorted(per_node, key=lambda x: x[2])
        nodes_neighbor_ids.append(np.array([x[0] for x in sorted_n], dtype=np.int64))
        nodes_neighbor_times.append(np.array([x[2] for x in sorted_n], dtype=np.float64))
        # edge feats: index by edge_id
        nodes_edge_feats.append(np.array([feats[x[1]] for x in sorted_n], dtype=np.float32))

    def dyglib_get_neighbors(node_ids_np, times_np, k):
        B = len(node_ids_np)
        out_ids = np.zeros((B, k), dtype=np.int64)
        out_times = np.zeros((B, k), dtype=np.float64)
        out_feats = np.zeros((B, k, d_edge), dtype=np.float32)
        out_mask = np.zeros((B, k), dtype=bool)
        for idx in range(B):
            nid = int(node_ids_np[idx])
            t = times_np[idx]
            i = np.searchsorted(nodes_neighbor_times[nid], t)
            start = max(0, i - k)
            nbr_slice = nodes_neighbor_ids[nid][start:i]
            time_slice = nodes_neighbor_times[nid][start:i]
            feat_slice = nodes_edge_feats[nid][start:i]
            n = len(nbr_slice)
            if n > 0:
                out_ids[idx, k-n:] = nbr_slice
                out_times[idx, k-n:] = time_slice
                out_feats[idx, k-n:] = feat_slice
                out_mask[idx, k-n:] = True
        return out_ids, out_times, out_feats, out_mask

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2, patch_size=PATCH_SIZE, K=K, num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    torch.cuda.synchronize()
    dyglib_start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()
        for step_start in range(0, train_end, BS):
            step_end = min(step_start + BS, train_end)
            b = step_end - step_start

            batch_src_np = src[step_start:step_end]
            batch_dst_np = dst[step_start:step_end]
            batch_time_np = timestamps[step_start:step_end]
            batch_feat_np = feats[step_start:step_end]

            # CPU neighbor sampling (DyGLib style)
            all_nodes_np = np.concatenate([batch_src_np, batch_dst_np])
            all_times_np = np.concatenate([batch_time_np, batch_time_np])

            # src neighbors
            s_ids, s_times, s_feats, s_mask = dyglib_get_neighbors(batch_src_np, batch_time_np, K)
            d_ids, d_times, d_feats, d_mask = dyglib_get_neighbors(batch_dst_np, batch_time_np, K)

            # Random neg
            neg_np = np.random.randint(0, num_nodes, b)
            n_ids, n_times, n_feats, n_mask = dyglib_get_neighbors(neg_np, batch_time_np, K)

            # Transfer to GPU
            batch_src_g = torch.from_numpy(batch_src_np).long().to(device)
            batch_dst_g = torch.from_numpy(batch_dst_np).long().to(device)
            batch_time_g = torch.from_numpy(batch_time_np).float().to(device)
            neg_g = torch.from_numpy(neg_np).long().to(device)

            prepared = PreparedBatch(
                src=batch_src_g, dst=batch_dst_g, neg=neg_g, time=batch_time_g,
                src_neighbors=NeighborData(
                    neighbor_ids=torch.from_numpy(s_ids).long().to(device),
                    timestamps=torch.from_numpy(s_times).float().to(device),
                    edge_feats=torch.from_numpy(s_feats).float().to(device),
                    mask=torch.from_numpy(s_mask).to(device)),
                dst_neighbors=NeighborData(
                    neighbor_ids=torch.from_numpy(d_ids).long().to(device),
                    timestamps=torch.from_numpy(d_times).float().to(device),
                    edge_feats=torch.from_numpy(d_feats).float().to(device),
                    mask=torch.from_numpy(d_mask).to(device)),
                neg_neighbors=NeighborData(
                    neighbor_ids=torch.from_numpy(n_ids).long().to(device),
                    timestamps=torch.from_numpy(n_times).float().to(device),
                    edge_feats=torch.from_numpy(n_feats).float().to(device),
                    mask=torch.from_numpy(n_mask).to(device)),
            )
            out = model(prepared)

            loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
                   criterion(out.neg_score, torch.zeros(b, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        print(f"  Epoch {epoch+1}: {epoch_time:.1f}s (loss={loss.item():.4f})")

    torch.cuda.synchronize()
    dyglib_total = time.perf_counter() - dyglib_start
    print(f"  Total: {dyglib_total:.1f}s ({dyglib_total/N_EPOCHS:.1f}s/epoch)")

    del model, optimizer
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("SUMMARY (LastFM, BS={}, K={}, {} epochs)".format(BS, K, N_EPOCHS))
    print("=" * 60)
    print(f"  TGEngine (T-CSR):    {tge_total:>6.1f}s total, {tge_total/N_EPOCHS:.1f}s/epoch")
    print(f"  TGM-style (Ring):    {tgm_total:>6.1f}s total, {tgm_total/N_EPOCHS:.1f}s/epoch")
    print(f"  DyGLib-style (CPU):  {dyglib_total:>6.1f}s total, {dyglib_total/N_EPOCHS:.1f}s/epoch")
    print()
    print(f"  TGEngine vs TGM:   {tgm_total/tge_total:.2f}x")
    print(f"  TGEngine vs DyGLib: {dyglib_total/tge_total:.2f}x")
    print()
    print(f"  Note: TGM results are INCORRECT on this dense graph (truncated history)")
    print(f"  TGEngine and DyGLib are both semantically correct (full history)")


if __name__ == "__main__":
    main()
