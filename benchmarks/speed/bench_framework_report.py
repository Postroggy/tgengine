"""TGEngine Framework Performance Report — Wikipedia & Reddit.

Compares TGEngine vs DyGLib-style training on standard benchmarks.
Measures: pipeline time, forward time, backward time, total epoch time.
Uses DyGLib's exact hyperparameters for each dataset.

Wikipedia: K=32, patch_size=1, BS=200, d_edge=172
Reddit:    K=64, patch_size=2, BS=200, d_edge=172
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
N_PROFILE_STEPS = 50
N_WARMUP = 10


def load_dataset(name):
    import pandas as pd
    df = pd.read_csv(Path(DATA_ROOT) / name / f"ml_{name}.csv")
    feats = np.load(Path(DATA_ROOT) / name / f"ml_{name}.npy")
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = feats.shape[1]
    train_end = int(num_edges * 0.7)
    return src, dst, timestamps, feats, num_nodes, num_edges, d_edge, train_end


def build_dyglib_sampler(src, dst, timestamps, feats, train_end, num_nodes, d_edge):
    """Build DyGLib-style CPU adjacency list for neighbor sampling."""
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(train_end):
        adj_list[src[i]].append((dst[i], i, timestamps[i]))
        adj_list[dst[i]].append((src[i], i, timestamps[i]))

    nodes_neighbor_ids = []
    nodes_neighbor_times = []
    nodes_edge_feats = []
    for per_node in adj_list:
        sorted_n = sorted(per_node, key=lambda x: x[2])
        nodes_neighbor_ids.append(np.array([x[0] for x in sorted_n], dtype=np.int64))
        nodes_neighbor_times.append(np.array([x[2] for x in sorted_n], dtype=np.float64))
        nodes_edge_feats.append(np.array([feats[x[1]] for x in sorted_n], dtype=np.float32))

    def get_neighbors(node_ids_np, times_np, k):
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
                out_ids[idx, k - n:] = nbr_slice
                out_times[idx, k - n:] = time_slice
                out_feats[idx, k - n:] = feat_slice
                out_mask[idx, k - n:] = True
        return out_ids, out_times, out_feats, out_mask

    return get_neighbors


def run_tgengine(src, dst, timestamps, feats, num_nodes, d_edge, train_end, K, patch_size, BS, device):
    """Run TGEngine training for 1 epoch and profile."""
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline import DataPipeline

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t[:train_end], dst_t[:train_end], time_t[:train_end], feat_t[:train_end])
    graph.freeze_csr()

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2,
                      patch_size=patch_size, K=K, num_nodes=num_nodes).to(device)
    pipeline = DataPipeline(model.gather_spec, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    # Profiling: detailed breakdown
    t_prep_list = []
    t_fwd_list = []
    t_bwd_list = []

    # Warmup
    for i in range(N_WARMUP):
        idx = slice(i * BS, (i + 1) * BS)
        b = min(BS, train_end - i * BS)
        neg = torch.randint(0, num_nodes, (b,), device=device)
        raw = RawBatch(src=src_t[idx][:b], dst=dst_t[idx][:b], time=time_t[idx][:b],
                       edge_feat=feat_t[idx][:b], neg=neg)
        prepared = pipeline.prepare(raw)
        out = model(prepared)
        loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
               criterion(out.neg_score, torch.zeros(b, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Profile N steps
    for i in range(N_PROFILE_STEPS):
        start = (N_WARMUP + i) * BS
        end = min(start + BS, train_end)
        b = end - start
        if b <= 0:
            break

        neg = torch.randint(0, num_nodes, (b,), device=device)
        raw = RawBatch(src=src_t[start:end], dst=dst_t[start:end], time=time_t[start:end],
                       edge_feat=feat_t[start:end], neg=neg)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        prepared = pipeline.prepare(raw)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out = model(prepared)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
               criterion(out.neg_score, torch.zeros(b, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()

        t_prep_list.append(t1 - t0)
        t_fwd_list.append(t2 - t1)
        t_bwd_list.append(t3 - t2)

    # Full epoch timing
    model.train()
    torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    for step_start in range(0, train_end, BS):
        step_end = min(step_start + BS, train_end)
        b = step_end - step_start
        neg = torch.randint(0, num_nodes, (b,), device=device)
        raw = RawBatch(src=src_t[step_start:step_end], dst=dst_t[step_start:step_end],
                       time=time_t[step_start:step_end], edge_feat=feat_t[step_start:step_end], neg=neg)
        prepared = pipeline.prepare(raw)
        out = model(prepared)
        loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
               criterion(out.neg_score, torch.zeros(b, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start

    del model, pipeline, graph, optimizer
    torch.cuda.empty_cache()

    return {
        "prep_ms": np.mean(t_prep_list) * 1000,
        "fwd_ms": np.mean(t_fwd_list) * 1000,
        "bwd_ms": np.mean(t_bwd_list) * 1000,
        "epoch_s": epoch_time,
        "steps_per_epoch": (train_end + BS - 1) // BS,
    }


def run_dyglib(src, dst, timestamps, feats, num_nodes, d_edge, train_end, K, patch_size, BS, device):
    """Run DyGLib-style training for 1 epoch and profile."""
    from tgengine.models.dygformer import DyGFormer
    from tgengine.core.batch import PreparedBatch, NeighborData

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    get_neighbors = build_dyglib_sampler(src, dst, timestamps, feats, train_end, num_nodes, d_edge)

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2,
                      patch_size=patch_size, K=K, num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    # Profiling
    t_prep_list = []
    t_fwd_list = []
    t_bwd_list = []

    # Warmup
    for i in range(N_WARMUP):
        start = i * BS
        end = min(start + BS, train_end)
        b = end - start
        batch_src_np = src[start:end]
        batch_dst_np = dst[start:end]
        batch_time_np = timestamps[start:end]
        neg_np = np.random.randint(0, num_nodes, b)

        s_ids, s_times, s_feats, s_mask = get_neighbors(batch_src_np, batch_time_np, K)
        d_ids, d_times, d_feats, d_mask = get_neighbors(batch_dst_np, batch_time_np, K)
        n_ids, n_times, n_feats, n_mask = get_neighbors(neg_np, batch_time_np, K)

        prepared = PreparedBatch(
            src=src_t[start:end], dst=dst_t[start:end], neg=torch.from_numpy(neg_np).long().to(device),
            time=time_t[start:end],
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

    # Profile N steps
    for i in range(N_PROFILE_STEPS):
        start = (N_WARMUP + i) * BS
        end = min(start + BS, train_end)
        b = end - start
        if b <= 0:
            break

        batch_src_np = src[start:end]
        batch_dst_np = dst[start:end]
        batch_time_np = timestamps[start:end]
        neg_np = np.random.randint(0, num_nodes, b)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        s_ids, s_times, s_feats, s_mask = get_neighbors(batch_src_np, batch_time_np, K)
        d_ids, d_times, d_feats, d_mask = get_neighbors(batch_dst_np, batch_time_np, K)
        n_ids, n_times, n_feats, n_mask = get_neighbors(neg_np, batch_time_np, K)

        prepared = PreparedBatch(
            src=src_t[start:end], dst=dst_t[start:end], neg=torch.from_numpy(neg_np).long().to(device),
            time=time_t[start:end],
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
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out = model(prepared)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        loss = criterion(out.pos_score, torch.ones(b, device=device)) + \
               criterion(out.neg_score, torch.zeros(b, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()

        t_prep_list.append(t1 - t0)
        t_fwd_list.append(t2 - t1)
        t_bwd_list.append(t3 - t2)

    # Full epoch timing
    model.train()
    torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    for step_start in range(0, train_end, BS):
        step_end = min(step_start + BS, train_end)
        b = step_end - step_start
        batch_src_np = src[step_start:step_end]
        batch_dst_np = dst[step_start:step_end]
        batch_time_np = timestamps[step_start:step_end]
        neg_np = np.random.randint(0, num_nodes, b)

        s_ids, s_times, s_feats, s_mask = get_neighbors(batch_src_np, batch_time_np, K)
        d_ids, d_times, d_feats, d_mask = get_neighbors(batch_dst_np, batch_time_np, K)
        n_ids, n_times, n_feats, n_mask = get_neighbors(neg_np, batch_time_np, K)

        prepared = PreparedBatch(
            src=src_t[step_start:step_end], dst=dst_t[step_start:step_end],
            neg=torch.from_numpy(neg_np).long().to(device), time=time_t[step_start:step_end],
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

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "prep_ms": np.mean(t_prep_list) * 1000,
        "fwd_ms": np.mean(t_fwd_list) * 1000,
        "bwd_ms": np.mean(t_bwd_list) * 1000,
        "epoch_s": epoch_time,
        "steps_per_epoch": (train_end + BS - 1) // BS,
    }


def print_report(ds_name, ds_info, tge_result, dyglib_result, K, patch_size, BS):
    num_edges, num_nodes, d_edge, train_end = ds_info
    steps = tge_result["steps_per_epoch"]

    print()
    print("=" * 70)
    print(f"  {ds_name.upper()} — DyGFormer (BS={BS}, K={K}, patch_size={patch_size})")
    print("=" * 70)
    print(f"  Dataset: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    print(f"  Train: {train_end:,} edges, {steps} steps/epoch")
    print()

    # Per-step breakdown
    tge_total_ms = tge_result["prep_ms"] + tge_result["fwd_ms"] + tge_result["bwd_ms"]
    dyg_total_ms = dyglib_result["prep_ms"] + dyglib_result["fwd_ms"] + dyglib_result["bwd_ms"]

    print("  Per-step breakdown (ms):")
    print(f"  {'Component':<20} {'TGEngine':>10} {'DyGLib':>10} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Pipeline/Sampling':<20} {tge_result['prep_ms']:>10.2f} {dyglib_result['prep_ms']:>10.2f} {dyglib_result['prep_ms']/tge_result['prep_ms']:>9.2f}x")
    print(f"  {'Model Forward':<20} {tge_result['fwd_ms']:>10.2f} {dyglib_result['fwd_ms']:>10.2f} {dyglib_result['fwd_ms']/tge_result['fwd_ms']:>9.2f}x")
    print(f"  {'Backward+Optim':<20} {tge_result['bwd_ms']:>10.2f} {dyglib_result['bwd_ms']:>10.2f} {dyglib_result['bwd_ms']/tge_result['bwd_ms']:>9.2f}x")
    print(f"  {'TOTAL':<20} {tge_total_ms:>10.2f} {dyg_total_ms:>10.2f} {dyg_total_ms/tge_total_ms:>9.2f}x")
    print()

    # Epoch time
    print("  Epoch time:")
    print(f"    TGEngine:  {tge_result['epoch_s']:.1f}s")
    print(f"    DyGLib:    {dyglib_result['epoch_s']:.1f}s")
    print(f"    Speedup:   {dyglib_result['epoch_s']/tge_result['epoch_s']:.2f}x")
    print()

    # Percentage breakdown
    print("  TGEngine time distribution:")
    print(f"    Pipeline:  {tge_result['prep_ms']/tge_total_ms*100:>5.1f}%")
    print(f"    Forward:   {tge_result['fwd_ms']/tge_total_ms*100:>5.1f}%")
    print(f"    Backward:  {tge_result['bwd_ms']/tge_total_ms*100:>5.1f}%")
    print()
    print("  DyGLib time distribution:")
    print(f"    Pipeline:  {dyglib_result['prep_ms']/dyg_total_ms*100:>5.1f}%")
    print(f"    Forward:   {dyglib_result['fwd_ms']/dyg_total_ms*100:>5.1f}%")
    print(f"    Backward:  {dyglib_result['bwd_ms']/dyg_total_ms*100:>5.1f}%")


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ===== Wikipedia: K=32, patch_size=1, BS=200 =====
    configs = [
        ("wikipedia", 32, 1, 200),
        ("reddit", 64, 2, 200),
        ("lastfm", 512, 16, 200),
    ]

    all_results = []

    for ds_name, K, patch_size, BS in configs:
        print(f"\n{'#'*70}")
        print(f"# Running: {ds_name} (K={K}, patch_size={patch_size}, BS={BS})")
        print(f"{'#'*70}")

        src, dst, timestamps, feats, num_nodes, num_edges, d_edge, train_end = load_dataset(ds_name)
        ds_info = (num_edges, num_nodes, d_edge, train_end)

        print(f"\n  [TGEngine] Training...")
        tge = run_tgengine(src, dst, timestamps, feats, num_nodes, d_edge, train_end, K, patch_size, BS, device)

        print(f"  [DyGLib]   Training...")
        dyg = run_dyglib(src, dst, timestamps, feats, num_nodes, d_edge, train_end, K, patch_size, BS, device)

        print_report(ds_name, ds_info, tge, dyg, K, patch_size, BS)
        all_results.append((ds_name, K, tge, dyg))

    # ===== Summary =====
    print()
    print()
    print("=" * 70)
    print("  SUMMARY: TGEngine vs DyGLib — End-to-End Epoch Speedup")
    print("=" * 70)
    print(f"  {'Dataset':<12} {'K':>4} {'TGEngine':>10} {'DyGLib':>10} {'Speedup':>10} {'Pipeline':>10}")
    print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for ds_name, K, tge, dyg in all_results:
        pipeline_speedup = dyg["prep_ms"] / tge["prep_ms"]
        print(f"  {ds_name:<12} {K:>4} {tge['epoch_s']:>8.1f}s {dyg['epoch_s']:>8.1f}s {dyg['epoch_s']/tge['epoch_s']:>9.2f}x {pipeline_speedup:>9.1f}x")

    print()
    print("  Notes:")
    print("  - TGEngine: GPU-resident T-CSR + searchsorted neighbor sampling")
    print("  - DyGLib:   CPU Python-loop neighbor sampling + numpy→GPU transfer")
    print("  - Same model (DyGFormer), same hyperparameters, same loss function")
    print("  - Pipeline speedup = neighbor sampling + data transfer speedup")
    print("  - Forward/Backward identical (same model on same GPU)")


if __name__ == "__main__":
    main()
