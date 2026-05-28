"""Profile: breakdown of training step time on LastFM.

Measures each component: neighbor sampling, negative sampling, model forward, backward.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")

    import pandas as pd
    df = pd.read_csv(Path(DATA_ROOT) / "lastfm" / "ml_lastfm.csv")
    feats = np.load(Path(DATA_ROOT) / "lastfm" / "ml_lastfm.npy")
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = feats.shape[1]

    print(f"LastFM: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    print(f"Avg degree: {2*num_edges/num_nodes:.0f}")
    print()

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    # Build graph
    from tgengine.core.temporal_graph import TemporalGraph
    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t, dst_t, time_t, feat_t)
    graph.freeze_csr()

    # Build model (DyGFormer)
    from tgengine.models.dygformer import DyGFormer
    model = DyGFormer(
        d_edge=d_edge, d_model=172,
        n_layers=2, n_heads=2, patch_size=8, K=32
    ).to(device)
    model.train()

    from tgengine.pipeline import DataPipeline
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.core.batch import RawBatch

    pipeline = DataPipeline(model.gather_spec, graph)
    neg_sampler = RandomNegative(num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Profile
    BS = 200
    K = 32
    N_WARMUP = 5
    N_ITER = 30

    t_neighbor = []
    t_neg = []
    t_prepare = []
    t_forward = []
    t_backward = []
    t_total = []

    for i in range(N_WARMUP + N_ITER):
        idx = torch.randint(0, num_edges, (BS,), device=device)
        batch_src = src_t[idx]
        batch_dst = dst_t[idx]
        batch_time = time_t[idx]
        batch_feat = feat_t[idx]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Negative sampling
        neg = neg_sampler.sample(batch_src, batch_dst, batch_time, graph)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Prepare batch (includes neighbor sampling)
        raw = RawBatch(src=batch_src, dst=batch_dst, time=batch_time,
                       edge_feat=batch_feat, neg=neg)
        prepared = pipeline.prepare(raw)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Forward
        out = model(prepared)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Backward
        pos_labels = torch.ones(BS, device=device)
        neg_labels = torch.zeros(BS, device=device)
        loss = criterion(out.pos_score, pos_labels) + criterion(out.neg_score, neg_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        if i >= N_WARMUP:
            t_neg.append(t1 - t0)
            t_prepare.append(t2 - t1)
            t_forward.append(t3 - t2)
            t_backward.append(t4 - t3)
            t_total.append(t4 - t0)

    # Results
    neg_ms = np.mean(t_neg) * 1000
    prep_ms = np.mean(t_prepare) * 1000
    fwd_ms = np.mean(t_forward) * 1000
    bwd_ms = np.mean(t_backward) * 1000
    total_ms = np.mean(t_total) * 1000

    print(f"=" * 60)
    print(f"Training Step Breakdown (BS={BS}, K={K}, DyGFormer)")
    print(f"=" * 60)
    print(f"  Negative sampling:   {neg_ms:>8.2f} ms  ({neg_ms/total_ms*100:>5.1f}%)")
    print(f"  Pipeline (nbr samp): {prep_ms:>8.2f} ms  ({prep_ms/total_ms*100:>5.1f}%)")
    print(f"  Model forward:       {fwd_ms:>8.2f} ms  ({fwd_ms/total_ms*100:>5.1f}%)")
    print(f"  Backward + optim:    {bwd_ms:>8.2f} ms  ({bwd_ms/total_ms*100:>5.1f}%)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total per step:      {total_ms:>8.2f} ms")
    print()

    # Now estimate: if neighbor sampling were as fast as TGM (0.75ms)
    tgm_nbr_ms = 0.75
    total_if_tgm = neg_ms + tgm_nbr_ms + fwd_ms + bwd_ms
    print(f"  Hypothetical with TGM-speed sampling: {total_if_tgm:.2f} ms/step")
    print(f"  Speedup vs current: {total_ms / total_if_tgm:.2f}x")
    print(f"  Time saved per step: {prep_ms - tgm_nbr_ms:.2f} ms")
    print()

    # Also test with larger BS and K
    for BS2, K2 in [(600, 32), (200, 256), (600, 256)]:
        print(f"\n--- BS={BS2}, K={K2} ---")
        # Just measure pipeline.prepare time
        from tgengine.core.gather_spec import GatherSpec, NeighborSpec
        model2 = DyGFormer(
            d_edge=d_edge, d_model=172,
            n_layers=2, n_heads=2, patch_size=8, K=K2
        ).to(device)
        pipeline2 = DataPipeline(model2.gather_spec, graph)

        prep_times = []
        fwd_times = []
        bwd_times = []
        model2.train()
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)

        for i in range(N_WARMUP + 20):
            idx = torch.randint(0, num_edges, (BS2,), device=device)
            raw = RawBatch(src=src_t[idx], dst=dst_t[idx], time=time_t[idx],
                           edge_feat=feat_t[idx],
                           neg=torch.randint(0, num_nodes, (BS2,), device=device))

            torch.cuda.synchronize()
            ta = time.perf_counter()
            prepared = pipeline2.prepare(raw)
            torch.cuda.synchronize()
            tb = time.perf_counter()
            out = model2(prepared)
            torch.cuda.synchronize()
            tc = time.perf_counter()
            loss = criterion(out.pos_score, torch.ones(BS2, device=device)) + \
                   criterion(out.neg_score, torch.zeros(BS2, device=device))
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            torch.cuda.synchronize()
            td = time.perf_counter()

            if i >= N_WARMUP:
                prep_times.append(tb - ta)
                fwd_times.append(tc - tb)
                bwd_times.append(td - tc)

        p = np.mean(prep_times) * 1000
        f = np.mean(fwd_times) * 1000
        b = np.mean(bwd_times) * 1000
        tot = p + f + b
        print(f"  Pipeline: {p:.2f}ms ({p/tot*100:.1f}%) | Forward: {f:.2f}ms ({f/tot*100:.1f}%) | Backward: {b:.2f}ms ({b/tot*100:.1f}%) | Total: {tot:.2f}ms")

        del model2, pipeline2, opt2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
