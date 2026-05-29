"""Per-batch forward internals breakdown for Wikipedia, Reddit, LastFM."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
N_WARMUP = 10
N_ITER = 30


def profile_dataset(ds_name, K, patch_size, BS):
    device = "cuda"
    import pandas as pd

    df = pd.read_csv(Path(DATA_ROOT) / ds_name / f"ml_{ds_name}.csv")
    feats = np.load(Path(DATA_ROOT) / ds_name / f"ml_{ds_name}.npy")
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = feats.shape[1]
    train_end = int(num_edges * 0.7)

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline import DataPipeline

    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t[:train_end], dst_t[:train_end], time_t[:train_end], feat_t[:train_end])
    graph.freeze_csr()

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2,
                      patch_size=patch_size, K=K, num_nodes=num_nodes).to(device)
    pipeline = DataPipeline(model.gather_spec, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    # Warmup
    for _ in range(N_WARMUP):
        idx = torch.randint(0, train_end, (BS,), device=device)
        neg = torch.randint(0, num_nodes, (BS,), device=device)
        raw = RawBatch(src=src_t[idx], dst=dst_t[idx], time=time_t[idx],
                       edge_feat=feat_t[idx], neg=neg)
        prepared = pipeline.prepare(raw)
        out = model(prepared)
        loss = criterion(out.pos_score, torch.ones(BS, device=device)) + \
               criterion(out.neg_score, torch.zeros(BS, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # High-level breakdown
    t_prep = []
    t_fwd = []
    t_bwd = []

    for _ in range(N_ITER):
        idx = torch.randint(0, train_end, (BS,), device=device)
        neg = torch.randint(0, num_nodes, (BS,), device=device)
        raw = RawBatch(src=src_t[idx], dst=dst_t[idx], time=time_t[idx],
                       edge_feat=feat_t[idx], neg=neg)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        prepared = pipeline.prepare(raw)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out = model(prepared)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        loss = criterion(out.pos_score, torch.ones(BS, device=device)) + \
               criterion(out.neg_score, torch.zeros(BS, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()

        t_prep.append(t1 - t0)
        t_fwd.append(t2 - t1)
        t_bwd.append(t3 - t2)

    # Forward internals
    t_time_enc = []
    t_co_enc = []
    t_patchify = []
    t_transformer = []
    t_pool = []

    for _ in range(N_ITER):
        idx = torch.randint(0, train_end, (BS,), device=device)
        neg = torch.randint(0, num_nodes, (BS,), device=device)
        raw = RawBatch(src=src_t[idx], dst=dst_t[idx], time=time_t[idx],
                       edge_feat=feat_t[idx], neg=neg)
        prepared = pipeline.prepare(raw)
        batch = prepared

        time_enc_total = 0
        co_enc_total = 0
        patchify_total = 0
        transformer_total = 0
        pool_total = 0

        for a_nbrs, b_nbrs in [(batch.src_neighbors, batch.dst_neighbors),
                                (batch.src_neighbors, batch.neg_neighbors)]:
            a_ids = batch.src
            b_ids = batch.dst if a_nbrs is batch.src_neighbors and b_nbrs is batch.dst_neighbors else batch.neg

            # Time encoding
            torch.cuda.synchronize(); ta = time.perf_counter()
            a_dt = batch.time.unsqueeze(1).float() - a_nbrs.timestamps.float()
            b_dt = batch.time.unsqueeze(1).float() - b_nbrs.timestamps.float()
            a_time_nbr = model.time_enc(a_dt)
            b_time_nbr = model.time_enc(b_dt)
            a_time_nbr = a_time_nbr.masked_fill(~a_nbrs.mask.unsqueeze(-1), 0.0)
            b_time_nbr = b_time_nbr.masked_fill(~b_nbrs.mask.unsqueeze(-1), 0.0)
            t0_enc = model.time_enc(torch.zeros(BS, 1, device=device))
            a_time = torch.cat([t0_enc, a_time_nbr], dim=1)
            b_time = torch.cat([t0_enc, b_time_nbr], dim=1)
            torch.cuda.synchronize(); tb = time.perf_counter()
            time_enc_total += tb - ta

            # Co-occurrence
            torch.cuda.synchronize(); tc = time.perf_counter()
            a_nbr_ids = a_nbrs.neighbor_ids.long()
            b_nbr_ids = b_nbrs.neighbor_ids.long()
            a_full_ids = torch.cat([a_ids.long().unsqueeze(1), a_nbr_ids], dim=1)
            b_full_ids = torch.cat([b_ids.long().unsqueeze(1), b_nbr_ids], dim=1)
            a_co, b_co = model.co_enc(a_full_ids, b_full_ids)
            torch.cuda.synchronize(); td = time.perf_counter()
            co_enc_total += td - tc

            # Patchify + Project
            torch.cuda.synchronize(); te = time.perf_counter()
            d_e = a_nbrs.edge_feats.shape[-1]
            a_edge = torch.cat([torch.zeros(BS, 1, d_e, device=device), a_nbrs.edge_feats], dim=1)
            b_edge = torch.cat([torch.zeros(BS, 1, d_e, device=device), b_nbrs.edge_feats], dim=1)
            ones = torch.ones(BS, 1, dtype=torch.bool, device=device)
            a_mask = torch.cat([ones, a_nbrs.mask], dim=1)
            b_mask = torch.cat([ones, b_nbrs.mask], dim=1)
            S = 1 + K
            if S % patch_size != 0:
                pad = patch_size - S % patch_size
                a_edge = F.pad(a_edge, (0, 0, 0, pad))
                b_edge = F.pad(b_edge, (0, 0, 0, pad))
                a_time_p = F.pad(a_time, (0, 0, 0, pad))
                b_time_p = F.pad(b_time, (0, 0, 0, pad))
                a_co_p = F.pad(a_co, (0, 0, 0, pad))
                b_co_p = F.pad(b_co, (0, 0, 0, pad))
            else:
                a_time_p, b_time_p = a_time, b_time
                a_co_p, b_co_p = a_co, b_co
            a_tok = model._build_token(a_edge, a_time_p, a_co_p, None)
            b_tok = model._build_token(b_edge, b_time_p, b_co_p, None)
            torch.cuda.synchronize(); tf = time.perf_counter()
            patchify_total += tf - te

            # Transformer
            torch.cuda.synchronize(); tg = time.perf_counter()
            n_patches = a_tok.shape[1]
            joint = torch.cat([a_tok, b_tok], dim=1)
            for layer in model.layers:
                joint = layer(joint)
            torch.cuda.synchronize(); th = time.perf_counter()
            transformer_total += th - tg

            # Pool + Decode
            torch.cuda.synchronize(); ti = time.perf_counter()
            a_out = joint[:, :n_patches, :].mean(dim=1)
            b_out = joint[:, n_patches:, :].mean(dim=1)
            a_emb = model.out_proj(a_out)
            b_emb = model.out_proj(b_out)
            torch.cuda.synchronize(); tj = time.perf_counter()
            pool_total += tj - ti

        t_time_enc.append(time_enc_total)
        t_co_enc.append(co_enc_total)
        t_patchify.append(patchify_total)
        t_transformer.append(transformer_total)
        t_pool.append(pool_total)

    # Print results
    prep_ms = np.mean(t_prep) * 1000
    fwd_ms = np.mean(t_fwd) * 1000
    bwd_ms = np.mean(t_bwd) * 1000
    total_ms = prep_ms + fwd_ms + bwd_ms

    te_ms = np.mean(t_time_enc) * 1000
    co_ms = np.mean(t_co_enc) * 1000
    pp_ms = np.mean(t_patchify) * 1000
    tr_ms = np.mean(t_transformer) * 1000
    pl_ms = np.mean(t_pool) * 1000
    fwd_sum = te_ms + co_ms + pp_ms + tr_ms + pl_ms

    print(f"\n{'='*70}")
    print(f"  {ds_name.upper()} — BS={BS}, K={K}, patch_size={patch_size}")
    print(f"  {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    print(f"{'='*70}")

    print(f"\n  HIGH-LEVEL (per step):")
    print(f"    {'Pipeline':<20} {prep_ms:>7.2f} ms  ({prep_ms/total_ms*100:>5.1f}%)")
    print(f"    {'Forward':<20} {fwd_ms:>7.2f} ms  ({fwd_ms/total_ms*100:>5.1f}%)")
    print(f"    {'Backward+Optim':<20} {bwd_ms:>7.2f} ms  ({bwd_ms/total_ms*100:>5.1f}%)")
    print(f"    {'TOTAL':<20} {total_ms:>7.2f} ms")

    print(f"\n  FORWARD INTERNALS (2x _encode_pair per step):")
    print(f"    {'Time Encoding':<20} {te_ms:>7.2f} ms  ({te_ms/fwd_sum*100:>5.1f}%)")
    print(f"    {'Co-occurrence':<20} {co_ms:>7.2f} ms  ({co_ms/fwd_sum*100:>5.1f}%)")
    print(f"    {'Patchify+Project':<20} {pp_ms:>7.2f} ms  ({pp_ms/fwd_sum*100:>5.1f}%)")
    print(f"    {'Transformer':<20} {tr_ms:>7.2f} ms  ({tr_ms/fwd_sum*100:>5.1f}%)")
    print(f"    {'Pool+Decode':<20} {pl_ms:>7.2f} ms  ({pl_ms/fwd_sum*100:>5.1f}%)")
    print(f"    {'─'*45}")
    print(f"    {'Subtotal':<20} {fwd_sum:>7.2f} ms")
    print(f"\n  SHAPES:")
    seq_len = K + 1
    if seq_len % patch_size != 0:
        seq_len += patch_size - seq_len % patch_size
    n_patches = seq_len // patch_size
    print(f"    Sequence length: 1+K = {1+K} → padded {seq_len} → {n_patches} patches")
    print(f"    Transformer input: (BS={BS}, 2×{n_patches}={2*n_patches} patches, d_model=172)")
    print(f"    Co-occ input: (BS={BS}, {1+K}) × (BS={BS}, {1+K})")

    del model, pipeline, graph, optimizer
    torch.cuda.empty_cache()


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    configs = [
        ("wikipedia", 32, 1, 200),
        ("reddit", 64, 2, 200),
        ("lastfm", 512, 16, 200),
    ]

    for ds_name, K, patch_size, BS in configs:
        profile_dataset(ds_name, K, patch_size, BS)


if __name__ == "__main__":
    main()
