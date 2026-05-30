"""AMP vs no-AMP training throughput benchmark: GraphMixer on Reddit.

Measures training step throughput only (no eval), to isolate compute speed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/amp_speed_test.py
"""

from __future__ import annotations

import gc
import time

import torch
from tqdm import tqdm

from tgengine import GraphMixer, RandomNegative, TemporalGraph, TrainConfig, load_dataset
from tgengine.pipeline import DataPipeline

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
DEVICE = "cuda"
BATCH_SIZE = 600
K = 32
D_MODEL = 172
WARMUP_BATCHES = 10   # discard these batches before timing
MEASURE_BATCHES = 50  # time these batches


def run_one(use_amp: bool) -> tuple[float, int]:
    """Returns (seconds_per_batch, total_edges_processed)."""
    dataset = load_dataset("reddit", dataset_path=DATA_ROOT)
    N = dataset.num_nodes
    train_batches = dataset.get_batches("train", BATCH_SIZE, device=DEVICE)
    print(f"  reddit: {N} nodes, {len(train_batches)} train batches")

    graph = TemporalGraph(N, buffer_size=K, edge_feat_dim=dataset.edge_feat_dim, device=DEVICE)
    # preload all train edges (Engine does this at init)
    for rb in train_batches:
        graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
    graph.freeze_csr()

    model = GraphMixer(
        d_model=D_MODEL, d_edge=dataset.edge_feat_dim,
        d_time=100, K=K, num_layers=2, dropout=0.1,
    ).to(DEVICE)
    model.train()

    pipeline = DataPipeline(model.gather_spec, graph)
    neg_strategy = RandomNegative(N)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_batches = min(WARMUP_BATCHES + MEASURE_BATCHES, len(train_batches))

    total_edges = 0
    t_start = None

    for i in tqdm(range(n_batches), desc=f"AMP={'ON ' if use_amp else 'OFF'}"):
        rb = train_batches[i]
        neg = neg_strategy.sample(rb.src, rb.dst, rb.time, graph, rb.edge_indices)
        rb.neg = neg
        prepared = pipeline.prepare(rb)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(prepared)
        scaler.scale(output.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i == WARMUP_BATCHES - 1:
            # start timing after warm-up
            torch.cuda.synchronize()
            t_start = time.perf_counter()

        if i >= WARMUP_BATCHES:
            total_edges += rb.src.shape[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    del dataset, graph, model, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, total_edges


def main():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"Config: batch_size={BATCH_SIZE}  K={K}  d_model={D_MODEL}")
    print(f"        warmup={WARMUP_BATCHES} batches, measure={MEASURE_BATCHES} batches")
    print("=" * 60)

    results = {}
    for use_amp in (False, True):
        tag = "AMP=ON " if use_amp else "AMP=OFF"
        print(f"\n[{tag}]")
        elapsed, edges = run_one(use_amp)
        sec_per_batch = elapsed / MEASURE_BATCHES
        throughput = edges / elapsed
        results[tag] = (elapsed, sec_per_batch, throughput)
        print(f"  total={elapsed:.2f}s  {sec_per_batch*1000:.1f}ms/batch  {throughput:.0f} edges/s")

    print("\n" + "=" * 60)
    t_off, spb_off, thr_off = results["AMP=OFF"]
    t_on,  spb_on,  thr_on  = results["AMP=ON "]
    speedup = t_off / t_on
    print(f"AMP=OFF : {t_off:.2f}s  {spb_off*1000:.1f}ms/batch  {thr_off:.0f} edges/s")
    print(f"AMP=ON  : {t_on:.2f}s  {spb_on*1000:.1f}ms/batch  {thr_on:.0f} edges/s")
    print(f"Speedup : {speedup:.2f}x")


if __name__ == "__main__":
    main()
