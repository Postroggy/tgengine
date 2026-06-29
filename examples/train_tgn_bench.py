"""TGN storage-engine benchmark vs tgn.cpp.

Mirrors tgn.cpp link_pred defaults: K=10, batch_size=200, epochs=10,
embedding_dim=100, memory_dim=100. Reports end-to-end per-epoch wall time
and standalone DataPipeline.prepare() throughput (data-fetch only).
"""
from __future__ import annotations

import argparse
import time

import torch

from tgengine.core.batch import RawBatch  # noqa: F401
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.engine import APEval, Engine, TrainConfig
from tgengine.models.tgn import TGN
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import RandomNegative
from tgengine.utils import seed_everything

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=DATA_ROOT)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--d_model", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bench_prepare", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    seed_everything(0)
    print(f"Loading wikipedia from {args.data_root}...")
    ds = load_dataset("wikipedia", args.data_root)
    print(f"  {ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}")

    train_batches = ds.get_batches("train", batch_size=args.batch_size)
    val_batches = ds.get_batches("val", batch_size=args.batch_size)
    test_batches = ds.get_batches("test", batch_size=args.batch_size)

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=args.device)
    model = TGN(num_nodes=ds.num_nodes, d_model=args.d_model, d_edge=ds.edge_feat_dim)
    model.gather_spec = GatherSpec(neighbors=NeighborSpec(k=args.K, strategy="recency"))
    model = model.to(args.device)

    neg = RandomNegative(ds.num_nodes)
    config = TrainConfig(
        epochs=args.epochs + args.warmup,
        lr=1e-4,
        device=args.device,
        patience=args.epochs + args.warmup + 5,
    )
    engine = Engine(
        model, graph,
        train_batches=train_batches,
        val_batches=val_batches,
        test_batches=test_batches,
        neg_strategy=neg,
        eval_protocol=APEval(),
        config=config,
    )

    print(f"  params={sum(p.numel() for p in model.parameters()):,}  K={args.K}  "
          f"batch_size={args.batch_size}  epochs={args.epochs} (+{args.warmup} warmup)")

    t0 = time.time()
    results = engine.train()
    t1 = time.time()
    total = t1 - t0
    n_tb = len(train_batches)
    timed_epochs = args.epochs + args.warmup
    print("\n=== TGEngine TGN end-to-end ===")
    print(f"total wall: {total:.2f}s  avg per-epoch: {total/timed_epochs:.3f}s  "
          f"avg per-batch: {total/(timed_epochs*n_tb)*1000:.2f}ms")
    print(f"result: {results}")

    if args.bench_prepare:
        bench_prepare(model, graph, train_batches, neg, args)


def bench_prepare(model, graph, train_batches, neg, args):
    pipeline = DataPipeline(model.gather_spec, graph)
    n = min(50, len(train_batches))
    for i in range(3):
        rb = train_batches[i]
        rb.neg = neg.sample(rb.src, rb.dst, rb.time, graph, rb.edge_indices)
        pipeline.prepare(rb)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n):
        rb = train_batches[i]
        rb.neg = neg.sample(rb.src, rb.dst, rb.time, graph, rb.edge_indices)
        pipeline.prepare(rb)
    torch.cuda.synchronize()
    dt = time.time() - t0
    edges = n * args.batch_size
    print("\n=== TGEngine prepare() standalone (data-fetch only) ===")
    print(f"{n} batches  {dt:.3f}s  {dt/n*1000:.2f}ms/batch  {edges/dt:.0f} edges/s")


if __name__ == "__main__":
    main()
