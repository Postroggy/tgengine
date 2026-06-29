"""DDP multi-GPU training of TimeAwareMamba on reddit (K=512).

Launch with torchrun via run_mamba.sh (python passthrough form so glibc239
stays on LD_LIBRARY_PATH for all forked workers):
    scripts/run_mamba.sh python -m torch.distributed.run \
        --nproc_per_node=4 examples/train_ddp_reddit.py --K 512

Records per-epoch wall-clock + peak mem on rank 0.
"""
import argparse
import os
import sys
import time

import torch

# Make benchmarks/ablation importable (benchmarks/ has no __init__.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ablation"))


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist_avail = torch.distributed.is_available() and world_size > 1

    if dist_avail and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    torch.cuda.set_device(local_rank)

    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything
    from bench_mamba_async_amp import _MiniMambaModel

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="reddit")
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    args = p.parse_args()

    seed_everything(42)
    ds = load_dataset(args.dataset, args.data_root)
    if local_rank == 0:
        print(f"[DDP reddit] world_size={world_size} | {ds.num_edges:,} edges "
              f"| K={args.K}", flush=True)

    train = ds.get_batches("train", batch_size=args.batch_size)
    val = ds.get_batches("val", batch_size=args.batch_size)
    test = ds.get_batches("test", batch_size=args.batch_size)

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device="cuda")
    model = _MiniMambaModel(ds.num_nodes, d_model=args.d_model, K=args.K,
                            n_layers=args.n_layers).to("cuda")
    neg = RandomNegative(ds.num_nodes)

    config = TrainConfig(
        epochs=args.epochs, lr=1e-3, device="cuda",
        patience=args.epochs + 5, distributed=dist_avail,
    )
    engine = Engine(
        model, graph, train, val, test,
        neg_strategy=neg, eval_protocol=APEval(), config=config,
    )
    if local_rank == 0:
        print(f"[rank0] distributed={engine._distributed} "
              f"batches/rank={len(engine._sharded_batches())} "
              f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    metrics = engine.train()
    dt = time.time() - t0
    if local_rank == 0:
        print(f"[rank0] DONE AP={metrics.get('ap', 0):.4f} "
              f"time={dt:.1f}s ({dt/args.epochs:.1f}s/epoch) "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
              f"world={world_size}", flush=True)

    if dist_avail:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
