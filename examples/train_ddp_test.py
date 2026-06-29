"""DDP (multi-GPU) e2e test for Engine.

Launch with torchrun (2 GPUs minimum):
    torchrun --nproc_per_node=2 examples/train_ddp_test.py

Validates: DDP model wrapping, batch sharding, eval all_reduce, checkpoint
only on rank 0. Uses a tiny model + uci subset for speed (logic, not accuracy).
"""
import os
import sys

import torch


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist_avail = torch.distributed.is_available() and world_size > 1

    if dist_avail and not torch.distributed.is_initialized():
        import datetime
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=datetime.timedelta(minutes=30),
        )
    torch.cuda.set_device(local_rank)

    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.models.graphmixer import GraphMixer
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(42)
    ds = load_dataset("uci", "/mnt/home/gyq/CodeBase/Graph/DG_Data")
    if local_rank == 0:
        print(f"[DDP e2e] world_size={world_size} | {ds.num_edges} edges", flush=True)

    train = ds.get_batches("train", batch_size=200)
    val = ds.get_batches("val", batch_size=200)
    test = ds.get_batches("test", batch_size=200)

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device="cuda")
    model = GraphMixer(d_model=32, d_edge=ds.edge_feat_dim, d_time=8, K=4, num_layers=1)
    neg = RandomNegative(ds.num_nodes)

    config = TrainConfig(
        epochs=2,
        lr=1e-3,
        device="cuda",
        patience=5,
        distributed=dist_avail,
    )
    engine = Engine(
        model, graph, train, val, test,
        neg_strategy=neg, eval_protocol=APEval(), config=config,
    )
    if local_rank == 0:
        print(f"[rank0] distributed={engine._distributed} rank={engine._rank} "
              f"world={engine._world_size} batches/rank={len(engine._sharded_batches())}",
              flush=True)

    metrics = engine.train()
    if local_rank == 0:
        print(f"[rank0] DONE metrics={metrics} "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    if dist_avail:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
