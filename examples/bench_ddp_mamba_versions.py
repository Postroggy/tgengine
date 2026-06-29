"""DDP K=512: Mamba v1 (TimeAwareMambaBlock) vs v2 (Mamba2Block) training speed.

Validates the hypothesis: at long sequences (K=512), Mamba-2's SSD kernel is
faster than Mamba-1's selective scan. Same data, same DDP setup, same config —
only the backbone differs.

Requires mamba2 env (causal_conv1d + Mamba2). Launch 4 workers manually so
each goes through the glibc239 ld-linux (torchrun execs python directly and
loses the launcher). See /tmp/run_ddp_versions.sh pattern.

    MAMBA_PYTHON=<mamba2>/python3.11 scripts/run_mamba.sh python \
        examples/bench_ddp_mamba_versions.py --mamba v2 --K 512
"""
import argparse
import os
import sys
import time

import torch

# Mamba2's triton SSD kernels JIT via gcc subprocess. Order matters:
#   1. import mamba_ssm WHILE glibc239 is on LD_LIBRARY_PATH (selective_scan_cuda
#      + causal_conv1d_cuda need GLIBC_2.32)
#   2. setup_mamba_env() strips glibc239 so triton's gcc subprocess uses the
#      system glibc (else "GLIBC_2.35 not found" crash)
# The run_mamba.sh `python` form bypasses the runner, so do it here explicitly.
import mamba_ssm  # noqa: F401  — must precede setup_mamba_env
from tgengine.utils.mamba_env import setup_mamba_env
setup_mamba_env()

from tgengine.models.base import TemporalModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ablation"))


class _MiniMamba23Model(TemporalModel):
    """Generic Mamba2/Mamba3 backbone model (no external Δt; both blocks are
    pure sequence models). Used for v2 and v3 benchmarks — only the block
    class differs."""

    def __init__(self, num_nodes, block_class, d_model=128, K=512, n_layers=2,
                 d_state=128, **block_kwargs):
        super().__init__()
        from tgengine.core.gather_spec import GatherSpec, NeighborSpec

        self.K = K
        self.d_model = d_model
        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, for_nodes=("src", "dst", "neg"))
        )
        self.node_emb = torch.nn.Embedding(num_nodes, d_model)
        self.edge_time_proj = torch.nn.Linear(1, d_model)
        self.layers = torch.nn.ModuleList([
            block_class(d_model, d_state=d_state, **block_kwargs) for _ in range(n_layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model)

    def _encode_side(self, nbr, query_time):
        ids = nbr.neighbor_ids.long().clamp(min=0)
        x = self.node_emb(ids)
        dt = (query_time.unsqueeze(1) - nbr.timestamps).clamp(min=0).float().unsqueeze(-1)
        x = x + self.edge_time_proj(dt)
        h = x
        for layer in self.layers:
            h = layer(h)  # Mamba2/Mamba3 blocks have no dt arg
        h = self.norm(h)
        last_idx = nbr.mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        return h[torch.arange(h.size(0), device=h.device), last_idx]

    def encode(self, batch):
        from tgengine.models.base import EmbeddingBundle

        src = self._encode_side(batch.src_neighbors, batch.time)
        dst = self._encode_side(batch.dst_neighbors, batch.time)
        neg = self._encode_side(batch.neg_neighbors, batch.time)
        return EmbeddingBundle(src=src, dst=dst, neg=neg)

    def forward(self, batch):
        from tgengine.models.base import ModelOutput
        import torch.nn.functional as F
        bundle = self.encode(batch)
        pos = (bundle.src * bundle.dst).sum(-1)
        neg = (bundle.src * bundle.neg).sum(-1)
        loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        loss = loss + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        return ModelOutput(loss=loss, pos_score=pos, neg_score=neg)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist_avail = torch.distributed.is_available() and world_size > 1

    if dist_avail and not torch.distributed.is_initialized():
        import datetime
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=datetime.timedelta(minutes=60),
        )
    torch.cuda.set_device(local_rank)

    p = argparse.ArgumentParser()
    p.add_argument("--mamba", choices=["v1", "v2", "v3"], default="v3")
    p.add_argument("--dataset", default="reddit")
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    args = p.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything
    from bench_mamba_async_amp import _MiniMambaModel

    seed_everything(42)
    ds = load_dataset(args.dataset, args.data_root)
    if local_rank == 0:
        print(f"[DDP {args.mamba}] world={world_size} | {ds.num_edges:,} edges | K={args.K}",
              flush=True)

    train = ds.get_batches("train", batch_size=args.batch_size)
    val = ds.get_batches("val", batch_size=args.batch_size)
    test = ds.get_batches("test", batch_size=args.batch_size)

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device="cuda")
    if args.mamba == "v1":
        model = _MiniMambaModel(ds.num_nodes, d_model=args.d_model, K=args.K,
                                n_layers=args.n_layers).to("cuda")
    else:
        from tgengine.nn.mamba_block import Mamba2Block, Mamba3Block
        block_class = Mamba3Block if args.mamba == "v3" else Mamba2Block
        model = _MiniMamba23Model(ds.num_nodes, block_class,
                                  d_model=args.d_model, K=args.K,
                                  n_layers=args.n_layers, expand=2).to("cuda")
    neg = RandomNegative(ds.num_nodes)

    config = TrainConfig(epochs=args.epochs, lr=1e-3, device="cuda",
                         patience=args.epochs + 5, distributed=dist_avail)
    engine = Engine(model, graph, train, val, test,
                    neg_strategy=neg, eval_protocol=APEval(), config=config)
    if local_rank == 0:
        print(f"[rank0] {args.mamba} batches/rank={len(engine._sharded_batches())} "
              f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    metrics = engine.train()
    dt = time.time() - t0
    if local_rank == 0:
        print(f"[rank0] DONE {args.mamba} AP={metrics.get('ap', 0):.4f} "
              f"time={dt:.1f}s ({dt/args.epochs:.1f}s/ep) "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    if dist_avail:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
