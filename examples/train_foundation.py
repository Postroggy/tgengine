"""Foundation model pretraining via the TGEngine (not a custom loop).

Uses the Engine's standard train loop so we get for free:
  - DDP multi-GPU (rank-0-only eval, gradient sync, barrier)
  - AMP mixed precision (Mamba conv1d/ssm wrapped in autocast=False)
  - async prefetch pipeline
  - grad clip, LR scheduler, checkpointing
  - APEval / ThreeWayEval protocols

The FoundationModel.forward() returns the 3-task pretraining loss
(MTM+NTP+LP) when pretrain_mode=True, so the Engine drives pretraining
through its normal _step() → model(prepared) → ModelOutput path.

Usage (single GPU):
    scripts/run_mamba.sh python examples/train_foundation.py \
        --datasets enron BitcoinAlpha uci \
        --epochs 20 --K 64 --d_model 512

Usage (4-GPU DDP):
    CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_mamba.sh python -m torch.distributed.run \
        --nproc_per_node=4 examples/train_foundation.py \
        --datasets enron BitcoinAlpha uci --epochs 20 --distributed
"""
import argparse
import sys
import os
import time

import torch

# Make benchmarks/ablation importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ablation"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["enron", "BitcoinAlpha", "uci"])
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_state", type=int, default=32)
    p.add_argument("--d_time", type=int, default=64)
    p.add_argument("--n_mamba_layers", type=int, default=8)
    p.add_argument("--gca_every", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--async_pipeline", action="store_true", default=False)
    # Pretraining task weights
    p.add_argument("--w_mtm", type=float, default=1.0)
    p.add_argument("--w_ntp", type=float, default=1.0)
    p.add_argument("--w_lp", type=float, default=1.0)
    p.add_argument("--mtm_mask_ratio", type=float, default=0.15)
    p.add_argument("--mtm_block_size", type=int, default=4)
    p.add_argument("--ema_momentum", type=float, default=0.999)
    # DDP
    p.add_argument("--distributed", action="store_true",
                   help="Enable DDP (use with torch.distributed.run)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # --- DDP setup (if launched via torchrun) ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist_avail = torch.distributed.is_available() and world_size > 1
    if dist_avail and not torch.distributed.is_initialized():
        import datetime
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=datetime.timedelta(minutes=120),
        )
    if dist_avail:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    from tgengine.core.dataset import load_dataset
    from tgengine.core.mixed_dataset import MixedDataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.models.foundation import FoundationModel
    from tgengine.utils import seed_everything

    seed_everything(42)

    # --- Load and mix datasets ---
    ds_list = []
    names = []
    for name in args.datasets:
        ds = load_dataset(name, args.data_root)
        ds_list.append(ds)
        names.append(name)
        if local_rank == 0:
            print(f"  [{name:15s}]  {ds.num_nodes:6d} nodes  {ds.num_edges:7d} edges  "
                  f"d_edge={ds.edge_feat_dim}", flush=True)

    mixed = MixedDataset(ds_list, names=names)
    if local_rank == 0:
        print(mixed.summary(), flush=True)

    # --- Build graph (Engine preloads train edges at init) ---
    graph = mixed.make_graph(buffer_size=args.K, device=device)

    # --- Build model ---
    model = FoundationModel(
        d_edge=mixed.d_edge,
        d_model=args.d_model,
        d_state=args.d_state,
        K=args.K,
        d_time=args.d_time,
        n_mamba_layers=args.n_mamba_layers,
        gca_every=args.gca_every,
        pretrain_mode=True,
        task_weights={"mtm": args.w_mtm, "ntp": args.w_ntp, "lp": args.w_lp},
        mtm_mask_ratio=args.mtm_mask_ratio,
        mtm_block_size=args.mtm_block_size,
    ).to(device)
    model.init_ema(momentum=args.ema_momentum)

    n_params = sum(p.numel() for p in model.parameters())
    mamba_count = sum(1 for b in model.blocks if hasattr(b, "ssm"))
    gca_count = sum(1 for b in model.blocks if hasattr(b, "q_proj"))
    if local_rank == 0:
        print(f"Model: {n_params:,} params  "
              f"{mamba_count} Mamba + {gca_count} GCA = {len(model.blocks)} blocks  "
              f"K={args.K}  d={args.d_model}  "
              f"weights(mtms={args.w_mtm},ntp={args.w_ntp},lp={args.w_lp})", flush=True)

    # --- Batches ---
    train_batches = mixed.get_batches(
        "train", batch_size=args.batch_size, balance=True, device=device,
    )
    val_batches = mixed.get_batches(
        "val", batch_size=args.batch_size, device=device,
    )
    test_batches = mixed.get_batches(
        "test", batch_size=args.batch_size, device=device,
    )
    if local_rank == 0:
        print(f"Batches: train={len(train_batches)} (balanced)  "
              f"val={len(val_batches)}  test={len(test_batches)}", flush=True)

    # --- Engine config ---
    neg = RandomNegative(mixed.num_nodes)
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, device=device,
        patience=args.epochs + 5,  # no early stopping for pretraining
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
        async_pipeline=args.async_pipeline,
        distributed=dist_avail,
        eval_strategy="every_n", eval_every=5,
    )

    engine = Engine(
        model, graph, train_batches, val_batches, test_batches,
        neg_strategy=neg, eval_protocol=APEval(), config=cfg,
    )

    if local_rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"  Foundation pretraining via Engine  ({'+'.join(names)})", flush=True)
        print(f"  distributed={dist_avail}  amp={args.use_amp}  "
              f"async={args.async_pipeline}", flush=True)
        print(f"{'='*60}", flush=True)

    # --- Train via Engine ---
    t0 = time.time()
    result = engine.train()
    dt = time.time() - t0

    if local_rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"  Engine pretraining complete in {dt:.1f}s", flush=True)
        print(f"  Test metrics (merged): {result}", flush=True)
        print(f"{'='*60}", flush=True)

        # Save best model (Engine tracks best_val, but for pretraining we
        # save the final state — the Engine's checkpoint logic saves to
        # config.checkpoint_dir if set)
        save_path = "checkpoints/foundation_engine.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Saved final model to {save_path}", flush=True)

    if dist_avail:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
