"""Reddit large-graph mamba training: sync vs async wall-clock comparison.

Validates whether async prefetch delivers a real speedup on a larger graph
(reddit: 592K edges, vs uci's ~20K). Same model, same GPU, same config —
only the data pipeline differs. This is the empirical answer to "does the
pipeline get fully utilized on big datasets."

Uses _MiniMambaModel (TimeAwareMambaBlock backbone) from bench_mamba_async_amp.
Shrunk d_model for the 16GB 4080; NOT an accuracy benchmark — the goal is
relative sync-vs-async timing.

Run on scnu:
  scripts/run_mamba.sh benchmarks/ablation/bench_mamba_reddit.py [--epochs N]
"""
import argparse
import time

import torch

from benchmarks.ablation.bench_mamba_async_amp import _MiniMambaModel


def build_engine(ds, async_pipe, amp, args):
    from tgengine.utils import seed_everything
    seed_everything(0)
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative

    train = ds.get_batches("train", batch_size=args.batch_size)
    val = ds.get_batches("val", batch_size=args.batch_size)
    test = ds.get_batches("test", batch_size=args.batch_size)

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device="cuda")
    model = _MiniMambaModel(ds.num_nodes, d_model=args.d_model, K=args.K,
                            n_layers=args.n_layers).to("cuda")
    cfg = TrainConfig(epochs=args.epochs, lr=1e-3, device="cuda",
                      patience=args.epochs + 5, use_amp=amp,
                      async_pipeline=async_pipe)
    return Engine(model, graph, train, val, test,
                  neg_strategy=RandomNegative(ds.num_nodes),
                  eval_protocol=APEval(), config=cfg), model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="reddit")
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--skip_async", action="store_true",
                   help="only run sync (for a quick probe)")
    args = p.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.utils import seed_everything

    seed_everything(0)
    ds = load_dataset(args.dataset, args.data_root)
    print(f"{ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}")
    print(f"train batches: {ds.train_size // args.batch_size}")

    # --- sync fp32 ---
    print("\n[sync + fp32]")
    eng, model = build_engine(ds, async_pipe=False, amp=False, args=args)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m_sync = eng.train()
    t_sync = time.time() - t0
    print(f"  AP={m_sync.get('ap', 0):.4f}  time={t_sync:.1f}s  "
          f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    if args.skip_async:
        print("\n[--skip_async] done.")
        return

    # --- async fp32 ---
    print("[async + fp32]")
    torch.cuda.reset_peak_memory_stats()
    eng, model = build_engine(ds, async_pipe=True, amp=False, args=args)
    t0 = time.time()
    m_async = eng.train()
    t_async = time.time() - t0
    print(f"  AP={m_async.get('ap', 0):.4f}  time={t_async:.1f}s  "
          f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    print(f"\n=== reddit sync vs async ===")
    print(f"sync  AP={m_sync.get('ap',0):.4f}  {t_sync:.1f}s")
    print(f"async AP={m_async.get('ap',0):.4f}  {t_async:.1f}s")
    print(f"speedup: {t_sync/t_async:.2f}x   AP diff: {abs(m_async.get('ap',0)-m_sync.get('ap',0)):.4f}")


if __name__ == "__main__":
    main()
