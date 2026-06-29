"""Profile Mamba fwd/bwd to locate the bottleneck for task 5 optimization.

Measures per-op time + memory for TimeAwareMambaBlock under realistic
foundation-model shapes (the 100M config: d_model=1024, but shrunk for the
RTX 4080 16GB). Compares fp32 vs AMP, and isolates SSM scan vs projections.

Run on scnu under glibc 2.39:
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=... ~/glibc239/lib64/ld-linux-x86-64.so.2 \
      python3.11 benchmarks/ablation/bench_mamba_fwdbwd.py
"""
import argparse
import time

import torch
import torch.nn as nn

from tgengine.nn.mamba_block import MambaBlock, TimeAwareMambaBlock


def _time_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def profile_block(blk_cls, d_model, K, B, n_layers, amp, device, dt_aware):
    blk = nn.Sequential(*[
        (blk_cls(d_model, d_state=64) if blk_cls is MambaBlock
         else blk_cls(d_model, d_state=64, dt_scale=0.1))
        for _ in range(n_layers)
    ]).to(device)

    x = torch.randn(B, K, d_model, device=device, requires_grad=True)
    dt = torch.rand(B, K, device=device) * 100.0 if dt_aware else None
    grad = torch.randn(B, K, d_model, device=device)

    def fwd_bwd():
        with torch.amp.autocast("cuda", enabled=amp):
            if dt_aware:
                y = blk(x, dt=dt) if n_layers == 1 else _stacked_dt_aware(blk, x, dt)
            else:
                y = blk(x)
        y.backward(grad)

    t = _time_fn(fwd_bwd)
    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    params = sum(p.numel() for p in blk.parameters())
    return t, mem, params


def _stacked_dt_aware(seq, x, dt):
    h = x
    for layer in seq:
        h = layer(h, dt=dt)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--B", type=int, default=200)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"Config: d_model={args.d_model} K={args.K} B={args.B} "
          f"n_layers={args.n_layers}")
    print(f"{'variant':<28} {'amp':<5} {'fwd+bwd(s)':>11} {'mem(GB)':>9} {'params':>10}")
    print("-" * 70)

    configs = [
        ("MambaBlock fp32", MambaBlock, False, False),
        ("MambaBlock amp", MambaBlock, True, False),
        ("TimeAwareMamba fp32", TimeAwareMambaBlock, False, True),
        ("TimeAwareMamba amp", TimeAwareMambaBlock, True, True),
    ]
    for name, cls, amp, dt_aware in configs:
        t, mem, params = profile_block(
            cls, args.d_model, args.K, args.B, args.n_layers, amp, args.device, dt_aware
        )
        print(f"{name:<28} {str(amp):<5} {t:>11.4f} {mem:>9.3f} {params:>10,}")

    # torch.compile IS viable under glibc 2.39 if LD_LIBRARY_PATH is cleaned
    # after importing mamba_ssm (see benchmarks/ablation/bench_compile_verify.py).
    # But it gives 0.96x on Mamba — no speedup, because scan kernel (87.7%) is
    # a closed mamba_ssm CUDA op that compile can't touch. Not included here;
    # use bench_compile_verify.py for the compile-vs-eager comparison.
    print("\n[todo] torch.compile: see bench_compile_verify.py "
          "(viable but 0.96x — no speedup on Mamba).")

    # Per-op breakdown for one TimeAware block via profiler
    print("\n--- Per-op breakdown (TimeAwareMamba, AMP) ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        blk = TimeAwareMambaBlock(args.d_model, d_state=64).to(args.device)
        x = torch.randn(args.B, args.K, args.d_model, device=args.device, requires_grad=True)
        dt = torch.rand(args.B, args.K, device=args.device) * 100.0
        grad = torch.randn(args.B, args.K, args.d_model, device=args.device)
        for _ in range(5):
            with torch.amp.autocast("cuda", enabled=True):
                y = blk(x, dt=dt)
            y.backward(grad)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
