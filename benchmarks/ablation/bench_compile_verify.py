"""Verify torch.compile works under glibc239 once LD_LIBRARY_PATH is cleaned.

Hypothesis: the earlier "torch.compile not viable" conclusion was wrong.
The crash was inductor shelling out to /usr/bin/gcc, which inherited
glibc239's LD_LIBRARY_PATH and couldn't load glibc239's libc (system
ld-linux is 2.31, needs GLIBC_2.35). conftest.py already fixes this for
pytest by stripping glibc239 from LD_LIBRARY_PATH after collection — and
test_crossmamba.py's triton JIT works as a result. The bench script ran
plain python (no conftest), so it hit the gcc crash.

Fix to test: after importing mamba_ssm (which needs glibc239 to load),
strip glibc239 from LD_LIBRARY_PATH. Subsequent inductor subprocesses
(gcc) then use the system glibc and succeed. The compiled .so links
libcuda (system), not glibc239, so it loads fine back into the process.

Run on scnu:
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=<glibc239...> \
      ~/glibc239/lib64/ld-linux-x86-64.so.2 python3.11 bench_compile_verify.py
"""
import os
import time

import torch
import torch._dynamo

# 1) import mamba_ssm WHILE glibc239 is still on LD_LIBRARY_PATH (needed)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401

# 2) NOW strip glibc239 so inductor's gcc subprocess uses system glibc
_lp = os.environ.get("LD_LIBRARY_PATH", "")
if "glibc239" in _lp:
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in _lp.split(":") if p and "glibc239" not in p
    )
    print(f"[env] stripped glibc239 from LD_LIBRARY_PATH")

from tgengine.nn.mamba_block import TimeAwareMambaBlock


def _time(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    dev = "cuda"
    d_model, K, B = 256, 32, 200
    torch.manual_seed(0)

    blk = TimeAwareMambaBlock(d_model, d_state=64, expand=2, dt_scale=0.1).to(dev)
    x = torch.randn(B, K, d_model, device=dev, requires_grad=True)
    dt = torch.rand(B, K, device=dev) * 100.0
    grad = torch.randn(B, K, d_model, device=dev)

    def run(mod):
        def fn():
            with torch.amp.autocast("cuda", enabled=True):
                y = mod(x, dt=dt)
            y.backward(grad)
        return _time(fn)

    print("eager warmup...")
    t_eager = run(blk)
    print(f"eager:      {t_eager*1000:.2f} ms/iter")

    print("compiling (reduce-overhead)...")
    torch._dynamo.reset()
    blk_c = torch.compile(blk, mode="reduce-overhead")
    t_compiled = run(blk_c)
    print(f"compiled:   {t_compiled*1000:.2f} ms/iter")
    print(f"speedup:    {t_eager/t_compiled:.2f}x")


if __name__ == "__main__":
    main()
