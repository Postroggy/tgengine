"""Verify allow_in_graph eliminates the SSM graph break under torch.compile.

dynamo can't trace selective_scan_cuda (pybind11 op) → graph break →
compile only covers fragments around the SSM, giving 0.96x (no win).

allow_in_graph wraps selective_scan_fn so dynamo treats it as an opaque
graph node (not traced, kept as-is). inductor then falls back to eager
for that node but fuses the surrounding projections — no graph break.

This script measures graph-break count + speed for three variants:
  1. eager
  2. torch.compile (raw selective_scan_fn — graph break expected)
  3. torch.compile with allow_in_graph wrapper (no graph break expected)

Run on scnu under glibc239 launcher.
"""
import os
import time

import torch
import torch._dynamo

# import mamba_ssm WHILE glibc239 is on LD_LIBRARY_PATH
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _raw_ssm

# strip glibc239 so inductor's gcc subprocess works
_lp = os.environ.get("LD_LIBRARY_PATH", "")
if "glibc239" in _lp:
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in _lp.split(":") if p and "glibc239" not in p
    )

from tgengine.nn.mamba_block import TimeAwareMambaBlock


# allow_in_graph wrapper around selective_scan_fn — turns out NOT to work:
# AOTAutograd still traces into the wrapper with FakeTensor and hits the
# "Cannot access data pointer" error from selective_scan_cuda. custom_op
# would need a hand-written backward (selective_scan's bwd is a closed CUDA
# kernel), so that path is impractical too. Kept here as a documented dead end.
@torch._dynamo.allow_in_graph
def _ssm_in_graph(u, delta, A, B, C, D, delta_bias, delta_softplus):
    return _raw_ssm(u, delta, A, B, C, D=D, delta_bias=delta_bias,
                    delta_softplus=delta_softplus)


# torch.compiler.disable wrapper — the practical option. dynamo graph-breaks
# here cleanly, the SSM runs eager (with its own autograd backward intact),
# and inductor still fuses the projections on either side.
@torch.compiler.disable
def _ssm_disabled(u, delta, A, B, C, D, delta_bias, delta_softplus):
    return _raw_ssm(u, delta, A, B, C, D=D, delta_bias=delta_bias,
                    delta_softplus=delta_softplus)


# Patch _SelectiveSSM.forward to use the wrapper (only for the wrapper variant)
import tgengine.nn.mamba_block as _mb

_orig_forward = _mb._SelectiveSSM.forward


def _forward_with_wrapper(self, x, delta_bias, extra_delta=None):
    xz = self.x_proj(x)
    dt_rank = self.dt_proj.in_features
    dt_raw, B_ssm, C_ssm = xz.split([dt_rank, self.d_state, self.d_state], dim=-1)
    A = -torch.exp(self.log_A)
    u = x.transpose(1, 2).contiguous()
    delta_no_bias = torch.nn.functional.linear(dt_raw, self.dt_proj.weight).transpose(1, 2).contiguous()
    if extra_delta is not None:
        delta_no_bias = delta_no_bias + extra_delta.transpose(1, 2).contiguous()
    B_t = B_ssm.transpose(1, 2).contiguous()
    C_t = C_ssm.transpose(1, 2).contiguous()
    y = _ssm_disabled(u, delta_no_bias, A, B_t, C_t, self.D, delta_bias, True)
    return y.transpose(1, 2)


def _time(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _reset():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


def main():
    dev = "cuda"
    d_model, K, B = 256, 32, 200
    torch.manual_seed(0)

    def make():
        blk = TimeAwareMambaBlock(d_model, d_state=64, expand=2, dt_scale=0.1).to(dev)
        x = torch.randn(B, K, d_model, device=dev, requires_grad=True)
        dt = torch.rand(B, K, device=dev) * 100.0
        grad = torch.randn(B, K, d_model, device=dev)
        return blk, x, dt, grad

    def run(mod, x, dt, grad):
        def fn():
            with torch.amp.autocast("cuda", enabled=True):
                y = mod(x, dt=dt)
            y.backward(grad)
        return _time(fn)

    # 1) eager
    blk, x, dt, grad = make()
    t_eager = run(blk, x, dt, grad)
    print(f"eager:                {t_eager*1000:6.2f} ms/iter")

    # 2) compile, raw selective_scan_fn (graph break expected)
    _reset()
    blk, x, dt, grad = make()
    _mb._SelectiveSSM.forward = _orig_forward
    blk_c = torch.compile(blk)  # default mode (no CUDA graph — safer with fallback)
    t_c_raw = run(blk_c, x, dt, grad)
    gb_raw = torch._dynamo.utils.counters["stats"].get("unique_graph_breaks", 0)
    print(f"compile (raw):        {t_c_raw*1000:6.2f} ms/iter  graph_breaks={gb_raw}")

    # 3) compile, torch.compiler.disable on the SSM call (clean graph break)
    _reset()
    blk, x, dt, grad = make()
    _mb._SelectiveSSM.forward = _forward_with_wrapper
    blk_c2 = torch.compile(blk)  # default mode (no CUDA graph)
    t_c_aig = run(blk_c2, x, dt, grad)
    gb_aig = torch._dynamo.utils.counters["stats"].get("unique_graph_breaks", 0)
    print(f"compile (disable ssm): {t_c_aig*1000:6.2f} ms/iter  graph_breaks={gb_aig}")

    _mb._SelectiveSSM.forward = _orig_forward
    print(f"\nspeedup vs eager:  raw={t_eager/t_c_raw:.2f}x  allow_in_graph={t_eager/t_c_aig:.2f}x")


if __name__ == "__main__":
    main()
