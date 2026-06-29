#!/usr/bin/env bash
# Launch any python entry point under the glibc 2.39 ld-linux so that
# mamba_ssm's selective_scan_cuda.so (GLIBC_2.32) loads — the fast-path.
# Also sets TRITON_LIBCUDA_PATH so triton JIT finds libcuda without ldconfig.
#
# Usage:
#   scripts/run_mamba.sh examples/train_dygformer_mamba.py --dataset uci --K 32
#   scripts/run_mamba.sh -m pytest tests/ -q
#   scripts/run_mamba.sh python -c "import mamba_ssm; print('ok')"
#
# The runner (tgengine.utils._mamba_runner) calls setup_mamba_env() after
# python starts, stripping glibc239 from LD_LIBRARY_PATH so triton/inductor
# subprocesses (gcc) use the system glibc. This is what makes torch.compile
# and triton JIT work alongside the fast-path. See AGENTS.md §2.
set -euo pipefail

SYSROOT="${MAMBA_GLIBC_SYSROOT:-$HOME/glibc239}"
PY="${MAMBA_PYTHON:-/mnt/home/gyq/.conda/envs/PyGBase/bin/python3.11}"
TORCH_LIB="/mnt/home/gyq/.conda/envs/PyGBase/lib/python3.11/site-packages/torch/lib"
CUDA_LIB="${MAMBA_CUDA_LIB:-$HOME/cuda128/lib64}"

if [ ! -x "$SYSROOT/lib64/ld-linux-x86-64.so.2" ]; then
    echo "error: glibc 2.39 sysroot not found at $SYSROOT/lib64/ld-linux-x86-64.so.2" >&2
    echo "       set MAMBA_GLIBC_SYSROOT or rebuild from sysroot_linux-64=2.39 (AGENTS.md §2)" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "error: python not found at $PY (set MAMBA_PYTHON)" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/lib/x86_64-linux-gnu}"
export LD_LIBRARY_PATH="$SYSROOT/lib64:$SYSROOT/lib:$TORCH_LIB:$CUDA_LIB"

# Decide form: `python` passthrough bypasses the runner (raw interpreter
# semantics for -c / REPL, no triton/inductor so no env strip needed).
# `script.py` and `-m module` go through the runner which calls setup_mamba_env.
LD="$SYSROOT/lib64/ld-linux-x86-64.so.2"
case "${1:-}" in
    "" )
        echo "usage: scripts/run_mamba.sh <script.py|-m module|python> [args...]" >&2
        exit 1
        ;;
    python|python3|python3.11 )
        shift
        exec "$LD" "$PY" "$@"
        ;;
    * )
        exec "$LD" "$PY" -m tgengine.utils._mamba_runner "$@"
        ;;
esac
