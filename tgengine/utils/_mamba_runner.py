"""Entry-point wrapper that prepares the mamba fast-path env then runs a script.

Used by scripts/run_mamba.sh. Splits responsibilities:
  - run_mamba.sh: launches python under the glibc 2.39 ld-linux (needed for
    selective_scan_cuda.so to load) + sets TRITON_LIBCUDA_PATH.
  - this runner: calls setup_mamba_env() to strip glibc239 from
    LD_LIBRARY_PATH (so triton/inductor gcc subprocesses work), then runs
    the user's script with a clean sys.argv.

Supports two invocation forms (the `python` passthrough is handled directly
by run_mamba.sh, not here, since `python -c`/REPL need raw interpreter
semantics that runpy can't reproduce):
    run_mamba.sh script.py [args...]      # run a .py file
    run_mamba.sh -m module [args...]      # run a module (e.g. pytest)
"""

from __future__ import annotations

import runpy
import sys

from tgengine.utils.mamba_env import setup_mamba_env


def main() -> None:
    setup_mamba_env()
    argv = sys.argv[1:]
    if not argv:
        print("usage: run_mamba.sh <script.py|-m module> [args...]", file=sys.stderr)
        print("       (for `python -c` / REPL, run_mamba.sh python ... bypasses this runner)",
              file=sys.stderr)
        sys.exit(2)

    if argv[0] == "-m":
        module = argv[1]
        sys.argv = [module, *argv[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = argv[0]
        sys.argv = argv  # script sees [script, *args]
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
