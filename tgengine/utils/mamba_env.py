"""Environment setup for mamba_ssm fast-path + triton + torch.compile.

The mamba_ssm selective_scan_cuda.so requires GLIBC_2.32, which scnu's
system glibc (2.31) cannot provide. The workaround (AGENTS.md §2) is to
launch the whole python process under a glibc 2.39 ld-linux, with
LD_LIBRARY_PATH pointing at ~/glibc239/lib64.

That creates a secondary problem: any subprocess that shells out to a
system binary (gcc, ldconfig) — used by triton JIT and torch.compile's
inductor backend — inherits LD_LIBRARY_PATH, loads glibc239's libc.so.6,
and crashes because the system ld-linux (2.31) needs GLIBC_2.35.

The fix: after mamba_ssm is imported (its .so is already loaded into the
process), strip the glibc239 entries from LD_LIBRARY_PATH. Subsequent
subprocesses then use the system glibc and succeed. The already-loaded
mamba_ssm/torch extensions are unaffected — glibc is backwards-compatible
and the symbols they need remain in the running process.

Usage (call once, after importing mamba_ssm):

    from tgengine.utils.mamba_env import setup_mamba_env
    setup_mamba_env()   # idempotent; safe to call in every entry point

This is what tests/conftest.py calls at collection time, and what
training entry points should call after `import mamba_ssm`.
"""

from __future__ import annotations

import os

_STRIPPED = False


def setup_mamba_env() -> bool:
    """Strip glibc239 from LD_LIBRARY_PATH so triton/inductor subprocesses work.

    Safe to call multiple times (idempotent). No-op when not running under
    the glibc239 launcher (e.g. local macOS, or an env with native glibc ≥2.32).

    Returns:
        True if LD_LIBRARY_PATH was modified, False if it was already clean.
    """
    global _STRIPPED
    if _STRIPPED:
        return False
    lp = os.environ.get("LD_LIBRARY_PATH", "")
    if not lp or "glibc239" not in lp:
        _STRIPPED = True
        return False
    cleaned = ":".join(p for p in lp.split(":") if p and "glibc239" not in p)
    os.environ["LD_LIBRARY_PATH"] = cleaned
    _STRIPPED = True
    return True


def mamba_fast_path_available() -> bool:
    """Whether selective_scan_fn can be imported (fast-path ready).

    Does NOT call setup_mamba_env — the caller controls when to strip
    LD_LIBRARY_PATH (must be after mamba_ssm import).
    """
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import (  # noqa: F401
            selective_scan_fn,
        )
        return True
    except Exception:
        return False
