"""Pytest config: env fix for scnu glibc 2.39 fast-path.

When tests run under the glibc 2.39 ld-linux launcher (required by mamba_ssm),
LD_LIBRARY_PATH points at ~/glibc239/lib64. That makes system binaries invoked
as subprocesses (gcc, ldconfig — used by triton JIT) crash, because the system
ld-linux (2.31) cannot load glibc 2.39's libc.so.6 (needs GLIBC_2.35).

By the time collection finishes, all glibc239-dependent extensions (mamba_ssm,
tgengine_cuda) are already imported and loaded into the process, so the
glibc239 entries in LD_LIBRARY_PATH are no longer needed by Python itself.
Strip them so subprocess calls to system tooling succeed. No-op when not
running under glibc239 (e.g. local macOS).

Logic lives in tgengine.utils.mamba_env so training entry points reuse it.
"""
from tgengine.utils.mamba_env import setup_mamba_env


def pytest_collection_modifyitems(items):
    setup_mamba_env()
