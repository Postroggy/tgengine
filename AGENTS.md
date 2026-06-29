# AGENTS.md — Agent 环境与运行约束

本文件记录在 scnu 服务器上跑 TGEngine（尤其 Mamba fast-path）必须遵守的环境约束。
这些约束来自 2026-06-28 的实战修复，违反会导致卡死、OOM 或 ABI 错误。

## 1. GPU 使用

- 只用 `CUDA_VISIBLE_DEVICES=1`（RTX 4080，16.7GB）。GPU 0/2/3 是同学的，禁止占用。
- 不设置 `CUDA_DEVICE_ORDER`。
- GPU 1 可能被其他用户（如 hzx）占用，运行前先 `nvidia-smi` 查剩余显存。显存紧时降 batch_size / K，不要硬冲 OOM。
- 详见 CLAUDE.md「GPU 使用约束」。

## 2. Mamba fast-path 必须用 glibc 2.39 启动

**问题**：`mamba_ssm` 的 `selective_scan_cuda.so` 有 `__libc_single_threaded@GLIBC_2.32` version requirement，scnu 系统 glibc 2.31 无法加载。Python loop fallback 能跑但慢 5.8x（77s/epoch vs 13s/epoch）。**走 fast-path 必须整进程切换 glibc 2.39。**

**启动方式（fast-path 训练）**：
```bash
SYSROOT=~/glibc239
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$SYSROOT/lib64:$SYSROOT/lib:/mnt/home/gyq/.conda/envs/PyGBase/lib/python3.11/site-packages/torch/lib:~/cuda128/lib64"
$SYSROOT/lib64/ld-linux-x86-64.so.2 /mnt/home/gyq/.conda/envs/PyGBase/bin/python3.11 <script.py> [args]
```

**约束**：
- 任何 import `mamba_ssm` / `selective_scan_fn` / CrossMamba / DyGFormerMamba 的脚本，**必须用上面的 ld-linux 启动**，否则 ImportError: GLIBC_2.32 not found。
- 不要用 `python` 直接跑（系统 glibc 2.31，会 fallback 到 Python loop 或直接报错）。
- 不要尝试重编译 mamba_ssm 消除 GLIBC_2.32——试过 10 种方案（system gcc / conda gcc / 源码 / --no-binary），torch extension build 工具链必然引入这个 version node，消不掉。直接用 glibc 2.39 切换。
- 不要替换系统 /lib/.../libc.so.6（会 brick 服务器）。本方案只整进程切换，不动系统 glibc。

**glibc 2.39 sysroot 位置**：`~/glibc239/`（已持久化）。若丢失，从 conda-forge `sysroot_linux-64=2.39` 的 .conda 包解压重建（见 memory `project_mamba_fastpath_glibc.md`）。

**triton JIT / torch.compile 的 subprocess 坑（2026-06-29 补）**：
glibc239 启动后 `LD_LIBRARY_PATH` 含 `~/glibc239/lib64`，任何 subprocess 调系统二进制（`gcc`、`ldconfig`——triton JIT 和 inductor 编译要用）会继承该 env，加载 glibc239 的 libc.so.6 时系统 ld-linux(2.31) 报 `GLIBC_2.35 not found` 崩溃。

**解法**：mamba_ssm import 后，调 `tgengine.utils.mamba_env.setup_mamba_env()` 清掉 LD_LIBRARY_PATH 的 glibc239 部分。mamba_ssm 的 .so 已加载进进程，glibc 向后兼容，清掉后 subprocess 用系统 glibc 正常工作。`tests/conftest.py` 在 collection 后自动调；训练入口用 `scripts/run_mamba.sh`（它经 `tgengine.utils._mamba_runner` 自动调）。

**torch.compile + mamba**：selective_scan_cuda 是闭源 pybind11 op，dynamo 无法 trace。`allow_in_graph` 不行（AOTAutograd 仍 trace 撞 FakeTensor），`custom_op` 要手写 backward（selective_scan bwd 是闭源 CUDA kernel，不实际）。正解是 `tgengine/nn/mamba_block.py` 的 `_selective_scan_call` 用 `@torch.compiler.disable` 包裹——dynamo 在 SSM 处干净 graph-break，SSM eager 跑（autograd backward 完整），inductor 融合周围 projection。实测 default mode 1.06x。**不要用 `mode="reduce-overhead"`**（CUDA graph + graph break 会 tensor 覆写崩）。

**启动器**：`scripts/run_mamba.sh <script.py|-m module|python> [args]` 一行启动，自动 glibc239 ld-linux + `TRITON_LIBCUDA_PATH` + `setup_mamba_env`。例：
```bash
scripts/run_mamba.sh examples/train_dygformer_mamba.py --dataset uci --K 32
scripts/run_mamba.sh -m pytest tests/ -q
```

## 3. CUDA extension（tgengine_cuda）加载

`tgengine/core/kernels.py` 的 `_ensure_cuda_ext()` 已 patch 处理三个坑，不要回退：

1. **stale lock 文件**：`~/.cache/torch_extensions/py311_cu128/tgengine_cuda/lock` 若残留（0 字节），cpp_extension.load 会永久等待。若 _ensure_cuda_ext 卡住，先 `rm -f` 该 lock。
2. **libcudart preload**：.so 加载时找 `libcudart.so.12`，但 `os.environ LD_LIBRARY_PATH` 对 dlopen post-start 无效。代码已用 `ctypes.CDLL(..., RTLD_GLOBAL)` 从 `torch/../nvidia/cuda_runtime/lib/libcudart.so.12` 预加载。不要删这段。
3. **CUDA_HOME**：代码优先用 `~/cuda128`（CUDA 12.8 软链），不用系统 `/usr/local/cuda`（11.7，会编出链接 11.7 的损坏 .so）。
4. **flag 仅成功时设**：`_cuda_ext_loaded` 在 load 成功后才设 True，允许失败重试。不要在 load 前设。

**验证**：`from tgengine.core import kernels; kernels._ensure_cuda_ext(); print(kernels.HAS_CUDA_EXT)` 应为 True。

## 4. conda 环境

- 主环境：`PyGBase`（PyTorch 2.9.1+cu128，Python 3.11）。
- 编译 tgn.cpp 等需 C++20：用单独的 `tgn` conda env（gcc-13.4 + cmake 3.28 + sysroot 2.28）+ 系统 glibc 2.31 链接（LDFLAGS=-L/usr/lib/x86_64-linux-gnu）。详见 memory `project_tgn_cpp_benchmark.md`。
- setuptools 必须 `<81`（mamba_ssm setup.py 用 `pkg_resources`，setuptools 81+ 移除了）。

## 5. 长任务运行模式

不要让 Claude Code 直接持有 SSH 前台训练进程。用 tmux + 日志：
```bash
ssh scnu 'tmux new-session -d -s train "<run_script.sh> 2>&1 | tee /tmp/train.log"'
# 查进度: ssh scnu 'tail -20 /tmp/train.log'
# 等完成: Monitor 工具 until ssh scnu 'grep -q "Best" /tmp/train.log'
```
tmux 内必须显式 `eval "$(conda shell.bash hook)"` + `conda activate`。用 `PYTHONUNBUFFERED=1` + `python -u` 实时输出。

## 6. 相关 memory

- `project_mamba_fastpath_glibc.md` — fast-path + glibc 2.39 修复全过程
- `project_tgn_cpp_benchmark.md` — tgn.cpp 编译环境适配
- `reference_gpu_mapping.md` — scnu GPU 映射
