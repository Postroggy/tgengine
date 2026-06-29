"""DyGFormer with Mamba v1 backbone (3 blocks) — GPU validation on uci.

Backbone swap: TransformerBlock → MambaBlock. Reuses DyGFormer's patch encoding.
Mamba selective scan: CUDA fast-path via mamba_ssm.selective_scan_fn.
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.models.dygformer import DyGFormer

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ssm_fn
    _HAS_FAST = True
except Exception:
    _ssm_fn = None
    _HAS_FAST = False


class _SelectiveSSM(nn.Module):
    """S6 selective scan. CUDA fast-path via mamba_ssm.selective_scan_fn; Python loop fallback."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: Tensor) -> Tensor:
        xz = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        dt_raw, B_s, C_s = xz.split([dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.log_A)  # (d, n)

        if _HAS_FAST and x.is_cuda:
            u = x.transpose(1, 2).contiguous()
            delta_no_bias = F.linear(dt_raw, self.dt_proj.weight).transpose(1, 2).contiguous()
            B_t = B_s.transpose(1, 2).contiguous()
            C_t = C_s.transpose(1, 2).contiguous()
            y = _ssm_fn(
                u, delta_no_bias, A, B_t, C_t,
                D=self.D,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            return y.transpose(1, 2)

        delta = F.softplus(F.linear(dt_raw, self.dt_proj.weight) + self.dt_proj.bias)
        u = x
        B_b, L, d = u.shape
        n = self.d_state
        h = torch.zeros(B_b, d, n, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            dt = delta[:, t]
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            h = A_bar * h + dt.unsqueeze(-1) * (u[:, t].unsqueeze(-1) * B_s[:, t].unsqueeze(1))
            ys.append((h * C_s[:, t].unsqueeze(1)).sum(-1) + self.D * u[:, t])
        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)
        self.ssm = _SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :residual.shape[1]]
        x_branch = F.silu(x_branch.transpose(1, 2))
        y = self.ssm(x_branch) * F.silu(z)
        return residual + self.out_proj(y)


class DyGFormerMamba(DyGFormer):
    """DyGFormer with Mamba v1 backbone instead of TransformerBlock."""

    def __init__(self, n_layers: int = 3, **kwargs):
        super().__init__(n_layers=n_layers, **kwargs)
        d_joint = self.n_channels * self.d_channel
        self.layers = nn.ModuleList([MambaBlock(d_joint) for _ in range(n_layers)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="uci")
    p.add_argument("--data_root", default="/mnt/home/gyq/CodeBase/Graph/DG_Data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from tgengine.core.dataset import load_dataset
    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.engine import APEval, Engine, TrainConfig
    from tgengine.pipeline.negatives import RandomNegative
    from tgengine.utils import seed_everything

    seed_everything(0)
    ds = load_dataset(args.dataset, args.data_root)
    print(f"{ds.num_edges:,} edges  {ds.num_nodes:,} nodes  d_edge={ds.edge_feat_dim}")

    train = ds.get_batches("train", batch_size=args.batch_size)
    val = ds.get_batches("val", batch_size=args.batch_size)
    test = ds.get_batches("test", batch_size=args.batch_size)

    graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=args.device)
    model = DyGFormerMamba(
        d_model=172, d_edge=ds.edge_feat_dim, d_time=100, d_channel=50,
        K=args.K, patch_size=1, n_layers=args.n_layers, n_heads=2,
        node_feat=ds.node_feat, num_nodes=ds.num_nodes,
    ).to(args.device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}  "
          f"backbone=Mamba×{args.n_layers}  K={args.K}")

    neg = RandomNegative(ds.num_nodes)
    cfg = TrainConfig(epochs=args.epochs, lr=1e-4, device=args.device, patience=args.epochs + 5)
    eng = Engine(model, graph, train_batches=train, val_batches=val,
                 test_batches=test, neg_strategy=neg, eval_protocol=APEval(), config=cfg)

    t0 = time.time()
    res = eng.train()
    dt = time.time() - t0
    print(f"\n=== DyGFormerMamba {args.dataset} K={args.K} Mamba×{args.n_layers} ===")
    print(f"total {dt:.1f}s  avg per-epoch {dt / args.epochs:.2f}s")
    print(f"result: {res}")
    if args.device.startswith("cuda"):
        print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
