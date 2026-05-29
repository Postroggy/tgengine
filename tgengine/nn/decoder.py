from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BilinearDecoder(nn.Module):
    """Bilinear scoring: score = src^T W dst."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)

    def forward(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        """
        Args:
            src_emb: (B, d) or (B, N, d)
            dst_emb: (B, d) or (B, N, d)

        Returns:
            (B,) or (B, N) scores.
        """
        return (self.W(src_emb) * dst_emb).sum(dim=-1)


class MergeDecoder(nn.Module):
    """DyGFormer-style MergeLayer: MLP([src; dst; src*dst])."""

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        combined = torch.cat([src_emb, dst_emb, src_emb * dst_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)


class ConcatMLPDecoder(nn.Module):
    """Simple concat + MLP decoder."""

    def __init__(self, d_model: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(d_model * 2, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)


class ConcatDecoder(nn.Module):
    """DyGLib MergeLayer: cat(src, dst) -> Linear(2d, d) -> ReLU -> Linear(d, 1)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        x = torch.cat([src_emb, dst_emb], dim=-1)
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)
