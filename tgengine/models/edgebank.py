"""EdgeBank: heuristic baseline for link prediction.

Predicts a link exists if the (src, dst) pair appeared in the history window.
No learnable parameters — purely memory-based.

Two variants:
  - "unlimited": remembers all historical edges (score = 1 if seen, 0 otherwise)
  - "tw" (time window): only remembers edges within a sliding time window

Reference: Poursafaei et al., "Towards Better Evaluation for Dynamic Link Prediction"
"""

from __future__ import annotations

import torch
from torch import Tensor

from tgengine.core.batch import PreparedBatch, NeighborData
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import ModelOutput, TemporalModel


class EdgeBank(TemporalModel):
    """EdgeBank heuristic baseline.

    Args:
        mode: "unlimited" (remember all edges) or "tw" (time-window).
        time_window: for "tw" mode, how far back to look (in time units).
    """

    gather_spec = GatherSpec(neighbors=NeighborSpec(k=1, strategy="recency"))

    def __init__(self, mode: str = "unlimited", time_window: float = 0.0):
        super().__init__()
        self.mode = mode
        self.time_window = time_window
        self._edges: set[tuple[int, int]] = set()
        self._timed_edges: list[tuple[float, int, int]] = []
        # Dummy parameter so .to(device) works and optimizer doesn't crash
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src = batch.src
        dst = batch.dst
        neg = batch.neg
        time = batch.time
        device = src.device

        if self.mode == "tw":
            current_time = time.max().item()
            cutoff = current_time - self.time_window
            # Evict expired edges
            while self._timed_edges and self._timed_edges[0][0] < cutoff:
                t, s, d = self._timed_edges.pop(0)
                self._edges.discard((s, d))

        # Score: 1.0 if edge seen in memory, 0.0 otherwise
        pos_score = torch.tensor(
            [1.0 if (s.item(), d.item()) in self._edges else 0.0
             for s, d in zip(src, dst)],
            device=device,
        )
        neg_score = torch.tensor(
            [1.0 if (s.item(), d.item()) in self._edges else 0.0
             for s, d in zip(src, neg)],
            device=device,
        )

        loss = torch.tensor(0.0, device=device)
        return ModelOutput(loss=loss, pos_score=pos_score, neg_score=neg_score)

    def evolve(self, src: Tensor, dst: Tensor, time: Tensor, edge_feat=None):
        """Add observed edges to memory."""
        for s, d, t in zip(src.cpu().tolist(), dst.cpu().tolist(), time.cpu().tolist()):
            self._edges.add((s, d))
            if self.mode == "tw":
                self._timed_edges.append((t, s, d))

    def reset(self):
        """Clear all stored edges."""
        self._edges.clear()
        self._timed_edges.clear()

    @property
    def supports_independent_encode(self) -> bool:
        return False
