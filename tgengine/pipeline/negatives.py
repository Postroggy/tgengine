from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from tgengine.core.temporal_graph import TemporalGraph


class NegativeStrategy(ABC):
    """Base class for negative sampling strategies."""

    @abstractmethod
    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        """Sample negative destination nodes.

        Args:
            src: (B,) source nodes.
            dst: (B,) positive destination nodes.
            time: (B,) timestamps.
            graph: the temporal graph (for historical neg).

        Returns:
            Tensor of shape (B,) or (B, N_neg) negative node IDs.
        """
        ...


class RandomNegative(NegativeStrategy):
    """Uniform random negative sampling. O(1), GPU-native."""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        return torch.randint(0, self.num_nodes, (src.shape[0],), device=src.device)


class HistoricalNegative(NegativeStrategy):
    """Sample negatives from src's historical neighbors."""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        # Get src's recent neighbors, randomly pick one as negative
        nbrs = graph.recent(src, time, k=32)
        valid_counts = nbrs.mask.sum(dim=1)  # (B,)
        # Random index within valid range
        rand_idx = (torch.rand(src.shape[0], device=src.device) * valid_counts.float()).long()
        rand_idx = rand_idx.clamp(max=nbrs.seq_len - 1)
        neg = nbrs.neighbor_ids.gather(1, rand_idx.unsqueeze(1)).squeeze(1)
        # Fallback to random for nodes with no history
        no_history = valid_counts == 0
        if no_history.any():
            neg[no_history] = torch.randint(0, self.num_nodes, (no_history.sum(),), device=src.device)
        return neg


class InductiveNegative(NegativeStrategy):
    """Sample negatives from nodes never seen in training (inductive split).

    Nodes are those that have total degree zero in the training graph — i.e.,
    they never appeared as src or dst in the training period.
    """

    def __init__(self, inductive_nodes: Tensor):
        """
        Args:
            inductive_nodes: 1-D tensor of node IDs unseen during training.
        """
        self.inductive_nodes = inductive_nodes

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        n = src.shape[0]
        rand_idx = torch.randint(0, len(self.inductive_nodes), (n,), device=src.device)
        return self.inductive_nodes.to(src.device)[rand_idx]


class FixedNegative(NegativeStrategy):
    """Fixed negative lists (for TGB evaluation)."""

    def __init__(self, neg_lists: Tensor):
        """
        Args:
            neg_lists: (N_edges, N_neg) pre-loaded negative node IDs.
        """
        self.neg_lists = neg_lists

    def sample(self, src: Tensor, dst: Tensor, time: Tensor, graph: TemporalGraph) -> Tensor:
        raise NotImplementedError("FixedNegative requires edge_indices in RawBatch")
