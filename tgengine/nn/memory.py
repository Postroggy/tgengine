from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NodeMemory(nn.Module):
    """Persistent node memory for TGN-style models.

    Maintains a memory vector per node that evolves via GRU.
    Also tracks last_updated_times so models can compute delta time in messages.
    """

    def __init__(self, num_nodes: int, d_model: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        # Non-parameter buffers: updated via detached assignment
        self.register_buffer("memory", torch.zeros(num_nodes, d_model))
        self.register_buffer("last_updated_times", torch.zeros(num_nodes))
        self.updater = nn.GRUCell(d_model, d_model)

    def read(self, node_ids: Tensor) -> Tensor:
        """(B,) → (B, d_model)."""
        return self.memory[node_ids]

    def last_times(self, node_ids: Tensor) -> Tensor:
        """(B,) → (B,) last update timestamps for given nodes."""
        return self.last_updated_times[node_ids]

    def update(self, node_ids: Tensor, messages: Tensor, timestamps: Tensor):
        """Update memory for given nodes and record their timestamps.

        Args:
            node_ids: (B,) nodes to update.
            messages: (B, d_model) input messages.
            timestamps: (B,) current interaction timestamps.
        """
        current = self.memory[node_ids]
        new_memory = self.updater(messages, current)
        self.memory[node_ids] = new_memory.detach()
        self.last_updated_times[node_ids] = timestamps.detach()

    def checkpoint(self):
        """Save memory + timestamp state."""
        return self.memory.clone(), self.last_updated_times.clone()

    def restore(self, state):
        """Restore saved state."""
        mem, times = state
        self.memory.copy_(mem)
        self.last_updated_times.copy_(times)

    def reset(self):
        """Reset all memory and timestamps to zeros."""
        self.memory.zero_()
        self.last_updated_times.zero_()
