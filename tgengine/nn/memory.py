from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class NodeMemory(nn.Module):
    """Persistent node memory for TGN-style models.

    Maintains a learnable embedding per node that evolves over time.
    """

    def __init__(self, num_nodes: int, d_model: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        # Non-parameter buffer: memory is updated via detached assignment
        self.register_buffer("memory", Tensor(num_nodes, d_model).zero_())
        self.updater = nn.GRUCell(d_model, d_model)

    def read(self, node_ids: Tensor) -> Tensor:
        """Read memory for given nodes. (B,) → (B, d_model)."""
        return self.memory[node_ids]

    def update(self, node_ids: Tensor, messages: Tensor):
        """Update memory for given nodes with new messages.

        Args:
            node_ids: (B,) nodes to update.
            messages: (B, d_model) input messages.
        """
        current = self.memory[node_ids]
        new_memory = self.updater(messages, current)
        self.memory[node_ids] = new_memory.detach()

    def checkpoint(self) -> Tensor:
        """Save memory state."""
        return self.memory.clone()

    def restore(self, state: Tensor):
        """Restore memory state."""
        self.memory.copy_(state)

    def reset(self):
        """Reset all memory to zeros."""
        self.memory.zero_()
