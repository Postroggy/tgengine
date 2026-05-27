from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeighborSpec:
    """Specification for temporal neighbor sampling."""

    k: int = 32
    k2: int = 0  # 2nd-hop neighbor count; 0 disables 2-hop
    strategy: str = "recency"  # recency | uniform | time_weighted
    include_edge_feat: bool = True
    for_nodes: tuple[str, ...] = ("src", "dst", "neg")

    @property
    def hops(self) -> int:
        return 2 if self.k2 > 0 else 1


@dataclass
class GatherSpec:
    """Static declaration of what data a model needs.

    The DataPipeline reads this at initialization time and builds an optimized
    execution plan. All data access must be declared here — models cannot access
    TemporalGraph directly during forward().
    """

    neighbors: NeighborSpec = field(default_factory=NeighborSpec)
    co_occurrence: bool = False
    memory: bool = False
