from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeighborSpec:
    """Specification for temporal neighbor sampling."""

    k: int = 32
    strategy: str = "recency"  # recency | uniform | time_weighted
    hops: int = 1
    include_edge_feat: bool = True
    for_nodes: tuple[str, ...] = ("src", "dst", "neg")


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
