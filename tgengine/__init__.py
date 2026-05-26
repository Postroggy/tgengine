"""TGEngine: High-performance Continuous-Time Dynamic Graph learning."""

__version__ = "0.1.0"

from .core.batch import NeighborData, PreparedBatch, RawBatch
from .core.gather_spec import GatherSpec, NeighborSpec
from .core.temporal_graph import TemporalGraph
from .models.base import ModelOutput, TemporalModel

__all__ = [
    "GatherSpec",
    "ModelOutput",
    "NeighborData",
    "NeighborSpec",
    "PreparedBatch",
    "RawBatch",
    "TemporalGraph",
    "TemporalModel",
]
