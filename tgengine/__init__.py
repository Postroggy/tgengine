"""TGEngine: High-performance Continuous-Time Dynamic Graph learning."""

__version__ = "0.1.0"

from .core.batch import NeighborData, PreparedBatch, RawBatch
from .core.dataset import TemporalDataset, load_dataset
from .core.gather_spec import GatherSpec, NeighborSpec
from .core.temporal_graph import TemporalGraph
from .engine import APEval, Engine, EvalProtocol, MRREval, ThreeWayEval, TrainConfig
from .pipeline.async_pipeline import AsyncDataPipeline
from .models.base import ModelOutput, TemporalModel
from .models.dygformer import DyGFormer
from .models.dygmamba import DyGMamba
from .models.freedyg import FreeDyG
from .models.graphmixer import GraphMixer
from .models.tgn import TGN
from .pipeline.negatives import (
    FixedNegative,
    HistoricalNegative,
    HistoricalNegPool,
    InductiveNegative,
    NegativeStrategy,
    RandomNegative,
)

__all__ = [
    # core
    "GatherSpec",
    "NeighborData",
    "NeighborSpec",
    "PreparedBatch",
    "RawBatch",
    "TemporalDataset",
    "TemporalGraph",
    "load_dataset",
    # models
    "DyGFormer",
    "DyGMamba",
    "FreeDyG",
    "GraphMixer",
    "ModelOutput",
    "TGN",
    "TemporalModel",
    # engine
    "APEval",
    "AsyncDataPipeline",
    "Engine",
    "EvalProtocol",
    "MRREval",
    "ThreeWayEval",
    "TrainConfig",
    # negatives
    "FixedNegative",
    "HistoricalNegative",
    "HistoricalNegPool",
    "InductiveNegative",
    "NegativeStrategy",
    "RandomNegative",
]
