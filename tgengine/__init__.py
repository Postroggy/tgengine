"""TGEngine: High-performance Continuous-Time Dynamic Graph learning."""

__version__ = "0.1.0"

from .core.batch import NeighborData, PreparedBatch, RawBatch
from .core.dataset import TemporalDataset, load_dataset
from .core.gather_spec import GatherSpec, NeighborSpec
from .core.temporal_graph import TemporalGraph
from .engine import APEval, AUCEval, Engine, EvalProtocol, HitsEval, MRREval, RankingEval, ThreeWayEval, TrainConfig, run_experiment
from .pipeline.async_pipeline import AsyncDataPipeline
from .models.base import ModelOutput, TemporalModel
from .models.dygformer import DyGFormer
from .models.dygmamba import DyGMamba
from .models.freedyg import FreeDyG
from .models.graphmixer import GraphMixer
from .models.tgn import TGN
from .pipeline.negatives import (
    DyGLibHistoricalNegative,
    DyGLibInductiveNegative,
    FixedNegative,
    HistoricalNegative,
    HistoricalNegPool,
    InBatchNegative,
    InductiveNegative,
    NegativeStrategy,
    RandomNegative,
    VectorizedHistoricalNegative,
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
    "AUCEval",
    "AsyncDataPipeline",
    "Engine",
    "EvalProtocol",
    "MRREval",
    "ThreeWayEval",
    "TrainConfig",
    "run_experiment",
    # negatives
    "DyGLibHistoricalNegative",
    "DyGLibInductiveNegative",
    "FixedNegative",
    "HistoricalNegative",
    "HistoricalNegPool",
    "InBatchNegative",
    "InductiveNegative",
    "NegativeStrategy",
    "RandomNegative",
    "VectorizedHistoricalNegative",
]
