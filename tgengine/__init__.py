"""TGEngine: High-performance Continuous-Time Dynamic Graph learning."""

__version__ = "0.1.0"

from .core.batch import NeighborData, PreparedBatch, RawBatch
from .core.dataset import TemporalDataset, load_dataset
from .core.mixed_dataset import MixedDataset
from .utils.download import download_dataset, list_available_datasets, list_datasets_by_family, set_proxy
from .utils import prepare_ranking_negs
from .core.gather_spec import GatherSpec, NeighborSpec
from .core.temporal_graph import TemporalGraph
from .engine import (
    APEval,
    AUCEval,
    AnomalyEval,
    EdgeClsEval,
    EdgeRegEval,
    Engine,
    EvalProtocol,
    HitsEval,
    MRREval,
    NodeClsEval,
    NodeRegEval,
    RankingEval,
    ThreeWayEval,
    TrainConfig,
    run_experiment,
)
from .tasks import (
    AnomalyDetectionHead,
    EdgeBinaryClassificationHead,
    EdgeClassificationHead,
    EdgeRegressionHead,
    LinkPredHead,
    MultiTaskHead,
    NodeBinaryClassificationHead,
    NodeClassificationHead,
    NodeRegressionHead,
    TaskHead,
)
from .pipeline import DataPipeline
from .pipeline.async_pipeline import AsyncDataPipeline
from .models.base import EmbeddingBundle, ModelOutput, TemporalModel
from .models.dygformer import DyGFormer
from .models.freedyg import FreeDyG
from .models.graphmixer import GraphMixer
from .models.tgn import TGN

# Mamba-backed models imported lazily (require mamba_ssm CUDA build).
try:
    from .models.crossmamba import CrossMamba
    from .models.dygmamba import DyGMamba
    _HAS_MAMBA_MODELS = True
except ImportError:
    _HAS_MAMBA_MODELS = False
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
    "MixedDataset",
    # models
    "DyGFormer",
    "EmbeddingBundle",
    "FreeDyG",
    "GraphMixer",
    "ModelOutput",
    "TGN",
    "TemporalModel",
    # engine
    "APEval",
    "AUCEval",
    "AnomalyEval",
    "AsyncDataPipeline",
    "DataPipeline",
    "EdgeClsEval",
    "EdgeRegEval",
    "Engine",
    "EvalProtocol",
    "HitsEval",
    "MRREval",
    "NodeClsEval",
    "NodeRegEval",
    "RankingEval",
    "ThreeWayEval",
    "TrainConfig",
    "run_experiment",
    # task heads
    "AnomalyDetectionHead",
    "EdgeBinaryClassificationHead",
    "EdgeClassificationHead",
    "EdgeRegressionHead",
    "LinkPredHead",
    "MultiTaskHead",
    "NodeBinaryClassificationHead",
    "NodeClassificationHead",
    "NodeRegressionHead",
    "TaskHead",
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
    # download + utils
    "download_dataset",
    "list_available_datasets",
    "list_datasets_by_family",
    "prepare_ranking_negs",
    "set_proxy",
]

if _HAS_MAMBA_MODELS:
    __all__ += ["CrossMamba", "DyGMamba"]
