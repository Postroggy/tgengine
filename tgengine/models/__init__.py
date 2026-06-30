from .base import ModelOutput, TemporalModel
from .dygformer import DyGFormer
from .freedyg import FreeDyG
from .graphmixer import GraphMixer
from .tgn import TGN

__all__ = ["DyGFormer", "FreeDyG", "GraphMixer", "ModelOutput", "TGN", "TemporalModel"]

# Mamba-backed models require mamba_ssm (CUDA selective_scan). Import lazily so
# that environments without a working mamba_ssm build (e.g. GLIBC mismatch) can
# still use all other models and run the test suite.
try:
    from .crossmamba import CrossMamba
    from .dygmamba import DyGMamba
    from .foundation import FoundationModel
    __all__ += ["CrossMamba", "DyGMamba", "FoundationModel"]
except ImportError:
    pass
