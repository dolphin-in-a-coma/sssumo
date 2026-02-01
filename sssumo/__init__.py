"""SSSUMO: Real-Time Semi-Supervised Submovement Decomposition."""

from .models import (
    Detector,
    Reconstructor,
    TDNNDetector,
    STEContinuousReconstructor,
    STEBinarizer,
    ContinuousPrimitive,
)

from .data import (
    OrganicDataset,
    SyntheticDataset,
    CombinedSyntheticDataset,
    Sampler1D,
    ConditionalSampler,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "Detector",
    "Reconstructor",
    "TDNNDetector",
    "STEContinuousReconstructor",
    "STEBinarizer",
    "ContinuousPrimitive",
    # Data
    "OrganicDataset",
    "SyntheticDataset",
    "CombinedSyntheticDataset",
    "Sampler1D",
    "ConditionalSampler",
]
