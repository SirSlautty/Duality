"""
DRAI (Dynamic Resonance AI) — V1 Production API
-----------------------------------------------

Stable, training-free dynamic resonance memory for transformer models.

This module exposes the public V1 API:
    • apply_v1()           – patch a model with DRAI V1
    • get_v1_stats()       – inspect attractor state
    • V1Config / V1Hyperparameters
    • Presets: get_v1_conservative(), get_v1_standard()

Supported architectures (V1):
    • GPT-NeoX / Pythia-family models
    • Other HF architectures with `.layers` will be supported in V2–V3
"""

# Core V1 attractor engine
from .resonance_layer_v1 import DraiResonanceLayerV1

# Configuration system
from .config import (
    V1Config,
    V1Hyperparameters,
    get_v1_conservative,
    get_v1_standard,
)

# High-level application API
from .apply_v1 import apply_v1, get_v1_stats

__version__ = "1.1.0"
__author__ = "Halcyon AI Research"

__all__ = [
    # Public API
    "apply_v1",
    "get_v1_stats",

    # Configuration system
    "V1Config",
    "V1Hyperparameters",
    "get_v1_conservative",
    "get_v1_standard",

    # Advanced internal components
    "DraiResonanceLayerV1",
]
