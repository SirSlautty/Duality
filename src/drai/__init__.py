"""
DRAI (Dynamic Resonance AI) – V1 Production Release
---------------------------------------------------

Internal-state augmentation for transformer models using persistent
attractor dynamics. This module exposes the public API for integrating
DRAI into any Hugging Face transformer model.

Quick Usage:
    >>> from drai import apply_drai_v1, get_v1_conservative_config
    >>> from transformers import AutoModelForCausalLM

    # Load any transformer
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

    # Add working memory (one line!)
    >>> model = apply_drai_v1(model, config=get_v1_conservative_config())

    # Use normally — now with internal state
    >>> outputs = model.generate(inputs)

See:
  - GitHub: https://github.com/HalcyonAIR/Duality
  - Theory docs: /docs/THE_CLOUD_MECHANISM.md
"""

# Core V1 components
from .resonance_layer_v1 import DraiResonanceLayerV1

# Configuration
from .config import (
    DraiV1Config,
    DraiV1Hyperparameters,
    get_v1_conservative_config,
    get_v1_standard_config,
)

# High-level public API
from .apply_v1 import apply_drai_v1, get_drai_v1_stats

__version__ = "0.1.0"
__author__ = "Halcyon AI Research"

__all__ = [
    # Main API
    "apply_drai_v1",
    "get_drai_v1_stats",

    # Configs
    "DraiV1Config",
    "DraiV1Hyperparameters",
    "get_v1_conservative_config",
    "get_v1_standard_config",

    # Advanced
    "DraiResonanceLayerV1",
]

