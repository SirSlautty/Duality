"""
DRAI (Dynamic Resonance AI) - V1 Production Release

Working memory for transformers through persistent attractor dynamics.

Usage:
    >>> from drai import apply_drai_v1, get_v1_conservative_config
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> # Load any transformer
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
    >>>
    >>> # Add working memory (one line!)
    >>> model = apply_drai_v1(model, config=get_v1_conservative_config())
    >>>
    >>> # Use normally - now with internal state
    >>> outputs = model.generate(inputs)
"""

# Core V1 components
from .resonance_layer_v1 import DraiResonanceLayerV1
from .config import (
    DraiV1Config,
    DraiV1Hyperparameters,
    get_v1_conservative_config,
    get_v1_standard_config,
)

# High-level V1 API
from .apply_v1 import apply_drai_v1, get_drai_v1_stats

# Version
__version__ = "1.0.0"

__all__ = [
    # V1 API (production-ready)
    "apply_drai_v1",
    "get_drai_v1_stats",

    # Configuration
    "DraiV1Config",
    "DraiV1Hyperparameters",
    "get_v1_conservative_config",
    "get_v1_standard_config",

    # Low-level components (for advanced users)
    "DraiResonanceLayerV1",
]
