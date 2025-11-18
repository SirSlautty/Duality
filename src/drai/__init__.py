"""
DRAI (Dynamic Resonance AI) Module

This module implements the resonance cortex - a memory layer that forms
stable attractors from recurring latent patterns and injects them back
into transformer attention as synthetic K/V pairs.

Usage:
    >>> from drai import apply_drai
    >>> model = apply_drai("EleutherAI/pythia-70m")

For more control:
    >>> from drai import apply_drai, DraiConfig
    >>> config = DraiConfig(num_drai_heads=2, max_attractors=64)
    >>> model = apply_drai("EleutherAI/pythia-70m", config=config)
"""

# Core components
from .resonance_layer import DraiResonanceLayer
from .config import DraiConfig, get_full_drai_config

# High-level API
from .apply import apply_drai, apply_drai_to_model, get_drai_stats

# Version
__version__ = "0.1.0"

__all__ = [
    # High-level API (recommended for end users)
    "apply_drai",
    "apply_drai_to_model",
    "get_drai_stats",

    # Configuration
    "DraiConfig",
    "get_full_drai_config",

    # Low-level components (for advanced users)
    "DraiResonanceLayer",
]
