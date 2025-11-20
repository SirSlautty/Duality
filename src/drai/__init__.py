"""
DRAI (Dynamic Resonance AI) — V1 Production Release
---------------------------------------------------

Dynamic, training-free attractor memory for transformer models.

This module exposes the stable public API for integrating DRAI V1 into
supported Hugging Face architectures (currently GPT-NeoX-style models,
with broader compatibility planned in V2–V3).

DRAI V1 adds a persistent, self-organizing internal memory field to
frozen language models using controlled attractor dynamics. It enables
stable reasoning over long contexts without modifying weights, training,
or requiring external memory systems.

Quick Start:
    >>> from drai import apply_drai_v1, get_v1_conservative_config
    >>> from transformers import AutoModelForCausalLM

    # Load a supported transformer model (e.g., GPT-NeoX family)
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

    # Add dynamic memory in one line
    >>> model = apply_drai_v1(model, config=get_v1_conservative_config())

    # Use normally — now with internal attractor memory
    >>> outputs = model.generate(inputs)

Documentation:
  - Repo: https://github.com/HalcyonAIR/Duality
  - DRAI V1 Overview: /docs/DRAI_V1_OVERVIEW.md
  - Attractor Memory Theory: /docs/ATTRACTOR_DYNAMICS.md

Compatibility Notes:
  - V1 is fully stable and production-ready.
  - API surface for standard use is stable.
  - Advanced APIs (ResonanceLayer, projections, internal hooks)
    may evolve in V2–V3 as concatenated K/V mechanisms are introduced.
"""

# Core V1 components
from .resonance_layer_v1 import DraiResonanceLayerV1

# Configuration system
from .config import (
    DraiV1Config,
    DraiV1Hyperparameters,
    get_v1_conservative_config,
    get_v1_standard_config,
)

# High-level public API
from .apply_v1 import apply_drai_v1, get_drai_v1_stats

# Package metadata
__version__ = "1.0.0"
__author__ = "Halcyon AI Research"

__all__ = [
    # Main API
    "apply_drai_v1",
    "get_drai_v1_stats",

    # Configurations
    "DraiV1Config",
    "DraiV1Hyperparameters",
    "get_v1_conservative_config",
    "get_v1_standard_config",

    # Advanced / Internal components
    "DraiResonanceLayerV1",
]
