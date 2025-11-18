"""
DRAI (Dynamic Resonance AI) Module

This module implements the resonance cortex - a memory layer that forms
stable attractors from recurring latent patterns and injects them back
into transformer attention as synthetic K/V pairs.
"""

from .resonance_layer import DraiResonanceLayer

__all__ = ["DraiResonanceLayer"]
