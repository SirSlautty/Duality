"""
DRAI Configuration System
==========================

This module defines the configuration structure for all DRAI algorithms,
including production-grade V1 and the earlier Phase-2 attractor system.

Goals:
    • Clean versioned configs
    • Consistent naming across algorithms
    • Architecture-aware layer selection
    • JSON-safe export for logging / reproducibility
    • Strict validation of all hyperparameters
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal, Dict, Any


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _auto_mid_layers(num_layers: int) -> List[int]:
    """
    Select mid-layer indices depending on model depth.

    • 1 layer → [0]
    • <= 4 layers → [middle]
    • <= 12 → [middle]
    • >= 24 → [middle-1, middle]
    """
    if num_layers <= 1:
        return [0]
    if num_layers <= 12:
        return [num_layers // 2]

    # 24/32-layer models (NeoX, Pythia)
    m = num_layers // 2
    return [m - 1, m]


# ---------------------------------------------------------------------------
# Base Configuration (shared across all algorithms)
# ---------------------------------------------------------------------------

@dataclass
class DRAIBaseConfig:
    """
    Base config for all DRAI algorithms.

    Required fields:
        • version
        • algorithm_type
        • layer_mode
        • num_drai_heads

    Layer modes:
        "all"       – inject DRAI into every transformer block
        "mid"       – auto-select central layers (safe default for small models)
        "none"      – no DRAI integration
        "selective" – custom injection into the provided list
        "specific"  – synonym for "selective"
    """

    version: str = "1.0"
    algorithm_type: Literal["phase2", "v1"] = "phase2"

    # Layer injection behavior
    layer_mode: Literal["all", "none", "mid", "selective", "specific"] = "all"
    specific_layers: Optional[List[int]] = None

    # Number of DRAI heads per injection layer
    num_drai_heads: int = 1

    # ----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe representation for logs or metadata storage."""
        return asdict(self)

    # ----------------------------------------------------------------------

    def resolve_layers(self, num_layers: int) -> List[int]:
        """
        Compute which layers should receive DRAI injection.

        Parameters
        ----------
        num_layers : int
            Number of transformer layers in the model.

        Returns
        -------
        List[int]
            Layer indices to inject DRAI into.
        """
        if self.layer_mode == "none":
            return []

        if self.layer_mode == "all":
            return list(range(num_layers))

        if self.layer_mode == "mid":
            return _auto_mid_layers(num_layers)

        if self.layer_mode in {"specific", "selective"}:
            return self.specific_layers or []

        raise ValueError(f"Unknown layer_mode: {self.layer_mode}")

    # ----------------------------------------------------------------------

    def __post_init__(self):
        """Perform strict, user-friendly validation."""
        if self.num_drai_heads < 1:
            raise ValueError("num_drai_heads must be >= 1")

        valid_modes = {"all", "none", "mid", "selective", "specific"}
        if self.layer_mode not in valid_modes:
            raise ValueError(
                f"Invalid layer_mode: '{self.layer_mode}'. "
                f"Must be one of {valid_modes}."
            )

        # Validate layer list use
        if self.layer_mode in {"selective", "specific"}:
            if not self.specific_layers:
                raise ValueError(
                    f"layer_mode='{self.layer_mode}' requires specific_layers list."
                )
            if not all(isinstance(x, int) and x >= 0 for x in self.specific_layers):
                raise ValueError(
                    "specific_layers must contain valid non-negative integer indices."
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"version={self.version}, "
            f"algorithm_type={self.algorithm_type}, "
            f"layer_mode={self.layer_mode})"
        )


# ---------------------------------------------------------------------------
# Phase-2 Algorithm (legacy research variant)
# ---------------------------------------------------------------------------

@dataclass
class Phase2Hyperparameters:
    """Hyperparameters for the older Phase-2 attractor algorithm."""

    max_attractors: int = 32
    coherence_threshold: float = 0.3
    formation_threshold: float = 0.5
    decay_rate: float = 0.01
    ema_momentum: float = 0.9
    use_strict_gating: bool = False
    activation_threshold: float = 0.7

    def __post_init__(self):
        if self.max_attractors < 1:
            raise ValueError("max_attractors must be >= 1")
        if not 0 < self.coherence_threshold < 1:
            raise ValueError("coherence_threshold must be in (0,1)")
        if not 0 < self.formation_threshold <= 1:
            raise ValueError("formation_threshold must be in (0,1]")
        if self.coherence_threshold > self.formation_threshold:
            raise ValueError(
                "coherence_threshold must be <= formation_threshold"
            )
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")
        if not 0 < self.ema_momentum < 1:
            raise ValueError("ema_momentum must be in (0,1)")


@dataclass
class Phase2Config(DRAIBaseConfig):
    """Configuration wrapper for Phase-2."""
    algorithm_type: Literal["phase2"] = "phase2"
    hyperparameters: Phase2Hyperparameters = field(default_factory=Phase2Hyperparameters)


def get_phase2_full() -> Phase2Config:
    """Full-strength Phase-2 attractor system."""
    return Phase2Config(layer_mode="all")


def get_phase2_conservative() -> Phase2Config:
    """Stable settings for smaller models."""
    return Phase2Config(
        layer_mode="all",
        hyperparameters=Phase2Hyperparameters(
            max_attractors=16,
            coherence_threshold=0.5,
            formation_threshold=0.7,
            decay_rate=0.02,
            ema_momentum=0.8,
        ),
    )


def get_phase2_gated() -> Phase2Config:
    """More selective, gated attractor behavior."""
    return Phase2Config(
        layer_mode="all",
        hyperparameters=Phase2Hyperparameters(
            use_strict_gating=True,
            activation_threshold=0.85,
        ),
    )


# ---------------------------------------------------------------------------
# V1 Algorithm (production, stable, recommended)
# ---------------------------------------------------------------------------

@dataclass
class V1Hyperparameters:
    """
    Hyperparameters for the V1 algorithm.
    Designed to guarantee stability on models <1B and reliability on larger ones.
    """

    max_attractors: int = 16
    theta_match: float = 0.8
    alpha_update: float = 0.05
    lambda_decay: float = 0.995
    strength_init: float = 0.5
    strength_min: float = 1e-3

    max_influence_scale: float = 0.15

    burn_in_threshold: float = 50.0
    burn_in_mode: Literal["strength", "tokens"] = "strength"
    burn_in_tokens: int = 10

    def __post_init__(self):
        if self.max_attractors < 1:
            raise ValueError("max_attractors must be >= 1")
        if not 0 < self.theta_match <= 1:
            raise ValueError("theta_match must be in (0,1]")
        if not 0 < self.alpha_update <= 1:
            raise ValueError("alpha_update must be in (0,1]")
        if not 0 < self.lambda_decay <= 1:
            raise ValueError("lambda_decay must be in (0,1]")
        if self.strength_min < 0:
            raise ValueError("strength_min must be >= 0")
        if self.max_influence_scale <= 0:
            raise ValueError("max_influence_scale must be > 0")


@dataclass
class V1Config(DRAIBaseConfig):
    """Configuration wrapper for DRAI V1 (stable production algorithm)."""

    algorithm_type: Literal["v1"] = "v1"
    hyperparameters: V1Hyperparameters = field(default_factory=V1Hyperparameters)


# Presets --------------------------------------------------------------------

def get_v1_conservative() -> V1Config:
    """
    Conservative settings for 410M–1B models.
    Prioritizes stability and safe influence scales.
    """
    return V1Config(
        layer_mode="mid",
        hyperparameters=V1Hyperparameters(
            max_attractors=16,
            theta_match=0.8,
            alpha_update=0.05,
            lambda_decay=0.995,
            max_influence_scale=0.15,
            burn_in_threshold=50.0,
        ),
    )


def get_v1_standard() -> V1Config:
    """
    Standard settings for larger (>1B) models.
    Allows stronger attractor influence and faster stabilization.
    """
    return V1Config(
        layer_mode="all",
        hyperparameters=V1Hyperparameters(
            max_attractors=32,
            theta_match=0.75,
            alpha_update=0.1,
            lambda_decay=0.99,
            max_influence_scale=0.3,
            burn_in_threshold=30.0,
            burn_in_tokens=8,
        ),
    )
