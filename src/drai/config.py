"""
Unified DRAI configuration system.

This module provides:
- A shared base configuration class (DRAIBaseConfig)
- Clean subclassing for Phase-2 and V1 algorithms
- Architecture-aware layer selection
- Versioned configuration structures
- JSON-safe representations
- Consistent hyperparameter naming and validation

Designed for long-term maintainability across:
- Phase-2 (resonance attractors)
- V1 (production-ready small-model algorithm)
- Future V2/V3 algorithms
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal, Dict, Any


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def _auto_mid_layers(num_layers: int) -> List[int]:
    """Return 1–2 middle layers depending on model depth."""
    if num_layers <= 4:
        return [num_layers // 2]
    if num_layers <= 12:
        return [num_layers // 2]
    # 24-layer NeoX etc.
    m = num_layers // 2
    return [m - 1, m]


# ---------------------------------------------------------------------------
# Base Classes (Versioned)
# ---------------------------------------------------------------------------

@dataclass
class DRAIBaseConfig:
    """
    Base config for all DRAI algorithms.

    All derived configs MUST include:
        - version
        - algorithm_type
        - layer_mode
        - num_drai_heads
    """
    version: str = "1.0"
    algorithm_type: Literal["phase2", "v1"] = "phase2"

    # Layer injection mode (same API across algorithms)
    layer_mode: Literal["all", "selective", "none", "mid", "specific"] = "all"
    specific_layers: Optional[List[int]] = None

    num_drai_heads: int = 1  # shared across all algorithms

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe global config export."""
        return asdict(self)

    # -------------------- Layer selection --------------------

    def resolve_layers(self, num_layers: int) -> List[int]:
        """
        Return the list of layers that should receive DRAI injection.
        Standardized across algorithms.
        """
        if self.layer_mode == "none":
            return []

        if self.layer_mode == "all":
            return list(range(num_layers))

        if self.layer_mode == "mid":
            return _auto_mid_layers(num_layers)

        if self.layer_mode == "specific":
            return self.specific_layers or []

        if self.layer_mode == "selective":
            return self.specific_layers or []

        raise ValueError(f"Unknown layer_mode: {self.layer_mode}")

    # ---------------------- Validation ------------------------

    def __post_init__(self):
        if self.num_drai_heads < 1:
            raise ValueError("num_drai_heads must be >= 1")

        valid_modes = {"all", "none", "mid", "selective", "specific"}
        if self.layer_mode not in valid_modes:
            raise ValueError(
                f"layer_mode must be one of {valid_modes}, got {self.layer_mode}"
            )

        # validate selective/specific lists
        if self.layer_mode in {"selective", "specific"}:
            if not self.specific_layers:
                raise ValueError(
                    f"layer_mode='{self.layer_mode}' requires specific_layers list"
                )
            if not all(isinstance(x, int) and x >= 0 for x in self.specific_layers):
                raise ValueError("specific_layers must be non-negative integers")

    # readable representation
    def __repr__(self):
        return f"{self.__class__.__name__}(version={self.version}, algorithm={self.algorithm_type}, layer_mode='{self.layer_mode}')"


# ---------------------------------------------------------------------------
# Phase-2 Hyperparameters (Legacy Algorithm)
# ---------------------------------------------------------------------------

@dataclass
class Phase2Hyperparameters:
    """Hyperparameters for legacy Phase-2 attractor system."""
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
            raise ValueError("coherence_threshold must <= formation_threshold")
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")
        if not 0 < self.ema_momentum < 1:
            raise ValueError("ema_momentum must be in (0,1)")


# ---------------------------------------------------------------------------
# Phase-2 Config
# ---------------------------------------------------------------------------

@dataclass
class Phase2Config(DRAIBaseConfig):
    """Clean Phase-2 configuration."""
    algorithm_type: Literal["phase2"] = "phase2"
    hyperparameters: Phase2Hyperparameters = field(default_factory=Phase2Hyperparameters)


# Presets
def get_phase2_full() -> Phase2Config:
    return Phase2Config(layer_mode="all")


def get_phase2_conservative() -> Phase2Config:
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
    return Phase2Config(
        layer_mode="all",
        hyperparameters=Phase2Hyperparameters(
            use_strict_gating=True,
            activation_threshold=0.85,
        ),
    )


# ---------------------------------------------------------------------------
# V1 Hyperparameters (Small-model production algorithm)
# ---------------------------------------------------------------------------

@dataclass
class V1Hyperparameters:
    """
    Hyperparameters for the V1 algorithm.
    Designed specifically to avoid catastrophic degradation
    on small models (<1B params).
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
        if not 0 < self.lambda_decay <= 1.0:
            raise ValueError("lambda_decay must be in (0,1]")
        if self.strength_min < 0:
            raise ValueError("strength_min must be >= 0")
        if self.max_influence_scale <= 0:
            raise ValueError("max_influence_scale must be > 0")


# ---------------------------------------------------------------------------
# V1 Config
# ---------------------------------------------------------------------------

@dataclass
class V1Config(DRAIBaseConfig):
    algorithm_type: Literal["v1"] = "v1"
    hyperparameters: V1Hyperparameters = field(default_factory=V1Hyperparameters)


# Presets
def get_v1_conservative() -> V1Config:
    """Best for 410M–1B models."""
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
    """Standard for 1B+ models."""
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

