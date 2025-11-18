"""Configuration system for DRAI integration into transformer models.

This module provides a comprehensive configuration system for controlling
DRAI behavior, injection strategy, and hyperparameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal


@dataclass
class DraiHyperparameters:
    """DRAI attractor dynamics hyperparameters.

    These parameters control the behavior of the attractor field:
    - Formation: How easily new attractors are created
    - Reinforcement: How attractors are strengthened
    - Decay: How attractors fade over time
    - Capacity: How many attractors can exist

    See docs/ATTRACTOR_MATHEMATICS.md for mathematical details.
    """

    max_attractors: int = 32
    """Maximum number of attractors per layer. Higher = more memory, more patterns."""

    coherence_threshold: float = 0.3
    """Minimum similarity to match existing attractor. Lower = looser matching."""

    formation_threshold: float = 0.5
    """Initial coherence for new attractors. Higher = stricter formation criteria."""

    decay_rate: float = 0.01
    """Exponential decay rate for unused attractors. Higher = faster forgetting."""

    ema_momentum: float = 0.9
    """EMA momentum for attractor updates. Higher = more stable, slower adaptation."""

    # Phase 5+ Gating parameters
    use_strict_gating: bool = False
    """Enable strict activation gating (threshold 0.85). CRITICAL for small models to prevent random injection."""

    activation_threshold: float = 0.7
    """Similarity threshold for activating attractors. Only inject if cosine_similarity > this."""

    def __post_init__(self):
        """Validate hyperparameters."""
        if self.max_attractors < 1:
            raise ValueError(f"max_attractors must be >= 1, got {self.max_attractors}")

        if not 0 < self.coherence_threshold < 1:
            raise ValueError(
                f"coherence_threshold must be in (0, 1), got {self.coherence_threshold}"
            )

        if not 0 < self.formation_threshold <= 1:
            raise ValueError(
                f"formation_threshold must be in (0, 1], got {self.formation_threshold}"
            )

        if self.coherence_threshold > self.formation_threshold:
            raise ValueError(
                f"coherence_threshold ({self.coherence_threshold}) must be <= "
                f"formation_threshold ({self.formation_threshold})"
            )

        if self.decay_rate < 0:
            raise ValueError(f"decay_rate must be >= 0, got {self.decay_rate}")

        if not 0 < self.ema_momentum < 1:
            raise ValueError(f"ema_momentum must be in (0, 1), got {self.ema_momentum}")

    def to_dict(self):
        """Convert to dictionary for easy passing to DraiResonanceLayer."""
        return asdict(self)


@dataclass
class DraiConfig:
    """Complete DRAI configuration for transformer integration.

    This configuration controls:
    1. Whether DRAI is enabled
    2. Which phase to use (1=zeros, 2=attractors)
    3. Which layers receive DRAI
    4. DRAI architecture (heads, dimensions)
    5. Attractor dynamics hyperparameters
    6. Monitoring and logging options

    Example usage:
        # Full DRAI on all layers
        config = DraiConfig(enabled=True, phase=2, layer_mode="all")

        # DRAI only on deeper layers
        config = DraiConfig(
            enabled=True,
            phase=2,
            layer_mode="selective",
            selective_layers=[6, 7, 8, 9, 10, 11]
        )

        # Custom hyperparameters
        config = DraiConfig(
            enabled=True,
            phase=2,
            hyperparameters=DraiHyperparameters(
                max_attractors=64,
                decay_rate=0.005,
            )
        )
    """

    # Core settings
    enabled: bool = False
    """Enable DRAI injection. If False, model behaves as standard transformer."""

    phase: int = 2
    """DRAI phase: 1=zeros (no impact), 2=full attractor dynamics."""

    # Layer selection
    layer_mode: Literal["all", "selective", "none"] = "all"
    """
    Which layers get DRAI:
    - "all": Inject into all layers
    - "selective": Only layers in selective_layers list
    - "none": No injection (equivalent to enabled=False)
    """

    selective_layers: Optional[List[int]] = None
    """Layer indices for selective injection. Only used if layer_mode="selective"."""

    # DRAI architecture
    num_drai_heads: int = 1
    """Number of DRAI heads per layer. Acts like extra attention positions."""

    drai_head_dim: Optional[int] = None
    """
    Dimension of each DRAI head. If None, uses model's head_dim.
    Typically should match model's head_dim for proper integration.
    """

    # Hyperparameters
    hyperparameters: DraiHyperparameters = field(default_factory=DraiHyperparameters)
    """Attractor dynamics hyperparameters. See DraiHyperparameters for details."""

    # Monitoring
    verbose_logging: bool = False
    """Enable verbose logging of DRAI operations (for debugging)."""

    collect_statistics: bool = True
    """Collect attractor statistics during forward passes."""

    def __post_init__(self):
        """Validate configuration."""
        # Validate phase
        if self.phase not in [1, 2]:
            raise ValueError(f"phase must be 1 or 2, got {self.phase}")

        # Validate layer_mode
        valid_modes = ["all", "selective", "none"]
        if self.layer_mode not in valid_modes:
            raise ValueError(
                f"layer_mode must be one of {valid_modes}, got {self.layer_mode}"
            )

        # Validate selective_layers
        if self.layer_mode == "selective":
            if self.selective_layers is None or len(self.selective_layers) == 0:
                raise ValueError(
                    "selective_layers must be provided when layer_mode='selective'"
                )
            if not all(isinstance(i, int) and i >= 0 for i in self.selective_layers):
                raise ValueError(
                    "selective_layers must contain non-negative integers"
                )

        # Validate num_drai_heads
        if self.num_drai_heads < 1:
            raise ValueError(
                f"num_drai_heads must be >= 1, got {self.num_drai_heads}"
            )

        # Validate drai_head_dim
        if self.drai_head_dim is not None and self.drai_head_dim < 1:
            raise ValueError(
                f"drai_head_dim must be >= 1 or None, got {self.drai_head_dim}"
            )

    def should_inject_layer(self, layer_idx: int) -> bool:
        """Determine if DRAI should be injected into a specific layer.

        Args:
            layer_idx: Index of the layer (0-indexed)

        Returns:
            True if DRAI should be injected, False otherwise
        """
        if not self.enabled:
            return False

        if self.layer_mode == "none":
            return False

        if self.layer_mode == "all":
            return True

        if self.layer_mode == "selective":
            return layer_idx in (self.selective_layers or [])

        return False

    def __repr__(self):
        """Readable string representation."""
        if not self.enabled:
            return "DraiConfig(enabled=False)"

        lines = [
            "DraiConfig(",
            f"  enabled={self.enabled}",
            f"  phase={self.phase}",
            f"  layer_mode='{self.layer_mode}'",
        ]

        if self.layer_mode == "selective":
            lines.append(f"  selective_layers={self.selective_layers}")

        lines.extend([
            f"  num_drai_heads={self.num_drai_heads}",
            f"  hyperparameters={self.hyperparameters}",
            ")"
        ])

        return "\n".join(lines)


# Preset configurations for common use cases

def get_baseline_config() -> DraiConfig:
    """Baseline configuration: DRAI disabled.

    Use this for baseline comparisons without DRAI.
    """
    return DraiConfig(enabled=False)


def get_full_drai_config() -> DraiConfig:
    """Full DRAI configuration: All layers, default hyperparameters.

    This is the standard configuration for testing DRAI impact.
    """
    return DraiConfig(
        enabled=True,
        phase=2,
        layer_mode="all",
        num_drai_heads=1,
    )


def get_selective_drai_config(num_layers: int, fraction: float = 0.5) -> DraiConfig:
    """Selective DRAI configuration: Only deeper layers.

    Args:
        num_layers: Total number of layers in the model
        fraction: Fraction of layers to use (starting from deeper layers)
            For example, 0.5 uses the second half of layers

    Returns:
        DraiConfig with selective layer injection
    """
    start_layer = int(num_layers * (1 - fraction))
    selective_layers = list(range(start_layer, num_layers))

    return DraiConfig(
        enabled=True,
        phase=2,
        layer_mode="selective",
        selective_layers=selective_layers,
        num_drai_heads=1,
    )


def get_aggressive_drai_config() -> DraiConfig:
    """Aggressive DRAI configuration: Fast formation, slow decay.

    This configuration encourages quick attractor formation and
    long-term memory retention.
    """
    return DraiConfig(
        enabled=True,
        phase=2,
        layer_mode="all",
        num_drai_heads=1,
        hyperparameters=DraiHyperparameters(
            max_attractors=64,  # More capacity
            coherence_threshold=0.2,  # Easier matching
            formation_threshold=0.3,  # Easier formation
            decay_rate=0.005,  # Slower decay
            ema_momentum=0.95,  # More stable
        ),
    )


def get_conservative_drai_config() -> DraiConfig:
    """Conservative DRAI configuration: Slow formation, fast decay.

    This configuration requires strong patterns for attractor formation
    and quickly forgets unused patterns.
    """
    return DraiConfig(
        enabled=True,
        phase=2,
        layer_mode="all",
        num_drai_heads=1,
        hyperparameters=DraiHyperparameters(
            max_attractors=16,  # Less capacity
            coherence_threshold=0.5,  # Stricter matching
            formation_threshold=0.7,  # Harder formation
            decay_rate=0.02,  # Faster decay
            ema_momentum=0.8,  # Less stable
        ),
    )


def get_gated_drai_config() -> DraiConfig:
    """Gated DRAI configuration: Strict activation gating for small models.

    This configuration adds threshold gating to prevent random attractor
    injection. CRITICAL for small models (< 1B params) to prevent
    performance degradation.

    Key feature: Only injects attractors that strongly match current query
    (cosine similarity > 0.85), preventing random noise injection that
    would degrade generation quality.

    Use this for:
    - Small models (pythia-410m, pythia-1b)
    - Generation tasks
    - Tasks requiring coherent outputs

    Phase 5+ feature validated through tractable task experiments.
    """
    return DraiConfig(
        enabled=True,
        phase=2,
        layer_mode="all",
        num_drai_heads=1,
        hyperparameters=DraiHyperparameters(
            max_attractors=32,  # Reasonable capacity
            coherence_threshold=0.3,  # Standard matching
            formation_threshold=0.5,  # Standard formation
            decay_rate=0.01,  # Standard decay
            ema_momentum=0.9,  # Standard stability
            use_strict_gating=True,  # CRITICAL: Enable gating
            activation_threshold=0.85,  # High threshold for injection
        ),
    )


# Example configurations for documentation
if __name__ == "__main__":
    print("=== DRAI Configuration Examples ===\n")

    print("1. Baseline (no DRAI):")
    print(get_baseline_config())
    print()

    print("2. Full DRAI (all layers):")
    print(get_full_drai_config())
    print()

    print("3. Selective DRAI (50% of 12 layers):")
    print(get_selective_drai_config(num_layers=12, fraction=0.5))
    print()

    print("4. Aggressive DRAI:")
    print(get_aggressive_drai_config())
    print()

    print("5. Conservative DRAI:")
    print(get_conservative_drai_config())
    print()

    # Validation examples
    print("=== Validation Examples ===\n")

    try:
        bad_config = DraiConfig(phase=3)
    except ValueError as e:
        print(f"✓ Caught invalid phase: {e}")

    try:
        bad_config = DraiConfig(
            layer_mode="selective",
            selective_layers=None
        )
    except ValueError as e:
        print(f"✓ Caught missing selective_layers: {e}")

    try:
        bad_config = DraiConfig(
            hyperparameters=DraiHyperparameters(
                coherence_threshold=0.8,
                formation_threshold=0.5
            )
        )
    except ValueError as e:
        print(f"✓ Caught invalid threshold ordering: {e}")
