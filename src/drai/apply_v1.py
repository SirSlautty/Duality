"""
High-level API for applying DRAI V1 to transformer models.

DRAI V1 provides a stable, production-ready attractor-memory mechanism
for GPT-NeoX models (e.g., Pythia). It adds pattern detection, attractor
accumulation, exponential averaging, and gated memory injection directly
into the model’s attention system — without training and without degrading
baseline behavior.

This module exposes two primary user-facing functions:

    apply_drai_v1(model_or_name, config=None, verbose=False)
    get_drai_v1_stats(model)

The rest of the module handles safety, loader wiring, architecture
inspection, and consistent statistics extraction.
"""

from typing import Optional, Union, Dict, Any, List
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
)

from .config import (
    DraiV1Config,
    get_v1_conservative_config,
)
from .neox_integration_v1 import inject_drai_v1_into_model


# ============================================================================
# Internal utilities
# ============================================================================

def _load_model_from_name(name: str, **kwargs) -> PreTrainedModel:
    """
    Internal model loader used when the user calls:
        apply_drai_v1("EleutherAI/pythia-410m")

    This keeps apply_v1 self-contained without needing a separate module.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"DRAI could not load model '{name}'. "
            f"Pass a PreTrainedModel instance instead. HF Error: {e}"
        )


def _get_layers_for_stats(model: PreTrainedModel) -> List:
    """
    Identify the transformer block list depending on architecture.

    Supported families:
      - GPT-NeoX (Pythia/NeoX-style): model.gpt_neox.layers
      - HF models exposing model.model.layers
      - Generic models exposing .layers

    Returns a list of layers or raises NotImplementedError.
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "layers"):
        return model.layers

    raise NotImplementedError(
        "DRAI V1 does not recognize the model architecture. "
        "Supported: GPT-NeoX, Pythia, or models exposing `.layers`."
    )


def _validate_not_already_injected(model: PreTrainedModel):
    """Prevent accidental double-injection."""
    if getattr(model, "_drai_v1_applied", False):
        raise RuntimeError(
            "DRAI V1 has already been applied to this model. "
            "Applying twice would corrupt attention weights."
        )


def _mark_injected(model: PreTrainedModel):
    """Mark the model as DRAI-patched."""
    setattr(model, "_drai_v1_applied", True)


# ============================================================================
# Public API
# ============================================================================

def apply_drai_v1(
    model: Union[PreTrainedModel, str],
    config: Optional[DraiV1Config] = None,
    verbose: bool = False,
    **kwargs,
) -> PreTrainedModel:
    """
    Apply DRAI V1 to a transformer model in-place.

    Args:
        model:
            - A loaded PreTrainedModel instance, OR
            - A HF model name/path string (e.g. "EleutherAI/pythia-410m")
              which will be loaded automatically.

        config:
            Optional DraiV1Config. If omitted, uses conservative V1 defaults
            suitable for small models (<1B).

        verbose:
            Enable per-layer printouts during injection.

        **kwargs:
            Passed directly to AutoModelForCausalLM.from_pretrained() when
            model is a string.

    Returns:
        A model modified with DRAI V1 attractor layers.
    """

    # Auto-load name-based models
    if isinstance(model, str):
        model = _load_model_from_name(model, **kwargs)

    # Prevent accidental double-application
    _validate_not_already_injected(model)

    # Default configuration
    if config is None:
        config = get_v1_conservative_config()

    # Perform injection
    model = inject_drai_v1_into_model(model, config, verbose=verbose)

    # Tag the model
    _mark_injected(model)

    return model


def get_drai_v1_stats(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Extract attractor statistics from a DRAI-modified model.

    Returns:
        {
            'num_drai_layers': int,
            'total_active': int,
            'total_strength': float,
            'mean_strength': float,
            'layers': [
                {
                    'layer_idx': int,
                    'num_alive': int,
                    'total_strength': float,
                    'mean_strength': float,
                    ...
                },
                ...
            ]
        }

    Notes:
        - If no layers contain DRAI V1 attention, returns zeros.
        - This function does NOT mutate the model.
    """

    from .neox_integration_v1 import DraiGPTNeoXAttentionV1

    layers = _get_layers_for_stats(model)

    stats = {
        "num_drai_layers": 0,
        "total_active": 0,
        "total_strength": 0.0,
        "mean_strength": 0.0,
        "layers": [],
    }

    for idx, layer in enumerate(layers):
        attn = getattr(layer, "attention", None)

        if isinstance(attn, DraiGPTNeoXAttentionV1):
            layer_stats = attn.drai.get_stats()

            stats["num_drai_layers"] += 1
            stats["total_active"] += layer_stats.get("num_alive", 0)
            stats["total_strength"] += layer_stats.get("total_strength", 0.0)

            stats["layers"].append({
                "layer_idx": idx,
                **layer_stats,
            })

    if stats["num_drai_layers"] > 0:
        stats["mean_strength"] = (
            stats["total_strength"] / stats["num_drai_layers"]
        )

    return stats
