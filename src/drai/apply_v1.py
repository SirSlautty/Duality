"""High-level API for applying DRAI V1 to transformer models.

DRAI V1 is a production-ready attractor mechanism designed primarily for
GPT-NeoX architectures (e.g., Pythia). It adds pattern-tracking, field-vector
accumulation, and soft gating to attention heads without destabilizing
generation.

This wrapper handles:
- Model loading (if a string path/name is given)
- Architecture detection
- Safety checks (prevents double-injection)
- Uniform statistics extraction
"""

from typing import Optional, Union, Dict, Any, List

from transformers import PreTrainedModel

from .config import DraiV1Config, get_v1_conservative_config
from .neox_integration_v1 import inject_drai_v1_into_model


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _get_layers_for_stats(model: PreTrainedModel) -> List:
    """Return the layer list for architectures supported by DRAI V1.

    Currently supported:
        - GPT-NeoX family (Pythia, GPT-NeoX)
        - Partial fallback for models exposing `.layers`

    Returns:
        List of layer modules.

    Raises:
        NotImplementedError for unsupported architectures.
    """
    # GPT-NeoX (Pythia, NeoX 20B etc.)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    # Some HF models use `model.model.layers`
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # Generic fallback: direct `.layers`
    if hasattr(model, "layers"):
        return model.layers

    raise NotImplementedError(
        "DRAI V1 stats extraction is not supported for this model architecture. "
        "Supported families: GPT-NeoX, Pythia, models exposing `.layers`."
    )


def _validate_not_already_injected(model: PreTrainedModel):
    """Prevent applying DRAI V1 twice."""
    if hasattr(model, "_drai_v1_applied") and model._drai_v1_applied:
        raise RuntimeError(
            "DRAI V1 has already been applied to this model. "
            "Double-injection would corrupt attention weights."
        )


def _mark_injected(model: PreTrainedModel):
    """Mark the model as patched with DRAI V1."""
    model._drai_v1_applied = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_drai_v1(
    model: Union[PreTrainedModel, str],
    config: Optional[DraiV1Config] = None,
    verbose: bool = False,
    **kwargs
) -> PreTrainedModel:
    """
    Apply DRAI V1 to a transformer model.

    Args:
        model: A loaded model or a HuggingFace model name/path.
        config: Optional DraiV1Config. If None, uses conservative defaults.
        verbose: Whether injection should log layer-by-layer details.
        **kwargs: Passed to HF model loader when `model` is a string.

    Returns:
        Model with DRAI V1 modifications applied.
    """
    # Load if user provided a model name
    if isinstance(model, str):
        from .apply import load_model
        model = load_model(model, **kwargs)

    # Safety check: avoid double injection
    _validate_not_already_injected(model)

    # Apply conservative defaults
    if config is None:
        config = get_v1_conservative_config()

    # Perform injection
    model = inject_drai_v1_into_model(model, config, verbose=verbose)

    # Tag the model so we don't re-apply
    _mark_injected(model)

    return model


def get_drai_v1_stats(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Extract statistics from DRAI V1 modified attention layers.

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
    """
    from .neox_integration_v1 import DraiGPTNeoXAttentionV1

    layers = _get_layers_for_stats(model)

    stats = {
        'num_drai_layers': 0,
        'total_active': 0,
        'total_strength': 0.0,
        'mean_strength': 0.0,
        'layers': []
    }

    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "attention", None)
        if isinstance(attn, DraiGPTNeoXAttentionV1):
            layer_stats = attn.drai.get_stats()

            stats['num_drai_layers'] += 1
            stats['total_active'] += layer_stats.get('num_alive', 0)
            stats['total_strength'] += layer_stats.get('total_strength', 0.0)

            stats['layers'].append({
                'layer_idx': layer_idx,
                **layer_stats
            })

    if stats['num_drai_layers'] > 0:
        stats['mean_strength'] = (
            stats['total_strength'] / stats['num_drai_layers']
        )

    return stats

