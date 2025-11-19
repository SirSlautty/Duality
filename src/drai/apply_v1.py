"""High-level API for applying DRAI V1 to models.

V1 is a production-ready attractor algorithm that:
- Doesn't wreck small models (tested on pythia-410m)
- Has real pattern detection / accumulation / decay
- Only injects latent-valid, gated K/V

Use V1 instead of Phase 2 for generation tasks on small models.

Example:
    >>> from drai import apply_drai_v1, get_v1_conservative_config
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
    >>> config = get_v1_conservative_config()
    >>> model = apply_drai_v1(model, config=config)
    >>>
    >>> # Now use the model normally - accuracy should stay near baseline!
    >>> outputs = model.generate(...)
"""

from typing import Optional, Union
from transformers import PreTrainedModel

from .config import DraiV1Config, get_v1_conservative_config
from .neox_integration_v1 import inject_drai_v1_into_model


def apply_drai_v1(
    model: Union[PreTrainedModel, str],
    config: Optional[DraiV1Config] = None,
    **kwargs
) -> PreTrainedModel:
    """Apply DRAI V1 to a transformer model.

    V1 is a production-ready algorithm designed for small models.
    It uses proper pattern detection, field vector approach, and soft gating
    to avoid catastrophic degradation on generation tasks.

    Args:
        model: Either a loaded model or a model name/path
        config: DRAI V1 configuration (if None, uses conservative default)
        **kwargs: Additional arguments for model loading (if model is a string)

    Returns:
        Model with DRAI V1 applied

    Example:
        >>> from drai import apply_drai_v1
        >>> model = apply_drai_v1("EleutherAI/pythia-410m")
        >>>
        >>> # Or with custom config
        >>> from drai import get_v1_standard_config
        >>> config = get_v1_standard_config()
        >>> model = apply_drai_v1("EleutherAI/pythia-1b", config=config)
    """
    # Load model if string
    if isinstance(model, str):
        from .apply import load_model
        model = load_model(model, **kwargs)

    # Use conservative config if not provided (safe for 410M)
    if config is None:
        config = get_v1_conservative_config()

    # Apply DRAI V1
    model = inject_drai_v1_into_model(model, config, verbose=True)

    return model


def get_drai_v1_stats(model: PreTrainedModel) -> dict:
    """Get statistics from DRAI V1 layers in a model.

    Args:
        model: Model with DRAI V1 applied

    Returns:
        Dictionary with DRAI V1 statistics

    Example:
        >>> stats = get_drai_v1_stats(model)
        >>> print(f"Active attractors: {stats['total_active']}")
        >>> print(f"Total strength: {stats['total_strength']:.2f}")
    """
    from .neox_integration_v1 import DraiGPTNeoXAttentionV1

    stats = {
        'num_drai_layers': 0,
        'total_active': 0,
        'total_strength': 0.0,
        'mean_strength': 0.0,
        'layers': []
    }

    # Iterate through layers
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        # Check if this layer has V1 wrapper
        if isinstance(layer.attention, DraiGPTNeoXAttentionV1):
            drai = layer.attention.drai

            # Get stats from V1 layer
            layer_stats = drai.get_stats()

            stats['num_drai_layers'] += 1
            stats['total_active'] += layer_stats['num_alive']
            stats['total_strength'] += layer_stats['total_strength']

            stats['layers'].append({
                'layer_idx': layer_idx,
                **layer_stats
            })

    # Compute mean
    if stats['num_drai_layers'] > 0:
        stats['mean_strength'] = stats['total_strength'] / stats['num_drai_layers']

    return stats
