"""High-level API for applying DRAI to models.

This module provides a simple interface for end users to augment
their models with DRAI self-organizing memory.

Example:
    >>> from drai import apply_drai
    >>> from transformers import AutoModel
    >>>
    >>> model = AutoModel.from_pretrained("EleutherAI/pythia-70m")
    >>> model = apply_drai(model)  # That's it!
    >>>
    >>> # Now use the model normally
    >>> outputs = model(input_ids)
"""

from typing import Optional, Union
import torch.nn as nn
from transformers import PreTrainedModel

from .config import DraiConfig, get_full_drai_config
from .build_drai_neox import inject_drai_into_model


def apply_drai(
    model: Union[PreTrainedModel, str],
    config: Optional[DraiConfig] = None,
    **kwargs
) -> PreTrainedModel:
    """Apply DRAI to a transformer model.

    This is the main entry point for end users. It augments a model
    with self-organizing memory through attractor dynamics.

    Args:
        model: Either a loaded model or a model name/path
        config: DRAI configuration (if None, uses default)
        **kwargs: Additional arguments for model loading

    Returns:
        Model with DRAI applied

    Example:
        >>> from drai import apply_drai
        >>> model = apply_drai("EleutherAI/pythia-70m")
        >>>
        >>> # Or with custom config
        >>> from drai import DraiConfig
        >>> config = DraiConfig(num_drai_heads=2, max_attractors=64)
        >>> model = apply_drai("EleutherAI/pythia-70m", config=config)
    """
    # Load model if string
    if isinstance(model, str):
        model = load_model(model, **kwargs)

    # Use default config if not provided
    if config is None:
        config = get_full_drai_config()

    # Apply DRAI
    model = inject_drai_into_model(model, config, verbose=True)

    return model


def apply_drai_to_model(
    model_name: str,
    config: Optional[DraiConfig] = None,
    **kwargs
) -> PreTrainedModel:
    """Apply DRAI to a model by name.

    Alias for apply_drai() that always loads from name.

    Args:
        model_name: HuggingFace model name or path
        config: DRAI configuration (if None, uses default)
        **kwargs: Additional arguments for model loading

    Returns:
        Model with DRAI applied
    """
    return apply_drai(model_name, config, **kwargs)


def load_model(model_name: str, **kwargs) -> PreTrainedModel:
    """Load a model from HuggingFace.

    Args:
        model_name: Model name or path
        **kwargs: Additional arguments (dtype, device_map, etc.)

    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM

    print(f"[DRAI] Loading model: {model_name}")

    # Extract our custom args
    torch_dtype = kwargs.pop('dtype', kwargs.pop('torch_dtype', None))
    device_map = kwargs.pop('device_map', None)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs
    )

    return model


def get_drai_stats(model: PreTrainedModel) -> dict:
    """Get statistics from DRAI layers in a model.

    Args:
        model: Model with DRAI applied

    Returns:
        Dictionary with DRAI statistics

    Example:
        >>> stats = get_drai_stats(model)
        >>> print(f"Active attractors: {stats['total_active']}")
    """
    stats = {
        'num_drai_layers': 0,
        'total_active': 0,
        'total_created': 0,
        'total_reinforced': 0,
        'layers': []
    }

    # Iterate through layers
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        if hasattr(layer.attention, 'drai'):
            drai = layer.attention.drai

            # Get stats
            active = (drai.attractor_coherence > 0.1).sum().item()
            created = drai._attractors_created.item()
            reinforced = drai._attractors_reinforced.item()

            stats['num_drai_layers'] += 1
            stats['total_active'] += active
            stats['total_created'] += created
            stats['total_reinforced'] += reinforced

            stats['layers'].append({
                'layer_idx': layer_idx,
                'active_attractors': active,
                'attractors_created': created,
                'attractors_reinforced': reinforced,
            })

    return stats


# Convenience exports
__all__ = [
    'apply_drai',
    'apply_drai_to_model',
    'get_drai_stats',
]
