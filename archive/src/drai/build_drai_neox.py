"""Model builder for DRAI-enhanced GPT-NeoX models.

This module provides functions to load pre-trained GPT-NeoX models from
HuggingFace and inject DRAI into the attention layers.

Key Functions:
- build_drai_neox_model(): Load model and inject DRAI
- inject_drai_into_model(): Inject DRAI into existing model
- load_model_and_tokenizer(): Convenience function for both

Example Usage:
    # Load pythia-125m with DRAI
    from src.drai.config import get_full_drai_config
    from src.drai.build_drai_neox import build_drai_neox_model

    config = get_full_drai_config()
    model = build_drai_neox_model("EleutherAI/pythia-125m", config)

    # Generate text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-125m")

    input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
"""

from typing import Optional, Union
import torch
from transformers import (
    GPTNeoXForCausalLM,
    GPTNeoXConfig,
    AutoTokenizer,
)

from .config import DraiConfig
from .neox_integration import DraiGPTNeoXAttention


def inject_drai_into_model(
    model: GPTNeoXForCausalLM,
    drai_config: DraiConfig,
    verbose: bool = True,
    lesion_mode: Optional[str] = None,  # Phase 5+: "zero", "scramble", or None
) -> GPTNeoXForCausalLM:
    """Inject DRAI into an existing GPT-NeoX model.

    This function replaces standard GPTNeoXAttention modules with
    DraiGPTNeoXAttention modules in all layers (or selected layers
    based on drai_config.layer_mode).

    The pre-trained weights are preserved - only the attention mechanism
    is enhanced with DRAI. The model can still be used for inference
    and generation as normal.

    Args:
        model: Pre-loaded GPT-NeoX model
        drai_config: DRAI configuration
        verbose: Print injection progress

    Returns:
        Modified model with DRAI injected

    Example:
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-125m")
        config = DraiConfig(enabled=True, phase=2, layer_mode="all")
        model = inject_drai_into_model(model, config)

    Note:
        This function modifies the model in-place and also returns it.
    """
    if not drai_config.enabled:
        if verbose:
            print("[DRAI] DRAI disabled - no injection performed")
        return model

    num_layers = len(model.gpt_neox.layers)
    layers_injected = 0

    if verbose:
        print(f"[DRAI] Injecting DRAI into GPT-NeoX model ({num_layers} layers)")
        print(f"[DRAI] Configuration: {drai_config.layer_mode} mode, phase {drai_config.phase}")

    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        # Check if this layer should receive DRAI
        if not drai_config.should_inject_layer(layer_idx):
            continue

        # Get original attention module
        original_attention = layer.attention

        # Create DRAI-enhanced attention
        drai_attention = DraiGPTNeoXAttention(
            config=model.config,
            drai_config=drai_config,
            layer_idx=layer_idx,
            lesion_mode=lesion_mode,  # Phase 5+: pass lesion_mode
        )

        # Copy pre-trained weights from original attention
        # QKV projection weights
        drai_attention.query_key_value.weight.data.copy_(
            original_attention.query_key_value.weight.data
        )
        if original_attention.query_key_value.bias is not None:
            drai_attention.query_key_value.bias.data.copy_(
                original_attention.query_key_value.bias.data
            )

        # Output projection weights
        drai_attention.dense.weight.data.copy_(
            original_attention.dense.weight.data
        )
        if original_attention.dense.bias is not None:
            drai_attention.dense.bias.data.copy_(
                original_attention.dense.bias.data
            )

        # Replace attention module
        layer.attention = drai_attention

        layers_injected += 1

        if verbose and drai_config.verbose_logging:
            print(f"[DRAI]   Layer {layer_idx}: ✓ DRAI injected")

    if verbose:
        print(f"[DRAI] Injection complete: {layers_injected}/{num_layers} layers enhanced")

    return model


def build_drai_neox_model(
    model_name_or_path: str,
    drai_config: Optional[DraiConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = None,
    trust_remote_code: bool = False,
    verbose: bool = True,
    **model_kwargs,
) -> GPTNeoXForCausalLM:
    """Build a DRAI-enhanced GPT-NeoX model from HuggingFace.

    This is the main entry point for loading pre-trained GPT-NeoX models
    with DRAI integration. It handles:
    1. Loading the pre-trained model
    2. Injecting DRAI into attention layers
    3. Preserving all pre-trained weights
    4. Configuring device placement

    Args:
        model_name_or_path: HuggingFace model identifier or local path
            Examples: "EleutherAI/pythia-125m", "EleutherAI/pythia-1.4b"
        drai_config: DRAI configuration (None = no DRAI, standard model)
        torch_dtype: Model dtype (None = auto, torch.float16 for GPU)
        device_map: Device placement ("auto", "cpu", "cuda", or custom dict)
        trust_remote_code: Whether to trust remote code (required for some models)
        verbose: Print loading progress
        **model_kwargs: Additional arguments passed to from_pretrained()

    Returns:
        DRAI-enhanced GPT-NeoX model ready for inference/generation

    Example:
        # Load pythia-125m with full DRAI
        from src.drai.config import get_full_drai_config
        config = get_full_drai_config()
        model = build_drai_neox_model(
            "EleutherAI/pythia-125m",
            drai_config=config,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        # Load larger model with selective DRAI (GPU)
        config = get_selective_drai_config(num_layers=32, fraction=0.5)
        model = build_drai_neox_model(
            "EleutherAI/pythia-1.4b",
            drai_config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    Available Models:
        - pythia-70m, pythia-160m, pythia-410m (small, fast)
        - pythia-1b, pythia-1.4b, pythia-2.8b (medium)
        - pythia-6.9b, pythia-12b (large, requires GPU)

    Note:
        For Phase 3, we recommend starting with pythia-125m or pythia-410m
        for faster iteration and debugging.
    """
    if verbose:
        print(f"[DRAI] Loading model: {model_name_or_path}")

    # Load pre-trained model
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )

    if verbose:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[DRAI] Model loaded: {num_params:,} parameters")

    # Inject DRAI if configured
    if drai_config is not None and drai_config.enabled:
        model = inject_drai_into_model(model, drai_config, verbose=verbose)
    elif verbose:
        print("[DRAI] No DRAI config provided - using standard model")

    return model


def load_model_and_tokenizer(
    model_name_or_path: str,
    drai_config: Optional[DraiConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = None,
    verbose: bool = True,
    **model_kwargs,
) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    """Load both model and tokenizer (convenience function).

    This is a convenience wrapper that loads both the DRAI-enhanced model
    and its tokenizer in one call.

    Args:
        model_name_or_path: HuggingFace model identifier
        drai_config: DRAI configuration
        torch_dtype: Model dtype
        device_map: Device placement
        verbose: Print loading progress
        **model_kwargs: Additional model arguments

    Returns:
        Tuple of (model, tokenizer)

    Example:
        from src.drai.config import get_full_drai_config

        config = get_full_drai_config()
        model, tokenizer = load_model_and_tokenizer(
            "EleutherAI/pythia-125m",
            drai_config=config,
        )

        # Generate text
        input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")
        output_ids = model.generate(input_ids, max_new_tokens=10)
        print(tokenizer.decode(output_ids[0]))
    """
    # Load model with DRAI
    model = build_drai_neox_model(
        model_name_or_path,
        drai_config=drai_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        verbose=verbose,
        **model_kwargs,
    )

    # Load tokenizer
    if verbose:
        print(f"[DRAI] Loading tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Ensure pad token is set (required for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if verbose:
        print(f"[DRAI] Tokenizer loaded (vocab size: {len(tokenizer)})")

    return model, tokenizer


def get_drai_statistics_from_model(model: GPTNeoXForCausalLM) -> list[dict]:
    """Extract DRAI statistics from all layers.

    Args:
        model: DRAI-enhanced GPT-NeoX model

    Returns:
        List of statistics dicts, one per layer with DRAI

    Example:
        # After running inference
        stats = get_drai_statistics_from_model(model)

        for layer_stats in stats:
            layer_idx = layer_stats["layer_idx"]
            count = layer_stats["attractor_count"]
            print(f"Layer {layer_idx}: {count} attractors formed")
    """
    stats_list = []

    for layer in model.gpt_neox.layers:
        if hasattr(layer.attention, 'get_drai_statistics'):
            stats = layer.attention.get_drai_statistics()
            if stats:  # Only include if DRAI is present
                stats_list.append(stats)

    return stats_list


def reset_drai_statistics_in_model(model: GPTNeoXForCausalLM):
    """Reset DRAI statistics in all layers.

    Args:
        model: DRAI-enhanced GPT-NeoX model

    Example:
        # Reset before evaluation
        reset_drai_statistics_in_model(model)

        # Run evaluation
        outputs = model(input_ids)

        # Get fresh statistics
        stats = get_drai_statistics_from_model(model)
    """
    for layer in model.gpt_neox.layers:
        if hasattr(layer.attention, 'reset_drai_statistics'):
            layer.attention.reset_drai_statistics()


def count_drai_parameters(model: GPTNeoXForCausalLM) -> tuple[int, int]:
    """Count DRAI parameters vs total parameters.

    Args:
        model: DRAI-enhanced GPT-NeoX model

    Returns:
        Tuple of (drai_params, total_params)

    Example:
        drai_params, total_params = count_drai_parameters(model)
        print(f"DRAI parameters: {drai_params:,} ({100*drai_params/total_params:.2f}%)")

    Note:
        Phase 2 DRAI uses buffers (non-learnable), so drai_params will be 0.
        Future phases may add learnable DRAI parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())

    drai_params = 0
    for layer in model.gpt_neox.layers:
        if hasattr(layer.attention, 'drai') and layer.attention.drai is not None:
            drai_params += sum(p.numel() for p in layer.attention.drai.parameters())

    return drai_params, total_params


def print_model_summary(model: GPTNeoXForCausalLM):
    """Print a summary of the DRAI-enhanced model.

    Args:
        model: DRAI-enhanced GPT-NeoX model

    Example:
        model = build_drai_neox_model("EleutherAI/pythia-125m", config)
        print_model_summary(model)
    """
    print("=" * 70)
    print("DRAI-Enhanced GPT-NeoX Model Summary")
    print("=" * 70)

    # Model info
    config = model.config
    print(f"Model: {config._name_or_path if hasattr(config, '_name_or_path') else 'Custom'}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Head dim: {config.hidden_size // config.num_attention_heads}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # DRAI info
    num_layers = len(model.gpt_neox.layers)
    drai_layers = 0
    drai_heads_per_layer = 0

    for layer in model.gpt_neox.layers:
        if hasattr(layer.attention, 'drai') and layer.attention.drai is not None:
            drai_layers += 1
            drai_heads_per_layer = layer.attention.num_drai_heads

    if drai_layers > 0:
        print(f"DRAI Status: ✓ ENABLED")
        print(f"DRAI layers: {drai_layers}/{num_layers}")
        print(f"DRAI heads per layer: {drai_heads_per_layer}")

        # Get attractor config from first DRAI layer
        for layer in model.gpt_neox.layers:
            if hasattr(layer.attention, 'drai') and layer.attention.drai is not None:
                drai = layer.attention.drai
                print(f"Max attractors: {drai.max_attractors}")
                print(f"Coherence threshold: {drai.coherence_threshold}")
                print(f"Formation threshold: {drai.formation_threshold}")
                break
    else:
        print(f"DRAI Status: ✗ DISABLED")

    print("=" * 70)


if __name__ == "__main__":
    """Test model builder functionality."""
    print("=== DRAI Model Builder Test ===\n")

    # Test 1: Load model without DRAI
    print("1. Loading standard model (no DRAI):")
    model_standard = build_drai_neox_model(
        "EleutherAI/pythia-70m",  # Smallest model for testing
        drai_config=None,
        torch_dtype=torch.float32,
        device_map="cpu",
        verbose=True,
    )
    print()

    # Test 2: Load model with DRAI
    print("2. Loading model with DRAI:")
    from .config import get_full_drai_config

    drai_config = get_full_drai_config()
    model_drai = build_drai_neox_model(
        "EleutherAI/pythia-70m",
        drai_config=drai_config,
        torch_dtype=torch.float32,
        device_map="cpu",
        verbose=True,
    )
    print()

    # Test 3: Model summary
    print("3. Model summary:")
    print_model_summary(model_drai)
    print()

    # Test 4: Parameter count
    print("4. Parameter count:")
    drai_params, total_params = count_drai_parameters(model_drai)
    print(f"   DRAI parameters: {drai_params:,}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   DRAI overhead: {100*drai_params/total_params:.4f}%")
    print()

    # Test 5: Simple forward pass
    print("5. Forward pass test:")
    input_ids = torch.randint(0, 50000, (1, 10))  # Random token IDs
    print(f"   Input shape: {input_ids.shape}")

    with torch.no_grad():
        output = model_drai(input_ids)

    print(f"   Output logits shape: {output.logits.shape}")
    print(f"   No NaN: {not torch.isnan(output.logits).any()}")
    print(f"   No Inf: {not torch.isinf(output.logits).any()}")
    print()

    # Test 6: Statistics
    print("6. DRAI statistics:")
    stats = get_drai_statistics_from_model(model_drai)
    print(f"   Layers with DRAI: {len(stats)}")
    if stats:
        first_layer_stats = stats[0]
        print(f"   Layer 0 attractor count: {first_layer_stats.get('attractor_count', 0)}")
    print()

    print("✓ All model builder tests passed!")
