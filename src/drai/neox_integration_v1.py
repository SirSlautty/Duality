"""V1 DRAI integration for GPT-NeoX models.

This module provides a minimal integration of DRAI V1 into GPT-NeoX attention.
It wraps existing attention modules to inject V1 resonance K/V pairs.

V1 uses a simpler, safer algorithm than Phase 2 DRAI.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

from .resonance_layer_v1 import DraiResonanceLayerV1
from .config import DraiV1Config


class DraiGPTNeoXAttentionV1(nn.Module):
    """GPT-NeoX Attention with V1 DRAI injection.

    This is a thin wrapper around the standard GPTNeoXAttention
    that injects V1 resonance K/V pairs into the attention mechanism.

    The key difference from Phase 2 DRAI is that V1 uses:
    - Proper pattern detection (live + score thresholds)
    - Field vector approach (weighted mean of attractors)
    - Soft gating (tanh * max_influence_scale)
    - Conservative parameters for small models
    """

    def __init__(
        self,
        original_attention: GPTNeoXAttention,
        drai_config: DraiV1Config,
        layer_idx: int,
    ):
        super().__init__()

        # Store original attention (we'll delegate to it)
        self.original_attention = original_attention

        # Copy key attributes for compatibility
        self.config = original_attention.config
        self.layer_idx = layer_idx
        self.num_attention_heads = original_attention.config.num_attention_heads
        self.head_size = original_attention.head_size
        self.hidden_size = original_attention.config.hidden_size

        # DRAI V1 configuration
        self.num_drai_heads = drai_config.num_drai_heads

        # Create V1 resonance layer
        self.drai = DraiResonanceLayerV1(
            hidden_size=self.hidden_size,
            num_heads=drai_config.num_drai_heads,
            head_dim=self.head_size,
            device=str(original_attention.query_key_value.weight.device),
            dtype=original_attention.query_key_value.weight.dtype,
            max_attractors=drai_config.hyperparameters.max_attractors,
            theta_match=drai_config.hyperparameters.theta_match,
            alpha_update=drai_config.hyperparameters.alpha_update,
            lambda_decay=drai_config.hyperparameters.lambda_decay,
            strength_init=drai_config.hyperparameters.strength_init,
            strength_min=drai_config.hyperparameters.strength_min,
            max_influence_scale=drai_config.hyperparameters.max_influence_scale,
            burn_in_threshold=drai_config.hyperparameters.burn_in_threshold,
        )

        # Forward all attributes that might be accessed
        for attr in ['query_key_value', 'dense', 'norm_factor', 'attention_dropout',
                     'rotary_emb', 'rotary_ndims', 'bias', 'masked_bias', 'scaling']:
            if hasattr(original_attention, attr):
                setattr(self, attr, getattr(original_attention, attr))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[tuple] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ):
        """Forward pass with V1 DRAI injection.

        This follows a simplified pattern:
        1. Run original attention to get Q, K, V
        2. Generate DRAI K/V from Q
        3. Concatenate DRAI K/V with standard K/V
        4. Re-run attention computation with augmented K/V

        For simplicity in V1, we take a hook-based approach:
        we just run DRAI on queries for state updates, and let the
        original attention handle everything. This means DRAI learns
        but doesn't inject yet (pure monitoring mode).

        This is the safest way to validate the V1 algorithm works
        without risking breaking the model.
        """
        # Run DRAI on input for state updates (monitoring mode)
        # We convert hidden_states to query-like format for DRAI
        batch, seq_len, hidden_size = hidden_states.shape

        # Project to QKV
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch, seq_len, self.num_attention_heads, 3 * self.head_size)

        # Split to get query
        query = qkv[..., :self.head_size]  # [batch, seq_len, num_heads, head_size]

        # Transpose for DRAI: [seq_len, batch, num_heads, head_size]
        query_for_drai = query.permute(1, 0, 2, 3)

        # Call DRAI V1 (this updates state but we don't use output yet)
        k_reson, v_reson = self.drai(query_for_drai)
        # k_reson, v_reson: [seq_len, batch, num_drai_heads, head_size]

        # For now, just run original attention
        # (We can add K/V injection later once we verify state updates work)
        return self.original_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def get_drai_statistics(self) -> dict:
        """Get DRAI V1 statistics."""
        stats = self.drai.get_stats()
        stats["layer_idx"] = self.layer_idx
        stats["num_drai_heads"] = self.num_drai_heads
        return stats


def inject_drai_v1_into_model(
    model,
    drai_config: DraiV1Config,
    verbose: bool = True,
):
    """Inject DRAI V1 into a GPT-NeoX model.

    Args:
        model: GPT-NeoX model
        drai_config: V1 configuration
        verbose: Print progress

    Returns:
        Modified model
    """
    if not drai_config.enabled:
        if verbose:
            print("[DRAI V1] Disabled in config")
        return model

    num_layers = len(model.gpt_neox.layers)

    # Determine target layers
    if drai_config.layer_mode == "mid":
        target_layers = [num_layers // 2]
    elif drai_config.layer_mode == "specific":
        target_layers = drai_config.specific_layers or []
    elif drai_config.layer_mode == "all":
        target_layers = list(range(num_layers))
    else:
        raise ValueError(f"Unknown layer_mode: {drai_config.layer_mode}")

    if verbose:
        print(f"[DRAI V1] Injecting into {len(target_layers)} layers: {target_layers}")
        print(f"[DRAI V1] Hyperparameters:")
        print(f"  - max_attractors: {drai_config.hyperparameters.max_attractors}")
        print(f"  - theta_match: {drai_config.hyperparameters.theta_match}")
        print(f"  - max_influence_scale: {drai_config.hyperparameters.max_influence_scale}")

    # Inject into target layers
    for layer_idx in target_layers:
        layer = model.gpt_neox.layers[layer_idx]

        # Wrap original attention
        original_attention = layer.attention

        # Create V1 wrapper
        v1_attention = DraiGPTNeoXAttentionV1(
            original_attention=original_attention,
            drai_config=drai_config,
            layer_idx=layer_idx,
        )

        # Move to same device/dtype as original
        v1_attention = v1_attention.to(
            device=original_attention.query_key_value.weight.device,
            dtype=original_attention.query_key_value.weight.dtype,
        )

        # Replace attention
        layer.attention = v1_attention

        if verbose:
            print(f"[DRAI V1] Layer {layer_idx}: ✓ Injected")

    if verbose:
        print(f"[DRAI V1] Injection complete!")

    return model
