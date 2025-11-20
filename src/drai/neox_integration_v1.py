"""
DRAI V1 — GPT-NeoX Integration Layer
-------------------------------------

This module injects the Dynamic Resonance AI (V1) algorithm directly
into GPT-NeoX attention layers.

V1 is designed for:
    • Small/medium NeoX models (70M–1B)
    • Zero-training augmentation
    • Fully deterministic behavior
    • Safe attractor memory with burn-in and soft gating

It replaces each GPT-NeoX attention module with a wrapper that:
    1. Extracts Q vectors for resonance processing
    2. Runs DRAI V1 attractor dynamics
    3. Produces synthetic resonance V vectors
    4. Adds a stable, bounded influence vector to the original output
"""

from typing import Optional
import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from .resonance_layer_v1 import DraiResonanceLayerV1
from .config import V1Config


# ============================================================================
# Wrapped GPT-NeoX Attention
# ============================================================================

class DraiGPTNeoXAttentionV1(nn.Module):
    """
    A drop-in replacement for GPTNeoXAttention that includes:
        • attractor formation
        • resonance field generation
        • bounded output correction

    The original HF attention object is preserved internally and used
    for all baseline computations.
    """

    def __init__(self, original_attention: GPTNeoXAttention,
                 drai_config: V1Config, layer_idx: int):
        super().__init__()

        self.original_attention = original_attention
        self.layer_idx = layer_idx
        self.hidden_size = original_attention.config.hidden_size
        self.num_attention_heads = original_attention.config.num_attention_heads
        self.head_size = original_attention.head_size

        # --- Initialize the resonance engine ---
        hp = drai_config.hyperparameters

        self.drai = DraiResonanceLayerV1(
            hidden_size=self.hidden_size,
            num_heads=drai_config.num_drai_heads,
            head_dim=self.head_size,
            device=str(original_attention.query_key_value.weight.device),
            dtype=original_attention.query_key_value.weight.dtype,
            max_attractors=hp.max_attractors,
            theta_match=hp.theta_match,
            alpha_update=hp.alpha_update,
            lambda_decay=hp.lambda_decay,
            strength_init=hp.strength_init,
            strength_min=hp.strength_min,
            max_influence_scale=hp.max_influence_scale,
            burn_in_threshold=hp.burn_in_threshold,
            burn_in_mode=hp.burn_in_mode,
            burn_in_tokens=hp.burn_in_tokens,
        )

        # HF expects this wrapper to expose all the same public attributes
        passthrough = [
            "query_key_value", "dense", "norm_factor",
            "attention_dropout", "rotary_emb", "rotary_ndims",
            "bias", "masked_bias", "scaling"
        ]
        for name in passthrough:
            if hasattr(original_attention, name):
                setattr(self, name, getattr(original_attention, name))

    # ----------------------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[tuple] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs
    ):
        """
        Forward pass with a stability-preserving DRAI V1 influence vector.
        """

        batch, seq_len, hidden_dim = hidden_states.shape

        # ---- 1) Extract Q vectors ------------------------------------------------
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch, seq_len, self.num_attention_heads, 3 * self.head_size)

        # Only the Q slice is needed for attractor updates
        q = qkv[..., :self.head_size]                  # [B, S, H, D]
        q_for_drai = q.permute(1, 0, 2, 3)             # [S, B, H, D]

        # ---- 2) Run resonance ----------------------------------------------------
        k_reson, v_reson = self.drai(q_for_drai)       # [S, B, H_drai, D]

        # ---- 3) Run baseline GPT-NeoX attention ---------------------------------
        base_out = self.original_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )

        # Normalize output representation
        if isinstance(base_out, tuple):
            original_out = base_out[0]
            remainder = base_out[1:]
        else:
            original_out = base_out
            remainder = None

        # ---- 4) Skip correction if still in burn-in ------------------------------
        if k_reson.norm() < 1e-8 and v_reson.norm() < 1e-8:
            return base_out

        # ---- 5) Convert resonance field → influence vector -----------------------
        v_reson_b_s_h_d = v_reson.permute(1, 0, 2, 3)
        influence_vec = v_reson_b_s_h_d.mean(dim=2)   # collapse DRAI-heads → [B, S, D]

        # ---- 6) Match hidden dimension (simple, safe expansion) ------------------
        if influence_vec.shape[-1] < hidden_dim:
            factor = hidden_dim // influence_vec.shape[-1]
            influence_vec = influence_vec.repeat(1, 1, factor)
        else:
            influence_vec = influence_vec[..., :hidden_dim]

        # ---- 7) Combine ----------------------------------------------------------
        modified = original_out + influence_vec

        if remainder is None:
            return modified
        return (modified,) + remainder

    # ----------------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------------

    def get_drai_statistics(self):
        st = self.drai.get_stats()
        st["layer_idx"] = self.layer_idx
        st["num_drai_heads"] = self.drai.num_heads
        return st


# ============================================================================
# Injection API
# ============================================================================

def inject_drai_v1_into_model(model, config: V1Config, verbose: bool = True):
    """
    Replace GPT-NeoX attention modules with DRAI-enabled versions.

    Requirements:
        • model must have `model.gpt_neox.layers`
    """

    # ---- Architecture Check -----------------------------------------------------
    if not hasattr(model, "gpt_neox"):
        raise TypeError("DRAI V1 injection only supports GPT-NeoX models.")

    layers = model.gpt_neox.layers
    num_layers = len(layers)
    target_layers = config.resolve_layers(num_layers)

    if verbose:
        print(f"[DRAI V1] Injecting into layers {target_layers}")
        hp = config.hyperparameters
        print(f"   max_attractors      = {hp.max_attractors}")
        print(f"   theta_match         = {hp.theta_match}")
        print(f"   max_influence_scale = {hp.max_influence_scale}")
        print(f"   burn_in_threshold   = {hp.burn_in_threshold}")

    # ---- Inject wrapper ---------------------------------------------------------
    for idx in target_layers:
        orig_attn = layers[idx].attention
        wrapped = DraiGPTNeoXAttentionV1(orig_attn, config, idx)

        # Ensure matching device/dtype
        wrapped = wrapped.to(
            device=orig_attn.query_key_value.weight.device,
            dtype=orig_attn.query_key_value.weight.dtype,
        )

        layers[idx].attention = wrapped

        if verbose:
            print(f"[DRAI V1] ✓ Wrapped layer {idx}")

    if verbose:
        print("[DRAI V1] Injection complete.")

    return model
