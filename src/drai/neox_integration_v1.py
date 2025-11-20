"""
DRAI (Dynamic Resonance AI) — V1 NeoX Integration
-------------------------------------------------

Production-grade integration of the DRAI V1 resonance engine into
GPT-NeoX / Pythia-family attention modules.

This adapter:
    • extracts Q for attractor processing
    • runs V1 resonance dynamics
    • receives the synthetic V-resonance field
    • injects a strictly bounded influence vector into the baseline output

Aligned with the public V1 API:
    - DRAIV1ResonanceLayer
    - V1Config / V1Hyperparameters
"""

from typing import Optional
import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

from .resonance_layer_v1 import DRAIV1ResonanceLayer
from .config import V1Config


# =====================================================================
# DRAI V1 Attention Wrapper for GPT-NeoX
# =====================================================================

class DRAIV1NeoXAttention(nn.Module):
    """
    Drop-in replacement for GPTNeoXAttention adding:
        • attractor formation
        • resonance-field generation
        • bounded influence vector injection

    Behaviour is otherwise identical to the original GPT-NeoX attention.
    """

    def __init__(self, original_attention: GPTNeoXAttention,
                 config: V1Config,
                 layer_idx: int):
        super().__init__()

        self.original_attention = original_attention
        self.layer_idx = layer_idx

        # Hidden configuration
        self.hidden_size = original_attention.config.hidden_size
        self.num_attention_heads = original_attention.config.num_attention_heads
        self.head_size = original_attention.head_size

        # ------------------------------------------------------------------
        # Initialize the DRAI V1 resonance engine
        # ------------------------------------------------------------------
        hp = config.hyperparameters

        self.resonance_layer = DRAIV1ResonanceLayer(
            hidden_size=self.hidden_size,
            num_heads=config.num_drai_heads,
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

        # Attributes that HF expects to exist on attention modules
        passthrough = [
            "query_key_value", "dense", "norm_factor",
            "attention_dropout", "rotary_emb", "rotary_ndims",
            "bias", "masked_bias", "scaling"
        ]
        for name in passthrough:
            if hasattr(original_attention, name):
                setattr(self, name, getattr(original_attention, name))

    # =====================================================================
    # Forward Pass
    # =====================================================================
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
        batch, seq_len, hidden_dim = hidden_states.shape

        # -------------------------------------------------------
        # (1) Extract Q only (QKV fused in NeoX)
        # -------------------------------------------------------
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch, seq_len, self.num_attention_heads, 3 * self.head_size)

        q = qkv[..., :self.head_size]          # [B, S, H, D]
        q_for_resonance = q.permute(1, 0, 2, 3)  # [S, B, H, D]

        # -------------------------------------------------------
        # (2) Run DRAI V1 resonance dynamics
        # -------------------------------------------------------
        k_reson, v_reson = self.resonance_layer(q_for_resonance)

        # -------------------------------------------------------
        # (3) Baseline GPT-NeoX attention
        # -------------------------------------------------------
        base = self.original_attention(
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

        if isinstance(base, tuple):
            base_out, tail = base[0], base[1:]
        else:
            base_out, tail = base, None

        # -------------------------------------------------------
        # (4) No injection until burn-in completes
        # -------------------------------------------------------
        if k_reson.norm() < 1e-8:
            return base

        # -------------------------------------------------------
        # (5) Convert resonance field → influence vector
        # -------------------------------------------------------
        v_res = v_reson.permute(1, 0, 2, 3)   # [B, S, H_drai, D]
        influence = v_res.mean(dim=2)         # [B, S, D]

        # -------------------------------------------------------
        # (6) Normalize and match hidden dimension safely
        # -------------------------------------------------------
        if influence.shape[-1] < hidden_dim:
            k = hidden_dim // influence.shape[-1]
            influence = influence.repeat(1, 1, k)
        else:
            influence = influence[..., :hidden_dim]

        # -------------------------------------------------------
        # (7) Final output = baseline + influence
        # -------------------------------------------------------
        modified = base_out + influence

        if tail is None:
            return modified
        return (modified,) + tail

    # =====================================================================
    # Public statistics
    # =====================================================================
    def get_v1_statistics(self):
        stats = self.resonance_layer.get_stats()
        stats["layer_idx"] = self.layer_idx
        stats["num_drai_heads"] = self.resonance_layer.num_heads
        return stats


# =====================================================================
# Injection Utility
# =====================================================================

def apply_v1_neox_injection(model, config: V1Config, verbose: bool = True):
    """
    Patch a GPT-NeoX model in-place, replacing attention modules
    with DRAI V1-enabled variants.
    """

    if not hasattr(model, "gpt_neox"):
        raise TypeError("DRAI V1 injection only supports GPT-NeoX models.")

    layers = model.gpt_neox.layers
    num_layers = len(layers)
    target_layers = config.resolve_layers(num_layers)

    if verbose:
        hp = config.hyperparameters
        print(f"[DRAI V1] Injecting into layers: {target_layers}")
        print(f"   max_attractors      = {hp.max_attractors}")
        print(f"   theta_match         = {hp.theta_match}")
        print(f"   max_influence_scale = {hp.max_influence_scale}")
        print(f"   burn_in_threshold   = {hp.burn_in_threshold}")

    for idx in target_layers:
        orig = layers[idx].attention
        wrapped = DRAIV1NeoXAttention(orig, config, idx)

        wrapped = wrapped.to(
            device=orig.query_key_value.weight.device,
            dtype=orig.query_key_value.weight.dtype,
        )

        layers[idx].attention = wrapped

        if verbose:
            print(f"[DRAI V1] ✓ wrapped layer {idx}")

    if verbose:
        print("[DRAI V1] Injection complete.")

    return model
