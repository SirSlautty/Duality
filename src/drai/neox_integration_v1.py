"""
DRAI V1 — GPT-NeoX Integration Layer (Updated)
----------------------------------------------

Stable integration of the DRAI Resonance Layer V1 into GPT-NeoX architectures.

The wrapper:
    1. Extracts Q for attractor processing
    2. Runs V1 resonance dynamics
    3. Receives synthetic V-resonance field
    4. Adds a strictly bounded influence vector to the baseline output

This implementation is aligned with the final stable API of:
    - DraiResonanceLayerV1
    - V1Config
"""

from typing import Optional
import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from .resonance_layer_v1 import DraiResonanceLayerV1
from .config import V1Config


# ============================================================================
# Wrapped GPT-NeoX Attention (clean, V1-aligned)
# ============================================================================

class DraiGPTNeoXAttentionV1(nn.Module):
    """
    Drop-in replacement for GPTNeoXAttention with:
        • attractor formation
        • resonance-field generation
        • bounded influence vector addition
    """

    def __init__(self, original_attention: GPTNeoXAttention,
                 drai_config: V1Config,
                 layer_idx: int):
        super().__init__()

        self.original_attention = original_attention
        self.layer_idx = layer_idx

        # HF config references
        self.hidden_size = original_attention.config.hidden_size
        self.num_attention_heads = original_attention.config.num_attention_heads
        self.head_size = original_attention.head_size

        # -------------------------
        # Initialize resonance core
        # -------------------------
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

        # Pass-through attributes HF expects
        passthrough = [
            "query_key_value", "dense", "norm_factor",
            "attention_dropout", "rotary_emb", "rotary_ndims",
            "bias", "masked_bias", "scaling"
        ]
        for name in passthrough:
            if hasattr(original_attention, name):
                setattr(self, name, getattr(original_attention, name))

    # ============================================================================
    # Forward pass
    # ============================================================================
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
        # (1) Extract Q (only Q is needed for resonance dynamics)
        # -------------------------------------------------------
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch, seq_len, self.num_attention_heads, 3 * self.head_size)

        q = qkv[..., :self.head_size]           # [B, S, H, D]
        q_for_drai = q.permute(1, 0, 2, 3)      # [S, B, H, D]

        # -------------------------------------------------------
        # (2) Run DRAI resonance (updated Layer API)
        # -------------------------------------------------------
        k_reson, v_reson = self.drai(q_for_drai)

        # -------------------------------------------------------
        # (3) Baseline GPT-NeoX attention
        # -------------------------------------------------------
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

        if isinstance(base_out, tuple):
            original_out = base_out[0]
            tail = base_out[1:]
        else:
            original_out = base_out
            tail = None

        # -------------------------------------------------------
        # (4) Skip influence if still in burn-in
        # -------------------------------------------------------
        if k_reson.norm() < 1e-8:
            return base_out

        # -------------------------------------------------------
        # (5) Convert resonance → influence vector
        # -------------------------------------------------------
        v_res = v_reson.permute(1, 0, 2, 3)     # [B, S, H_drai, D]
        influence = v_res.mean(dim=2)           # [B, S, D]

        # -------------------------------------------------------
        # (6) Match hidden dimension (safe expansion)
        # -------------------------------------------------------
        if influence.shape[-1] < hidden_dim:
            k = hidden_dim // influence.shape[-1]
            influence = influence.repeat(1, 1, k)
        else:
            influence = influence[..., :hidden_dim]

        # -------------------------------------------------------
        # (7) Combine baseline + influence
        # -------------------------------------------------------
        modified = original_out + influence

        if tail is None:
            return modified
        return (modified,) + tail

    # ============================================================================
    # Stats
    # ============================================================================
    def get_drai_statistics(self):
        st = self.drai.get_stats()
        st["layer_idx"] = self.layer_idx
        st["num_drai_heads"] = self.drai.num_heads
        return st


# ============================================================================
# Injection Utility
# ============================================================================

def inject_drai_v1_into_model(model, config: V1Config, verbose: bool = True):
    """
    Replace GPT-NeoX attention modules with DRAI-enabled attention.
    """

    if not hasattr(model, "gpt_neox"):
        raise TypeError("DRAI V1 injection only supports GPT-NeoX models.")

    layers = model.gpt_neox.layers
    num_layers = len(layers)
    targets = config.resolve_layers(num_layers)

    if verbose:
        hp = config.hyperparameters
        print(f"[DRAI V1] Injecting into layers: {targets}")
        print(f"   max_attractors      = {hp.max_attractors}")
        print(f"   theta_match         = {hp.theta_match}")
        print(f"   max_influence_scale = {hp.max_influence_scale}")
        print(f"   burn_in_threshold   = {hp.burn_in_threshold}")

    for idx in targets:
        orig = layers[idx].attention
        wrapped = DraiGPTNeoXAttentionV1(orig, config, idx)

        # enforce device/dtype parity
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
