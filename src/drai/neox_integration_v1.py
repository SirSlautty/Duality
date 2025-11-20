"""
V1 DRAI integration for GPT-NeoX models.

This module wraps GPT-NeoX attention layers with the V1 DRAI algorithm,
using the unified config API from config.py.

V1 design goals:
- Conservative, production-safe memory for 410M–1B models
- Proper pattern detection
- Field-vector update rule
- Soft gating with influence-cap
- Burn-in (strength or token-based)
"""

import torch
import torch.nn as nn
from typing import Optional

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

from .resonance_layer_v1 import DraiResonanceLayerV1
from .config import V1Config


# ============================================================================
# Attention Wrapper
# ============================================================================

class DraiGPTNeoXAttentionV1(nn.Module):
    """
    GPT-NeoX Attention with V1 DRAI injection.
    Wraps an existing GPTNeoXAttention module.
    """

    def __init__(self, original_attention: GPTNeoXAttention, drai_config: V1Config, layer_idx: int):
        super().__init__()

        self.original_attention = original_attention
        self.config = original_attention.config
        self.layer_idx = layer_idx

        self.num_attention_heads = original_attention.config.num_attention_heads
        self.head_size = original_attention.head_size
        self.hidden_size = original_attention.config.hidden_size

        # --- DRAI V1 Hyperparams ---
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

        # Mirror attributes the HF attention code expects
        passthrough_attrs = [
            "query_key_value", "dense", "norm_factor", "attention_dropout",
            "rotary_emb", "rotary_ndims", "bias", "masked_bias", "scaling"
        ]
        for attr in passthrough_attrs:
            if hasattr(original_attention, attr):
                setattr(self, attr, getattr(original_attention, attr))

    # =======================================================================
    # Forward Pass
    # =======================================================================

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
        """
        Forward with V1 resonance injection.

        A fully mathematically-pure KV-concat version will come later.
        For now we use a controlled influence-pathway that preserves stability.
        """

        batch, seq_len, hidden_size = hidden_states.shape

        # 1) Compute Q,K,V in NeoX form
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch, seq_len, self.num_attention_heads, 3 * self.head_size)

        query = qkv[..., :self.head_size]                     # [B, S, H, D]
        query_for_drai = query.permute(1, 0, 2, 3)            # [S, B, H, D]

        # 2) Call DRAI V1
        k_reson, v_reson = self.drai(query_for_drai)          # each: [S, B, H_drai, D]

        # 3) Run original attention (QKV from original layer)
        attn_output = self.original_attention(
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

        # Extract base output (return may be tuple)
        if isinstance(attn_output, tuple):
            base_output = attn_output[0]
            tail = attn_output[1:]
        else:
            base_output = attn_output
            tail = None

        # If burn-in has not completed OR negligible resonance → return raw output
        if (k_reson.norm() < 1e-8) and (v_reson.norm() < 1e-8):
            return attn_output

        # 4) Compute low-risk correction vector
        v_reson_bhsd = v_reson.permute(1, 0, 2, 3)            # [B, S, H_drai, D]
        drai_vec = v_reson_bhsd.mean(dim=2)                   # [B, S, D]

        # 5) Match hidden size by projection-aligned expansion
        if drai_vec.shape[-1] < hidden_size:
            repeat_factor = hidden_size // drai_vec.shape[-1]
            drai_vec = drai_vec.repeat(1, 1, repeat_factor)
        elif drai_vec.shape[-1] > hidden_size:
            drai_vec = drai_vec[..., :hidden_size]

        modified_output = base_output + drai_vec

        if tail is None:
            return modified_output
        else:
            return (modified_output,) + tail

    # =======================================================================
    # Stats
    # =======================================================================

    def get_drai_statistics(self):
        stats = self.drai.get_stats()
        stats["layer_idx"] = self.layer_idx
        stats["num_drai_heads"] = self.drai.num_heads
        return stats


# ============================================================================
# Injection API
# ============================================================================

def inject_drai_v1_into_model(model, drai_config: V1Config, verbose: bool = True):
    """
    Inject V1 DRAI into GPT-NeoX.

    Uses the new config.resolve_layers() system.
    """

    if not drai_config:
        if verbose:
            print("[DRAI V1] No configuration provided — skipping.")
        return model

    if not drai_config.enabled:
        if verbose:
            print("[DRAI V1] Disabled.")
        return model

    num_layers = len(model.gpt_neox.layers)
    target_layers = drai_config.resolve_layers(num_layers)

    if verbose:
        print(f"[DRAI V1] Injecting into layers: {target_layers}")
        hp = drai_config.hyperparameters
        print(f"  max_attractors      = {hp.max_attractors}")
        print(f"  theta_match         = {hp.theta_match}")
        print(f"  max_influence_scale = {hp.max_influence_scale}")
        print(f"  burn_in_threshold   = {hp.burn_in_threshold}")

    for layer_idx in target_layers:
        original_attn = model.gpt_neox.layers[layer_idx].attention

        wrapper = DraiGPTNeoXAttentionV1(
            original_attention=original_attn,
            drai_config=drai_config,
            layer_idx=layer_idx,
        )

        # Move to same device/dtype
        wrapper = wrapper.to(
            device=original_attn.query_key_value.weight.device,
            dtype=original_attn.query_key_value.weight.dtype,
        )

        model.gpt_neox.layers[layer_idx].attention = wrapper

        if verbose:
            print(f"[DRAI V1] Layer {layer_idx}: ✓ wrapped")

    if verbose:
        print("[DRAI V1] Injection complete.")

    return model

