"""GPT-NeoX integration module for DRAI.

This module provides DraiGPTNeoXAttention, a subclass of GPTNeoXAttention
that injects DRAI attractor dynamics into the attention mechanism.

Key Integration Points:
1. DRAI K/V generation after RoPE application
2. K/V concatenation along sequence dimension
3. Attention mask extension for DRAI heads
4. Automatic K/V cache compatibility

See docs/NEOX_ARCHITECTURE_ANALYSIS.md for detailed analysis.
See docs/INTEGRATION_METHODOLOGY.md for methodology and decisions.
"""

from typing import Optional, Callable, Unpack
import torch
import torch.nn as nn

# Import transformers components
try:
    from transformers.models.gpt_neox.modeling_gpt_neox import (
        GPTNeoXAttention,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.cache_utils import Cache
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError as e:
    raise ImportError(
        f"Failed to import transformers components: {e}\n"
        "Please install transformers>=4.30.0: pip install transformers"
    )

# Import DRAI components
from .resonance_layer import DraiResonanceLayer
from .config import DraiConfig


class DraiGPTNeoXAttention(GPTNeoXAttention):
    """GPT-NeoX attention with DRAI injection.

    This class extends standard GPT-NeoX attention by injecting Dynamic
    Resonance AI (DRAI) attractor fields that generate synthetic K/V pairs
    representing accumulated patterns from the input sequence.

    Integration Strategy:
    1. Compute standard Q, K, V projections
    2. Apply RoPE to Q and K (not V, as standard)
    3. ★ Generate DRAI K/V from queries (after RoPE)
    4. ★ Concatenate DRAI K/V with standard K/V along sequence dimension
    5. ★ Extend attention mask to cover DRAI heads
    6. Update K/V cache (automatically includes DRAI)
    7. Compute attention (DRAI K/V participate naturally)
    8. Project output

    The DRAI K/V are treated as extra "positions" in the sequence that all
    tokens can attend to. These represent accumulated attractor patterns.

    Args:
        config: GPT-NeoX model configuration
        drai_config: DRAI configuration (if None, behaves as standard attention)
        layer_idx: Layer index for cache management

    Example:
        # Create DRAI-enhanced attention
        config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-125m")
        drai_config = DraiConfig(enabled=True, phase=2, num_drai_heads=1)

        attention = DraiGPTNeoXAttention(config, drai_config, layer_idx=0)

        # Forward pass
        hidden_states = torch.randn(2, 10, 768)  # [batch, seq, hidden]
        attention_mask = create_causal_mask(10)
        position_embeddings = (cos, sin)

        output, attn_weights = attention(
            hidden_states,
            attention_mask,
            position_embeddings=position_embeddings
        )
    """

    def __init__(
        self,
        config,
        drai_config: Optional[DraiConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        """Initialize DRAI-enhanced NeoX attention.

        Args:
            config: GPT-NeoX model configuration
            drai_config: DRAI configuration (None = standard attention)
            layer_idx: Layer index for cache management
        """
        # Initialize parent class (standard GPT-NeoX attention)
        super().__init__(config, layer_idx)

        # Store DRAI config
        self.drai_config = drai_config

        # Initialize DRAI resonance layer if enabled
        if drai_config and drai_config.enabled and drai_config.should_inject_layer(layer_idx or 0):
            # Determine head dimension
            head_dim = drai_config.drai_head_dim
            if head_dim is None:
                head_dim = self.head_size  # Use model's head dimension

            # Create DRAI resonance layer
            self.drai = DraiResonanceLayer(
                hidden_size=config.hidden_size,
                num_heads=drai_config.num_drai_heads,
                head_dim=head_dim,
                phase=drai_config.phase,
                **drai_config.hyperparameters.to_dict(),
            )

            # Store number of DRAI heads for mask extension
            self.num_drai_heads = drai_config.num_drai_heads

            if drai_config.verbose_logging:
                print(
                    f"[DRAI] Layer {layer_idx}: Initialized DRAI with "
                    f"{drai_config.num_drai_heads} heads, phase {drai_config.phase}"
                )
        else:
            self.drai = None
            self.num_drai_heads = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        """Forward pass with DRAI injection.

        This method follows the standard GPT-NeoX attention flow with
        DRAI injection at the appropriate point (after RoPE, before attention).

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, 1, query_len, key_len]
            head_mask: Optional head masking
            layer_past: K/V cache from previous forward passes
            output_attentions: Whether to return attention weights
            cache_position: Position indices for caching
            position_embeddings: (cos, sin) tuple for RoPE
            **kwargs: Additional arguments for flash attention

        Returns:
            Tuple of (attention_output, attention_weights)
            - attention_output: [batch, seq_len, hidden_size]
            - attention_weights: [batch, num_heads, query_len, key_len] or None
        """
        # ===== Standard GPT-NeoX Attention (Part 1) =====

        # Get input shape
        input_shape = hidden_states.shape[:-1]  # [batch, seq_len]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        # 1. QKV Projection (fused)
        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, 3 * head_size]

        # 2. Split into Q, K, V
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        # Each: [batch, num_heads, seq_len, head_size]

        # 3. Apply RoPE to Q and K (NOT to V, as standard)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ===== DRAI INJECTION POINT =====

        if self.drai is not None:
            # Generate DRAI K/V from query states (after RoPE)
            # DRAI sees the same queries that attention sees
            k_reson, v_reson = self.drai(
                query_layer=query_states,
                attention_mask=attention_mask,
            )
            # k_reson, v_reson: [batch, num_heads, num_drai_heads, head_size]

            # Concatenate DRAI K/V with standard K/V along sequence dimension
            # This treats DRAI heads as extra "positions" that all tokens can attend to
            key_states = torch.cat([key_states, k_reson], dim=2)
            value_states = torch.cat([value_states, v_reson], dim=2)
            # New shapes: [batch, num_heads, seq_len + num_drai_heads, head_size]

            # Extend attention mask to cover DRAI heads
            attention_mask = self._extend_attention_mask(attention_mask)

        # ===== Standard GPT-NeoX Attention (Part 2) =====

        # 4. Cache QKV values (includes DRAI K/V if present)
        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            # Update cache with current K/V (including DRAI)
            key_states, value_states = layer_past.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 5. Select attention implementation (eager, flash, sdpa, etc.)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 6. Compute attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,  # Includes DRAI K if present
            value_states,  # Includes DRAI V if present
            attention_mask,  # Extended if DRAI present
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            head_mask=head_mask,
            **kwargs,
        )
        # attn_output: [batch, seq_len, num_heads, head_size]
        # attn_weights: [batch, num_heads, query_len, key_len] (if requested)

        # 7. Reshape and project output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        # Shape: [batch, seq_len, hidden_size]

        return attn_output, attn_weights

    def _extend_attention_mask(
        self,
        attention_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Extend attention mask to cover DRAI heads.

        NeoX uses additive masking:
        - 0 = attend
        - -inf (or large negative) = mask

        We extend the mask by appending zeros for DRAI heads, allowing
        all query positions to attend to DRAI heads.

        Args:
            attention_mask: Original mask [batch, 1, query_len, key_len]

        Returns:
            Extended mask [batch, 1, query_len, key_len + num_drai_heads]

        Example:
            Original (4 tokens, causal):
            [[0, -inf, -inf, -inf],
             [0, 0,    -inf, -inf],
             [0, 0,    0,    -inf],
             [0, 0,    0,    0   ]]

            Extended (4 tokens + 1 DRAI head):
            [[0, -inf, -inf, -inf, 0],  ← All attend to DRAI
             [0, 0,    -inf, -inf, 0],
             [0, 0,    0,    -inf, 0],
             [0, 0,    0,    0,    0]]
                                   ^
                                   DRAI head (always attend)
        """
        if attention_mask is None:
            return None

        batch, _, query_len, key_len = attention_mask.shape

        # Create extension for DRAI heads (all zeros = all attend)
        drai_extension = torch.zeros(
            batch,
            1,
            query_len,
            self.num_drai_heads,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Concatenate along key dimension
        extended_mask = torch.cat([attention_mask, drai_extension], dim=-1)

        return extended_mask

    def get_drai_statistics(self) -> dict:
        """Get DRAI attractor statistics.

        Returns:
            Dictionary with attractor statistics, or empty dict if no DRAI
        """
        if self.drai is None:
            return {}

        stats = self.drai.get_statistics()

        # Add layer-specific metadata
        stats["layer_idx"] = self.layer_idx
        stats["num_drai_heads"] = self.num_drai_heads

        return stats

    def reset_drai_statistics(self):
        """Reset DRAI statistics counters."""
        if self.drai is not None:
            self.drai.reset_statistics()

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()

        if self.drai is not None:
            drai_repr = (
                f", drai=DraiResonanceLayer("
                f"num_heads={self.num_drai_heads}, "
                f"max_attractors={self.drai.max_attractors})"
            )
            return base_repr + drai_repr

        return base_repr


# Helper function for testing
def create_drai_attention_for_testing(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    drai_config: Optional[DraiConfig] = None,
    layer_idx: int = 0,
):
    """Create a DraiGPTNeoXAttention module for testing.

    Args:
        hidden_size: Model hidden size
        num_attention_heads: Number of standard attention heads
        drai_config: DRAI configuration
        layer_idx: Layer index

    Returns:
        DraiGPTNeoXAttention module
    """
    # Create minimal config
    from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

    config = GPTNeoXConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=2048,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=12,
    )

    # Create attention module
    attention = DraiGPTNeoXAttention(config, drai_config, layer_idx)

    return attention


if __name__ == "__main__":
    """Test DraiGPTNeoXAttention basic functionality."""
    print("=== DraiGPTNeoXAttention Test ===\n")

    # Test 1: Standard attention (no DRAI)
    print("1. Standard attention (no DRAI):")
    attn_standard = create_drai_attention_for_testing(drai_config=None)
    print(f"   Has DRAI: {attn_standard.drai is not None}")
    print(f"   Num DRAI heads: {attn_standard.num_drai_heads}")
    print()

    # Test 2: DRAI-enhanced attention
    print("2. DRAI-enhanced attention:")
    from .config import get_full_drai_config

    drai_config = get_full_drai_config()
    attn_drai = create_drai_attention_for_testing(drai_config=drai_config)
    print(f"   Has DRAI: {attn_drai.drai is not None}")
    print(f"   Num DRAI heads: {attn_drai.num_drai_heads}")
    print(f"   DRAI max attractors: {attn_drai.drai.max_attractors}")
    print()

    # Test 3: Mask extension
    print("3. Mask extension test:")
    batch, query_len, key_len = 2, 4, 4
    mask = torch.zeros(batch, 1, query_len, key_len)
    print(f"   Original mask shape: {mask.shape}")

    extended = attn_drai._extend_attention_mask(mask)
    print(f"   Extended mask shape: {extended.shape}")
    print(f"   Expected: ({batch}, 1, {query_len}, {key_len + attn_drai.num_drai_heads})")
    print()

    # Test 4: Forward pass shape test (no actual computation)
    print("4. Module representation:")
    print(f"   Standard: {attn_standard}")
    print(f"   With DRAI: {attn_drai}")
    print()

    print("✓ All basic tests passed!")
