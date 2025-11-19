# GPT-NeoX Architecture Analysis

**Purpose:** Deep dive into GPT-NeoX attention implementation to identify DRAI injection points

**Date:** 2025-11-18
**Transformers Version:** 4.57.1
**Model:** GPT-NeoX (HuggingFace implementation)

---

## Overview

GPT-NeoX uses a standard transformer architecture with Rotary Position Embeddings (RoPE) for position encoding. The attention mechanism is implemented in `GPTNeoXAttention` class.

**Key Files:**
- `/usr/local/lib/python3.11/dist-packages/transformers/models/gpt_neox/modeling_gpt_neox.py`
- Model class: `GPTNeoXAttention` (lines 129-194)

---

## GPTNeoXAttention Class Structure

### Initialization (lines 130-141)

```python
class GPTNeoXAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.head_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.scaling = self.head_size**-0.5
        self.is_causal = True
        self.layer_idx = layer_idx

        # Single linear layer produces Q, K, V
        self.query_key_value = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,  # Q + K + V concatenated
            bias=config.attention_bias
        )

        # Output projection
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias
        )
```

**Key Parameters:**
- `head_size`: Dimension per attention head
- `rotary_ndims`: How many dimensions get RoPE (typically 50-100% of head_size)
- `scaling`: 1/√(head_size) for attention scores
- `is_causal`: Enable causal masking
- `layer_idx`: Layer index (for cache management)

**Architecture Note:**
NeoX uses a **fused QKV projection** - single Linear layer produces all three. This is more efficient than separate Q, K, V projections.

---

## Forward Pass Flow

### Complete Attention Flow

```
Input: hidden_states [batch, seq_len, hidden_size]
│
├─→ 1. QKV Projection (line 157)
│   query_key_value(hidden_states) → [batch, seq_len, 3 * hidden_size]
│
├─→ 2. Reshape and Split (lines 157-158)
│   Reshape to [batch, num_heads, seq_len, 3 * head_size]
│   Split into query_states, key_states, value_states
│
├─→ 3. Apply RoPE (line 161)
│   query_states, key_states = apply_rotary_pos_emb(query, key, cos, sin)
│   Note: value_states NOT rotated
│
├─→ ★ DRAI INJECTION POINT ★
│   [After RoPE, before attention computation]
│   This is where we'll concatenate DRAI K/V
│
├─→ 4. Update Cache (lines 164-171)
│   If layer_past exists:
│       key_states, value_states = layer_past.update(...)
│   Cache now includes: past_kv + current_kv
│
├─→ 5. Compute Attention (lines 178-188)
│   attention_interface(query, key, value, mask, ...)
│   Returns: attn_output, attn_weights
│
├─→ 6. Reshape and Project (lines 191-192)
│   Reshape: [batch, num_heads, seq_len, head_size] → [batch, seq_len, hidden_size]
│   Project: dense(attn_output)
│
└─→ Output: attn_output, attn_weights
```

---

## Rotary Position Embeddings (RoPE)

### RoPE Application (lines 59-94)

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Split into rotated and non-rotated parts
    rotary_dim = cos.shape[-1]  # = self.rotary_ndims
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotation to first part
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed
```

**Key Points:**
1. Only Q and K are rotated (V is NOT rotated)
2. Only partial rotation: first `rotary_ndims` dimensions
3. Typical `rotary_pct` = 0.25 to 1.0 (often 0.25 for NeoX)
4. RoPE encodes relative position between tokens

**DRAI Implication:**
DRAI K/V are synthetic (no position) → Should NOT apply RoPE to them

---

## Attention Computation

### Eager Attention (lines 97-126)

```python
def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, head_mask=None, **kwargs
):
    # Q @ K^T with scaling
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    # Apply causal mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask  # Additive mask (0 for attend, -inf for mask)

    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Head masking (optional)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # Dropout (training only)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Weighted sum
    attn_output = torch.matmul(attn_weights, value)

    # Transpose back: [batch, heads, seq, head_dim] → [batch, seq, heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

**Tensor Shapes:**
- `query`: [batch, num_heads, query_seq_len, head_size]
- `key`: [batch, num_heads, kv_seq_len, head_size]
- `value`: [batch, num_heads, kv_seq_len, head_size]
- `attn_weights`: [batch, num_heads, query_seq_len, kv_seq_len]
- `attn_output`: [batch, query_seq_len, num_heads * head_size]

**DRAI Impact:**
When we concatenate DRAI K/V, `kv_seq_len` increases by `num_drai_heads`:
- Standard: `kv_seq_len = current_seq + past_seq`
- With DRAI: `kv_seq_len = current_seq + past_seq + num_drai_heads`

---

## K/V Cache System

### Cache Update (lines 164-171)

```python
if layer_past is not None:
    cache_kwargs = {
        "sin": sin,
        "cos": cos,
        "partial_rotation_size": self.rotary_ndims,
        "cache_position": cache_position,
    }
    key_states, value_states = layer_past.update(
        key_states, value_states, self.layer_idx, cache_kwargs
    )
```

**Cache Behavior:**
1. **First forward pass (prompt):** `layer_past=None`
   - Compute K/V for all prompt tokens
   - Return cache with all K/V
2. **Subsequent forward passes (generation):** `layer_past` contains previous K/V
   - Compute K/V for only new token
   - Concatenate: `[past_kv, new_kv]` along sequence dimension
   - Return updated cache

**Cache Type:** `Cache` or `DynamicCache` (from `transformers.cache_utils`)
- `DynamicCache`: List of (key, value) tuples, one per layer
- Automatically manages concatenation

**DRAI Integration Challenge:**
DRAI generates K/V that must be included in the cache:
```python
# Standard cache update
key_states, value_states = layer_past.update(key_states, value_states, ...)

# With DRAI - need to concatenate DRAI K/V BEFORE caching
k_reson, v_reson = self.drai(query_states, ...)
key_states = torch.cat([key_states, k_reson], dim=2)  # Concat along seq dim
value_states = torch.cat([value_states, v_reson], dim=2)

# Now cache includes DRAI K/V
# Next generation step will have DRAI in layer_past
```

**Key Insight:**
Cache happens AFTER RoPE but BEFORE attention. This is perfect for DRAI injection.

---

## Attention Mask Structure

### Causal Mask

NeoX uses **additive masking**:
- Attend: 0
- Mask: -∞ (or very large negative number like -10000)

```python
# Mask shape: [batch, 1, query_seq_len, kv_seq_len]
# Applied as: attn_weights = attn_weights + mask

# Example causal mask (seq_len=4):
mask = [
    [0,   -inf, -inf, -inf],  # Token 0 attends only to itself
    [0,   0,    -inf, -inf],  # Token 1 attends to 0,1
    [0,   0,    0,    -inf],  # Token 2 attends to 0,1,2
    [0,   0,    0,    0   ],  # Token 3 attends to 0,1,2,3
]
```

**With DRAI:**
Need to extend mask to allow attending to DRAI heads:

```python
# Original mask: [batch, 1, query_seq_len, seq_len]
# Extended mask: [batch, 1, query_seq_len, seq_len + num_drai_heads]

# DRAI extension: all zeros (all tokens can attend to DRAI)
drai_mask = torch.zeros(batch, 1, query_seq_len, num_drai_heads, device=device)
extended_mask = torch.cat([mask, drai_mask], dim=-1)

# Example with 1 DRAI head (seq_len=4):
extended_mask = [
    [0,   -inf, -inf, -inf, 0],  # Token 0 → [0, DRAI]
    [0,   0,    -inf, -inf, 0],  # Token 1 → [0,1, DRAI]
    [0,   0,    0,    -inf, 0],  # Token 2 → [0,1,2, DRAI]
    [0,   0,    0,    0,    0],  # Token 3 → [0,1,2,3, DRAI]
]
                        ^^^^
                        DRAI head (always attend)
```

---

## Tensor Shape Reference

### Standard NeoX Shapes

**Initialization:**
- `hidden_states`: [batch, seq_len, hidden_size]
- `hidden_size`: Typically 2048, 4096, 6144, 8192 depending on model size
- `num_attention_heads`: Typically 16, 32, 40, 64
- `head_size`: hidden_size / num_attention_heads

**Example: pythia-125m**
- `hidden_size`: 768
- `num_attention_heads`: 12
- `head_size`: 64
- `rotary_pct`: 0.25
- `rotary_ndims`: 16 (25% of 64)

**After QKV Projection:**
- `qkv`: [batch, num_heads, seq_len, 3 * head_size]
- Split into 3 chunks of `head_size` each

**After Split:**
- `query_states`: [batch, num_heads, seq_len, head_size]
- `key_states`: [batch, num_heads, seq_len, head_size]
- `value_states`: [batch, num_heads, seq_len, head_size]

**During Generation (with cache):**
- Current input: [batch, 1, hidden_size] (single token)
- Current K/V: [batch, num_heads, 1, head_size]
- Past K/V: [batch, num_heads, past_seq_len, head_size]
- Combined K/V: [batch, num_heads, past_seq_len + 1, head_size]

**With DRAI (num_drai_heads=1):**
- DRAI K/V: [batch, num_heads, 1, head_size] (treated as extra "sequence" position)
- Combined K/V: [batch, num_heads, past_seq_len + 1 + 1, head_size]
  - past_seq_len: Previous tokens
  - +1: Current token
  - +1: DRAI head

---

## DRAI Injection Strategy

### Injection Point Identified

**Location:** Between RoPE application and attention computation (after line 161, before line 173)

```python
# Line 161: RoPE applied
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# ★ INJECT DRAI HERE ★
# Generate DRAI K/V from query_states (already rotated)
if self.drai is not None:
    k_reson, v_reson = self.drai(
        query_states,
        attention_mask,  # May need to pass for attractor formation
    )
    # Concatenate DRAI K/V along sequence dimension
    key_states = torch.cat([key_states, k_reson], dim=2)
    value_states = torch.cat([value_states, v_reson], dim=2)

# Line 164-171: Cache update (will now include DRAI K/V)
if layer_past is not None:
    key_states, value_states = layer_past.update(key_states, value_states, ...)

# Line 173-188: Attention computation (will attend to DRAI K/V)
attn_output, attn_weights = attention_interface(...)
```

### Why This Location?

**Advantages:**
1. ✓ RoPE already applied to standard K/V (don't apply to DRAI)
2. ✓ Before cache update (DRAI included in cache automatically)
3. ✓ Before attention (DRAI participates in attention naturally)
4. ✓ Clean integration point (single concatenation operation)
5. ✓ No modification to cache logic
6. ✓ No modification to attention computation

**Alternatives Considered:**
- **Before RoPE:** Would need to skip RoPE for DRAI K/V (more complex)
- **After cache:** Would need separate DRAI cache management (much more complex)
- **After attention:** Too late, can't participate in attention
- **Separate attention:** Requires merging two attention outputs (complex)

---

## Implementation Plan

### Step 1: Create DraiGPTNeoXAttention

Subclass `GPTNeoXAttention`:

```python
class DraiGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config, drai_config=None, layer_idx=None):
        super().__init__(config, layer_idx)

        # Add DRAI if configured
        if drai_config and drai_config.enabled:
            self.drai = DraiResonanceLayer(
                hidden_size=config.hidden_size,
                num_heads=drai_config.num_drai_heads,
                head_dim=self.head_size,
                phase=drai_config.phase,
                **drai_config.hyperparameters.to_dict()
            )
        else:
            self.drai = None

    def forward(self, hidden_states, attention_mask, ...):
        # ... standard QKV projection and RoPE ...

        # DRAI injection
        if self.drai is not None:
            k_reson, v_reson = self.drai(query_states, attention_mask)
            key_states = torch.cat([key_states, k_reson], dim=2)
            value_states = torch.cat([value_states, v_reson], dim=2)

            # Extend attention mask
            attention_mask = extend_attention_mask(attention_mask, self.drai.num_heads)

        # ... continue with standard attention ...
```

### Step 2: Replace Attention Modules

Function to replace standard attention with DRAI attention:

```python
def inject_drai_into_model(model, drai_config):
    """Replace GPTNeoXAttention with DraiGPTNeoXAttention in all layers."""
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        if drai_config.should_inject_layer(layer_idx):
            # Get original attention config
            original_attn = layer.attention

            # Create DRAI attention
            drai_attn = DraiGPTNeoXAttention(
                config=model.config,
                drai_config=drai_config,
                layer_idx=layer_idx
            )

            # Copy weights from original
            drai_attn.query_key_value.weight.data = original_attn.query_key_value.weight.data
            drai_attn.query_key_value.bias.data = original_attn.query_key_value.bias.data
            drai_attn.dense.weight.data = original_attn.dense.weight.data
            drai_attn.dense.bias.data = original_attn.dense.bias.data

            # Replace
            layer.attention = drai_attn

    return model
```

### Step 3: Extend Attention Mask

Helper function:

```python
def extend_attention_mask(mask, num_drai_heads):
    """Extend causal mask to include DRAI heads."""
    if mask is None:
        return None

    batch, _, query_len, key_len = mask.shape

    # DRAI extension: zeros (all can attend)
    drai_extension = torch.zeros(
        batch, 1, query_len, num_drai_heads,
        dtype=mask.dtype,
        device=mask.device
    )

    # Concatenate along key dimension
    extended_mask = torch.cat([mask, drai_extension], dim=-1)

    return extended_mask
```

---

## Configuration Requirements

### GPTNeoX Config Values

Relevant config parameters we need:
```python
config.hidden_size           # 768 for pythia-125m
config.num_attention_heads   # 12 for pythia-125m
config.attention_bias        # True/False
config.rotary_pct            # 0.25 typical
config.attention_dropout     # Dropout probability
config._attn_implementation  # "eager", "flash_attention_2", "sdpa"
```

### DRAI Config

Our DRAI config needs:
```python
@dataclass
class DraiConfig:
    enabled: bool = True
    phase: int = 2
    num_drai_heads: int = 1
    layer_mode: str = "all"  # or "selective"
    selective_layers: Optional[List[int]] = None
    hyperparameters: DraiHyperparameters = ...
```

---

## Testing Checklist

### Unit Tests

- [ ] `test_drai_attention_initialization` - DraiGPTNeoXAttention initializes correctly
- [ ] `test_drai_attention_forward_shape` - Output shapes match standard attention
- [ ] `test_mask_extension` - Attention mask extends correctly
- [ ] `test_inject_drai_into_model` - Model injection works
- [ ] `test_weight_copying` - Pre-trained weights copy correctly

### Integration Tests

- [ ] `test_load_pythia_with_drai` - Load pythia-125m with DRAI
- [ ] `test_forward_pass_no_crash` - Forward pass completes
- [ ] `test_generation_with_cache` - Generation with K/V cache works
- [ ] `test_no_nan_inf` - No numerical issues
- [ ] `test_attractors_form` - Attractors actually form

### Regression Tests

- [ ] `test_model_without_drai_unchanged` - Baseline unchanged
- [ ] `test_phase1_matches_baseline` - Phase 1 (zeros) matches baseline

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Integration Method** | Subclassing | Clean, maintainable, Pythonic |
| **Injection Point** | After RoPE, before attention | Clean, minimal changes |
| **RoPE on DRAI K/V** | No | DRAI K/V are position-agnostic |
| **DRAI Input** | Query states (after RoPE) | Consistent with what attention sees |
| **K/V Concatenation** | Along sequence dimension | Treats DRAI as extra positions |
| **Mask Extension** | Zeros for DRAI columns | All tokens can attend to DRAI |
| **Cache Strategy** | Include DRAI in standard cache | Automatic, no extra logic |

---

## Next Steps

1. ✓ Architecture analysis complete (this document)
2. → Implement `DraiGPTNeoXAttention` class
3. → Implement `inject_drai_into_model` function
4. → Create unit tests
5. → Test with pythia-125m
6. → Document results

---

**Document Status:** Complete
**Last Updated:** 2025-11-18
**Next Document:** `NEOX_INTEGRATION_IMPLEMENTATION.md` (implementation details)
