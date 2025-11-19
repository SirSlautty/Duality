# DRAI Integration Methodology

**Document Purpose:** Comprehensive record of the methodology used to integrate DRAI into transformer architectures, with emphasis on reproducibility and justification of all decisions.

**Target Audience:** Researchers and engineers who want to understand, reproduce, or adapt our integration approach.

**Status:** 🚧 Living Document - Updated in real-time during Phase 3

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Principles](#core-principles)
3. [Architecture Selection](#architecture-selection)
4. [Integration Strategy](#integration-strategy)
5. [Technical Implementation](#technical-implementation)
6. [Configuration Design](#configuration-design)
7. [Testing Methodology](#testing-methodology)
8. [Measurement Protocol](#measurement-protocol)
9. [Validation Criteria](#validation-criteria)
10. [Lessons Learned](#lessons-learned)
11. [Reproducibility Guide](#reproducibility-guide)

---

## Executive Summary

This document describes the complete methodology for integrating DRAI (Dynamic Resonance AI) attractor fields into transformer models. We detail every technical decision, alternative considered, implementation challenge, and solution developed.

**Key Integration Challenges:**
1. Injecting DRAI without breaking pre-trained models
2. Maintaining K/V cache compatibility for generation
3. Handling position encodings (RoPE) for synthetic K/V
4. Extending attention masks to cover DRAI heads
5. Preserving gradient flow for potential future training

**Integration Approach:**
- **Method:** Subclass existing attention modules
- **Scope:** All layers (with configuration option for selective)
- **Mode:** Inference-first (training in Phase 5)
- **Philosophy:** Minimal invasiveness, maximum compatibility

---

## Core Principles

### 1. Minimal Invasiveness
**Principle:** Modify as little existing code as possible.

**Rationale:**
- Reduces risk of breaking existing functionality
- Makes integration easier to understand
- Simplifies debugging and maintenance
- Facilitates porting to other architectures

**Implementation:**
- Subclass rather than modify existing classes
- Use composition where possible
- Preserve original behavior when DRAI disabled
- Keep DRAI code in separate modules

### 2. Backward Compatibility
**Principle:** Model must work with DRAI disabled or phase=1.

**Rationale:**
- Allows easy baseline comparisons
- Provides fallback if issues arise
- Enables gradual rollout per layer
- Maintains pre-trained model compatibility

**Implementation:**
- Configuration flag to enable/disable DRAI
- Phase 1 mode returns zeros (no behavior change)
- Default configuration preserves original behavior
- Easy toggle for A/B testing

### 3. Configuration Driven
**Principle:** All DRAI behavior controlled via configuration.

**Rationale:**
- Experiment with hyperparameters without code changes
- Easy to document different configurations
- Reproducible experiments via config files
- Enables automated hyperparameter search

**Implementation:**
- Comprehensive configuration schema
- Validation of configuration values
- Defaults that work well out-of-box
- Clear documentation of all options

### 4. Layer Selectivity
**Principle:** Choose which layers receive DRAI injection.

**Rationale:**
- Not all layers may benefit equally from DRAI
- Allows targeted experiments (e.g., only deeper layers)
- Reduces computational cost for testing
- Enables layer-wise analysis of attractor behavior

**Implementation:**
- Configuration modes: "all", "selective", "none"
- List of specific layer indices for selective mode
- Per-layer DRAI configuration support
- Runtime inspection of DRAI status per layer

### 5. Observable Behavior
**Principle:** DRAI behavior must be measurable and visible.

**Rationale:**
- Need to verify attractors are forming
- Essential for debugging integration issues
- Enables scientific analysis of behavior
- Provides evidence of correctness

**Implementation:**
- Logging hooks for attractor statistics
- Export functions for attractor states
- Visualization utilities
- Per-layer metrics collection

### 6. Reproducibility
**Principle:** All experiments must be exactly reproducible.

**Rationale:**
- Scientific rigor requires reproducibility
- Others must be able to verify our claims
- Debugging requires deterministic behavior
- Fair comparison requires controlled conditions

**Implementation:**
- Fixed random seeds
- Deterministic forward pass (no randomness)
- Version pinning of all dependencies
- Complete configuration logging
- Data and code versioning

---

## Architecture Selection

### Target: GPT-NeoX

**Why GPT-NeoX?**

1. **Open Source:** Fully open source with permissive license
2. **Well Documented:** Clear code structure and documentation
3. **Active Development:** Maintained by EleutherAI
4. **Pre-trained Models:** Many sizes available (125M to 20B)
5. **Standard Architecture:** Representative of modern transformers
6. **Research Focus:** Designed for research and experimentation

**Alternatives Considered:**

| Architecture | Pros | Cons | Decision |
|--------------|------|------|----------|
| GPT-NeoX | Open, well-documented, many sizes | Large codebase | ✓ Selected |
| nanoGPT | Simple, minimal code | Limited features, small models only | Future target |
| Llama | Popular, performant | Licensing constraints, complex | Future target |
| GPT-2 (HF) | Simple, well-known | Old architecture, limited sizes | Future target |
| BLOOM | Multilingual, large | Very large, complex | Not needed yet |

**Model Size Selection:**

Starting with **GPT-NeoX-125M** (pythia-125m):
- **Smallest available** (easiest to debug)
- **Fast inference** (quick iteration)
- **Low memory** (runs on CPU/small GPU)
- **Representative** (same architecture as larger models)
- **Pre-trained** (coherent outputs expected)

**Future Scaling:**
- 410M (next size up)
- 1.3B (medium size)
- 6.7B (large, if resources permit)

---

## Integration Strategy

### Approach: Subclassing

**Selected Method:** Subclass `GPTNeoXAttention` to create `DraiGPTNeoXAttention`

```python
class DraiGPTNeoXAttention(GPTNeoXAttention):
    """GPT-NeoX attention with DRAI injection.

    Extends standard NeoX attention by:
    1. Adding DRAI resonance layer
    2. Injecting synthetic K/V from attractors
    3. Concatenating DRAI K/V with standard attention K/V
    4. Preserving all original functionality
    """

    def __init__(self, config, drai_config=None):
        super().__init__(config)

        if drai_config and drai_config.enabled:
            self.drai = DraiResonanceLayer(
                hidden_size=config.hidden_size,
                num_heads=drai_config.num_drai_heads,
                phase=drai_config.phase,
                **drai_config.hyperparameters
            )
        else:
            self.drai = None

    def forward(self, hidden_states, attention_mask, ...):
        # 1. Standard attention computation
        query, key, value = self._compute_qkv(hidden_states)

        # 2. DRAI injection (if enabled)
        if self.drai is not None:
            k_reson, v_reson = self.drai(query, attention_mask)
            key = torch.cat([key, k_reson], dim=2)  # Concat along seq dim
            value = torch.cat([value, v_reson], dim=2)

        # 3. Continue with standard attention
        attn_output = self._apply_attention(query, key, value, attention_mask)
        return attn_output
```

**Why Subclassing?**

✓ **Clean separation** - DRAI code isolated from NeoX code
✓ **Inheritance** - Reuse all NeoX attention logic
✓ **Override only what's needed** - Minimal code duplication
✓ **Standard pattern** - Familiar to PyTorch developers
✓ **Easy to test** - Can test with/without DRAI easily
✓ **Maintainable** - NeoX updates don't break DRAI

**Rejected Alternatives:**

**Monkey Patching:**
```python
# Too fragile, hard to maintain
original_forward = GPTNeoXAttention.forward
GPTNeoXAttention.forward = lambda self, *args: drai_forward(self, *args)
```

**Direct Modification:**
```python
# Requires forking NeoX, hard to update
# Changes scattered throughout codebase
```

**Wrapper Module:**
```python
# Extra indirection, more complex
# Harder to integrate with model loading
```

### Integration Points

**Where to inject DRAI?**

```
┌─────────────────────────────────────────┐
│         GPTNeoXAttention Layer          │
├─────────────────────────────────────────┤
│                                         │
│  1. Input: hidden_states                │
│     ↓                                   │
│  2. Compute Q, K, V projections         │
│     ↓                                   │
│  3. Apply rotary position encoding      │  ← RoPE applied here
│     (only to Q and standard K)          │
│     ↓                                   │
│  ★  DRAI INJECTION POINT ★              │  ← Inject DRAI K/V here
│     - Generate K_reson, V_reson         │
│     - Concatenate with K, V             │
│     ↓                                   │
│  4. Compute attention scores            │  ← Extended K includes DRAI
│     scores = Q @ K^T / √d               │
│     ↓                                   │
│  5. Apply attention mask                │  ← Extend mask for DRAI heads
│     ↓                                   │
│  6. Apply softmax                       │
│     ↓                                   │
│  7. Weighted sum: attn @ V              │  ← Extended V includes DRAI
│     ↓                                   │
│  8. Output projection                   │
│     ↓                                   │
│  9. Output: attention output            │
│                                         │
└─────────────────────────────────────────┘
```

**Critical Decision: Inject AFTER RoPE**

**Rationale:**
- RoPE encodes position information
- DRAI K/V represent patterns, not specific positions
- Applying RoPE to DRAI K/V would incorrectly position synthetic patterns
- DRAI heads should be "position-agnostic" attractors

**Alternative Considered:**
- Apply RoPE to DRAI K/V → Rejected (semantically incorrect)
- Use different position encoding for DRAI → Rejected (too complex for Phase 3)

---

## Technical Implementation

### Challenge 1: K/V Cache Compatibility

**Problem:**
Transformers use K/V caching during generation to avoid recomputing past tokens. DRAI generates synthetic K/V that must be cached alongside standard K/V.

**Standard NeoX Generation:**
```python
# Step 1: Process prompt (all tokens)
past_kv = None
output = model(input_ids, past_key_values=past_kv)
past_kv = output.past_key_values  # Cache K/V for all layers

# Step 2+: Generate tokens one by one
for _ in range(max_new_tokens):
    output = model(next_token, past_key_values=past_kv)
    past_kv = output.past_key_values  # Update cache
    next_token = sample(output.logits)
```

**With DRAI:**
```python
# K/V cache must include DRAI heads
# Standard: [batch, num_heads, seq_len, head_dim]
# With DRAI: [batch, num_heads, seq_len + num_drai_heads, head_dim]
#                                       ^^^^^^^^^^^^^^^^^^^
#                                       Extended sequence length

# OR (depending on NeoX implementation):
# Standard: [batch, num_heads, seq_len, head_dim]
# With DRAI: [batch, num_heads + num_drai_heads, seq_len, head_dim]
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                    Extended head count
```

**Solution:**
Concatenate DRAI K/V along the sequence dimension, treating DRAI heads as "extra positions":

```python
def forward(self, hidden_states, past_key_value=None, ...):
    # Compute standard K/V
    key, value = self.compute_kv(hidden_states)

    # Use past K/V if available
    if past_key_value is not None:
        past_key, past_value = past_key_value
        # past_key includes previous DRAI K/V
        key = torch.cat([past_key, key], dim=2)  # Concat along seq
        value = torch.cat([past_value, value], dim=2)

    # Generate DRAI K/V (every step, not just first)
    if self.drai is not None:
        query = self.compute_query(hidden_states)
        k_reson, v_reson = self.drai(query)

        # Concatenate DRAI K/V
        # Important: DRAI K/V come AFTER standard K/V
        key = torch.cat([key, k_reson], dim=2)
        value = torch.cat([value, v_reson], dim=2)

    # Return new cache (includes current + DRAI K/V)
    present_key_value = (key, value)

    # Compute attention with extended K/V
    attn_output = self.compute_attention(query, key, value, ...)

    return attn_output, present_key_value
```

**Key Decisions:**
1. **Regenerate DRAI K/V every step** - Attractors evolve during generation
2. **Append to cache** - DRAI K/V included in past_key_value
3. **Position in cache: end** - Standard K/V first, then DRAI K/V

**Testing:**
- Verify generation doesn't crash
- Verify cache shapes are correct
- Verify generation is coherent
- Verify attractors evolve during generation

---

### Challenge 2: Attention Mask Extension

**Problem:**
Attention masks prevent tokens from attending to future positions (causal masking). When we add DRAI heads, the mask must extend to allow attending to them.

**Standard Causal Mask:**
```python
# Shape: [batch, 1, seq_len, seq_len]
# Upper triangular = True (attend), Lower = False (mask)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
mask = mask.unsqueeze(0).unsqueeze(0)

# Example for seq_len=4:
[[True,  False, False, False],   # Token 0 attends to 0
 [True,  True,  False, False],   # Token 1 attends to 0,1
 [True,  True,  True,  False],   # Token 2 attends to 0,1,2
 [True,  True,  True,  True ]]   # Token 3 attends to 0,1,2,3
```

**With DRAI (num_drai_heads=1):**
```python
# DRAI head is like an extra position that all tokens can attend to
# Shape: [batch, 1, seq_len, seq_len + 1]

mask_extended = torch.cat([
    mask,  # Standard causal mask
    torch.ones(batch, 1, seq_len, num_drai_heads)  # All attend to DRAI
], dim=-1)

# Example for seq_len=4, drai_heads=1:
[[True,  False, False, False, True ],   # Token 0 → [0, DRAI]
 [True,  True,  False, False, True ],   # Token 1 → [0,1, DRAI]
 [True,  True,  True,  False, True ],   # Token 2 → [0,1,2, DRAI]
 [True,  True,  True,  True,  True ]]   # Token 3 → [0,1,2,3, DRAI]
                                   ^^^^
                                   DRAI head (always attendable)
```

**Rationale:**
- DRAI heads represent accumulated patterns, not future tokens
- All tokens should be able to attend to attractors
- DRAI heads are "outside" the causal sequence

**Implementation:**
```python
def extend_attention_mask(mask, num_drai_heads):
    """Extend attention mask to cover DRAI heads."""
    batch, _, seq_len, _ = mask.shape

    # DRAI extension: all ones (all tokens can attend to DRAI)
    drai_mask = torch.ones(
        batch, 1, seq_len, num_drai_heads,
        dtype=mask.dtype,
        device=mask.device
    )

    # Concatenate along key dimension
    extended_mask = torch.cat([mask, drai_mask], dim=-1)

    return extended_mask
```

**Alternative Considered:**
- Treat DRAI as separate attention mechanism with its own mask
- Rejected: More complex, harder to integrate with standard attention

---

### Challenge 3: Rotary Position Embeddings (RoPE)

**Problem:**
NeoX uses RoPE for position encoding. RoPE rotates Q and K vectors based on position. DRAI generates synthetic K/V - what position should they have?

**RoPE Mechanism:**
```python
# Apply rotary position encoding
def apply_rope(x, position_ids):
    # x: [batch, heads, seq_len, head_dim]
    # Rotate based on position in sequence
    return rotate(x, position_ids)

# In standard attention:
query = apply_rope(query, position_ids)
key = apply_rope(key, position_ids)
# Value is NOT rotated (RoPE only on Q and K)
```

**Decision: Do NOT apply RoPE to DRAI K/V**

**Rationale:**
1. **Semantic Correctness:**
   - RoPE encodes "where" a token is in the sequence
   - DRAI K/V encode "what" patterns have been seen
   - Attractors are position-agnostic concepts
   - Applying RoPE would incorrectly position these concepts

2. **Technical Correctness:**
   - DRAI K/V are synthetic (not from input sequence)
   - They don't have a "true" position
   - What position would we assign? Last? Average? Arbitrary?

3. **Empirical Validation:**
   - Without RoPE, DRAI acts as "global" memory
   - This is the desired behavior for attractor fields

**Implementation:**
```python
def forward(self, hidden_states, position_ids, ...):
    # Compute Q, K, V
    query, key, value = self.compute_qkv(hidden_states)

    # Apply RoPE ONLY to standard Q and K
    query = self.apply_rotary_pos_emb(query, position_ids)
    key = self.apply_rotary_pos_emb(key, position_ids)
    # Note: value is not rotated (standard RoPE behavior)

    # Generate DRAI K/V (NO RoPE)
    if self.drai is not None:
        k_reson, v_reson = self.drai(query)  # query already rotated
        # DO NOT apply RoPE to k_reson or v_reson
        key = torch.cat([key, k_reson], dim=2)
        value = torch.cat([value, v_reson], dim=2)

    # Continue with attention...
```

**Alternative Considered:**
- Use position_ids[-1] for DRAI K/V (position of last token)
- Rejected: Arbitrary choice, no semantic meaning

---

### Challenge 4: Query Input to DRAI

**Problem:**
DRAI needs query input to extract patterns. Should we pass queries before or after RoPE?

**Option A: Pass queries AFTER RoPE (selected)**
```python
query = self.apply_rotary_pos_emb(query, position_ids)  # RoPE first
k_reson, v_reson = self.drai(query)  # DRAI sees rotated queries
```

**Option B: Pass queries BEFORE RoPE**
```python
k_reson, v_reson = self.drai(query)  # DRAI sees original queries
query = self.apply_rotary_pos_emb(query, position_ids)  # RoPE after
```

**Decision: Option A (after RoPE)**

**Rationale:**
1. **Consistency:** DRAI sees the same queries that attention sees
2. **Information:** Rotated queries contain position information
3. **Patterns:** DRAI can detect position-dependent patterns
4. **Simplicity:** Easier to reason about and debug

**Trade-off:**
- DRAI patterns include position information
- May make attractors more position-specific
- Could reduce generalization across positions

**Future Consideration:**
- Experiment with Option B in Phase 4
- May add configuration flag for this choice
- Could compare attractor behavior between options

---

### Challenge 5: Gradient Flow

**Problem:**
DRAI uses EMA updates (gradient-free) in Phase 2, but we want to preserve gradient flow for future Phase 5 training.

**Solution:**
All DRAI operations are differentiable:

```python
# Differentiable operations:
pattern = query.mean(dim=[0,1,2])  # ✓ Differentiable
pattern = F.normalize(pattern)     # ✓ Differentiable
similarity = F.cosine_similarity() # ✓ Differentiable
k_reson = attractor_centroids[...] # ✓ Differentiable (indexing)
v_reson = coherence * centroids    # ✓ Differentiable

# Gradient flow:
loss.backward()  # Gradients flow through:
# 1. Output
# 2. Attention output
# 3. V_reson (weighted attractor)
# 4. attractor_centroids (via indexing)
# 5. Query (via mean pooling)
# 6. Model parameters
```

**Non-differentiable operations** (EMA updates):
```python
# These are in-place buffer updates (no gradient needed):
self.attractor_centroids[idx] = new_centroid  # Buffer update
self.attractor_coherence[idx] = new_coherence  # Buffer update

# These happen in forward pass but don't affect gradients
# Gradients flow through the read operations, not the writes
```

**Testing:**
```python
def test_gradients_flow_through_drai():
    model = build_drai_neox_model()
    input_ids = torch.randint(0, 50000, (1, 10))

    output = model(input_ids)
    loss = output.logits.sum()  # Dummy loss
    loss.backward()

    # Verify gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
```

---

## Configuration Design

### Configuration Schema

```python
from dataclasses import dataclass, field
from typing import Optional, List, Literal

@dataclass
class DraiHyperparameters:
    """DRAI attractor dynamics hyperparameters."""
    max_attractors: int = 32
    coherence_threshold: float = 0.3
    formation_threshold: float = 0.5
    decay_rate: float = 0.01
    ema_momentum: float = 0.9

@dataclass
class DraiConfig:
    """Complete DRAI configuration."""

    # Core settings
    enabled: bool = False  # Enable DRAI?
    phase: int = 2  # 1=zeros, 2=attractors

    # Layer selection
    layer_mode: Literal["all", "selective", "none"] = "all"
    selective_layers: Optional[List[int]] = None  # Used if mode="selective"

    # DRAI architecture
    num_drai_heads: int = 1  # DRAI heads per layer
    drai_head_dim: Optional[int] = None  # None = use model's head_dim

    # Hyperparameters
    hyperparameters: DraiHyperparameters = field(default_factory=DraiHyperparameters)

    # Monitoring
    verbose_logging: bool = False
    collect_statistics: bool = True

    def should_inject_layer(self, layer_idx: int) -> bool:
        """Determine if DRAI should be injected into a specific layer."""
        if not self.enabled:
            return False
        if self.layer_mode == "none":
            return False
        if self.layer_mode == "all":
            return True
        if self.layer_mode == "selective":
            return layer_idx in (self.selective_layers or [])
        return False
```

### Configuration Examples

**Example 1: Full DRAI (all layers)**
```python
config = DraiConfig(
    enabled=True,
    phase=2,
    layer_mode="all",
    num_drai_heads=1,
)
```

**Example 2: Selective layers (deeper layers only)**
```python
config = DraiConfig(
    enabled=True,
    phase=2,
    layer_mode="selective",
    selective_layers=[6, 7, 8, 9, 10, 11],  # Last 6 layers (for 12-layer model)
    num_drai_heads=1,
)
```

**Example 3: Aggressive attractors (fast formation, slow decay)**
```python
config = DraiConfig(
    enabled=True,
    phase=2,
    layer_mode="all",
    hyperparameters=DraiHyperparameters(
        max_attractors=64,  # More capacity
        coherence_threshold=0.2,  # Easier matching
        formation_threshold=0.3,  # Easier formation
        decay_rate=0.005,  # Slower decay (longer memory)
        ema_momentum=0.95,  # More stable (less adaptation)
    )
)
```

**Example 4: Conservative attractors (slow formation, fast decay)**
```python
config = DraiConfig(
    enabled=True,
    phase=2,
    layer_mode="all",
    hyperparameters=DraiHyperparameters(
        max_attractors=16,  # Less capacity
        coherence_threshold=0.5,  # Stricter matching
        formation_threshold=0.7,  # Harder formation
        decay_rate=0.02,  # Faster decay (shorter memory)
        ema_momentum=0.8,  # Less stable (more adaptation)
    )
)
```

---

## Testing Methodology

### Test Pyramid

```
                    ┌─────────────────┐
                    │   Functional    │  ← Does it generate good text?
                    │     Tests       │
                    └─────────────────┘
                   ┌───────────────────┐
                   │   Integration     │  ← Does it work with NeoX?
                   │      Tests        │
                   └───────────────────┘
                ┌──────────────────────────┐
                │      Unit Tests          │  ← Do components work?
                └──────────────────────────┘
```

### Unit Tests
Test individual components in isolation.

**Test: DRAI Configuration**
```python
def test_drai_config_validation():
    """Test configuration validation."""
    # Valid config
    config = DraiConfig(enabled=True, phase=2)
    assert config.enabled
    assert config.phase == 2

    # Invalid phase
    with pytest.raises(ValueError):
        DraiConfig(phase=3)  # Only 1 and 2 supported

    # Selective without layer list
    with pytest.raises(ValueError):
        DraiConfig(layer_mode="selective", selective_layers=None)
```

**Test: Attention Subclass**
```python
def test_drai_attention_initialization():
    """Test DraiGPTNeoXAttention initializes correctly."""
    config = GPTNeoXConfig(hidden_size=768, num_attention_heads=12)
    drai_config = DraiConfig(enabled=True, phase=2)

    layer = DraiGPTNeoXAttention(config, drai_config)

    assert hasattr(layer, 'drai')
    assert layer.drai is not None
    assert layer.drai.phase == 2
```

**Test: Mask Extension**
```python
def test_attention_mask_extension():
    """Test mask extends correctly for DRAI heads."""
    mask = create_causal_mask(seq_len=4)  # [1, 1, 4, 4]
    extended = extend_attention_mask(mask, num_drai_heads=2)

    assert extended.shape == (1, 1, 4, 6)  # 4 + 2
    assert extended[..., -2:].all()  # DRAI columns all True
```

### Integration Tests
Test DRAI with real NeoX models.

**Test: Model Loading**
```python
def test_load_pretrained_with_drai():
    """Test loading pre-trained NeoX model with DRAI injection."""
    model = build_drai_neox_model(
        model_name="EleutherAI/pythia-125m",
        drai_config=DraiConfig(enabled=True, phase=2)
    )

    # Model should load without errors
    assert model is not None

    # DRAI should be injected
    first_attention = model.gpt_neox.layers[0].attention
    assert isinstance(first_attention, DraiGPTNeoXAttention)
    assert first_attention.drai is not None
```

**Test: Forward Pass**
```python
def test_forward_pass_with_drai():
    """Test forward pass completes without errors."""
    model = build_drai_neox_model(
        model_name="EleutherAI/pythia-125m",
        drai_config=DraiConfig(enabled=True, phase=2)
    )

    input_ids = torch.randint(0, 50000, (2, 10))  # [batch=2, seq=10]

    output = model(input_ids)

    assert output.logits.shape == (2, 10, 50304)  # [batch, seq, vocab]
    assert not torch.isnan(output.logits).any()
    assert not torch.isinf(output.logits).any()
```

**Test: K/V Cache**
```python
def test_generation_with_kv_cache():
    """Test generation with K/V caching."""
    model = build_drai_neox_model(
        model_name="EleutherAI/pythia-125m",
        drai_config=DraiConfig(enabled=True, phase=2)
    )
    tokenizer = load_tokenizer("EleutherAI/pythia-125m")

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate tokens
    output_ids = model.generate(
        input_ids,
        max_new_tokens=20,
        use_cache=True  # Enable K/V caching
    )

    # Should complete without errors
    assert output_ids.shape[1] > input_ids.shape[1]

    # Decode and check it's not garbage
    text = tokenizer.decode(output_ids[0])
    assert len(text) > len(prompt)
```

### Functional Tests
Test model behavior and output quality.

**Test: Coherent Generation**
```python
def test_model_generates_coherent_text():
    """Test model generates coherent text with DRAI."""
    model, tokenizer = load_model_and_tokenizer(
        "EleutherAI/pythia-125m",
        drai_config=DraiConfig(enabled=True, phase=2)
    )

    prompt = "The capital of France is"
    output = generate_text(model, tokenizer, prompt, max_length=10)

    # Basic coherence checks
    assert "paris" in output.lower() or "Paris" in output
    assert len(output.split()) >= 5  # At least a few words

    # No repeated tokens (common failure mode)
    tokens = output.split()
    repetition_ratio = len(tokens) / len(set(tokens))
    assert repetition_ratio < 2.0  # Some repetition OK, but not excessive
```

**Test: Attractor Formation**
```python
def test_attractors_form_during_generation():
    """Test that attractors actually form during text generation."""
    model, tokenizer = load_model_and_tokenizer(
        "EleutherAI/pythia-125m",
        drai_config=DraiConfig(enabled=True, phase=2)
    )

    # Long prompt to allow attractor formation
    prompt = "The quick brown fox jumps over the lazy dog. " * 5
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Forward pass
    _ = model(input_ids)

    # Check attractors formed
    first_layer_drai = model.gpt_neox.layers[0].attention.drai
    attractor_count = first_layer_drai.attractor_count.item()

    assert attractor_count > 0, "No attractors formed"
    assert attractor_count <= first_layer_drai.max_attractors
```

---

## Measurement Protocol

### Metric 1: Perplexity

**Definition:** Perplexity measures how well the model predicts the next token.

```
PPL = exp(average_cross_entropy_loss)
```

Lower is better. Perfect prediction = 1.0.

**Measurement Procedure:**

1. **Dataset:** WikiText-2 test set (standard benchmark)

2. **Evaluation Code:**
```python
def evaluate_perplexity(model, tokenizer, dataset):
    """Evaluate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in dataset:
            input_ids = tokenizer.encode(text, return_tensors="pt")

            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Accumulate
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity
```

3. **Comparison:**
```python
# Baseline (no DRAI)
ppl_baseline = evaluate_perplexity(model, tokenizer, dataset, drai=False)

# With DRAI
ppl_drai = evaluate_perplexity(model, tokenizer, dataset, drai=True)

# Delta
delta = ppl_drai - ppl_baseline
percent_change = (delta / ppl_baseline) * 100

print(f"Baseline: {ppl_baseline:.2f}")
print(f"DRAI:     {ppl_drai:.2f}")
print(f"Delta:    {delta:+.2f} ({percent_change:+.1f}%)")
```

4. **Success Criteria:**
   - Acceptable: `delta < +2.0` (no significant degradation)
   - Good: `delta < +1.0`
   - Excellent: `delta < 0` (improvement!)

---

### Metric 2: Attractor Statistics

**Purpose:** Understand attractor dynamics during inference.

**Statistics to Track:**

```python
@dataclass
class AttractorStats:
    """Statistics tracked for each layer during evaluation."""

    # Formation
    total_patterns_seen: int
    attractors_formed: int
    formation_rate: float  # attractors_formed / total_patterns_seen

    # Reinforcement
    reinforcement_count: int
    avg_reinforcements_per_attractor: float

    # Decay
    decay_events: int
    pruning_events: int

    # Coherence
    coherence_mean: float
    coherence_std: float
    coherence_max: float
    coherence_distribution: List[float]  # Histogram bins

    # Lifespan
    avg_attractor_lifespan: float  # In timesteps
    max_attractor_lifespan: int

    # Spatial
    attractor_centroids: np.ndarray  # For visualization
    attractor_similarities: np.ndarray  # Pairwise similarity matrix
```

**Collection Code:**
```python
def collect_attractor_stats(model, input_ids):
    """Collect attractor statistics during inference."""
    stats_per_layer = []

    # Forward pass with hooks
    _ = model(input_ids)

    # Extract stats from each layer
    for layer in model.gpt_neox.layers:
        if hasattr(layer.attention, 'drai') and layer.attention.drai:
            drai = layer.attention.drai
            stats = drai.get_statistics()
            stats_per_layer.append(stats)

    return stats_per_layer
```

---

### Metric 3: Generation Quality (Qualitative)

**Purpose:** Subjective assessment of text quality.

**Prompts:**
```python
EVALUATION_PROMPTS = [
    "Once upon a time",
    "The meaning of life is",
    "In a world where",
    "The scientific method",
    "Artificial intelligence will",
]
```

**Generation:**
```python
def compare_generations(model, tokenizer, prompts):
    """Generate with and without DRAI for comparison."""
    results = []

    for prompt in prompts:
        # Without DRAI
        output_baseline = generate(model, tokenizer, prompt, drai=False)

        # With DRAI
        output_drai = generate(model, tokenizer, prompt, drai=True)

        results.append({
            "prompt": prompt,
            "baseline": output_baseline,
            "drai": output_drai,
        })

    return results
```

**Evaluation Criteria:**
1. **Coherence:** Does the text make sense?
2. **Relevance:** Does it stay on topic?
3. **Repetition:** Does it repeat tokens/phrases?
4. **Diversity:** Does it use varied vocabulary?
5. **Grammaticality:** Is the grammar correct?

**Scoring:**
```python
# Manual scoring (1-5 scale for each criterion)
# Or automated metrics:
- Repetition: n-gram overlap
- Diversity: unique token ratio
- Relevance: cosine similarity with prompt
```

---

## Validation Criteria

### Must Pass (Required)

1. ✓ **Model loads** - No errors during loading
2. ✓ **Forward pass succeeds** - No crashes
3. ✓ **No NaN/Inf** - Numerical stability maintained
4. ✓ **Generates text** - Generation loop completes
5. ✓ **Attractors form** - count > 0 after sufficient input
6. ✓ **Perplexity acceptable** - delta < +2.0

### Should Pass (Important)

7. ✓ **Perplexity minimal impact** - delta < +1.0
8. ✓ **Coherent text** - Output makes sense
9. ✓ **No excessive repetition** - Repetition ratio < 2.0
10. ✓ **Attractors evolve** - Coherence changes over time
11. ✓ **Reasonable speed** - Not >2x slower than baseline
12. ✓ **Reasonable memory** - Not >2x memory of baseline

### Nice to Pass (Aspirational)

13. ◯ **Perplexity improvement** - delta < 0
14. ◯ **Quality improvement** - Subjective quality better
15. ◯ **Interesting patterns** - Novel attractor behaviors observed
16. ◯ **Long-range coherence** - Better consistency over long sequences

---

## Lessons Learned

**This section will be filled in as we proceed through Phase 3.**

### Technical Challenges
- TBD

### Implementation Insights
- TBD

### Configuration Tuning
- TBD

### Performance Observations
- TBD

---

## Reproducibility Guide

### Environment

**Dependencies:**
```
torch==2.9.1
transformers==4.36.0
gpt-neox==0.3.0  # Or latest compatible
```

**Hardware:**
```
Minimum: CPU, 8GB RAM
Recommended: GPU (any), 16GB RAM
```

**Random Seeds:**
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)  # Always use seed 42 for reproducibility
```

### Exact Reproduction Steps

**Step 1: Setup**
```bash
git clone https://github.com/HalcyonAIR/Duality
cd Duality
pip install -r requirements.txt
```

**Step 2: Download Model**
```python
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-125m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-125m")
```

**Step 3: Build DRAI Model**
```python
from src.drai.build_drai_neox import build_drai_neox_model
from src.drai.config import DraiConfig

config = DraiConfig(enabled=True, phase=2)
model = build_drai_neox_model("EleutherAI/pythia-125m", config)
```

**Step 4: Evaluate**
```python
from experiments.neox_baseline import evaluate_perplexity

ppl = evaluate_perplexity(model, tokenizer, "wikitext-2-test")
print(f"Perplexity: {ppl:.2f}")
```

### Configuration Files

All experiments will have associated configuration files:
```
experiments/configs/
├── baseline.yaml          # No DRAI
├── full_drai.yaml         # DRAI on all layers
├── selective_drai.yaml    # DRAI on some layers
└── aggressive_drai.yaml   # High attractor parameters
```

### Data Versioning

**Datasets:**
- WikiText-2: Standard HuggingFace version
- Version tracked via git LFS or documented download source

**Models:**
- pythia-125m: Specific revision hash documented
- All model weights from HuggingFace hub with version pins

---

**Document Status:** Living document, updated throughout Phase 3
**Last Updated:** 2025-11-18
**Next Update:** After completing Sub-Phase 3.2 (Architecture Analysis)
