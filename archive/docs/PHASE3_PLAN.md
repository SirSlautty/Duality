# Phase 3 Plan - Transformer Integration

**Objective:** Integrate DRAI resonance layer into GPT-NeoX transformer architecture
**Status:** 🚧 In Progress
**Start Date:** 2025-11-18
**Documentation Priority:** CRITICAL - This is the proof-of-concept phase

---

## Overview

Phase 3 represents the critical transition from isolated component testing to real-world transformer integration. This phase will demonstrate that DRAI can be successfully injected into a production transformer architecture and will provide our first measurements of its impact on model behavior.

**Why This Matters:**
- First validation that DRAI works in a real transformer
- Establishes injection methodology for other architectures
- Provides baseline performance metrics
- Demonstrates backward compatibility (model still works)
- Creates foundation for future training experiments

---

## Success Criteria

### Must Have (Required)
1. ✓ DRAI successfully injected into GPT-NeoX attention layers
2. ✓ Model loads and runs with pre-trained weights
3. ✓ Model produces coherent text output
4. ✓ No crashes or numerical instabilities
5. ✓ Attractor field functions during inference
6. ✓ Comprehensive documentation of methodology

### Should Have (Important)
1. ✓ Perplexity measurements (with vs without DRAI)
2. ✓ Attractor formation tracking during generation
3. ✓ Comparison of outputs (qualitative analysis)
4. ✓ Memory and speed benchmarks
5. ✓ Visualization of attractor evolution

### Nice to Have (Optional)
1. Multiple model sizes tested (e.g., 125M, 1.3B)
2. Different DRAI configurations compared
3. Layer-wise analysis (where are attractors most active?)
4. Long-context behavior analysis
5. Interactive demo notebook

---

## Phase 3 Sub-Phases

### 3.1: Environment Setup (Estimated: 30 min)
**Goal:** Get GPT-NeoX repository and dependencies

**Tasks:**
1. Clone GPT-NeoX repository into `models/gpt-neox/`
2. Review NeoX installation requirements
3. Identify minimal dependencies needed (no training deps)
4. Setup environment for inference only
5. Download a small pre-trained model (125M or 410M)

**Deliverables:**
- Working NeoX installation
- Pre-trained model downloaded
- Baseline inference verified

**Documentation:**
- Installation steps in `docs/NEOX_SETUP.md`
- Dependency decisions explained
- Model selection rationale

---

### 3.2: Architecture Analysis (Estimated: 1-2 hours)
**Goal:** Deeply understand NeoX attention implementation

**Tasks:**
1. Read and annotate NeoX attention code
2. Identify all attention layer types
3. Map out tensor shapes through attention
4. Understand positional encoding (RoPE)
5. Identify K/V cache mechanism
6. Find injection points for DRAI
7. Understand config system

**Key Files to Analyze:**
- `gpt-neox/megatron/model/transformer.py`
- `gpt-neox/megatron/model/gpt2_model.py`
- `gpt-neox/megatron/model/attention.py`
- Configuration loading system

**Questions to Answer:**
- Where exactly is attention computed?
- What are the tensor shapes at each step?
- How are K/V handled in generation (caching)?
- Where can we inject DRAI without breaking anything?
- How do we pass DRAI config through the model?
- Are there multiple attention implementations?

**Deliverables:**
- Annotated attention code
- Tensor shape diagrams
- Injection point identification
- Integration strategy document

**Documentation:**
- `docs/NEOX_ARCHITECTURE_ANALYSIS.md` - Deep dive into NeoX attention
- Diagrams of attention flow
- Injection point justification

---

### 3.3: Integration Strategy Design (Estimated: 1 hour)
**Goal:** Design clean, minimal integration approach

**Design Principles:**
1. **Minimal Invasiveness** - Change as little NeoX code as possible
2. **Backward Compatibility** - Model works with DRAI disabled
3. **Configuration Driven** - DRAI enabled via config flags
4. **Layer Selectivity** - Choose which layers get DRAI
5. **Easy Debugging** - Clear separation of DRAI vs standard attention

**Integration Options:**

**Option A: Monkey Patch**
```python
# Pros: No NeoX code changes, easy to test
# Cons: Fragile, hard to maintain

import gpt_neox.attention as neox_attn
original_forward = neox_attn.Attention.forward

def drai_forward(self, *args, **kwargs):
    # Inject DRAI here
    pass

neox_attn.Attention.forward = drai_forward
```

**Option B: Subclass**
```python
# Pros: Clean, maintainable, Pythonic
# Cons: Need to modify model construction

class DraiGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, *args, drai_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if drai_config:
            self.drai = DraiResonanceLayer(...)

    def forward(self, ...):
        # Standard attention + DRAI injection
        pass
```

**Option C: Wrapper Module**
```python
# Pros: Zero NeoX changes, composable
# Cons: More complex, extra module layer

class DraiAttentionWrapper(nn.Module):
    def __init__(self, attention_module, drai_config):
        self.attention = attention_module
        self.drai = DraiResonanceLayer(...)

    def forward(self, ...):
        # Wrap attention with DRAI
        pass
```

**Recommended Approach: Option B (Subclass)**
- Cleanest integration
- Easy to understand and maintain
- Follows standard PyTorch patterns
- Easy to toggle on/off per layer

**Deliverables:**
- Integration strategy document
- Code architecture diagrams
- Configuration schema
- Implementation checklist

**Documentation:**
- `docs/INTEGRATION_STRATEGY.md` - Full strategy explanation
- Decision rationale for each choice
- Alternative approaches considered

---

### 3.4: DRAI Configuration System (Estimated: 30 min)
**Goal:** Create configuration system for DRAI in NeoX

**Configuration Schema:**
```yaml
# Example DRAI configuration
drai:
  enabled: true
  phase: 2

  # Which layers get DRAI?
  layer_config:
    mode: "all"  # Options: "all", "selective", "none"
    selective_layers: [0, 6, 12, 18, 23]  # If mode="selective"

  # DRAI hyperparameters
  hyperparameters:
    max_attractors: 32
    coherence_threshold: 0.3
    formation_threshold: 0.5
    decay_rate: 0.01
    ema_momentum: 0.9

  # Integration settings
  integration:
    num_drai_heads: 1  # How many DRAI heads per layer?
    drai_head_dim: null  # null = use model's head_dim
    inject_all_layers: true
    verbose_logging: false
```

**Tasks:**
1. Create `src/drai/config.py` with configuration dataclass
2. Add validation for DRAI config
3. Create config loading utilities
4. Document all configuration options

**Deliverables:**
- Configuration system code
- Configuration validation
- Example configs for different use cases
- Configuration documentation

**Documentation:**
- `docs/DRAI_CONFIGURATION.md` - Complete config reference
- Example configurations with explanations
- Tuning guide for different scenarios

---

### 3.5: Implementation (Estimated: 2-3 hours)
**Goal:** Implement DRAI-enhanced NeoX attention

**Implementation Steps:**

1. **Create DRAI-NeoX Bridge Module**
   - File: `src/drai/neox_integration.py`
   - Purpose: Adaptation layer between DRAI and NeoX

2. **Create DraiGPTNeoXAttention Class**
   - Subclass NeoX attention
   - Add DRAI initialization
   - Modify forward pass to inject DRAI K/V
   - Handle K/V cache compatibility

3. **Create Model Builder**
   - File: `src/drai/build_drai_neox.py`
   - Function to construct NeoX model with DRAI
   - Replace attention modules with DRAI versions
   - Preserve pre-trained weights

4. **Add Monitoring Hooks**
   - Track attractor formation during inference
   - Log attractor statistics
   - Capture K/V contributions

**Key Implementation Challenges:**

**Challenge 1: K/V Cache Compatibility**
```python
# NeoX uses K/V cache for generation
# DRAI generates synthetic K/V that must be cached too

# Standard NeoX cache:
past_kv = (key, value)  # [batch, heads, seq, head_dim]

# With DRAI:
standard_kv = (K_attn, V_attn)
drai_kv = (K_reson, V_reson)
combined_kv = (
    torch.cat([K_attn, K_reson], dim=2),
    torch.cat([V_attn, V_reson], dim=2)
)
# Cache the combined K/V
```

**Challenge 2: Rotary Position Embeddings (RoPE)**
```python
# NeoX uses RoPE on Q and K
# DRAI K/V are synthetic - do they need RoPE?

# Decision: NO - DRAI K/V represent patterns, not positions
# Only apply RoPE to standard attention K/V
```

**Challenge 3: Attention Mask**
```python
# Attention mask must extend to DRAI heads
# DRAI heads should attend to all positions

# Solution: Extend mask to cover DRAI heads
# Or: Use separate mask for DRAI (all ones = attend everywhere)
```

**Challenge 4: Gradient Checkpointing**
```python
# NeoX uses gradient checkpointing for training
# DRAI must be checkpoint-compatible

# Solution: Ensure DRAI forward is deterministic
# Test with torch.utils.checkpoint.checkpoint()
```

**Deliverables:**
- `src/drai/neox_integration.py` (integration module)
- `src/drai/build_drai_neox.py` (model builder)
- `src/drai/config.py` (configuration)
- Unit tests for integration components
- Integration tests with real NeoX models

**Documentation:**
- `docs/NEOX_INTEGRATION_IMPLEMENTATION.md`
- Code annotations explaining each decision
- Diagram of data flow through DRAI-enhanced attention

---

### 3.6: Testing & Validation (Estimated: 2-3 hours)
**Goal:** Verify integration works correctly

**Test Levels:**

**Level 1: Unit Tests**
```python
# Test individual components
- test_drai_neox_attention_initialization()
- test_drai_neox_attention_forward()
- test_config_loading()
- test_model_building()
```

**Level 2: Integration Tests**
```python
# Test with real NeoX models
- test_load_pretrained_neox_with_drai()
- test_forward_pass_with_drai()
- test_generation_with_drai()
- test_kv_cache_compatibility()
```

**Level 3: Functional Tests**
```python
# Test model behavior
- test_model_produces_coherent_text()
- test_attractors_form_during_generation()
- test_no_numerical_instabilities()
```

**Level 4: Regression Tests**
```python
# Ensure we didn't break anything
- test_model_without_drai_unchanged()
- test_outputs_match_baseline() (with DRAI phase=1)
```

**Validation Checklist:**
- [ ] Model loads without errors
- [ ] Forward pass completes
- [ ] Generation produces text
- [ ] Text is coherent (subjective)
- [ ] No NaN or Inf values
- [ ] Attractors form (count > 0)
- [ ] Attractors evolve (coherence changes)
- [ ] Memory usage acceptable
- [ ] Speed acceptable (not 10x slower)
- [ ] K/V cache works in generation

**Deliverables:**
- Test suite for NeoX integration
- Validation report with checklist
- Screenshots of successful generation
- Error logs (if any issues found)

**Documentation:**
- `docs/INTEGRATION_TESTING.md`
- Test results and validation evidence
- Known issues and limitations

---

### 3.7: Baseline Measurements (Estimated: 2-3 hours)
**Goal:** Establish quantitative baseline metrics

**Metrics to Measure:**

**1. Perplexity**
```python
# Measure on standard benchmark (e.g., WikiText-2)
perplexity_baseline = evaluate_perplexity(model, dataset, drai_enabled=False)
perplexity_drai = evaluate_perplexity(model, dataset, drai_enabled=True)
perplexity_delta = perplexity_drai - perplexity_baseline

# Expected: perplexity should not increase significantly
# Goal: Δ perplexity < 1.0
```

**2. Attractor Statistics**
```python
# Track attractor behavior during evaluation
stats = {
    "attractors_formed": [],
    "coherence_distribution": [],
    "attractor_lifespan": [],
    "reinforcement_frequency": [],
    "decay_events": [],
}
```

**3. Generation Quality**
```python
# Qualitative comparison of generated text
prompts = ["Once upon a time", "The meaning of life is", ...]
outputs_baseline = [generate(model, p, drai=False) for p in prompts]
outputs_drai = [generate(model, p, drai=True) for p in prompts]

# Compare:
# - Coherence (subjective)
# - Repetition (n-gram overlap)
# - Diversity (unique tokens)
# - Relevance to prompt
```

**4. Performance Benchmarks**
```python
# Speed and memory
benchmark = {
    "inference_time": time_forward_pass(),
    "generation_speed": tokens_per_second(),
    "memory_usage": peak_memory_mb(),
    "memory_overhead_drai": memory_drai - memory_baseline,
}
```

**5. Attention Pattern Analysis**
```python
# How does DRAI change attention?
attention_weights_baseline = extract_attention_weights(model, input, drai=False)
attention_weights_drai = extract_attention_weights(model, input, drai=True)

# Analyze:
# - Entropy of attention distribution
# - Attention to DRAI heads vs standard heads
# - Position-wise attention patterns
```

**Deliverables:**
- Measurement scripts
- Benchmark results (CSV, JSON)
- Analysis notebooks
- Comparison visualizations

**Documentation:**
- `docs/BASELINE_MEASUREMENTS.md`
- Complete methodology for each metric
- Raw results with statistical analysis
- Interpretation and insights

---

### 3.8: Visualization & Analysis (Estimated: 2-3 hours)
**Goal:** Create insightful visualizations of DRAI behavior

**Visualizations to Create:**

**1. Attractor Evolution Timeline**
```python
# Plot attractor count and coherence over generation steps
# X-axis: Generation step
# Y-axis: Number of attractors, average coherence
# Shows: How attractors form and decay during text generation
```

**2. Attractor Manifold (t-SNE/UMAP)**
```python
# Project attractor centroids to 2D
# Color by coherence strength
# Shows: Clustering of similar patterns in latent space
```

**3. Token-to-Attractor Mapping**
```python
# Heatmap showing which tokens activate which attractors
# X-axis: Tokens in sequence
# Y-axis: Attractor index
# Color: Similarity / reinforcement strength
# Shows: Which words/concepts correspond to attractors
```

**4. Attention Weight Distribution**
```python
# Compare attention distributions with/without DRAI
# Histogram of attention weights to DRAI heads
# Shows: How much model relies on DRAI vs standard attention
```

**5. Perplexity Over Time**
```python
# Plot perplexity at each token position
# Compare DRAI vs baseline
# Shows: Where DRAI helps/hurts most
```

**6. Coherence Heatmap by Layer**
```python
# Heatmap: Layers × Generation steps
# Color: Average attractor coherence
# Shows: Which layers form strongest attractors
```

**Deliverables:**
- Jupyter notebooks with visualizations
- High-quality plots (publication ready)
- Interactive visualizations (Plotly)
- Visualization utilities in `src/drai/visualization.py`

**Documentation:**
- `docs/VISUALIZATION_GUIDE.md`
- Explanation of each visualization
- How to interpret results
- Code examples for creating visualizations

---

### 3.9: Results Documentation (Estimated: 2-3 hours)
**Goal:** Comprehensive documentation of findings

**Documents to Create:**

**1. PHASE3_RESULTS.md**
- Executive summary
- Methodology overview
- Key findings
- Quantitative results
- Qualitative observations
- Limitations and caveats
- Implications for future work

**2. INTEGRATION_METHODOLOGY.md**
- Detailed technical methodology
- Every decision explained
- Alternative approaches considered
- Implementation challenges
- Solutions and workarounds
- Best practices learned

**3. PERFORMANCE_ANALYSIS.md**
- Complete performance analysis
- Statistical tests
- Significance of results
- Comparison with baseline
- Performance vs hyperparameter plots
- Memory and speed analysis

**4. CASE_STUDIES.md**
- Specific examples of DRAI in action
- Interesting attractor behaviors
- Failure cases and analysis
- Edge cases and handling

**5. README Updates**
- Add Phase 3 completion status
- Link to all new documentation
- Update quick start with NeoX examples
- Add visualizations to README

**Documentation Standards:**
- Clear methodology section (reproducible)
- Raw data included or linked
- Statistical rigor (error bars, significance tests)
- Honest about limitations
- Speculative claims clearly marked
- Code snippets for key operations
- Visual aids (diagrams, plots)

---

## Risk Assessment

### High Risk
1. **NeoX compatibility issues** - Complex codebase, many dependencies
   - Mitigation: Start with minimal dependencies, inference only

2. **Numerical instability** - DRAI could cause NaN/Inf
   - Mitigation: Extensive testing, gradient clipping, careful initialization

3. **Performance degradation** - Perplexity increases significantly
   - Mitigation: Phase 1 fallback (zeros), hyperparameter tuning

### Medium Risk
4. **K/V cache bugs** - Generation could fail or produce garbage
   - Mitigation: Thorough testing of generation loop

5. **Memory issues** - DRAI adds memory overhead
   - Mitigation: Profile memory, optimize buffer sizes

6. **RoPE incompatibility** - Position encoding could conflict
   - Mitigation: Don't apply RoPE to DRAI K/V

### Low Risk
7. **Configuration complexity** - Hard to configure DRAI
   - Mitigation: Sensible defaults, clear documentation

8. **Slow inference** - DRAI adds computational cost
   - Mitigation: Profile and optimize, accept some slowdown

---

## Timeline Estimate

| Sub-Phase | Estimated Time | Priority |
|-----------|---------------|----------|
| 3.1 Environment Setup | 30 min | Critical |
| 3.2 Architecture Analysis | 1-2 hours | Critical |
| 3.3 Integration Strategy | 1 hour | Critical |
| 3.4 Configuration System | 30 min | High |
| 3.5 Implementation | 2-3 hours | Critical |
| 3.6 Testing & Validation | 2-3 hours | Critical |
| 3.7 Baseline Measurements | 2-3 hours | High |
| 3.8 Visualization | 2-3 hours | Medium |
| 3.9 Results Documentation | 2-3 hours | Critical |

**Total Estimated Time:** 12-18 hours
**Priority Order:** 3.1 → 3.2 → 3.3 → 3.5 → 3.6 → 3.4 → 3.7 → 3.9 → 3.8

---

## Success Metrics

**Minimum Viable Success:**
- ✓ Model runs with DRAI
- ✓ Generates coherent text
- ✓ Attractors form during inference
- ✓ Perplexity < baseline + 2.0

**Full Success:**
- ✓ All above
- ✓ Perplexity < baseline + 1.0
- ✓ Interesting attractor patterns observed
- ✓ Complete documentation
- ✓ Visualizations created

**Exceptional Success:**
- ✓ All above
- ✓ Perplexity improvement over baseline
- ✓ Measurable coherence improvement
- ✓ Novel insights about attractor behavior
- ✓ Published-quality documentation

---

## Deliverables Summary

### Code
- [ ] `src/drai/config.py` - Configuration system
- [ ] `src/drai/neox_integration.py` - NeoX integration module
- [ ] `src/drai/build_drai_neox.py` - Model builder
- [ ] `src/drai/visualization.py` - Visualization utilities
- [ ] `tests/integration/test_neox_integration.py` - Integration tests
- [ ] `experiments/neox_baseline.py` - Measurement scripts
- [ ] `experiments/notebooks/phase3_analysis.ipynb` - Analysis notebook

### Documentation
- [ ] `docs/NEOX_SETUP.md` - Setup guide
- [ ] `docs/NEOX_ARCHITECTURE_ANALYSIS.md` - Architecture deep dive
- [ ] `docs/INTEGRATION_STRATEGY.md` - Strategy document
- [ ] `docs/DRAI_CONFIGURATION.md` - Configuration reference
- [ ] `docs/NEOX_INTEGRATION_IMPLEMENTATION.md` - Implementation details
- [ ] `docs/INTEGRATION_TESTING.md` - Testing methodology
- [ ] `docs/BASELINE_MEASUREMENTS.md` - Measurement methodology
- [ ] `docs/VISUALIZATION_GUIDE.md` - Visualization guide
- [ ] `docs/PHASE3_RESULTS.md` - Results report
- [ ] `docs/INTEGRATION_METHODOLOGY.md` - Complete methodology
- [ ] `docs/PERFORMANCE_ANALYSIS.md` - Performance analysis
- [ ] `docs/CASE_STUDIES.md` - Case studies

---

## Notes for Future Phases

**Phase 4: Visualization & Analysis**
- Can use Phase 3 visualizations as foundation
- Add more sophisticated analysis tools
- Create interactive demos

**Phase 5: Training & Fine-tuning**
- Use Phase 3 integration as starting point
- Add training loop modifications
- Explore learnable DRAI parameters

---

## Open Questions

1. Should DRAI be applied to all layers or just some? (Start with all, then experiment)
2. How many DRAI heads per layer? (Start with 1, then try 2-4)
3. Should attractor field be shared across layers or per-layer? (Start per-layer)
4. Should we use Phase 1 (zeros) or Phase 2 (attractors) initially? (Phase 2 - we want to see behavior)
5. What model size to start with? (125M - smallest, easiest to debug)
6. Should we fine-tune after integration? (Not yet - Phase 3 is inference only)

---

**This plan will be updated as we progress through Phase 3.**
**All decisions, results, and learnings will be documented in real-time.**
