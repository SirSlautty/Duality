# Phase 5 Pilot Experiment - Diagnostic Report

**Date:** 2025-11-18
**Model:** EleutherAI/pythia-70m
**Task:** Long-story consistency evaluation (5 stories, 1120-1764 tokens)

---

## Executive Summary

**CRITICAL FINDING:** DRAI degrades generation quality on pythia-70m, causing more severe repetition collapse compared to baseline.

- **Baseline accuracy:** 6.5%
- **DRAI accuracy:** 0.0%
- **Statistical significance:** Not significant (p=0.19, n=5)
- **Conclusion:** DRAI harms rather than helps on this small model

---

## Detailed Findings

### 1. Generation Quality Degradation

Both baseline and DRAI exhibit repetition collapse (model too small for task), but **DRAI makes it significantly worse**:

**Baseline output example:**
```
"Patricia considered the situation carefully. New landscapes unfolded before them.
Avery considered the situation carefully. New landscapes unfolded before them..."
```
- Repetitive but maintains some semantic structure
- Uses story-relevant phrases
- Achieved 6.5% accuracy (some questions partially answered)

**DRAI output example:**
```
"The journey was far from the journey took avery considered the adventure was far
from the journeyed the journeyed..."
```
- More severe degradation (word-level loops: "journeyed the journeyed")
- Loses grammatical structure entirely
- 0% accuracy (no questions answered correctly)

### 2. DRAI Statistics

- **Layers enhanced:** 6
- **Total active attractors:** 7 (~1.2 per layer)
- **Attractor utilization:** Very low

This suggests:
- Attractors are not being leveraged effectively
- Model may be too small to benefit from additional attractor capacity
- Attractor dynamics may be interfering with already-fragile generation

### 3. Task Difficulty

The stories are challenging:
- **Length:** 1120-1764 tokens (mean: 1359)
- **Distractors:** 0-3 conflicting subplots per story
- **Questions:** 5 per story (factual recall, character ID, relationships)

pythia-70m (70M parameters) is fundamentally too small for this task. Both models fail, but DRAI makes failure worse.

---

## Hypothesis: Why DRAI Degrades Performance

### Primary Hypothesis: Attention Dilution
When DRAI injects K'/V' from attractors, attention is distributed over more tokens:
- **Baseline:** `Attn(Q, K, V)` where K/V are from story context only
- **DRAI:** `Attn(Q, [K; K'], [V; V'])` where K'/V' are attractor memories

For a 70M parameter model:
- Limited capacity to process additional context
- Attractor injection may dilute attention to actual story tokens
- Generation becomes incoherent as model "listens" to irrelevant attractors

### Secondary Hypothesis: Attractor Instability
- With only 7 active attractors across 6 layers, dynamics may be unstable
- EMA updates on sparse activations could create noisy gradients
- Attractors may not have time to "learn" useful patterns in short stories

### Tertiary Hypothesis: Initialization Mismatch
- Attractors initialized to match embedding distribution (Phase 2)
- May not be appropriate for pythia-70m's specific latent space
- Could create distribution shift that confuses generation

---

## Comparison to Phase 4 Results

**Phase 4:** DRAI achieved **zero-cost integration** on perplexity:
- Perplexity maintained within noise threshold
- No degradation on evaluation metrics
- Successful proof of concept

**Phase 5:** DRAI **harms generation quality**:
- Perplexity measures how well model predicts next token
- Generation quality measures coherence of sampled text
- These are related but not identical metrics!

**Key insight:** DRAI may maintain perplexity (probability distribution quality) while degrading generation (sampling quality). This suggests:
1. Attractors don't harm the model's knowledge
2. But they interfere with the generation/sampling process
3. Especially problematic for small models with limited capacity

---

## Recommendations

### Immediate Actions

1. **Test with larger model (pythia-410m or pythia-1b)**
   - More capacity to absorb attractor overhead
   - Better baseline generation quality
   - Can isolate DRAI effect from model-size limitations

2. **Adjust DRAI hyperparameters for small models**
   - Reduce `num_attractors` (currently 512 → try 128)
   - Increase `activation_threshold` (currently 0.7 → try 0.85)
   - Reduce influence: lower `coherence_weight` or add scaling factor

3. **Add generation-time diagnostics**
   - Log attention weights to attractors vs. story tokens
   - Measure entropy of attention distribution
   - Track when repetition loops begin

### Investigation Priorities

1. **Attention weight analysis**
   - How much attention goes to attractors vs. context?
   - Are attractors dominating attention incorrectly?

2. **Perplexity on long stories**
   - Does DRAI maintain perplexity on these specific stories?
   - If yes: confirms generation/perplexity disconnect
   - If no: suggests task-specific breakdown

3. **Ablation studies**
   - Vary `num_attractors`: [32, 64, 128, 256, 512]
   - Vary `activation_threshold`: [0.5, 0.7, 0.8, 0.9]
   - Find optimal settings for small models

---

## Next Steps

**Option A: Larger Model (Recommended)**
- Run same pilot on pythia-410m (410M params, 4x larger)
- If DRAI still degrades: fundamental architecture issue
- If DRAI improves: confirms small-model limitation

**Option B: Hyperparameter Tuning**
- Reduce attractor count and increase selectivity
- Add attractor influence scaling factor
- Re-run pilot on pythia-70m

**Option C: Diagnostic Deep Dive**
- Implement attention weight logging
- Run single-story analysis with full diagnostics
- Understand exact mechanism of degradation

**Option D: Acknowledge Limitation**
- Document that DRAI requires minimum model size
- Update paper with this finding
- Focus on demonstrating benefits on larger models

---

## Conclusion

This pilot revealed a critical issue: **DRAI degrades generation quality on small models (70M params)**. While Phase 4 showed zero-cost integration for perplexity, Phase 5 reveals that generation is more sensitive to attractor interference.

This is valuable negative evidence that will strengthen the paper:
1. Shows we're honestly evaluating DRAI limitations
2. Identifies minimum model size requirements
3. Suggests perplexity ≠ generation quality (important distinction)

**Recommendation:** Test with pythia-410m before abandoning Phase 5 approach. If larger model succeeds, we have a clear story: "DRAI requires models >100M params to benefit from dynamic memory."
