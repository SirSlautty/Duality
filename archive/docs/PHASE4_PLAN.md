# Phase 4 Plan - Analysis & Visualization

**Objective:** Quantitative evaluation and attractor behavior analysis to support paper publication

**Status:** 🚀 In Progress
**Start Date:** 2025-11-18
**Estimated Duration:** 2-3 weeks (accelerated with available compute)

---

## Overview

Phase 3 proved **feasibility** - DRAI integrates and works. Phase 4 proves **value** - measuring impact on model performance and understanding attractor behavior.

**Critical for Paper:** This phase provides the empirical evidence reviewers will demand.

---

## Goals

### Primary Goals (Must Have)

1. **Perplexity Evaluation**
   - Measure on WikiText-2 (standard benchmark)
   - Compare baseline vs DRAI
   - Statistical significance testing
   - Document methodology completely

2. **Attractor Behavior Analysis**
   - Track formation over time
   - Measure coherence evolution
   - Analyze decay patterns
   - Characterize attractor diversity

3. **Generation Quality Metrics**
   - Repetition analysis (n-gram overlap)
   - Diversity metrics (unique tokens)
   - Coherence measures
   - Comparison with baseline

### Secondary Goals (Should Have)

4. **Scaling Analysis**
   - Test with larger models (125M, 410M)
   - Measure scaling trends
   - Identify optimal configurations

5. **Hyperparameter Sensitivity**
   - Vary max_attractors (16, 32, 64)
   - Vary thresholds (coherence, formation)
   - Vary decay_rate and ema_momentum
   - Find best settings

6. **Layer-wise Analysis**
   - Compare attractor behavior per layer
   - Identify which layers benefit most
   - Test selective injection strategies

### Stretch Goals (Nice to Have)

7. **Visualization**
   - Attractor evolution plots
   - t-SNE/UMAP of attractor manifold
   - Token-to-attractor heatmaps
   - Interactive notebooks

8. **Long-Context Testing**
   - Behavior on sequences > 512 tokens
   - Memory persistence analysis
   - Coherence over long generation

---

## Phase 4 Sub-Tasks

### 4.1: Evaluation Infrastructure (Day 1)

**Goal:** Build reusable evaluation framework

**Tasks:**
- Create `experiments/evaluation/` directory structure
- Implement perplexity evaluation script
- Create attractor statistics collection
- Build comparison utilities
- Add logging and checkpointing

**Deliverables:**
- `experiments/evaluation/evaluate_perplexity.py`
- `experiments/evaluation/collect_statistics.py`
- `experiments/evaluation/compare_outputs.py`
- `experiments/evaluation/utils.py`

---

### 4.2: WikiText-2 Perplexity (Day 1-2)

**Goal:** Quantitative performance measurement

**Methodology:**
```python
# Standard protocol
1. Load WikiText-2 test set
2. Evaluate baseline model (no DRAI)
3. Evaluate DRAI model (phase 2, default config)
4. Compute perplexity for both
5. Statistical significance test
6. Document results
```

**Metrics to Track:**
- Perplexity (primary metric)
- Bits per character
- Token-level loss distribution
- Sequence length vs perplexity

**Success Criteria:**
- Acceptable: PPL_DRAI < PPL_baseline + 2.0
- Good: PPL_DRAI < PPL_baseline + 1.0
- Excellent: PPL_DRAI < PPL_baseline (improvement!)

**Deliverables:**
- Perplexity scores (baseline and DRAI)
- Statistical significance results
- Per-sequence analysis
- Methodology documentation

---

### 4.3: Attractor Statistics Collection (Day 2-3)

**Goal:** Understand attractor dynamics during evaluation

**Statistics to Track:**

**Formation:**
- Total attractors formed per layer
- Formation rate over time
- Pattern diversity at formation

**Reinforcement:**
- Reinforcement count per attractor
- Coherence growth curves
- EMA convergence behavior

**Decay:**
- Decay events per layer
- Attractor lifespan distribution
- Pruning frequency

**Spatial:**
- Pairwise attractor similarities
- Clustering behavior
- Centroid evolution

**Deliverables:**
- Complete statistics dataset (JSON/CSV)
- Per-layer analysis
- Temporal evolution tracking
- Distribution visualizations

---

### 4.4: Generation Quality Analysis (Day 3-4)

**Goal:** Qualitative and quantitative generation comparison

**Test Prompts:**
```python
PROMPTS = [
    # Factual
    "The capital of France is",
    "Albert Einstein was born in",
    "The chemical symbol for gold is",

    # Creative
    "Once upon a time",
    "In a world where",
    "The old wizard said",

    # Reasoning
    "To solve this problem, first we",
    "The main difference between X and Y is",
    "This happens because",
]
```

**Metrics:**

1. **Repetition (n-gram overlap):**
```python
def repetition_score(text):
    # Compute 3-gram and 4-gram self-overlap
    ngrams = extract_ngrams(text, n=3)
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return 1 - unique_ratio  # Higher = more repetitive
```

2. **Diversity (unique token ratio):**
```python
def diversity_score(text):
    tokens = tokenize(text)
    return len(set(tokens)) / len(tokens)
```

3. **Coherence (perplexity-based):**
```python
def coherence_score(text, model):
    # Low perplexity = high coherence
    return -compute_perplexity(text, model)
```

**Deliverables:**
- Side-by-side generation comparison
- Quantitative metric scores
- Human evaluation checklist
- Example outputs for paper

---

### 4.5: Scaling Experiments (Day 4-5)

**Goal:** Test with larger models

**Models to Test:**
- pythia-70m (6 layers) - already tested
- pythia-125m (12 layers) - medium
- pythia-410m (24 layers) - large (if time/compute permits)

**Comparison Metrics:**
- Perplexity impact vs model size
- Attractor behavior vs depth
- Memory overhead vs parameters
- Speed impact vs model size

**Hypothesis:**
Larger models may benefit more from DRAI (more capacity to organize)

**Deliverables:**
- Multi-model comparison table
- Scaling trend analysis
- Optimal configuration per size
- Recommendations

---

### 4.6: Hyperparameter Tuning (Day 5-6)

**Goal:** Find optimal DRAI configuration

**Parameters to Vary:**

1. **max_attractors:** [16, 32, 64, 128]
2. **coherence_threshold:** [0.2, 0.3, 0.4, 0.5]
3. **formation_threshold:** [0.4, 0.5, 0.6, 0.7]
4. **decay_rate:** [0.005, 0.01, 0.02, 0.05]
5. **ema_momentum:** [0.8, 0.9, 0.95]

**Search Strategy:**
- Grid search (exhaustive but thorough)
- Or random search (faster, good enough)
- Focus on perplexity as objective

**Deliverables:**
- Hyperparameter sweep results
- Best configuration identified
- Sensitivity analysis
- Configuration recommendations

---

### 4.7: Visualization (Day 6-7)

**Goal:** Create publication-quality visualizations

**Visualizations to Create:**

1. **Perplexity Comparison:**
```python
# Bar chart: Baseline vs DRAI
# With error bars (standard error)
# Significance stars (*, **, ***)
```

2. **Attractor Formation Timeline:**
```python
# Line plot: Attractor count over evaluation
# Per-layer curves
# Shows formation dynamics
```

3. **Coherence Distribution:**
```python
# Histogram: Attractor coherence values
# Before/after decay
# Shows memory organization
```

4. **Attractor Manifold (t-SNE):**
```python
# 2D projection of attractor centroids
# Color by coherence
# Shows pattern clustering
```

5. **Token-to-Attractor Heatmap:**
```python
# Which tokens activate which attractors
# Reveals semantic organization
```

**Deliverables:**
- High-resolution plots (PDF/PNG)
- Interactive notebooks (Jupyter)
- Figure captions for paper
- Supplementary visualizations

---

### 4.8: Documentation (Day 7)

**Goal:** Document complete methodology for paper

**Documents to Create:**

1. **PHASE4_RESULTS.md**
   - All quantitative results
   - Statistical analysis
   - Interpretation and insights

2. **EVALUATION_METHODOLOGY.md**
   - Complete evaluation protocol
   - Reproducibility guide
   - Benchmark details

3. **HYPERPARAMETER_GUIDE.md**
   - Best configurations
   - Sensitivity analysis
   - Tuning recommendations

4. **VISUALIZATION_GUIDE.md**
   - How to create each plot
   - Interpretation guide
   - Figure descriptions for paper

---

## Success Metrics

### Minimum Success (Acceptable for Paper)

- ✓ Perplexity measured on WikiText-2
- ✓ No catastrophic degradation (PPL < baseline + 2.0)
- ✓ Attractors demonstrably forming
- ✓ Complete methodology documented

### Full Success (Good Paper)

- ✓ All above
- ✓ Perplexity within 1.0 of baseline
- ✓ Interesting attractor patterns observed
- ✓ Quality metrics show no degradation
- ✓ Scaling trends documented
- ✓ Publication-quality visualizations

### Exceptional Success (Strong Paper)

- ✓ All above
- ✓ Perplexity improvement over baseline
- ✓ Clear quality improvements in generation
- ✓ Discovered optimal configurations
- ✓ Compelling attractor visualizations
- ✓ Ready for top-tier venue submission

---

## Timeline

**Week 1 (Days 1-3):**
- Infrastructure setup
- WikiText-2 evaluation
- Statistics collection

**Week 2 (Days 4-7):**
- Quality analysis
- Scaling experiments
- Hyperparameter tuning
- Visualization

**Week 3 (if needed):**
- Additional experiments
- Refinement
- Documentation polish

**Accelerated (with compute):**
- Can complete in 1 week with parallel experiments
- Run multiple configs simultaneously
- Faster iteration

---

## Resource Requirements

**Compute:**
- CPU sufficient for pythia-70m, 125m
- GPU helpful but not required (experiments are inference-only)
- Storage: ~10GB for models and results

**Software:**
- Already installed: PyTorch, transformers, datasets
- Additional: scipy (for stats), seaborn (for plots)

**Data:**
- WikiText-2: ~4MB (downloads automatically)
- Generated outputs: ~1GB

---

## Risks and Mitigation

### Risk 1: Perplexity Degradation

**Risk:** DRAI increases perplexity significantly

**Impact:** Harder to publish (looks like performance loss)

**Mitigation:**
- Hyperparameter tuning to minimize impact
- Phase 1 mode as fallback (zeros - no impact)
- Emphasize architectural contribution over performance
- Position as exploration vs optimization

### Risk 2: No Interesting Patterns

**Risk:** Attractors form but behavior is boring/random

**Impact:** Less compelling paper

**Mitigation:**
- Try different prompts/datasets
- Longer sequences (more time to form patterns)
- Manual inspection of attractor semantics
- Focus on other contributions (architecture, metacognition)

### Risk 3: Time/Compute Constraints

**Risk:** Experiments take too long

**Impact:** Can't complete Phase 4 fully

**Mitigation:**
- Prioritize core experiments (perplexity, basic analysis)
- Use smallest models first (pythia-70m)
- Parallelize where possible
- Accept partial results for initial paper

---

## Deliverables Summary

**Code:**
- [ ] `experiments/evaluation/evaluate_perplexity.py`
- [ ] `experiments/evaluation/collect_statistics.py`
- [ ] `experiments/evaluation/compare_outputs.py`
- [ ] `experiments/evaluation/visualize_attractors.py`
- [ ] `experiments/evaluation/hyperparameter_sweep.py`

**Data:**
- [ ] WikiText-2 perplexity results (baseline vs DRAI)
- [ ] Attractor statistics dataset (JSON/CSV)
- [ ] Generation quality metrics
- [ ] Hyperparameter sweep results

**Visualizations:**
- [ ] Perplexity comparison (bar chart)
- [ ] Attractor formation timeline (line plot)
- [ ] Coherence distribution (histogram)
- [ ] Attractor manifold (t-SNE)
- [ ] Token-attractor heatmap

**Documentation:**
- [ ] `docs/PHASE4_RESULTS.md` - Complete results
- [ ] `docs/EVALUATION_METHODOLOGY.md` - Protocol
- [ ] `docs/HYPERPARAMETER_GUIDE.md` - Tuning guide
- [ ] `docs/VISUALIZATION_GUIDE.md` - Figure guide

---

## Paper Integration

**How Phase 4 supports the paper:**

**Section 4 (Experiments):**
- Perplexity results → Table 1
- Quality metrics → Table 2
- Attractor statistics → Table 3
- Scaling trends → Figure 1
- Formation timeline → Figure 2
- Manifold visualization → Figure 3

**Section 5 (Discussion):**
- Interpretation of results
- Why patterns emerge
- Limitations observed
- Future work identified

**Appendix:**
- Complete hyperparameter details
- Full statistical tests
- Additional visualizations
- Reproducibility checklist

---

## Next Actions

**Immediate (Today):**
1. Create evaluation infrastructure
2. Run WikiText-2 baseline evaluation
3. Run WikiText-2 DRAI evaluation
4. Compute initial statistics

**This Week:**
5. Generate quality comparison
6. Run scaling experiments
7. Create visualizations
8. Document results

**Next Week (if needed):**
9. Hyperparameter tuning
10. Additional analysis
11. Paper draft integration

---

**Ready to start Phase 4! Let's get the quantitative results for the paper.** 🚀
