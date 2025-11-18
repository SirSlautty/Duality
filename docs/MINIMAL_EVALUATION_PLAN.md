# Minimal Phase 5 Evaluation Plan

**Goal:** Prove DRAI provides measurable benefits beyond zero-cost integration

**Timeline:** 3-4 weeks for core results

---

## Three Tasks (Doable Starting Block)

### Task 1: Long Story Consistency

**What:** Can DRAI track facts and characters better than baseline over 1-2k token stories?

**Setup:**
- 50-100 synthetic stories with:
  - Planted facts ("The code is 4932")
  - Characters with attributes
  - Distractor subplots
- Questions at the end testing recall and consistency

**Compare:**
- Baseline (pythia-125m)
- DRAI (pythia-125m + resonance)
- RAG (pythia-125m + vector retrieval)

**Metrics:**
- Factual accuracy (%)
- Consistency score (contradictions per story)
- Robustness (performance vs number of distractors)

**Success:** DRAI ≥ baseline on accuracy, > baseline on consistency

---

### Task 2: Resonance Lesion

**What:** Prove attractors are causally important (not just dead weights)

**Setup:**
- Same stories as Task 1
- Three DRAI variants:
  1. Full DRAI (normal)
  2. Zeroed attractors (lesioned)
  3. Scrambled attractors (randomized)

**Metrics:**
- Same as Task 1
- Plus: perplexity (should stay similar)

**Success:** Performance drops when attractors are lesioned/scrambled, but perplexity unchanged

**This proves:** Attractors are functionally relevant, not just noise

---

### Task 3: Instrumentation & Visualization

**What:** Make DRAI interpretable and visually compelling

**Deliverables:**
1. **Heatmap:** tokens (x) vs attractors (y), showing which attractors activate for which tokens
2. **Time-series:** Track top-k attractors across a story, show they persist for relevant concepts
3. **Illustrative example:** "This attractor tracked character Alice through the story"

**Success:** Clear visual evidence that attractors correspond to meaningful concepts

---

## Implementation Checklist

### Week 1: Infrastructure
- [ ] Add logging to `DraiResonanceLayer` (attractor activations, strengths, entropy)
- [ ] Create synthetic story generator with planted facts
- [ ] Build evaluation harness (load models, run inference, score answers)

### Week 2: Experiments
- [ ] Run Task 1 (baseline vs DRAI vs RAG)
- [ ] Run Task 2 (lesioning experiment)
- [ ] Collect data for visualizations

### Week 3: Analysis
- [ ] Statistical analysis (t-tests, effect sizes)
- [ ] Create visualizations (heatmaps, time-series, examples)
- [ ] Draft results section

### Week 4: Polish
- [ ] Human evaluation (N=10-20 if time permits)
- [ ] Final figures and tables
- [ ] Write up findings

---

## Expected Results

**Conservative (still publishable):**
- DRAI ≈ baseline on most metrics
- Small but significant advantage on consistency
- Lesioning shows attractors matter
- Visualizations show interpretable dynamics

**Optimistic (strong paper):**
- DRAI > baseline on consistency (10-20% improvement)
- DRAI ≈ RAG on recall, > RAG on coherence
- Clear causal evidence from lesioning
- Beautiful visualizations showing concept tracking

---

## Key Narrative

> "We added a resonance cortex that costs nothing in perplexity (Phase 4) but provides measurable, causal, interpretable gains on long-horizon memory and coherence (Phase 5)."

**This is enough for:**
- ArXiv preprint
- Conference submission
- Strong evidence section in README

---

## Files to Create

```
experiments/evaluation/
  ├── long_story_consistency.py   # Task 1
  ├── lesion_experiment.py        # Task 2
  └── generate_stories.py         # Synthetic data

experiments/visualization/
  ├── plot_attractor_heatmap.py   # Token×attractor heatmap
  ├── plot_time_series.py         # Attractor strength over time
  └── plot_concept_tracking.py    # Illustrative examples

results/phase5/
  ├── task1_consistency/
  ├── task2_lesion/
  └── visualizations/
```

---

## Open Questions to Resolve

1. **Model size:** pythia-125m (fast iteration) or pythia-410m (more convincing)?
2. **Number of stories:** 50 (quick proof) or 200 (stronger statistics)?
3. **RAG baseline:** Use simple BM25 or semantic retrieval?
4. **Human eval:** Skip for now, or include N=10 for qualitative evidence?

**Recommendation:** Start with pythia-125m, 50-100 stories, simple RAG, skip human eval initially. Can scale up if initial results are promising.

---

**Next:** Implement logging infrastructure in `DraiResonanceLayer`

**Then:** Build synthetic story generator

**Status:** Plan complete, ready to start implementation
