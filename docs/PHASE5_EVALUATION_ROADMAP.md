# Phase 5 Evaluation Roadmap: Beyond Perplexity

**Date:** 2025-11-18
**Status:** Planning
**Goal:** Prove DRAI provides measurable cognitive benefits beyond zero-cost integration

## Executive Summary

Phase 4 validated that DRAI adds zero performance degradation. **Phase 5 must prove it actually helps.** This document outlines experiments that exploit DRAI's unique properties:

1. **Long-horizon narrative/dialogue coherence** - "Can this thing keep track of what matters over time better than baseline/RAG?"
2. **Selective, local memory updates (attractor locality)** - "Does DRAI actually update only near-related concepts rather than smearing weights everywhere?"

**Key Insight:** We need tasks where DRAI's intrinsic, self-organizing memory should *theoretically* outperform both baseline (no memory) and RAG (external, discrete retrieval).

---

## 1. Core Hypotheses to Test

### Already Validated ✓
- ✅ **Zero-cost integration** - DRAI doesn't degrade perplexity
- ✅ **Attractor dynamics work** - Formation, reinforcement, decay all active
- ✅ **Scales across model sizes** - pythia-70m and pythia-125m both work

### To Validate (Phase 5+)
- ❓ **Reasoning benefits** - Does DRAI improve multi-step reasoning?
- ❓ **Contextual recall** - Better long-horizon memory than baseline?
- ❓ **Locality of adaptation** - Updates near-related concepts selectively?
- ❓ **Unique causal levers** - Can we manipulate resonance field to control behavior?
- ❓ **Meta-cognition** - Uncertainty awareness, subjective agency (Phase 6+)

---

## 2. Concrete Experiments

### 2.1 Targeted Cognitive Tasks

#### A. Long-Horizon Story / Dialogue Consistency

**Setup:**
Synthetic task with 1-2k token stories containing:
- **Characters:** A, B, C with attributes and relationships
- **Timelines:** Early events referenced later
- **Planted facts:** "Alice's brother died in X", "The key code is 4932"
- **Distractor arcs:** Later sections with conflicting facts

**At the end, ask:**
- Factual questions: "What was the original code?"
- Consistency questions: "Is Bob still alive at the end?"
- Resolution questions: "Why did X happen?"

**Conditions:**
1. **Baseline** - Pythia-125m
2. **DRAI** - Pythia-125m + DRAI resonance cortex
3. **RAG** - Pythia-125m + vector DB of the story

**Metrics:**
- **Accuracy** on factual questions (%)
- **Consistency score** - Contradictions / 10 (human or rubric-based)
- **Memory robustness** - Performance when inserting 1/2/3 irrelevant subplots

**Expected DRAI Advantage:**
- Similar perplexity
- Higher consistency/recall, especially with heavy distractors
- Less "drift" in late answers

**Implementation Plan:**
```python
# experiments/evaluation/long_story_consistency.py
def generate_story(num_facts=10, num_distractors=3):
    """Generate synthetic story with planted facts and distractors."""

def evaluate_consistency(model, stories, questions):
    """Evaluate factual accuracy and consistency."""

def compare_models(baseline, drai, rag=None):
    """Compare performance across conditions."""
```

---

#### B. Multi-Step Reasoning with Interruptions

**Task:**
Give model a multi-step reasoning problem (planning, math-ish puzzle).
- Between steps, inject irrelevant chatter ("BTW here's a random paragraph about cats/news")
- At the end, ask it to summarize the final plan/answer

**Measure:**
- How often does it stick to its own earlier reasoning vs get pulled off-track?
- Does DRAI preserve intermediate attractors for the main thread better?

**Example:**
```
Step 1: "To solve this, we first need to calculate X..."
[DISTRACTOR: 100 tokens about cats]
Step 2: "Next, we take X and apply Y..."
[DISTRACTOR: 100 tokens about weather]
Step 3: "Finally, combine to get Z..."
Question: "What was the final answer?"
```

**Expected DRAI Advantage:**
- Attractors should maintain "main thread" state across distractors
- Baseline should be more susceptible to topic drift

---

### 2.2 Exploit the Resonance Field (Causal Interventions)

**This is where we prove "it's not just an extra MLP."**

#### C. Resonance Lesioning Experiment

For a given task (e.g. long story consistency), run 3 variants of the same DRAI model:

1. **Full DRAI** - Normal attractor updates
2. **Lesioned** - Zero out `attractor_field` every N tokens or before answering
3. **Scrambled** - Randomly permute attractors before answering

**Prediction:**
If performance drops significantly in (2)/(3) while perplexity stays similar, that's strong evidence that:
- The attractors are functionally relevant
- Not just dead weights

**Implementation:**
```python
# In DraiResonanceLayer, add lesion modes:
class DraiResonanceLayer:
    def __init__(self, ..., lesion_mode=None):
        self.lesion_mode = lesion_mode  # None, "zero", "scramble"

    def forward(self, Q, ...):
        if self.lesion_mode == "zero":
            attractor_field = torch.zeros_like(self.attractor_field)
        elif self.lesion_mode == "scramble":
            attractor_field = self.attractor_field[torch.randperm(self.max_attractors)]
        else:
            attractor_field = self.attractor_field
        # ... continue with lesioned field
```

---

#### D. Attractor "Priming" Experiment

**Setup:**
- Before main story, feed a short "priming" snippet that creates attractors in a specific subspace (e.g. about space travel)
- Then give an ambiguous story where the model could interpret things in multiple ways
- See if the primed DRAI model is biased toward interpretations consistent with the primed attractors more than baseline

**Example:**
```
Priming: "Space travel is dangerous. Oxygen is critical. Mars missions fail often."
[Creates attractors in "space/danger" subspace]

Ambiguous story: "John's expedition failed. The team didn't make it back."
Question: "What kind of expedition was it?"

DRAI (primed): More likely to say "space expedition"
Baseline: No bias from priming
```

**This is a nice "causal bias via resonance" test.**

---

### 2.3 Compare Against Vector-DB RAG

For the same long context tasks:
- **Baseline** - No memory
- **RAG** - Baseline + retrieve top-k relevant chunks from vector DB
- **DRAI** - Intrinsic resonance memory
- **DRAI + RAG** (optional) - Hybrid approach

**Claims to test:**
RAG does okay on strict fact recall when retrieval works, but:
- ❌ Brittle under paraphrase / slight misalignment
- ❌ Can feel "cut-and-paste" / incoherent globally
- ❌ Discrete retrieval vs continuous integration

DRAI:
- May not beat RAG on raw recall percentage initially
- But shows:
  - ✅ Better narrative coherence
  - ✅ Smoother integration of old and new info
  - ✅ Less contradiction between "what was said earlier" and "what's said now"

**If we get parity on recall but better coherence, that's a big win:**
> "Internal resonance vs external patchwork."

---

### 2.4 Visualizations & Diagnostics

**Pre-define what to log from DRAI so plots aren't random art.**

For each token sequence, log:
- **Attractor strengths** over time
- **Attractor activations** - Which attractors matched each token
- **Entropy** of the attractor distribution
- **Norm and cosine distance** between successive `attractor_field` snapshots

**Visuals:**

1. **Heatmap:** tokens (x) vs attractors (y), intensity = match strength
   - Want to see bands forming for coherent themes/characters

2. **Strength time-series:** Track top-k attractors across a story
   - Do certain attractors "carry" through?
   - Does their strength spike in relevant sections?

3. **t-SNE / UMAP** of attractor vectors
   - Color by topic/concept, see if they cluster meaningfully

**If, during long-horizon tasks, you see:**
- Stable attractor traces corresponding to key story entities
- Those traces predict which facts the model recalls correctly

**…that's the "forests of meaning" story becoming concrete.**

---

### 2.5 Human-in-the-Loop

**This doesn't need to be huge; even N=10-20 people gives nice qualitative evidence.**

**Protocol:**
- Show people pairs of answers (baseline vs DRAI, anonymized + shuffled)
- Ask them to rate on:
  - Coherence
  - Consistency with earlier context
  - Perceived "thoughtfulness" or "stability"

**Track:**
- **Preference rates** - "Which feels like it 'remembers' better?"
- **Free-text comments** - These are gold for the paper

**Example survey item:**
```
Story: [1000 tokens with character Alice who loves cats]
Question: "What does Alice care about?"

Answer A: "Alice cares about her garden."
Answer B: "Alice cares about cats."

Which answer is more consistent with the story?
Which answer feels like the model "understood" better?
```

---

### 2.6 Ablation & Robustness

A lot of this overlaps with 2.2 (C/D), but formally:

**Ablations:**
- Turn off DRAI (lesion) ⇒ performance on long-horizon tasks drops
- Randomize attractors ⇒ performance worse than trained DRAI but similar perplexity

**Stress tests:**
- Add more distractors (1 → 2 → 3 → 5)
- Larger context (1k → 2k → 4k tokens)
- Slightly adversarial perturbations (rewording, reordering)

**We want curves where:**
- Baseline collapses earlier
- DRAI maintains performance longer as difficulty ramps up

**Example metric:**
```
Accuracy vs Number of Distractors:
Baseline:  90% → 75% → 60% → 40%
DRAI:      90% → 85% → 80% → 70%
```

---

## 3. Instrumentation: What to Add to DraiResonanceLayer

**Add a simple logging hook (even just to a dict / JSONL):**

**Per sequence:**
- `attractor_vectors` snapshot at key steps
- `attractor_strengths` at each step
- Token-wise `best_match_index` and `match_score`

**Aggregate metrics:**
- Attractor entropy
- Proportion of tokens that matched any attractor (vs novel)
- Mean similarity of query→attractor over time

**Implementation:**
```python
class DraiResonanceLayer:
    def __init__(self, ..., enable_logging=False):
        self.enable_logging = enable_logging
        self._log = [] if enable_logging else None

    def forward(self, Q, ...):
        # ... existing logic ...

        if self.enable_logging:
            self._log.append({
                'step': self._forward_count.item(),
                'best_match_idx': best_match_idx.cpu().numpy(),
                'match_scores': similarities.cpu().numpy(),
                'attractor_strengths': self.attractor_coherence.cpu().numpy(),
                'attractor_entropy': -torch.sum(
                    match_probs * torch.log(match_probs + 1e-8)
                ).item(),
            })

    def get_logs(self):
        return self._log

    def clear_logs(self):
        self._log = []
```

**These are then used in:**
- Plots (heatmaps, time-series)
- Ablation analysis (e.g. "when attractor entropy is low, consistency is high")

---

## 4. Minimal Phase 5 Evaluation Plan (Doable Starting Block)

**If you want a doable starting block for Phase 5:**

### Task 1: Long Story Consistency

**Goal:** Prove DRAI helps with long-horizon memory

**Dataset:**
- Synthetic 1-2k token stories with facts + distractors
- 50-100 stories with varying complexity

**Conditions:**
- Baseline vs DRAI vs RAG

**Metrics:**
- Factual accuracy (%)
- Contradiction rate (contradictions per story)
- Human coherence rating (1-5 scale)

**Success Criteria:**
- DRAI ≥ baseline on accuracy
- DRAI > baseline on consistency
- DRAI ≈ RAG on recall, > RAG on coherence

---

### Task 2: Resonance Lesion

**Goal:** Prove attractors are causally important

**Dataset:**
- Same stories from Task 1

**Conditions:**
- DRAI (full) vs DRAI (zeroed) vs DRAI (scrambled)

**Metrics:**
- Same as Task 1

**Success Criteria:**
- Performance drops when attractors are lesioned/scrambled
- Perplexity remains similar (proving it's not just model degradation)

---

### Task 3: Instrumentation & Visualization

**Goal:** Make DRAI interpretable

**Deliverables:**
- At least 1 heatmap of tokens vs attractors
- At least 1 time-series of top-k attractors over a story
- At least 1 illustrative example: "This attractor tracked character X"

**Success Criteria:**
- Visualizations clearly show attractor dynamics
- Can point to specific attractors and say "this one tracked concept X"
- Attractor strength correlates with task performance

---

## 5. Expected Outcomes

**Minimal success (publishable):**
- DRAI maintains zero-cost integration on new tasks
- Shows *some* advantage on long-horizon consistency
- Lesioning proves attractors are functionally relevant
- Visualizations show interpretable attractor dynamics

**Strong success (high-impact paper):**
- DRAI significantly outperforms baseline on consistency/coherence
- Matches or beats RAG on recall, exceeds on coherence
- Clear causal evidence from lesioning/priming
- Beautiful visualizations showing "forests of meaning"
- Human evaluators prefer DRAI outputs

**Moonshot (top-tier venue):**
- All of above +
- Scales to larger models (pythia-410m, 1B)
- Works on real-world tasks (not just synthetic)
- Theoretical analysis of why DRAI helps
- Opens new research direction in intrinsic memory

---

## 6. Implementation Roadmap

### Phase 5.1: Infrastructure (1-2 weeks)
- [ ] Add logging to DraiResonanceLayer
- [ ] Create synthetic story generator
- [ ] Build evaluation harness for consistency tasks
- [ ] Set up visualization pipeline

### Phase 5.2: Core Experiments (2-3 weeks)
- [ ] Task 1: Long story consistency (baseline vs DRAI vs RAG)
- [ ] Task 2: Resonance lesioning
- [ ] Task 3: Visualizations

### Phase 5.3: Analysis & Writing (1-2 weeks)
- [ ] Statistical analysis of results
- [ ] Create publication-quality figures
- [ ] Draft paper sections (methods, results, discussion)

### Phase 5.4: Scaling & Robustness (1-2 weeks)
- [ ] Test on larger models (pythia-410m)
- [ ] Stress tests (more distractors, longer context)
- [ ] Human evaluation (N=10-20)

**Total estimated time:** 5-9 weeks for comprehensive Phase 5 evaluation

---

## 7. Key Narrative for Paper

> **"We added a resonance cortex that cost nothing in perplexity but gave measurable, causal, interpretable gains on long-horizon memory and coherence."**

**Three-part story:**

1. **Zero-cost foundation (Phase 4):** DRAI integrates without degrading performance
2. **Functional benefits (Phase 5):** DRAI improves long-horizon memory and coherence
3. **Causal mechanism (Phase 5):** Lesioning proves attractors are doing the work

**This is enough for:**
- A strong arXiv paper
- Conference submission (e.g. NeurIPS, ICML, ICLR)
- A very convincing README "Evidence it matters" section

---

## 8. Open Questions

**Before starting Phase 5:**

1. **What model size?** pythia-125m is good for iteration, but may need pythia-410m for convincing results
2. **Synthetic vs real tasks?** Synthetic is easier to control, but real tasks are more convincing
3. **How many samples?** Balance between statistical power and compute budget
4. **Human evaluation scope?** N=10 is doable, N=100 is expensive but stronger

**During Phase 5:**

1. **If DRAI doesn't win on first task?** Try different tasks, or accept "no degradation" as the win
2. **How to present null results?** Be honest about limitations, frame as proof-of-concept
3. **RAG baseline design?** What retrieval method? How many chunks? Needs careful design

---

## Next Steps

1. **Decide:** Start Phase 5 now, or write up Phase 4 results first?
2. **If starting Phase 5:** Begin with instrumentation (logging in DraiResonanceLayer)
3. **If writing first:** Draft paper with Phase 4 results, add Phase 5 as "future work"

**Recommendation:** Add instrumentation now (it's non-invasive), then decide whether to run experiments or write first.

---

**Status:** Roadmap complete, ready for Phase 5 implementation

**Next:** Implement logging infrastructure and build first evaluation task
