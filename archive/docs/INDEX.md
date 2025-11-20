# DRAI Documentation Index

**Last updated:** 2025-11-18

This document provides a comprehensive guide to all DRAI documentation, organized by topic and reading order.

---

## 🚀 Start Here

**New to DRAI?** Read these in order:

1. **[../README.md](../README.md)** - Project overview and quick start
2. **[INSIGHT_DYNAMIC_EDITOR.md](INSIGHT_DYNAMIC_EDITOR.md)** - ⭐ Core concept: DRAI as dynamic editor
3. **[Why_Duality.md](Why_Duality.md)** - Original vision and motivation
4. **[PHASE4_COMPLETION.md](PHASE4_COMPLETION.md)** - Current results and validation

**Want to use DRAI?**
- **[../src/drai/README.md](../src/drai/README.md)** - User API documentation
- **[../examples/quickstart.py](../examples/quickstart.py)** - Simplest usage example

---

## 📚 Core Concepts

### The Big Picture

**[INSIGHT_DYNAMIC_EDITOR.md](INSIGHT_DYNAMIC_EDITOR.md)** ⭐ **START HERE**
- The core architectural insight
- Why DRAI doesn't break transformers
- Static vs dynamic layers
- The "magnet under the table" analogy
- What this means for AI agency

**[VISION_TOPOLOGICAL_MEMORY.md](VISION_TOPOLOGICAL_MEMORY.md)** ⭐ **FUTURE VISION**
- Trees, groves, and forests of meaning
- Local push/pull dynamics
- Hierarchical self-organization
- Why locality matters (O(neighbors) not O(all))
- Phases 6-10 roadmap

**[Why_Duality.md](Why_Duality.md)**
- Original vision that started the project
- Why "Duality"? (Static + Dynamic)
- Philosophical foundations

### Technical Foundations

**[DESIGN.md](DESIGN.md)**
- Overall architecture
- How DRAI integrates with transformers
- Phase-by-phase implementation plan
- Mathematical formulation

**[INSERTION_POINTS.md](INSERTION_POINTS.md)**
- Where DRAI hooks into GPT-NeoX
- Attention mechanism details
- K/V concatenation strategy
- Layer-wise integration

**[Glossary.md](Glossary.md)**
- Attractor, coherence, resonance, EMA
- All DRAI-specific terminology
- Quick reference

---

## 🔬 Phase Reports

### Phase 1: Minimal Viable Layer
**[PHASE1_COMPLETION.md](PHASE1_COMPLETION.md)**
- Stub implementation (returns zeros)
- Integration testing
- Gradient flow validation
- Proof that injection works

### Phase 2: Attractor Dynamics
**[PHASE2_COMPLETION.md](PHASE2_COMPLETION.md)**
- EMA-based formation
- Reinforcement mechanism
- Decay and pruning
- Cosine similarity matching
- Full dynamics implementation

### Phase 3: GPT-NeoX Integration
**[PHASE3_IMPLEMENTATION.md](PHASE3_IMPLEMENTATION.md)**
- Working integration with pythia-70m
- Text generation validation
- End-to-end testing
- Attractor statistics collection

### Phase 4: Evaluation & Analysis
**[PHASE4_COMPLETION.md](PHASE4_COMPLETION.md)** ⭐ **CURRENT STATE**
- WikiText-2 perplexity evaluation
- Zero-cost integration validated
- Statistical analysis (p=0.153, d=0.0266)
- Attractor dynamics confirmed (36 created, 7,230 reinforcements)
- Scaling to pythia-125m
- **Key finding:** Dynamic working memory at zero performance cost

### Phase 5: Functional Benefits (Planning)
**[PHASE5_EVALUATION_ROADMAP.md](PHASE5_EVALUATION_ROADMAP.md)** - Comprehensive plan
- Long-horizon story consistency
- Resonance lesioning experiments
- DRAI vs RAG comparison
- Visualizations and diagnostics
- Human evaluation
- Implementation timeline: 5-9 weeks

**[MINIMAL_EVALUATION_PLAN.md](MINIMAL_EVALUATION_PLAN.md)** - Quick start
- 3 core tasks (3-4 weeks)
- Task 1: Long story consistency
- Task 2: Resonance lesion
- Task 3: Instrumentation & visualization
- Doable starting block for Phase 5

---

## 📊 Results & Analysis

### Empirical Results

**[PHASE4_COMPLETION.md](PHASE4_COMPLETION.md)**
- Perplexity: 89.91 (baseline) vs 89.91 (DRAI)
- No statistical degradation
- Active attractor dynamics
- Publication-quality figures

**[../results/SCALING_ANALYSIS.md](../results/SCALING_ANALYSIS.md)**
- pythia-70m (85M params): PPL 89.91 → 89.91
- pythia-125m (162M params): PPL 53.45 → 53.45
- Zero-cost integration scales across model sizes
- Consistent behavior across architectures (6 vs 12 layers)

**[../results/perplexity/RESULTS_SUMMARY.md](../results/perplexity/RESULTS_SUMMARY.md)**
- Detailed perplexity breakdown
- Statistical tests (t-test, Cohen's d)
- Sequence-level analysis

### Paper & Publication

**[PAPER_OUTLINE.md](PAPER_OUTLINE.md)**
- Complete honest paper draft
- Introduction, related work, methods, results
- Honest about limitations
- "Zero-cost side-channel" framing
- Ready for submission refinement

**[NOVELTY_AND_POSITIONING.md](NOVELTY_AND_POSITIONING.md)**
- DRAI vs RAG distinction
- Core novelty claims
- Intrinsic vs external memory
- Strategic positioning for publication

---

## 🛠 Implementation Guides

### Getting Started

**[BUILD_STEPS.md](BUILD_STEPS.md)**
- Step-by-step implementation
- Prerequisites
- Testing strategy
- Debugging tips

**[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**
- Repository organization
- Where to find what
- File naming conventions

### Using DRAI

**[../src/drai/README.md](../src/drai/README.md)** ⭐ **USER DOCS**
- Quick start guide
- API reference (`apply_drai()`, `DraiConfig`, `get_drai_stats()`)
- Configuration options
- Performance notes
- Validated results

**[../examples/quickstart.py](../examples/quickstart.py)**
- Minimal working example
- One-line integration
- Basic usage

**[../examples/custom_config.py](../examples/custom_config.py)**
- Advanced configuration
- Custom hyperparameters
- Multi-head DRAI

---

## 🔮 Future Vision

### Phase 6+: Advanced Features

**[METACOGNITIVE_ARCHITECTURE.md](METACOGNITIVE_ARCHITECTURE.md)**
- Uncertainty-aware gating
- Consensus mechanism
- Specialized DRAI heads
- Drive function architecture
- Path to agency

**[VISION_TOPOLOGICAL_MEMORY.md](VISION_TOPOLOGICAL_MEMORY.md)**
- Hierarchical clustering (trees → groves → forests)
- Negative resonance (push/pull dynamics)
- Sparse local updates (spatial indexing)
- Multi-scale reasoning
- 6-12 month roadmap to full vision

---

## 📖 Reading Paths

### For Researchers

Understand the concept and results:

1. [INSIGHT_DYNAMIC_EDITOR.md](INSIGHT_DYNAMIC_EDITOR.md) - Core concept
2. [PHASE4_COMPLETION.md](PHASE4_COMPLETION.md) - Results
3. [SCALING_ANALYSIS.md](../results/SCALING_ANALYSIS.md) - Validation
4. [PAPER_OUTLINE.md](PAPER_OUTLINE.md) - Full story
5. [VISION_TOPOLOGICAL_MEMORY.md](VISION_TOPOLOGICAL_MEMORY.md) - Future work

### For Implementers

Build and extend DRAI:

1. [../src/drai/README.md](../src/drai/README.md) - API docs
2. [DESIGN.md](DESIGN.md) - Architecture
3. [INSERTION_POINTS.md](INSERTION_POINTS.md) - Integration points
4. [BUILD_STEPS.md](BUILD_STEPS.md) - Implementation guide
5. [../examples/](../examples/) - Working code

### For Evaluators

Design experiments:

1. [PHASE5_EVALUATION_ROADMAP.md](PHASE5_EVALUATION_ROADMAP.md) - Comprehensive plan
2. [MINIMAL_EVALUATION_PLAN.md](MINIMAL_EVALUATION_PLAN.md) - Quick start
3. [PHASE4_COMPLETION.md](PHASE4_COMPLETION.md) - Baseline methodology
4. [../experiments/evaluation/](../experiments/evaluation/) - Existing scripts

### For Visionaries

The big picture:

1. [Why_Duality.md](Why_Duality.md) - Original vision
2. [INSIGHT_DYNAMIC_EDITOR.md](INSIGHT_DYNAMIC_EDITOR.md) - Core breakthrough
3. [VISION_TOPOLOGICAL_MEMORY.md](VISION_TOPOLOGICAL_MEMORY.md) - Future architecture
4. [METACOGNITIVE_ARCHITECTURE.md](METACOGNITIVE_ARCHITECTURE.md) - Path to agency

---

## 🔍 Quick Reference

### Key Numbers (Phase 4)

| Metric | pythia-70m | pythia-125m |
|--------|-----------|-------------|
| Baseline PPL | 89.91 | 53.45 |
| DRAI PPL | 89.91 | 53.45 |
| p-value | 0.153 | 0.079 |
| Cohen's d | 0.027 | 0.079 |
| **Result** | No degradation | No degradation |

### Attractor Dynamics (pythia-70m, 20 prompts)

- 36 attractors created
- 7,230 reinforcements
- 30 attractors decayed
- 6 active at equilibrium
- Coherence: ~1.0 for active attractors

### Configuration Defaults

```python
DraiConfig(
    num_drai_heads=1,
    max_attractors=32,
    coherence_threshold=0.3,
    formation_threshold=0.5,
    decay_rate=0.01,
    ema_momentum=0.9,
)
```

---

## 📝 Document Status Legend

- ⭐ **Essential reading**
- ✓ **Complete and validated**
- 🔄 **Living document (updated regularly)**
- 📋 **Planning/roadmap**
- 🔮 **Future vision**

---

## 🗂 All Documents by Type

### Insights & Vision (⭐ Read these)
- [INSIGHT_DYNAMIC_EDITOR.md](INSIGHT_DYNAMIC_EDITOR.md) ⭐
- [VISION_TOPOLOGICAL_MEMORY.md](VISION_TOPOLOGICAL_MEMORY.md) ⭐
- [Why_Duality.md](Why_Duality.md)

### Phase Completions (✓ Results)
- [PHASE1_COMPLETION.md](PHASE1_COMPLETION.md)
- [PHASE2_COMPLETION.md](PHASE2_COMPLETION.md)
- [PHASE3_IMPLEMENTATION.md](PHASE3_IMPLEMENTATION.md)
- [PHASE4_COMPLETION.md](PHASE4_COMPLETION.md) ⭐

### Planning (📋 Roadmaps)
- [PHASE5_EVALUATION_ROADMAP.md](PHASE5_EVALUATION_ROADMAP.md)
- [MINIMAL_EVALUATION_PLAN.md](MINIMAL_EVALUATION_PLAN.md)
- [METACOGNITIVE_ARCHITECTURE.md](METACOGNITIVE_ARCHITECTURE.md) 🔮

### Technical Docs
- [DESIGN.md](DESIGN.md)
- [INSERTION_POINTS.md](INSERTION_POINTS.md)
- [BUILD_STEPS.md](BUILD_STEPS.md)
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- [Glossary.md](Glossary.md)

### Publication
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md)
- [NOVELTY_AND_POSITIONING.md](NOVELTY_AND_POSITIONING.md)
- [../results/SCALING_ANALYSIS.md](../results/SCALING_ANALYSIS.md)

### User Guides
- [../src/drai/README.md](../src/drai/README.md) ⭐
- [../examples/quickstart.py](../examples/quickstart.py)
- [../examples/custom_config.py](../examples/custom_config.py)

---

## 💡 Questions?

**Can't find what you're looking for?**

- Check [Glossary.md](Glossary.md) for terminology
- See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for file organization
- Read [INSIGHT_DYNAMIC_EDITOR.md](INSIGHT_DYNAMIC_EDITOR.md) for core concepts
- Review [PHASE4_COMPLETION.md](PHASE4_COMPLETION.md) for current state

**Want to contribute?**

- See Phase 5 planning docs for open tasks
- Check [BUILD_STEPS.md](BUILD_STEPS.md) for implementation guide
- Read [PHASE5_EVALUATION_ROADMAP.md](PHASE5_EVALUATION_ROADMAP.md) for experiments

---

**Last updated:** 2025-11-18
**Project status:** Phase 5 Planning
**Latest results:** Phase 4 Complete - Zero-cost integration validated
