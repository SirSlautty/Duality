# Duality  
  
Welcome to **Project Duality** – an experimental research repository exploring hybrid cognitive architectures. The goal is to augment an open‑source transformer model with a second “resonance cortex” based on the Dynamic Resonance AI (DRAI) principles we’ve been discussing.  
  
## Vision

**DRAI is not a retrieval system. It's a dynamic editor of static knowledge.**

By layering attractor-based working memory on top of transformer long-term memory, DRAI adds temporal coherence and state maintenance without modifying learned weights. This creates a hybrid architecture:

- **Static layer (Transformer):** Permanent semantic knowledge learned via backprop
- **Dynamic layer (DRAI):** Temporary emphasis and continuity via attractor dynamics
- **Together:** A proto-agent with both stable semantics and working memory

> "You're not rewriting the learned landscape. You're tilting the energy flow inside it."

This is the foundation for coherent, agent-like behavior in language models.  
  
## Objectives  
  
- Prototype a simple hook into an open transformer (e.g. GPT‑NeoX) that routes a head’s Q/K/V through a resonance module.  
- Implement the resonance accumulator that detects recurring latent vectors and forms stable attractors.  
- Inject the attractor outputs back into the attention mixing as synthetic K/V pairs.  
- Visualise the evolution of the resonance manifold as new concepts are stabilised.  
- Document findings, pitfalls, and emergent behaviours along the way.  
  
## Project Status

**Current Phase:** Phase 5 - INVESTIGATION (Generation Quality Issue Identified)
**Environment:** ✓ PyTorch 2.9.1 + transformers 4.57.1

**Completed Phases:**
- **Phase 1:** ✓ Minimal viable DRAI layer
- **Phase 2:** ✓ Full attractor dynamics (EMA, reinforcement, decay, pruning)
- **Phase 3:** ✓ GPT-NeoX integration - WORKING!
- **Phase 4:** ✓ Evaluation & analysis - VALIDATED!

**Key Results (Phase 4):**
- **Zero-cost integration:** Perplexity 89.91 (baseline) vs 89.91 (DRAI) across pythia-70m and pythia-125m
- **Active dynamics:** 36 attractors created, 7,230 reinforcements, 30 decayed - equilibrium reached
- **Statistical validation:** p=0.153 (no significant degradation), Cohen's d=0.0266 (negligible effect)
- **Scalability confirmed:** Consistent zero-cost integration from 85M to 162M parameters
- **Core finding:** DRAI adds dynamic working memory as a zero-cost side-channel to static semantics

**Next: Phase 5 (Functional Benefits)**
- Prove DRAI improves long-horizon memory and coherence
- Test tasks requiring working memory (DRAI's designed strength)
- Demonstrate causal importance via lesioning experiments

**Test Results:**
- Unit tests: 52/52 passing (100%)
- Integration: ✓ Text generation validated
- Attractors: ✓ Forming, reinforcing, and pruning during inference
- Stability: ✓ No NaN/Inf, no crashes
- Performance: ✓ Maintains baseline perplexity

**Phase Reports:**
- [`docs/PHASE1_COMPLETION.md`](docs/PHASE1_COMPLETION.md) - Phase 1: Minimal DRAI implementation
- [`docs/PHASE2_COMPLETION.md`](docs/PHASE2_COMPLETION.md) - Phase 2: Attractor dynamics
- [`docs/PHASE3_IMPLEMENTATION.md`](docs/PHASE3_IMPLEMENTATION.md) - Phase 3: Transformer integration
- [`docs/PHASE4_COMPLETION.md`](docs/PHASE4_COMPLETION.md) - Phase 4: Evaluation & analysis
- [`docs/PHASE5_EVALUATION_ROADMAP.md`](docs/PHASE5_EVALUATION_ROADMAP.md) - Phase 5: Functional benefits plan
- [`docs/MINIMAL_EVALUATION_PLAN.md`](docs/MINIMAL_EVALUATION_PLAN.md) - Phase 5: Quick-start 3-4 week plan
- **[`results/phase5/PHASE5_FINDINGS_FINAL.md`](results/phase5/PHASE5_FINDINGS_FINAL.md) - Phase 5: Pilot results & generation issue**

**Core Insights:**
- [`docs/INSIGHT_DYNAMIC_EDITOR.md`](docs/INSIGHT_DYNAMIC_EDITOR.md) - **DRAI as dynamic editor of static knowledge**
- [`docs/VISION_TOPOLOGICAL_MEMORY.md`](docs/VISION_TOPOLOGICAL_MEMORY.md) - **Trees, groves, forests: Hierarchical memory vision**
- [`results/SCALING_ANALYSIS.md`](results/SCALING_ANALYSIS.md) - Scaling from pythia-70m to pythia-125m

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HalcyonAIR/Duality.git
cd Duality

# Install in editable mode
pip install -e .

# For development (includes pytest, black, etc.)
pip install -e ".[dev]"
```

### Basic Usage

```python
from drai import apply_drai

# Apply DRAI to any GPT-NeoX model (one line!)
model = apply_drai("EleutherAI/pythia-70m")

# Use normally
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(inputs['input_ids'], max_length=50)

print(tokenizer.decode(outputs[0]))
```

**That's it!** Your model now has self-organizing working memory.

See [`src/drai/README.md`](src/drai/README.md) for complete API documentation and [`examples/`](examples/) for more examples.

### Repository Structure

```
Duality/
├── src/               # DRAI implementation
│   ├── drai/         # Resonance layer core
│   └── models/       # Transformer integration
├── tests/            # Test suite (unit, integration, environment)
├── experiments/      # Notebooks and scripts
├── models/           # External model repositories
└── docs/             # Design docs and specifications
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for detailed documentation.

## Key Documentation

### Core Concepts
- **[INSIGHT_DYNAMIC_EDITOR.md](docs/INSIGHT_DYNAMIC_EDITOR.md)** - ⭐ **Core insight: DRAI as dynamic editor**
- **[VISION_TOPOLOGICAL_MEMORY.md](docs/VISION_TOPOLOGICAL_MEMORY.md)** - ⭐ **Future vision: Trees, groves, forests**
- **[Why_Duality.md](docs/Why_Duality.md)** - Original vision and philosophy
- **[Glossary.md](docs/Glossary.md)** - Terminology and concepts

### Implementation
- **[DESIGN.md](docs/DESIGN.md)** - Architecture and theory
- **[INSERTION_POINTS.md](docs/INSERTION_POINTS.md)** - Technical implementation strategy
- **[BUILD_STEPS.md](docs/BUILD_STEPS.md)** - Step-by-step build guide
- **[src/drai/README.md](src/drai/README.md)** - User API documentation

### Results & Analysis
- **[PHASE4_COMPLETION.md](docs/PHASE4_COMPLETION.md)** - Evaluation results (Phase 4)
- **[SCALING_ANALYSIS.md](results/SCALING_ANALYSIS.md)** - pythia-70m to pythia-125m scaling
- **[PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md)** - Honest paper draft

### Future Phases
- **[PHASE5_EVALUATION_ROADMAP.md](docs/PHASE5_EVALUATION_ROADMAP.md)** - Comprehensive Phase 5 plan
- **[MINIMAL_EVALUATION_PLAN.md](docs/MINIMAL_EVALUATION_PLAN.md)** - Quick-start 3-4 week plan
- **[METACOGNITIVE_ARCHITECTURE.md](docs/METACOGNITIVE_ARCHITECTURE.md)** - Phase 6+ vision

## Development Status

### Phase 1-4: Foundation (Complete ✓)
- [x] Environment setup and verification
- [x] Project structure and organization
- [x] DRAI resonance layer implementation
- [x] Unit tests (52 tests, 100% passing)
- [x] Attractor dynamics (formation, reinforcement, decay, pruning)
- [x] Transformer integration (GPT-NeoX)
- [x] Text generation validation
- [x] End-to-end integration testing
- [x] Perplexity evaluation (WikiText-2)
- [x] Attractor statistics collection
- [x] Visualization for publication
- [x] **User-facing API (`apply_drai()` - one-line integration)**
- [x] **Scaling validation (pythia-70m → pythia-125m)**
- [x] **Logging infrastructure for analysis**

### Phase 5: Functional Benefits (In Progress - Debugging)
- [x] Synthetic story generation (fixed: removed repetition bug)
- [x] Evaluation harness for memory tasks (working)
- [x] Lesioning experiment infrastructure (ready)
- [x] Attractor visualization tools (heatmaps, time-series)
- [ ] **CRITICAL:** Fix DRAI generation quality degradation (attention weight analysis needed)
- [ ] Re-run pilots after fix
- [ ] DRAI vs RAG comparison (blocked until generation fixed)
- [ ] Human evaluation (blocked until generation fixed)

**Status:** Pilot experiments revealed DRAI degrades generation quality (word salad, incoherence) despite maintaining perplexity. Infrastructure validated. Investigating attention dilution hypothesis. See [`results/phase5/PHASE5_FINDINGS_FINAL.md`](results/phase5/PHASE5_FINDINGS_FINAL.md)

### Phase 6+: Advanced Features (Future)
- [ ] Negative resonance (push/pull dynamics)
- [ ] Hierarchical clustering (trees → groves → forests)
- [ ] Sparse local updates (spatial indexing)
- [ ] Learned attractor projections (geometric alignment)
- [ ] Metacognitive layer (uncertainty-aware gating)
- [ ] Training from scratch with DRAI

## Contributing

This is a research playground. Feel free to open issues or discussions as the project evolves. Contributions and critiques are welcome.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details. 
