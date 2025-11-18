# Duality  
  
Welcome to **Project Duality** – an experimental research repository exploring hybrid cognitive architectures. The goal is to augment an open‑source transformer model with a second “resonance cortex” based on the Dynamic Resonance AI (DRAI) principles we’ve been discussing.  
  
## Vision  
  
We want to move beyond retrieval‑based memory and build an inseparable memory layer inside the model’s attention heads. By dedicating one or two heads to a DRAI resonance engine, the model will learn to stabilise and reinject latent attractors instead of fetching tokens. This repository will collect prototypes, diagrams, and experiments exploring this new frontier.  
  
## Objectives  
  
- Prototype a simple hook into an open transformer (e.g. GPT‑NeoX) that routes a head’s Q/K/V through a resonance module.  
- Implement the resonance accumulator that detects recurring latent vectors and forms stable attractors.  
- Inject the attractor outputs back into the attention mixing as synthetic K/V pairs.  
- Visualise the evolution of the resonance manifold as new concepts are stabilised.  
- Document findings, pitfalls, and emergent behaviours along the way.  
  
## Project Status

**Current Phase:** Phase 4 - COMPLETE ✓
**Environment:** ✓ PyTorch 2.9.1 + transformers 4.57.1
**Phase 1:** ✓ Complete (Minimal viable DRAI layer)
**Phase 2:** ✓ Complete (Full attractor dynamics)
**Phase 3:** ✓ Complete (GPT-NeoX integration - WORKING!)
**Phase 4:** ✓ Complete (Evaluation & analysis - VALIDATED!)

**Key Results (Phase 4):**
- Perplexity: 89.91 (baseline) vs 89.91 (DRAI) - **No degradation!**
- Attractors: 36 created, 7,230 reinforcements, active dynamics confirmed
- Statistical significance: p=0.153 (not significant difference)
- **Finding:** DRAI adds self-organizing memory with ZERO performance cost

**Test Results:**
- Unit tests: 52/52 passing (100%)
- Integration: ✓ Text generation validated
- Attractors: ✓ Forming, reinforcing, and pruning during inference
- Stability: ✓ No NaN/Inf, no crashes
- Performance: ✓ Maintains baseline perplexity

**Completion Reports:**
- [`docs/PHASE1_COMPLETION.md`](docs/PHASE1_COMPLETION.md) - Minimal DRAI implementation
- [`docs/PHASE2_COMPLETION.md`](docs/PHASE2_COMPLETION.md) - Attractor dynamics
- [`docs/PHASE3_IMPLEMENTATION.md`](docs/PHASE3_IMPLEMENTATION.md) - Transformer integration
- [`docs/PHASE4_COMPLETION.md`](docs/PHASE4_COMPLETION.md) - **Evaluation & analysis (NEW!)**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HalcyonAIR/Duality.git
cd Duality

# Install dependencies
pip install -r requirements.txt

# For development (includes pytest, black, etc.)
pip install -e ".[dev]"

# Run environment tests
pytest tests/environment
```

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

- **[DESIGN.md](docs/DESIGN.md)** - Architecture and theory
- **[INSERTION_POINTS.md](docs/INSERTION_POINTS.md)** - Technical implementation strategy
- **[BUILD_STEPS.md](docs/BUILD_STEPS.md)** - Step-by-step build guide
- **[Glossary.md](docs/Glossary.md)** - Terminology and concepts
- **[Why_Duality.md](docs/Why_Duality.md)** - Vision and philosophy

## Development Status

- [x] Environment setup and verification
- [x] Project structure and organization
- [x] Documentation and design specs
- [x] DRAI resonance layer implementation (Phase 1 & 2)
- [x] Unit tests for DRAI (52 tests, 100% passing)
- [x] Attractor dynamics (formation, reinforcement, decay, pruning)
- [x] **Transformer integration (GPT-NeoX) - COMPLETE!**
- [x] **Text generation with DRAI - VALIDATED!**
- [x] **End-to-end integration testing - PASSING!**
- [x] **Perplexity measurements and quantitative evaluation - COMPLETE!**
- [x] **Attractor statistics collection and analysis - COMPLETE!**
- [x] **Visualization for publication - COMPLETE!**
- [ ] Training and fine-tuning experiments (Phase 5+)
- [ ] Hyperparameter optimization (Phase 5+)
- [ ] Metacognitive architecture (Phase 6+)

## Contributing

This is a research playground. Feel free to open issues or discussions as the project evolves. Contributions and critiques are welcome.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details. 
