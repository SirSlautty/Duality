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

**Current Phase:** Phase 2 - DRAI Core Implementation
**Environment:** ✓ PyTorch 2.9.1 installed and tested
**Structure:** ✓ Complete (Phase 1 finished)
**Implementation:** 🚧 In progress

See [`docs/CODEBASE_AUDIT.md`](docs/CODEBASE_AUDIT.md) for the complete roadmap.

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
- [ ] DRAI resonance layer implementation
- [ ] Unit tests for DRAI
- [ ] Transformer integration
- [ ] Experimental validation

## Contributing

This is a research playground. Feel free to open issues or discussions as the project evolves. Contributions and critiques are welcome.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details. 
