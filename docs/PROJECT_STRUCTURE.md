# Duality Project Structure

**Last Updated:** 2025-11-18
**Status:** Phase 1 Complete - Ready for Implementation

## Directory Organization

```
Duality/
├── README.md                   # Project overview and quick start
├── LICENSE                     # Apache 2.0 license
├── requirements.txt            # Core dependencies
├── pyproject.toml             # Project metadata and tool configuration
├── pytest.ini                 # Pytest configuration
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── drai/                  # DRAI resonance module
│   │   ├── __init__.py
│   │   └── resonance_layer.py # Core DRAI implementation
│   └── models/                # Model integration code
│       └── __init__.py
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── unit/                  # Unit tests for components
│   │   ├── __init__.py
│   │   └── test_drai_resonance.py
│   ├── integration/           # Integration tests with transformers
│   │   └── __init__.py
│   └── environment/           # Environment verification
│       ├── __init__.py
│       └── test_pytorch_installation.py
│
├── experiments/               # Research notebooks and scripts
│   ├── notebooks/             # Jupyter notebooks
│   └── scripts/               # Standalone experiment scripts
│
├── models/                    # External model repositories
│   └── README.md              # Instructions for cloning models
│
└── docs/                      # Documentation
    ├── PROJECT_STRUCTURE.md   # This file
    ├── DESIGN.md              # Architecture design
    ├── BUILD_STEPS.md         # Build instructions
    ├── INSERTION_POINTS.md    # Technical implementation details
    ├── CODEBASE_AUDIT.md      # Audit report and action plan
    ├── INSTALLATION_VERIFICATION.md  # Environment setup verification
    ├── AGENT_INSTRUCTIONS.md  # Instructions for AI agents
    ├── Diagrams.md            # Architecture diagrams
    ├── Glossary.md            # Terminology and concepts
    └── Why_Duality.md         # Vision and philosophy
```

## Directory Purposes

### `/src` - Source Code

The main implementation code for the Duality project.

**`/src/drai`** - DRAI Resonance Module
- Core implementation of Dynamic Resonance AI
- `resonance_layer.py` - The main DraiResonanceLayer class
- Attractor accumulation and pattern detection
- Synthetic K/V generation logic

**`/src/models`** - Model Integration
- Integration wrappers for different transformer architectures
- Future: `neox_integration.py`, `gptj_integration.py`
- Utilities for injecting DRAI into attention layers

### `/tests` - Test Suite

Comprehensive testing infrastructure following pytest conventions.

**`/tests/unit`** - Unit Tests
- Test individual components in isolation
- Fast, focused tests for DRAI modules
- Test tensor operations, shapes, gradients
- Run with: `pytest tests/unit`

**`/tests/integration`** - Integration Tests
- Test DRAI integration with full transformer models
- Verify attention mechanisms work correctly
- Test with GPT-NeoX, GPT-J, etc.
- Run with: `pytest tests/integration`

**`/tests/environment`** - Environment Tests
- Verify PyTorch installation
- Check CUDA availability
- Test dependencies
- Run with: `pytest tests/environment`

### `/experiments` - Research and Prototyping

Exploratory work, notebooks, and experimental scripts.

**`/experiments/notebooks`** - Jupyter Notebooks
- Interactive exploration of DRAI behavior
- Visualization of attractor formation
- Small-scale prototyping
- Results analysis

**`/experiments/scripts`** - Standalone Scripts
- Reproducible experiments
- Training/fine-tuning scripts
- Evaluation and benchmarking
- Data generation utilities

### `/models` - External Repositories

Git submodules or clones of transformer model repositories.

- GPT-NeoX (HalcyonAIR/gpt-neox)
- GPT-J (HalcyonAIR/mesh-transformer-jax)
- Model weights (downloaded separately, not committed)
- See `models/README.md` for setup instructions

### `/docs` - Documentation

All design documents, specifications, and research notes.

**Core Documentation:**
- `PROJECT_STRUCTURE.md` - This file
- `DESIGN.md` - Architecture and theory
- `INSERTION_POINTS.md` - Technical implementation strategy
- `BUILD_STEPS.md` - Step-by-step build guide

**Research Documentation:**
- `Glossary.md` - Terminology and concepts
- `Why_Duality.md` - Vision and motivation
- `Diagrams.md` - Architecture diagrams

**Development Documentation:**
- `CODEBASE_AUDIT.md` - Audit results and roadmap
- `INSTALLATION_VERIFICATION.md` - Environment setup
- `AGENT_INSTRUCTIONS.md` - AI agent guidance

## Package Structure

The project follows standard Python packaging conventions:

```python
# Import DRAI components
from src.drai import DraiResonanceLayer

# Import model integrations (future)
from src.models import NeoXIntegration
```

## Configuration Files

### `pyproject.toml`
- Project metadata (name, version, dependencies)
- Build system configuration
- Tool configurations (pytest, black, mypy)
- Development dependencies

### `pytest.ini`
- Pytest test discovery settings
- Fallback configuration (pyproject.toml takes precedence)

### `requirements.txt`
- Core runtime dependencies
- Used for simple pip installation
- Kept minimal (PyTorch, NumPy, etc.)

## Development Workflow

### Installing Dependencies

```bash
# Core dependencies only
pip install -r requirements.txt

# Development dependencies (recommended)
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[dev,experiments,models]"
```

### Running Tests

```bash
# All tests
pytest

# Specific test categories
pytest tests/unit
pytest tests/integration
pytest tests/environment

# With coverage
pytest --cov=src --cov-report=html

# Specific markers
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Experiments

```bash
# Start Jupyter
jupyter notebook experiments/notebooks/

# Run experiment script
python experiments/scripts/train_simple_drai.py
```

## Phase 1 Completion Status

✓ **Directory structure created**
- All directories in place
- `__init__.py` files for Python packages
- Proper separation of concerns

✓ **Documentation organized**
- All docs moved to `/docs`
- File extensions added
- Logical grouping maintained

✓ **Testing infrastructure**
- Pytest configured
- Test directories created
- Environment tests in place

✓ **Project configuration**
- `pyproject.toml` created
- Build system configured
- Tool settings defined

## Next Steps (Phase 2)

1. Implement `src/drai/resonance_layer.py`
2. Create unit tests in `tests/unit/test_drai_resonance.py`
3. Define attractor mathematics
4. Verify gradient flow

See `CODEBASE_AUDIT.md` for the complete implementation roadmap.

---

**Note:** This structure is designed to support both research exploration and production-quality implementation. The separation of concerns allows for clean development while maintaining flexibility for experimentation.
