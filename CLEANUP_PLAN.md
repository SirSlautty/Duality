# Repository Cleanup Plan - Phase 1

## Files to KEEP (Clean, Essential)

### Core Implementation
- ✅ `src/drai/__init__.py`
- ✅ `src/drai/apply_v1.py` (main API)
- ✅ `src/drai/config.py`
- ✅ `src/drai/resonance_layer_v1.py`
- ✅ `src/drai/neox_integration_v1.py`

### Tests
- ✅ `tests/` (all - 52 passing tests)

### Essential Documentation
- ✅ `docs/THE_CLOUD_MECHANISM.md`
- ✅ `docs/COMPUTATIONAL_SIGNATURES.md`
- ✅ `docs/DRAI_V1_INVARIANT.md`
- ✅ `Lessons_Learned` (top-level file)
- ✅ `README.md` (will be rewritten)

### Examples (to verify/clean)
- ✅ `examples/quickstart.py` (verify it works)

### Benchmarks (to reorganize)
- ✅ `experiments/evaluation/test_v1_final.py` → `benchmarks/story_comprehension_410m/`

---

## Files to ARCHIVE (move to archive/)

### Phase 2 code (outdated)
- 📦 `src/drai/apply.py`
- 📦 `src/drai/resonance_layer.py`
- 📦 `src/drai/neox_integration.py`
- 📦 `src/drai/build_drai_neox.py`
- 📦 `src/drai/apply_v1.py` → keep
- 📦 `src/drai/__main__.py`

### Old documentation
- 📦 `docs/AGENT_INSTRUCTIONS.md`
- 📦 `docs/ATTRACTOR_MATHEMATICS.md`
- 📦 `docs/BUILD_STEPS.md`
- 📦 `docs/CODEBASE_AUDIT.md`
- 📦 `docs/DESIGN.md`
- 📦 `docs/Diagrams.md`
- 📦 `docs/Glossary.md`
- 📦 `docs/IMPLEMENTATION_STATUS.md`
- 📦 `docs/INDEX.md`
- 📦 `docs/INSERTION_POINTS.md`
- 📦 `docs/INSIGHT_DYNAMIC_EDITOR.md`
- 📦 `docs/INSTALLATION_VERIFICATION.md`
- 📦 `docs/INTEGRATION_METHODOLOGY.md`
- 📦 `docs/METACOGNITIVE_ARCHITECTURE.md`
- 📦 `docs/MINIMAL_EVALUATION_PLAN.md`
- 📦 `docs/NEOX_ARCHITECTURE_ANALYSIS.md`
- 📦 `docs/NOVELTY_AND_POSITIONING.md`
- 📦 `docs/PAPER_OUTLINE.md`
- 📦 `docs/PHASE1_COMPLETION.md`
- 📦 `docs/PHASE2_COMPLETION.md`
- 📦 `docs/PHASE3_IMPLEMENTATION.md`
- 📦 `docs/PHASE3_PLAN.md`
- 📦 `docs/PHASE4_COMPLETION.md`
- 📦 `docs/PHASE4_PLAN.md`
- 📦 `docs/PHASE5_EVALUATION_ROADMAP.md`
- 📦 `docs/PROJECT_STRUCTURE.md`
- 📦 `docs/THEORY_SYNMAIC_RESONANCE.md`
- 📦 `docs/VISION_TOPOLOGICAL_MEMORY.md`
- 📦 `docs/Why_Duality.md`

### Experimental/incomplete evaluation scripts
- 📦 `experiments/evaluation/test_v1_*.py` (except final)
- 📦 `experiments/evaluation/lesion_experiment.py` (incomplete)
- 📦 `experiments/evaluation/long_story_consistency.py` (old)
- 📦 `experiments/evaluation/simple_*.py` (drafts)

### Scratch files
- 📦 `drai_v1_standalone.py` (was demonstration, not core)

---

## New Files to CREATE

### Phase 2: Reproducibility
- 📝 `notebooks/basic_usage.ipynb`
- 📝 `scripts/run_reproduction_410m.sh`
- 📝 `benchmarks/story_comprehension_410m/README.md`
- 📝 `benchmarks/story_comprehension_410m/dataset_sample.json`
- 📝 `requirements.txt`

### Phase 3: Public Release
- 📝 `README.md` (complete rewrite - engineering-first)
- 📝 `CONTRIBUTING.md` (community guidelines)
- 📝 `LICENSE` (already exists - Apache 2.0)

---

## Clean Directory Structure (Target)

```
Duality/
├── README.md                    # Public-facing, engineering-first
├── LICENSE                      # Apache 2.0
├── requirements.txt             # Clean dependencies
├── setup.py                     # Installation
├── Lessons_Learned              # Key insights from development
│
├── src/
│   └── drai/
│       ├── __init__.py
│       ├── apply_v1.py          # Main API
│       ├── config.py            # Hyperparameters
│       ├── resonance_layer_v1.py
│       └── neox_integration_v1.py
│
├── tests/                       # 52 passing tests
│   ├── unit/
│   ├── integration/
│   └── environment/
│
├── examples/
│   └── quickstart.py            # 5-minute getting started
│
├── notebooks/
│   └── basic_usage.ipynb        # Interactive tutorial
│
├── benchmarks/
│   └── story_comprehension_410m/
│       ├── README.md
│       ├── run.py               # The +3.3% test
│       └── dataset_sample.json  # Small test set
│
├── scripts/
│   └── run_reproduction_410m.sh # One-command reproduction
│
├── docs/
│   ├── THE_CLOUD_MECHANISM.md          # Core mechanism
│   ├── COMPUTATIONAL_SIGNATURES.md      # Theory grounding
│   └── DRAI_V1_INVARIANT.md            # Mathematics
│
└── archive/                     # Historical/speculative docs
    └── (everything else)
```

---

## Action Items

1. ✅ Create `archive/` directory
2. ✅ Move non-essential docs to `archive/docs/`
3. ✅ Move Phase 2 code to `archive/src/`
4. ✅ Move experimental scripts to `archive/experiments/`
5. ✅ Reorganize `experiments/evaluation/test_v1_final.py` → `benchmarks/`
6. ✅ Clean up `examples/quickstart.py` and verify it works
7. ✅ Create `requirements.txt`
8. ✅ Create `notebooks/basic_usage.ipynb`
9. ✅ Create `scripts/run_reproduction_410m.sh`
10. ✅ Rewrite `README.md` (engineering-first, no consciousness)
