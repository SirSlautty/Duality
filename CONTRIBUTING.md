# Contributing to DRAI

Thank you for your interest in contributing to DRAI! This project is in active development, and we welcome contributions from the community.

## How to Contribute

### 1. Testing on New Models

The most valuable contribution right now is **testing DRAI V1 on different model families**:

- Try DRAI on models you're working with (GPT-2, LLaMA variants, etc.)
- Report both successes and failures
- Share your configurations and hyperparameters

**What to report:**
- Model name and size
- Task/benchmark used
- DRAI configuration (conservative/standard/custom)
- Baseline vs DRAI performance
- Any unexpected behaviors

### 2. Benchmarks and Evaluations

We need more diverse benchmarks:

- Story comprehension (currently covered)
- Long-context tasks
- Multi-turn dialogue
- Reasoning tasks
- Code generation

**Submit a benchmark:**
1. Create a directory under `benchmarks/your_benchmark_name/`
2. Include `run.py` (standalone evaluation script)
3. Include `README.md` (description, expected results)
4. Use the same structure as `benchmarks/story_comprehension_410m/`

### 3. Bug Reports

Found a bug? Please report it with:

- Minimal reproducible example
- Model and configuration used
- Expected vs actual behavior
- DRAI statistics output (`get_drai_v1_stats()`)

### 4. Feature Requests

Have an idea? Open an issue describing:

- Use case or problem it solves
- Proposed approach (if you have one)
- Why it matters for the community

### 5. Code Contributions

**Before submitting a PR:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run existing benchmarks to ensure no regressions
5. Add tests if applicable
6. Update documentation

**Code style:**
- Follow existing code structure
- Use type hints
- Add docstrings for public functions
- Keep changes focused (one feature/fix per PR)

**PR checklist:**
- [ ] Code follows existing style
- [ ] Tests pass (run `pytest`)
- [ ] Benchmarks still work
- [ ] Documentation updated
- [ ] Commit messages are clear

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Duality.git
cd Duality

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run quick validation
bash scripts/run_reproduction_410m.sh
```

## Questions?

- Open an issue for general questions
- Check existing issues before creating new ones
- Be respectful and constructive

## What We're Looking For

**High priority:**
- Testing on 1B-7B models
- Long-context benchmarks (2k+ tokens)
- Multi-turn dialogue evaluations
- Efficiency improvements

**Medium priority:**
- Additional configuration presets
- Visualization tools for attractor dynamics
- Integration examples (LangChain, etc.)

**Lower priority (but welcome):**
- Documentation improvements
- Code cleanup/refactoring
- Type hint improvements

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Attribution

Contributors will be acknowledged in release notes and documentation. Significant contributions may warrant co-authorship on future papers.

---

Thank you for helping make DRAI better! 🚀
