# Quick Validation (Smoke Test)

A lightweight smoke test for DRAI V1 development and CI.

## Purpose

- **Fast validation** (~1-2 minutes vs 5-10 minutes for full benchmark)
- **Smoke test** to verify DRAI doesn't catastrophically break models
- **Development workflow** sanity check before running full benchmarks

**This is NOT a replacement for full benchmarks** - it only validates basic functionality.

## What It Tests

1. Model loads successfully
2. DRAI V1 applies without errors
3. Model still generates text after DRAI
4. DRAI layers are active
5. Attractors form during generation

## What It Doesn't Test

- Accuracy improvements
- Long-context performance
- Multi-turn generation
- Stability over many samples

## Usage

```bash
cd benchmarks/quick_validation
python smoke_test.py
```

Expected output:
```
✅ PASS: Model generates text with DRAI

Key validations:
  ✓ Model didn't crash
  ✓ DRAI layers applied successfully
  ✓ Generation still works
  ✓ 1 DRAI layer(s) active
```

## When to Use

- **Before pushing commits** - Quick sanity check
- **In CI/CD** - Fast validation on every PR
- **During development** - Iterate quickly without waiting for full benchmarks

## When NOT to Use

- **Validating accuracy claims** - Use full benchmarks
- **Testing on new model families** - Use full benchmarks
- **Before publishing results** - Use full benchmarks

## Model Used

- **pythia-70m** (smaller/faster than 410m)
- **3 simple prompts** (not comprehensive)
- **Deterministic generation** (for reproducibility)

## Runtime

- **~1-2 minutes** on CPU
- **~30 seconds** on GPU

Compare to full benchmark: **~5-10 minutes**
