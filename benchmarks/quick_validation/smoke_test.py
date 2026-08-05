#!/usr/bin/env python3
"""
DRAI V1 Quick Smoke Test

A lightweight validation that DRAI V1 doesn't catastrophically break models.
Uses pythia-70m (smaller/faster) with minimal test cases.

Purpose:
- Fast CI validation (~1-2 minutes)
- Smoke test before running full benchmarks
- Development workflow sanity check

NOT a replacement for full benchmarks - just verifies basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# DRAI V1
from src.drai import apply_v1, get_v1_conservative, get_v1_stats


def quick_test():
    """Quick smoke test to verify DRAI doesn't break model."""
    print("\n" + "="*70)
    print("DRAI V1 - Quick Smoke Test")
    print("="*70)
    print("\nThis is a FAST validation (not a full benchmark)")
    print("Purpose: Verify DRAI doesn't catastrophically break the model")
    print("\n" + "="*70)

    # Configuration
    model_name = "EleutherAI/pythia-70m"
    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The sky is",
    ]

    print(f"\n[1/3] Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Baseline test
    print(f"\n[2/3] Testing baseline (no DRAI)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    baseline_outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_outputs.append(generated)
        print(f"  Baseline: '{prompt}' → '{generated}'")

    # DRAI test
    print(f"\n[3/3] Testing with DRAI V1...")
    config = get_v1_conservative()
    model = apply_v1(model, config=config)

    drai_outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        drai_outputs.append(generated)
        print(f"  DRAI:     '{prompt}' → '{generated}'")

    # Check DRAI stats
    stats = get_v1_stats(model)
    print(f"\nDRAI Statistics:")
    print(f"  - Layers with DRAI: {stats['num_drai_layers']}")
    print(f"  - Total active attractors: {stats['total_active']}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    # Simple check: model still generates reasonable text
    all_valid = all(len(out) > len(test_prompts[i]) for i, out in enumerate(drai_outputs))

    if all_valid:
        print("✅ PASS: Model generates text with DRAI")
        print("\nKey validations:")
        print("  ✓ Model didn't crash")
        print("  ✓ DRAI layers applied successfully")
        print("  ✓ Generation still works")
        print(f"  ✓ {stats['num_drai_layers']} DRAI layer(s) active")
        print("\nNote: This is NOT a performance benchmark!")
        print("Run full benchmarks to validate accuracy improvements.")
        verdict = "PASS"
    else:
        print("❌ FAIL: Model didn't generate properly with DRAI")
        verdict = "FAIL"

    print(f"{'='*70}\n")

    return verdict == "PASS"


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
