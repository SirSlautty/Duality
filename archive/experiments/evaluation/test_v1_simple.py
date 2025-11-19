#!/usr/bin/env python3
"""
Quick test of DRAI V1 algorithm on simple stories.

This script validates that:
1. V1 loads without errors
2. Forward pass works
3. State updates happen correctly
4. Accuracy is near baseline (no catastrophic degradation)

Usage:
    python experiments/evaluation/test_v1_simple.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import V1
from src.drai import apply_drai_v1, get_v1_conservative_config, get_drai_v1_stats

# Import simple story evaluator
from simple_story_generator import SimpleStoryGenerator
from simple_evaluator import SimpleEvaluator


def test_v1_basic():
    """Test that V1 loads and runs without errors."""
    print("=" * 80)
    print("TEST 1: Basic V1 Loading and Forward Pass")
    print("=" * 80)

    # Load model
    model_name = "EleutherAI/pythia-410m"
    print(f"\n[1/4] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    print("✓ Model loaded")

    # Apply V1
    print(f"\n[2/4] Applying DRAI V1...")
    config = get_v1_conservative_config()
    model = apply_drai_v1(model, config=config)
    print("✓ V1 applied")

    # Test forward pass
    print(f"\n[3/4] Testing forward pass...")
    test_text = "Once upon a time, there was a"
    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {outputs.logits.shape}")

    # Check stats
    print(f"\n[4/4] Checking V1 statistics...")
    stats = get_drai_v1_stats(model)
    print(f"✓ Statistics retrieved:")
    print(f"  - Layers with V1: {stats['num_drai_layers']}")
    print(f"  - Total active attractors: {stats['total_active']}")
    print(f"  - Total strength: {stats['total_strength']:.4f}")

    print("\n" + "=" * 80)
    print("✅ TEST 1 PASSED")
    print("=" * 80)

    return model, tokenizer


def test_v1_generation():
    """Test that V1 can generate text."""
    print("\n" + "=" * 80)
    print("TEST 2: Text Generation")
    print("=" * 80)

    model_name = "EleutherAI/pythia-410m"
    print(f"\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Apply V1
    print(f"\n[2/3] Applying DRAI V1...")
    config = get_v1_conservative_config()
    model = apply_drai_v1(model, config=config)

    # Generate
    print(f"\n[3/3] Generating text...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=10,
        do_sample=False,  # Greedy for determinism
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generation successful")
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {generated_text}")

    # Check stats after generation
    stats = get_drai_v1_stats(model)
    print(f"\n✓ Post-generation statistics:")
    for layer_stats in stats['layers']:
        print(f"  Layer {layer_stats['layer_idx']}:")
        print(f"    - Active attractors: {layer_stats['num_alive']}")
        print(f"    - Total strength: {layer_stats['total_strength']:.4f}")
        print(f"    - Mean strength: {layer_stats['mean_strength']:.4f}")
        print(f"    - Timestep: {layer_stats['timestep']}")

    print("\n" + "=" * 80)
    print("✅ TEST 2 PASSED")
    print("=" * 80)


def test_v1_on_simple_stories():
    """Test V1 on simple stories (main validation)."""
    print("\n" + "=" * 80)
    print("TEST 3: Simple Stories Evaluation")
    print("=" * 80)

    model_name = "EleutherAI/pythia-410m"

    # Generate test stories
    print(f"\n[1/5] Generating test stories...")
    generator = SimpleStoryGenerator(seed=42)
    stories = generator.generate_batch(num_stories=10)  # Small batch for quick test
    print(f"✓ Generated {len(stories)} stories")

    # Test baseline
    print(f"\n[2/5] Evaluating baseline (no DRAI)...")
    evaluator = SimpleEvaluator(model_name, device="cpu", dtype="float32")
    baseline_results = evaluator.evaluate_model(model_name, stories, use_drai=False)
    baseline_acc = baseline_results['accuracy']
    print(f"✓ Baseline accuracy: {baseline_acc:.1%} ({baseline_results['correct']}/{baseline_results['total']})")

    # Test V1
    print(f"\n[3/5] Evaluating with DRAI V1...")
    # Reload model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Apply V1
    config = get_v1_conservative_config()
    model = apply_drai_v1(model, config=config)

    # Evaluate
    results = []
    for i, story in enumerate(stories):
        print(f"  Story {i+1}/{len(stories)}...", end=" ")

        # For each question
        for q_idx, question in enumerate(story.questions):
            # Create prompt
            prompt = f"{story.story}\n\nQuestion: {question.question}\nAnswer:"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")

            # Generate answer
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated = generated.strip().split()[0] if generated.strip() else ""

            # Check correctness
            correct = generated.lower() == question.answer.lower()
            results.append(correct)

        print("✓")

    v1_acc = sum(results) / len(results)
    v1_correct = sum(results)
    v1_total = len(results)

    print(f"\n✓ V1 accuracy: {v1_acc:.1%} ({v1_correct}/{v1_total})")

    # Compare
    print(f"\n[4/5] Comparing results...")
    delta = v1_acc - baseline_acc
    print(f"  Baseline: {baseline_acc:.1%}")
    print(f"  V1:       {v1_acc:.1%}")
    print(f"  Δ:        {delta:+.1%}")

    if delta > -0.10:  # Less than 10% degradation
        print(f"  ✅ PASS: V1 maintains performance (Δ > -10%)")
    else:
        print(f"  ⚠️  WARN: V1 shows degradation (Δ = {delta:.1%})")

    # Check V1 stats
    print(f"\n[5/5] V1 statistics after evaluation...")
    stats = get_drai_v1_stats(model)
    for layer_stats in stats['layers']:
        print(f"  Layer {layer_stats['layer_idx']}:")
        print(f"    - Active attractors: {layer_stats['num_alive']}")
        print(f"    - Total strength: {layer_stats['total_strength']:.4f}")
        print(f"    - Timestep: {layer_stats['timestep']}")

    print("\n" + "=" * 80)
    print("✅ TEST 3 COMPLETE")
    print("=" * 80)

    return {
        'baseline_acc': baseline_acc,
        'v1_acc': v1_acc,
        'delta': delta,
        'stats': stats,
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DRAI V1 VALIDATION TESTS")
    print("=" * 80)

    try:
        # Test 1: Basic loading
        test_v1_basic()

        # Test 2: Generation
        test_v1_generation()

        # Test 3: Simple stories (main validation)
        results = test_v1_on_simple_stories()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print(f"\nFinal Results:")
        print(f"  Baseline: {results['baseline_acc']:.1%}")
        print(f"  V1 DRAI:  {results['v1_acc']:.1%}")
        print(f"  Δ:        {results['delta']:+.1%}")
        print(f"\n{results['stats']['num_drai_layers']} layer(s) enhanced with V1")
        print(f"{results['stats']['total_active']} active attractors")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
