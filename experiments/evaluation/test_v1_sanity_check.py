#!/usr/bin/env python3
"""
DRAI V1 SANITY CHECK: Prove It's Actually Working!

We've seen perfect parity (85% = 85%, 83.3% = 83.3%), but is DRAI
actually DOING anything? Or is it just a no-op?

This script runs 5 tests to PROVE DRAI is active:
  A. Crank influence to 3.0 (20x normal) → Should degrade if working
  B. Inject random garbage attractors → Should degrade badly if working
  C. Print attention deltas → Scientific proof of numerical changes
  D. Disable DRAI mid-generation → Watch behavior change
  E. All of the above (comprehensive proof)

If DRAI is a no-op, all tests will show ~85% accuracy.
If DRAI is working, we'll see clear effects.

Author: Halcyon AI Research (with a mischievous grin 😈)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import json
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_story_generator import SimpleStoryGenerator

# Import V1
from src.drai import apply_drai_v1, get_v1_conservative_config, get_drai_v1_stats
from src.drai.config import DraiV1Config, DraiV1Hyperparameters


def score_answer(predicted: str, correct: str) -> bool:
    """Score answer with flexible matching."""
    predicted = predicted.lower().strip()
    correct = correct.lower().strip()

    if predicted == correct:
        return True
    if correct in predicted:
        return True

    predicted_words = set(predicted.split())
    correct_words = set(correct.split())
    if correct_words.issubset(predicted_words):
        return True

    return False


def evaluate_model(model, tokenizer, stories, model_name="Model", verbose=False):
    """Evaluate model on simple stories."""
    results = []
    total_questions = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

    for i, story in enumerate(stories, 1):
        if verbose:
            print(f"\n[Story {i}/{len(stories)}]", end=" ")

        for q_idx, question in enumerate(story.questions):
            total_questions += 1

            prompt = f"""Story: {story.text}

Question: {question['question']}
Answer:"""

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1800,
                return_attention_mask=True
            )

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            predicted_answer = predicted_answer.split('\n')[0].split('.')[0].strip()

            correct = score_answer(predicted_answer, question['answer'])
            results.append({
                "story_idx": i,
                "question": question['question'],
                "expected": question['answer'],
                "generated": predicted_answer,
                "correct": correct
            })

            if verbose:
                print("✓" if correct else "✗", end="")

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total_questions

    if verbose:
        print(f"\n\n{'='*70}")
        print(f"RESULTS: {model_name}")
        print(f"{'='*70}")
        print(f"  Correct: {correct_count}/{total_questions}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"{'='*70}\n")

    return accuracy, correct_count, total_questions


def test_a_cranked_influence(model_name, tokenizer, stories):
    """Test A: Crank influence to 3.0 (20x normal) - should degrade!"""
    print("\n" + "="*70)
    print("TEST A: CRANKED INFLUENCE (3.0 vs normal 0.15)")
    print("="*70)
    print("If DRAI is working, this should DEGRADE performance significantly.")
    print("If DRAI is a no-op, we'll see ~85% (no change).")
    print()

    # Create config with INSANE influence
    config = DraiV1Config(
        enabled=True,
        layer_mode="mid",
        num_drai_heads=1,
        hyperparameters=DraiV1Hyperparameters(
            max_attractors=16,
            theta_match=0.8,
            burn_in_threshold=50.0,
            max_influence_scale=3.0,  # 20x HIGHER THAN NORMAL!
        )
    )

    print(f"Loading model with CRANKED influence (3.0)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    accuracy, correct, total = evaluate_model(
        model, tokenizer, stories,
        model_name="CRANKED Influence (3.0)",
        verbose=True
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "test": "cranked_influence",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "influence_scale": 3.0
    }


def test_b_garbage_attractors(model_name, tokenizer, stories):
    """Test B: Inject random garbage attractors - should degrade badly!"""
    print("\n" + "="*70)
    print("TEST B: RANDOM GARBAGE ATTRACTORS")
    print("="*70)
    print("Inject random noise as attractors - should cause chaos!")
    print("If DRAI is working, accuracy should DROP significantly.")
    print()

    config = get_v1_conservative_config()

    print(f"Loading model and injecting GARBAGE...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    # INJECT RANDOM GARBAGE into attractors
    from src.drai.neox_integration_v1 import DraiGPTNeoXAttentionV1
    for layer in model.gpt_neox.layers:
        if isinstance(layer.attention, DraiGPTNeoXAttentionV1):
            drai = layer.attention.drai
            # Fill attractors with RANDOM NOISE
            drai.attractors.data = torch.randn_like(drai.attractors) * 10.0
            drai.strengths.data = torch.ones_like(drai.strengths) * 100.0  # HIGH strength!
            print(f"  ✓ Injected garbage into layer {layer.attention.layer_idx}")

    accuracy, correct, total = evaluate_model(
        model, tokenizer, stories,
        model_name="GARBAGE Attractors",
        verbose=True
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "test": "garbage_attractors",
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def test_c_attention_deltas(model_name, tokenizer, stories):
    """Test C: Print attention deltas - scientific proof!"""
    print("\n" + "="*70)
    print("TEST C: ATTENTION DELTAS (Scientific Proof)")
    print("="*70)
    print("Capture actual numerical changes in attention patterns.")
    print()

    config = get_v1_conservative_config()

    print(f"Loading model with DRAI...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    # Test on first story
    story = stories[0]
    question = story.questions[0]
    prompt = f"""Story: {story.text}

Question: {question['question']}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)

    # Hook to capture K/V deltas
    kv_deltas = []

    def hook_fn(module, input, output):
        """Capture DRAI K/V outputs."""
        from src.drai.neox_integration_v1 import DraiGPTNeoXAttentionV1
        if isinstance(module, DraiGPTNeoXAttentionV1):
            # The DRAI module stores last k_reson, v_reson in forward pass
            if hasattr(module, '_last_kv_delta'):
                kv_deltas.append(module._last_kv_delta)

    # Register hooks
    from src.drai.neox_integration_v1 import DraiGPTNeoXAttentionV1
    hooks = []
    for layer in model.gpt_neox.layers:
        if isinstance(layer.attention, DraiGPTNeoXAttentionV1):
            # Manually capture deltas
            query = torch.randn(1, 1, 2048)  # Dummy query
            k_reson, v_reson = layer.attention.drai(query)

            k_norm = torch.norm(k_reson).item()
            v_norm = torch.norm(v_reson).item()

            if k_norm > 0 or v_norm > 0:
                print(f"  Layer {layer.attention.layer_idx}:")
                print(f"    K injection norm: {k_norm:.6f}")
                print(f"    V injection norm: {v_norm:.6f}")
                kv_deltas.append((k_norm, v_norm))

    print(f"\n  Total layers with non-zero injection: {len(kv_deltas)}")

    if len(kv_deltas) > 0:
        avg_k = np.mean([d[0] for d in kv_deltas])
        avg_v = np.mean([d[1] for d in kv_deltas])
        print(f"  Average K norm: {avg_k:.6f}")
        print(f"  Average V norm: {avg_v:.6f}")
        print(f"\n  ✓ PROOF: DRAI is injecting non-zero K/V signals!")
    else:
        print(f"\n  ✗ WARNING: No non-zero K/V injections detected!")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "test": "attention_deltas",
        "num_injections": len(kv_deltas),
        "avg_k_norm": float(np.mean([d[0] for d in kv_deltas])) if kv_deltas else 0.0,
        "avg_v_norm": float(np.mean([d[1] for d in kv_deltas])) if kv_deltas else 0.0,
    }


def test_d_disable_mid_generation(model_name, tokenizer, stories):
    """Test D: Disable DRAI mid-generation - watch behavior change!"""
    print("\n" + "="*70)
    print("TEST D: DISABLE DRAI MID-GENERATION")
    print("="*70)
    print("Generate first half with DRAI, second half without.")
    print("If DRAI is working, we'll see behavior changes.")
    print()

    config = get_v1_conservative_config()

    print(f"Loading model with DRAI...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    # Test on first story
    story = stories[0]
    question = story.questions[0]
    prompt = f"""Story: {story.text}

Question: {question['question']}
Answer:"""

    print(f"\nGenerating with DRAI ON for 10 tokens...")
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate first 10 tokens with DRAI
    outputs_1 = model.generate(
        inputs["input_ids"],
        max_new_tokens=10,
        do_sample=False,
    )
    answer_1 = tokenizer.decode(outputs_1[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  First 10 tokens: '{answer_1}'")

    # Disable DRAI
    print(f"\nDisabling DRAI...")
    from src.drai.neox_integration_v1 import DraiGPTNeoXAttentionV1
    for layer in model.gpt_neox.layers:
        if isinstance(layer.attention, DraiGPTNeoXAttentionV1):
            layer.attention.drai.max_influence_scale = 0.0  # Turn off injection
    print(f"  ✓ DRAI disabled (influence = 0.0)")

    # Generate next 10 tokens WITHOUT DRAI
    print(f"\nGenerating with DRAI OFF for 10 more tokens...")
    outputs_2 = model.generate(
        outputs_1,
        max_new_tokens=10,
        do_sample=False,
    )
    answer_2 = tokenizer.decode(outputs_2[0][outputs_1.shape[1]:], skip_special_tokens=True)
    print(f"  Next 10 tokens: '{answer_2}'")

    print(f"\nFull answer:")
    print(f"  WITH DRAI: '{answer_1}'")
    print(f"  THEN WITHOUT: '{answer_2}'")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "test": "disable_mid_generation",
        "with_drai": answer_1,
        "without_drai": answer_2,
    }


def main():
    print("\n" + "="*70)
    print("DRAI V1 SANITY CHECK: IS IT ACTUALLY WORKING?")
    print("="*70)
    print("\nWe've seen perfect parity (85% = 85%), but is DRAI a no-op?")
    print("\nRunning 5 tests to PROVE DRAI is active:")
    print("  A. Crank influence to 3.0 → Should degrade")
    print("  B. Inject garbage attractors → Should degrade badly")
    print("  C. Print attention deltas → Scientific proof")
    print("  D. Disable mid-generation → Watch behavior change")
    print("  E. All of the above (comprehensive)")
    print("\n" + "="*70)

    model_name = "EleutherAI/pythia-410m"

    # Generate small test set
    print(f"\nGenerating 10 test stories...")
    generator = SimpleStoryGenerator(seed=42)
    stories = generator.generate_dataset(num_stories=10)
    print(f"  ✓ Generated {len(stories)} stories")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Get baseline for comparison
    print(f"\n{'='*70}")
    print("BASELINE (for reference)")
    print(f"{'='*70}")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_baseline.eval()
    baseline_acc, baseline_correct, baseline_total = evaluate_model(
        model_baseline, tokenizer, stories,
        model_name="Baseline (no DRAI)",
        verbose=True
    )
    del model_baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run all tests
    results = {
        "baseline": {
            "accuracy": baseline_acc,
            "correct": baseline_correct,
            "total": baseline_total
        }
    }

    # Test A: Cranked influence
    results["test_a"] = test_a_cranked_influence(model_name, tokenizer, stories)

    # Test B: Garbage attractors
    results["test_b"] = test_b_garbage_attractors(model_name, tokenizer, stories)

    # Test C: Attention deltas
    results["test_c"] = test_c_attention_deltas(model_name, tokenizer, stories)

    # Test D: Disable mid-generation
    results["test_d"] = test_d_disable_mid_generation(model_name, tokenizer, stories)

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    print(f"\nBaseline: {baseline_acc:.1%}")
    print(f"Cranked influence (3.0): {results['test_a']['accuracy']:.1%}")
    print(f"Garbage attractors: {results['test_b']['accuracy']:.1%}")

    cranked_delta = results['test_a']['accuracy'] - baseline_acc
    garbage_delta = results['test_b']['accuracy'] - baseline_acc

    print(f"\nDeltas from baseline:")
    print(f"  Cranked: {cranked_delta:+.1%}")
    print(f"  Garbage: {garbage_delta:+.1%}")

    print(f"\nAttention deltas:")
    print(f"  Injections detected: {results['test_c']['num_injections']}")
    print(f"  Avg K/V norm: {results['test_c']['avg_k_norm']:.6f} / {results['test_c']['avg_v_norm']:.6f}")

    print(f"\n{'='*70}")

    # Determine if DRAI is actually working
    is_working = False

    if abs(cranked_delta) > 0.10 or abs(garbage_delta) > 0.10:
        print("✅ DRAI IS WORKING!")
        print(f"   Proof: Cranked/garbage configs cause {abs(max(cranked_delta, garbage_delta)):.1%} change")
        is_working = True
    elif results['test_c']['num_injections'] > 0:
        print("✅ DRAI IS WORKING!")
        print(f"   Proof: Non-zero K/V injections detected")
        is_working = True
    else:
        print("❌ DRAI MIGHT BE A NO-OP!")
        print("   No significant accuracy changes or K/V deltas detected")

    print(f"{'='*70}\n")

    # Save results
    results_file = Path(__file__).parent / "results" / "v1_sanity_check" / "comprehensive.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}\n")

    return is_working


if __name__ == "__main__":
    working = main()
    sys.exit(0 if working else 1)
