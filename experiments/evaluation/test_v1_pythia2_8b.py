#!/usr/bin/env python3
"""
DRAI V1 Evaluation on pythia-2.8b

Testing V1 on a significantly larger model (6.8x bigger than 410M)
to see if larger capacity enables actual performance GAINS from
attractor memory, not just stability.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_story_generator import SimpleStoryGenerator

# Import V1
from src.drai import apply_drai_v1, get_v1_standard_config, get_drai_v1_stats


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


def evaluate_model(model, tokenizer, stories, model_name="Model"):
    """Evaluate model on simple stories."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    results = []
    total_questions = 0

    for i, story in enumerate(stories, 1):
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
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

            print("✓" if correct else "✗", end="")

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total_questions

    print(f"\n\n{'='*70}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"  Correct: {correct_count}/{total_questions}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"{'='*70}\n")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_questions,
        "results": results
    }


def main():
    print("\n" + "="*70)
    print("DRAI V1 EVALUATION: pythia-2.8b")
    print("="*70)
    print("\nTesting whether larger model capacity enables")
    print("performance GAINS from attractor memory.")
    print("\nModel size: 2.8B parameters (6.8x larger than 410M)")
    print("\n" + "="*70)

    model_name = "EleutherAI/pythia-2.8b"
    num_stories = 20

    # Generate test stories
    print(f"\n[1/4] Generating {num_stories} test stories...")
    generator = SimpleStoryGenerator(seed=42)
    stories = generator.generate_dataset(num_stories=num_stories)
    total_questions = sum(len(story.questions) for story in stories)
    print(f"  ✓ Generated {num_stories} stories ({total_questions} questions)")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Test baseline
    print(f"\n[2/4] Testing BASELINE (no DRAI)...")
    print(f"  Loading {model_name}...")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model_baseline.eval()
    print(f"  ✓ Model loaded")

    baseline_results = evaluate_model(
        model_baseline, tokenizer, stories,
        model_name=f"Baseline (pythia-2.8b)"
    )

    del model_baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test V1 DRAI
    print(f"\n[3/4] Testing V1 DRAI...")

    config = get_v1_standard_config()
    print(f"  Configuration (standard - aggressive):")
    print(f"    - Burn-in threshold: {config.hyperparameters.burn_in_threshold}")
    print(f"    - Max influence scale: {config.hyperparameters.max_influence_scale}")
    print(f"    - Layer mode: {config.layer_mode}")
    print(f"    - Max attractors: {config.hyperparameters.max_attractors}")
    print(f"  Loading {model_name}...")

    model_v1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model_v1.eval()
    print(f"  ✓ Model loaded")

    model_v1 = apply_drai_v1(model_v1, config=config)

    v1_results = evaluate_model(
        model_v1, tokenizer, stories,
        model_name=f"V1 DRAI (pythia-2.8b)"
    )

    v1_stats = get_drai_v1_stats(model_v1)

    del model_v1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compare
    print(f"\n[4/4] COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Model':<40} {'Accuracy':>12} {'Correct':>15}")
    print(f"{'-'*70}")
    print(f"{'Baseline (pythia-2.8b)':<40} {baseline_results['accuracy']:>11.1%} "
          f"{baseline_results['correct']:>6}/{baseline_results['total']:<6}")
    print(f"{'V1 DRAI (pythia-2.8b)':<40} {v1_results['accuracy']:>11.1%} "
          f"{v1_results['correct']:>6}/{v1_results['total']:<6}")
    print(f"{'-'*70}")

    delta = v1_results['accuracy'] - baseline_results['accuracy']
    delta_pct = delta * 100

    print(f"\nDelta: {delta:+.1%} ({delta_pct:+.1f} percentage points)")

    print(f"\nV1 Attractor Statistics:")
    print(f"  - Layers with DRAI: {v1_stats['num_drai_layers']}")
    print(f"  - Total active attractors: {v1_stats['total_active']}")
    print(f"  - Total strength: {v1_stats['total_strength']:.2f}")

    if len(v1_stats['layers']) > 0:
        avg_strength = v1_stats['total_strength'] / v1_stats['num_drai_layers']
        print(f"  - Avg strength per layer: {avg_strength:.2f}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT for pythia-2.8b")
    print(f"{'='*70}")

    if delta >= 0.05:
        print(f"🎉 IMPROVEMENT! V1 boosts performance (Δ = {delta:+.1%})")
        print(f"\n   This is the breakthrough:")
        print(f"   Larger model can leverage attractor memory for")
        print(f"   actual performance gains, not just stability!")
        verdict = "IMPROVEMENT"
    elif delta >= -0.05:
        print(f"✅ SUCCESS! V1 maintains performance (Δ = {delta:+.1%})")
        print(f"   V1 is stable and production-ready on 2.8B!")
        verdict = "SUCCESS"
    elif delta >= -0.15:
        print(f"⚠️  PARTIAL: Mild degradation (Δ = {delta:+.1%})")
        verdict = "PARTIAL"
    else:
        print(f"❌ FAILURE: Significant degradation (Δ = {delta:+.1%})")
        verdict = "FAILURE"

    print(f"{'='*70}\n")

    # Save results
    results_dir = Path(__file__).parent / "results" / "v1_larger_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "pythia_2_8b_results.json", 'w') as f:
        json.dump({
            "model_name": model_name,
            "baseline": baseline_results,
            "v1_drai": v1_results,
            "v1_stats": v1_stats,
            "delta": delta,
            "verdict": verdict,
            "config": {
                "burn_in_threshold": config.hyperparameters.burn_in_threshold,
                "max_influence_scale": config.hyperparameters.max_influence_scale,
                "layer_mode": config.layer_mode,
                "max_attractors": config.hyperparameters.max_attractors,
            }
        }, f, indent=2)

    print(f"Results saved to: {results_dir / 'pythia_2_8b_results.json'}\n")

    return verdict in ["SUCCESS", "IMPROVEMENT"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
