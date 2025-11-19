#!/usr/bin/env python3
"""
DRAI V1 Evaluation on pythia-1b

Test V1 on a larger model to see if it provides performance GAINS
rather than just maintaining parity.

Expected:
- Baseline: ~90-95% (larger models perform better)
- V1 DRAI: ~90-98% (possible improvement from memory!)

Model: pythia-1b (1.4B params, 2.4x larger than 410m)
Config: Standard V1 config (less conservative than 410m)
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
                    do_sample=False,  # Greedy
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
    print("DRAI V1 EVALUATION: pythia-1b")
    print("="*70)
    print("\nTesting V1 on larger model to see if it provides GAINS.")
    print("\nExpected:")
    print("  - Baseline: ~90-95% (larger models better)")
    print("  - V1 DRAI: ~90-98% (possible improvement!)")
    print("\n" + "="*70)

    model_name = "EleutherAI/pythia-1b"
    num_stories = 20

    print(f"\n[1/4] Generating {num_stories} test stories...")
    generator = SimpleStoryGenerator(seed=42)
    stories = generator.generate_dataset(num_stories=num_stories)
    total_questions = sum(len(story.questions) for story in stories)
    print(f"  ✓ Generated {num_stories} stories ({total_questions} questions)")

    print(f"\n[2/4] Testing BASELINE (no DRAI)...")
    print(f"  Loading pythia-1b (this may take a minute)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model_baseline.eval()

    baseline_results = evaluate_model(
        model_baseline, tokenizer, stories,
        model_name="Baseline pythia-1b (no DRAI)"
    )

    del model_baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[3/4] Testing V1 DRAI...")
    print(f"  Configuration: STANDARD (for 1B models)")

    config = get_v1_standard_config()
    print(f"    - Max attractors: {config.hyperparameters.max_attractors}")
    print(f"    - Burn-in threshold: {config.hyperparameters.burn_in_threshold}")
    print(f"    - Max influence scale: {config.hyperparameters.max_influence_scale}")
    print(f"    - Layer mode: {config.layer_mode}")
    print(f"    - Theta match: {config.hyperparameters.theta_match}")

    print(f"  Loading pythia-1b (this may take a minute)...")
    model_v1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model_v1.eval()

    model_v1 = apply_drai_v1(model_v1, config=config)

    v1_results = evaluate_model(
        model_v1, tokenizer, stories,
        model_name="V1 DRAI pythia-1b (standard config)"
    )

    v1_stats = get_drai_v1_stats(model_v1)

    del model_v1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[4/4] FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Model':<35} {'Accuracy':>12} {'Correct':>15}")
    print(f"{'-'*70}")
    print(f"{'Baseline pythia-1b':<35} {baseline_results['accuracy']:>11.1%} "
          f"{baseline_results['correct']:>6}/{baseline_results['total']:<6}")
    print(f"{'V1 DRAI pythia-1b':<35} {v1_results['accuracy']:>11.1%} "
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
        layer_stats = v1_stats['layers'][0]
        print(f"  - Burn-in complete: {'YES' if not layer_stats['burn_in_active'] else 'NO'}")
        print(f"  - Timesteps processed: {int(layer_stats['timestep'])}")

    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if delta >= 0.05:  # 5% improvement
        print(f"🎉 OUTSTANDING! V1 IMPROVES performance (Δ = {delta:+.1%})")
        print(f"\nV1 provides tangible benefits on 1B models!")
        print(f"The attractor memory is helping the model!")
        verdict = "IMPROVEMENT"
    elif delta >= -0.02:  # Within 2% (tighter bound for 1B)
        print(f"✅ SUCCESS! V1 maintains performance (Δ = {delta:+.1%})")
        print(f"\nV1 is stable on 1B models!")
        verdict = "SUCCESS"
    elif delta >= -0.10:  # 2-10% degradation
        print(f"⚠️  PARTIAL: Mild degradation (Δ = {delta:+.1%})")
        print(f"\nNeeds tuning for 1B models.")
        verdict = "PARTIAL"
    else:
        print(f"❌ FAILURE: Degradation (Δ = {delta:+.1%})")
        verdict = "FAILURE"

    print(f"{'='*70}\n")

    results_file = Path(__file__).parent / "results" / "v1_pythia1b" / "comparison.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "model": "pythia-1b",
            "baseline": baseline_results,
            "v1_drai": v1_results,
            "v1_stats": v1_stats,
            "delta": delta,
            "verdict": verdict,
            "config": {
                "max_attractors": config.hyperparameters.max_attractors,
                "burn_in_threshold": config.hyperparameters.burn_in_threshold,
                "max_influence_scale": config.hyperparameters.max_influence_scale,
                "theta_match": config.hyperparameters.theta_match,
                "layer_mode": config.layer_mode,
            }
        }, f, indent=2)

    print(f"Results saved to: {results_file}\n")

    return verdict in ["SUCCESS", "IMPROVEMENT"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
