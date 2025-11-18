#!/usr/bin/env python3
"""
Comprehensive V1 DRAI Evaluation on pythia-2.8b

Full test suite including:
1. Baseline (no DRAI)
2. V1 with burn-in enabled (threshold=30.0, standard config)
3. V1 with burn-in disabled (threshold=0.0, to see immediate effects)
4. Cranked influence (3.0, proof of injection)
5. Garbage attractors (sanity check)

This will show if the +3.3% improvement on 410M scales to larger models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_story_generator import SimpleStoryGenerator

from src.drai import apply_drai_v1, get_v1_standard_config, get_drai_v1_stats
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


def evaluate_model(model, tokenizer, stories, model_name="Model"):
    """Evaluate model on simple stories."""
    results = []
    total_questions = 0

    print(f"\nEvaluating: {model_name}")
    for i, story in enumerate(stories, 1):
        print(f"[Story {i}/{len(stories)}]", end=" ")

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
                "question": question['question'],
                "expected": question['answer'],
                "generated": predicted_answer,
                "correct": correct
            })

            print("✓" if correct else "✗", end="")

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total_questions

    print(f"\n  Accuracy: {accuracy:.1%} ({correct_count}/{total_questions})\n")

    return accuracy, correct_count, total_questions


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE V1 DRAI EVALUATION: pythia-2.8b")
    print("="*70)
    print("\nTesting full suite on larger model (6.8x bigger than 410M)")
    print("to see if +3.3% improvement scales with model size.\n")
    print("="*70)

    model_name = "EleutherAI/pythia-2.8b"

    # Generate test stories
    print(f"\n[1/6] Generating 20 test stories...")
    generator = SimpleStoryGenerator(seed=42)
    stories = generator.generate_dataset(num_stories=20)
    total_questions = sum(len(story.questions) for story in stories)
    print(f"  ✓ Generated {len(stories)} stories ({total_questions} questions)")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    results = {}

    # Test 1: Baseline
    print(f"\n{'='*70}")
    print("[2/6] BASELINE (no DRAI)")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()

    baseline_acc, baseline_correct, baseline_total = evaluate_model(
        model, tokenizer, stories, "Baseline pythia-2.8b"
    )
    results["baseline"] = {
        "accuracy": baseline_acc,
        "correct": baseline_correct,
        "total": baseline_total
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 2: V1 with burn-in ENABLED (standard config)
    print(f"\n{'='*70}")
    print("[3/6] V1 DRAI (burn-in ENABLED, threshold=30.0)")
    print(f"{'='*70}")

    config = get_v1_standard_config()
    print(f"  Configuration:")
    print(f"    - Burn-in threshold: {config.hyperparameters.burn_in_threshold}")
    print(f"    - Max influence scale: {config.hyperparameters.max_influence_scale}")
    print(f"    - Layer mode: {config.layer_mode}")
    print(f"    - Max attractors: {config.hyperparameters.max_attractors}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    v1_burn_in_acc, v1_burn_in_correct, v1_burn_in_total = evaluate_model(
        model, tokenizer, stories, "V1 (burn-in enabled)"
    )
    v1_burn_in_stats = get_drai_v1_stats(model)
    results["v1_burn_in"] = {
        "accuracy": v1_burn_in_acc,
        "correct": v1_burn_in_correct,
        "total": v1_burn_in_total,
        "stats": v1_burn_in_stats
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 3: V1 with burn-in DISABLED
    print(f"\n{'='*70}")
    print("[4/6] V1 DRAI (burn-in DISABLED, threshold=0.0)")
    print(f"{'='*70}")

    config_no_burn_in = DraiV1Config(
        enabled=True,
        layer_mode="all",
        num_drai_heads=1,
        hyperparameters=DraiV1Hyperparameters(
            max_attractors=32,
            theta_match=0.75,
            burn_in_threshold=0.0,  # DISABLED!
            max_influence_scale=0.3,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    model = apply_drai_v1(model, config=config_no_burn_in)

    v1_no_burn_in_acc, v1_no_burn_in_correct, v1_no_burn_in_total = evaluate_model(
        model, tokenizer, stories, "V1 (no burn-in)"
    )
    results["v1_no_burn_in"] = {
        "accuracy": v1_no_burn_in_acc,
        "correct": v1_no_burn_in_correct,
        "total": v1_no_burn_in_total
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 4: CRANKED influence
    print(f"\n{'='*70}")
    print("[5/6] CRANKED INFLUENCE (3.0, no burn-in)")
    print(f"{'='*70}")
    print("  This should show clear degradation if injection works!")

    config_cranked = DraiV1Config(
        enabled=True,
        layer_mode="all",
        num_drai_heads=1,
        hyperparameters=DraiV1Hyperparameters(
            max_attractors=32,
            theta_match=0.75,
            burn_in_threshold=0.0,
            max_influence_scale=3.0,  # 10x HIGHER!
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    model = apply_drai_v1(model, config=config_cranked)

    cranked_acc, cranked_correct, cranked_total = evaluate_model(
        model, tokenizer, stories, "CRANKED (3.0)"
    )
    results["cranked"] = {
        "accuracy": cranked_acc,
        "correct": cranked_correct,
        "total": cranked_total
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 5: GARBAGE attractors
    print(f"\n{'='*70}")
    print("[6/6] GARBAGE ATTRACTORS (no burn-in)")
    print(f"{'='*70}")
    print("  Random noise attractors - should degrade if injection works!")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    model = apply_drai_v1(model, config=config_no_burn_in)

    # Inject garbage
    from src.drai.neox_integration_v1 import DraiGPTNeoXAttentionV1
    garbage_count = 0
    for layer in model.gpt_neox.layers:
        if isinstance(layer.attention, DraiGPTNeoXAttentionV1):
            drai = layer.attention.drai
            drai.attractor_vectors.data = torch.randn_like(drai.attractor_vectors) * 10.0
            drai.attractor_strengths.data = torch.ones_like(drai.attractor_strengths) * 100.0
            garbage_count += 1
    print(f"  ✓ Injected garbage into {garbage_count} layers")

    garbage_acc, garbage_correct, garbage_total = evaluate_model(
        model, tokenizer, stories, "GARBAGE"
    )
    results["garbage"] = {
        "accuracy": garbage_acc,
        "correct": garbage_correct,
        "total": garbage_total
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n{'='*70}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Test':<35} {'Accuracy':>12} {'Delta':>10}")
    print(f"{'-'*70}")
    print(f"{'Baseline':<35} {baseline_acc:>11.1%} {'':<10}")
    print(f"{'V1 (burn-in enabled)':<35} {v1_burn_in_acc:>11.1%} {(v1_burn_in_acc - baseline_acc):>+10.1%}")
    print(f"{'V1 (no burn-in)':<35} {v1_no_burn_in_acc:>11.1%} {(v1_no_burn_in_acc - baseline_acc):>+10.1%}")
    print(f"{'Cranked (3.0)':<35} {cranked_acc:>11.1%} {(cranked_acc - baseline_acc):>+10.1%}")
    print(f"{'Garbage':<35} {garbage_acc:>11.1%} {(garbage_acc - baseline_acc):>+10.1%}")
    print(f"{'-'*70}")

    # Determine verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    v1_delta = v1_burn_in_acc - baseline_acc
    cranked_delta = abs(cranked_acc - baseline_acc)
    garbage_delta = abs(garbage_acc - baseline_acc)

    # Check if injection works
    if cranked_delta > 0.10 or garbage_delta > 0.10:
        print(f"✅ INJECTION CONFIRMED WORKING!")
        print(f"   Cranked/garbage shows {max(cranked_delta, garbage_delta):.1%} effect")
    else:
        print(f"⚠️  WARNING: Injection effects unclear")

    # Check if V1 helps
    if v1_delta >= 0.05:
        print(f"\n🎉 V1 IMPROVES PERFORMANCE! (Δ = {v1_delta:+.1%})")
        print(f"   pythia-2.8b benefits from V1 DRAI!")
        verdict = "IMPROVEMENT"
    elif v1_delta >= -0.05:
        print(f"\n✅ V1 MAINTAINS PERFORMANCE (Δ = {v1_delta:+.1%})")
        print(f"   pythia-2.8b is stable with V1 DRAI")
        verdict = "SUCCESS"
    elif v1_delta >= -0.15:
        print(f"\n⚠️  MILD DEGRADATION (Δ = {v1_delta:+.1%})")
        verdict = "PARTIAL"
    else:
        print(f"\n❌ DEGRADATION (Δ = {v1_delta:+.1%})")
        verdict = "FAILURE"

    print(f"\n{'='*70}\n")

    # Save results
    results_dir = Path(__file__).parent / "results" / "v1_pythia2_8b_comprehensive"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "comprehensive_results.json", 'w') as f:
        json.dump({
            "model": model_name,
            "results": results,
            "verdict": verdict,
            "summary": {
                "baseline": baseline_acc,
                "v1_burn_in": v1_burn_in_acc,
                "v1_no_burn_in": v1_no_burn_in_acc,
                "cranked": cranked_acc,
                "garbage": garbage_acc,
                "v1_delta": v1_delta,
            }
        }, f, indent=2)

    print(f"Results saved to: {results_dir / 'comprehensive_results.json'}\n")

    return verdict in ["SUCCESS", "IMPROVEMENT"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
