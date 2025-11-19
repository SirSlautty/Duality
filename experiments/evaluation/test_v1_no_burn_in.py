#!/usr/bin/env python3
"""
V1 DRAI Test with BURN-IN DISABLED

Now that we know burn-in is preventing ALL injection during short evals,
let's test with burn-in disabled to see what DRAI actually does!

Tests:
1. Baseline (no DRAI)
2. V1 with burn-in DISABLED (threshold=0.0)
3. V1 with CRANKED influence (3.0) and no burn-in
4. V1 with GARBAGE attractors and no burn-in

This will show if DRAI can actually affect the model when active.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_story_generator import SimpleStoryGenerator

from src.drai import apply_drai_v1, get_drai_v1_stats
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
    print("V1 DRAI with BURN-IN DISABLED")
    print("="*70)
    print("\nBurn-in was preventing ALL injection during short evals.")
    print("Let's see what DRAI actually does when active!\n")
    print("="*70)

    model_name = "EleutherAI/pythia-410m"

    # Generate test stories
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

    results = {}

    # Test 1: Baseline
    print(f"\n{'='*70}")
    print("TEST 1: BASELINE (no DRAI)")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    baseline_acc, baseline_correct, baseline_total = evaluate_model(
        model, tokenizer, stories, "Baseline"
    )
    results["baseline"] = {
        "accuracy": baseline_acc,
        "correct": baseline_correct,
        "total": baseline_total
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 2: V1 with NO burn-in
    print(f"\n{'='*70}")
    print("TEST 2: V1 DRAI (burn-in DISABLED)")
    print(f"{'='*70}")

    config = DraiV1Config(
        enabled=True,
        layer_mode="mid",
        num_drai_heads=1,
        hyperparameters=DraiV1Hyperparameters(
            max_attractors=16,
            theta_match=0.8,
            burn_in_threshold=0.0,  # DISABLED!
            max_influence_scale=0.15,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    v1_acc, v1_correct, v1_total = evaluate_model(
        model, tokenizer, stories, "V1 (no burn-in)"
    )
    v1_stats = get_drai_v1_stats(model)
    results["v1_no_burn_in"] = {
        "accuracy": v1_acc,
        "correct": v1_correct,
        "total": v1_total,
        "stats": v1_stats
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 3: CRANKED influence
    print(f"\n{'='*70}")
    print("TEST 3: CRANKED INFLUENCE (3.0, no burn-in)")
    print(f"{'='*70}")

    config_cranked = DraiV1Config(
        enabled=True,
        layer_mode="mid",
        num_drai_heads=1,
        hyperparameters=DraiV1Hyperparameters(
            max_attractors=16,
            theta_match=0.8,
            burn_in_threshold=0.0,  # DISABLED!
            max_influence_scale=3.0,  # 20x HIGHER!
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
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

    # Test 4: GARBAGE attractors
    print(f"\n{'='*70}")
    print("TEST 4: GARBAGE ATTRACTORS (no burn-in)")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = apply_drai_v1(model, config=config)

    # Inject garbage
    from src.drai.neox_integration_v1 import DraiGPTNeoXAttentionV1
    for layer in model.gpt_neox.layers:
        if isinstance(layer.attention, DraiGPTNeoXAttentionV1):
            drai = layer.attention.drai
            drai.attractor_vectors.data = torch.randn_like(drai.attractor_vectors) * 10.0
            drai.attractor_strengths.data = torch.ones_like(drai.attractor_strengths) * 100.0

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
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Test':<30} {'Accuracy':>12} {'Delta':>10}")
    print(f"{'-'*70}")
    print(f"{'Baseline':<30} {baseline_acc:>11.1%} {'':<10}")
    print(f"{'V1 (no burn-in)':<30} {v1_acc:>11.1%} {(v1_acc - baseline_acc):>+10.1%}")
    print(f"{'Cranked (3.0)':<30} {cranked_acc:>11.1%} {(cranked_acc - baseline_acc):>+10.1%}")
    print(f"{'Garbage':<30} {garbage_acc:>11.1%} {(garbage_acc - baseline_acc):>+10.1%}")
    print(f"{'-'*70}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    v1_delta = abs(v1_acc - baseline_acc)
    cranked_delta = abs(cranked_acc - baseline_acc)
    garbage_delta = abs(garbage_acc - baseline_acc)

    if v1_delta < 0.05 and cranked_delta < 0.05 and garbage_delta < 0.05:
        print("❌ DRAI STILL APPEARS TO BE A NO-OP!")
        print("   Even with burn-in disabled, no effects detected.")
        verdict = "NO-OP"
    else:
        print("✅ DRAI IS WORKING!")
        print(f"   Effects detected:")
        if v1_delta > 0.05:
            print(f"   - V1 changes accuracy by {v1_delta:.1%}")
        if cranked_delta > 0.05:
            print(f"   - Cranked influence causes {cranked_delta:.1%} change")
        if garbage_delta > 0.05:
            print(f"   - Garbage attractors cause {garbage_delta:.1%} change")
        verdict = "WORKING"

    print(f"\n{'='*70}\n")

    # Save results
    results_file = Path(__file__).parent / "results" / "v1_no_burn_in" / "results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "verdict": verdict
        }, f, indent=2)

    print(f"Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
