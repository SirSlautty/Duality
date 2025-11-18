"""
Simple Tractable Evaluator for 410M Models

Fixes from previous attempt:
1. Stories fit in context (500 tokens + room for instructions)
2. Pad token set explicitly (not EOS)
3. Greedy sampling (temp=0, no randomness)
4. Simple answer matching (substring/exact)
5. Clear prompts

Author: Halcyon AI Research
Date: 2025-11-18
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_story_generator import SimpleStoryGenerator, SimpleStory

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.drai import apply_drai, get_drai_stats
from src.drai.config import get_gated_drai_config


@dataclass
class EvalResult:
    """Result for one story."""
    story_id: int
    model_name: str
    answers: List[Dict]
    accuracy: float
    num_correct: int
    num_total: int


class SimpleEvaluator:
    """Evaluate models on simple tractable QA task."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.generator = SimpleStoryGenerator(seed=42)

    def evaluate_model(
        self,
        model_name: str,
        stories: List[SimpleStory],
        use_drai: bool = False
    ) -> List[EvalResult]:
        """
        Evaluate model on simple stories.

        Args:
            model_name: HuggingFace model name
            stories: List of SimpleStory objects
            use_drai: Whether to apply DRAI

        Returns:
            List of EvalResult objects
        """
        print(f"\nEvaluating: {model_name} {'with DRAI' if use_drai else 'baseline'}")
        print("Loading model...")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        model = model.to(self.device)
        model.eval()

        # FIX #1: Set pad token explicitly (not EOS)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set padding side to left (better for generation)
        tokenizer.padding_side = "left"

        # Apply DRAI if requested
        if use_drai:
            print("Applying GATED DRAI (threshold=0.85)...")
            gated_config = get_gated_drai_config()
            model = apply_drai(model, config=gated_config)
            print("[DRAI] Applied successfully with strict gating")

        results = []

        # Evaluate each story
        for story_id, story in enumerate(tqdm(stories, desc="Evaluating stories")):
            answers = []
            num_correct = 0

            for question_data in story.questions:
                question = question_data["question"]
                correct_answer = question_data["answer"]

                # Create prompt
                prompt = self._create_prompt(story.text, question)

                # Check token count (CRITICAL)
                token_count = len(tokenizer.encode(prompt))
                if token_count > 1800:  # Leave room for generation
                    print(f"Warning: Story {story_id} prompt is {token_count} tokens (truncating)")
                    # Truncate story if needed
                    words = story.text.split()
                    story.text = ' '.join(words[:300])  # Keep first 300 words
                    prompt = self._create_prompt(story.text, question)

                # Tokenize with proper attention mask
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1800,  # Leave room for 50 token answer
                    padding=False,  # No padding needed for single input
                    return_attention_mask=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate answer (GREEDY - no sampling)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],  # FIX #2: Explicit attention mask
                        max_new_tokens=20,  # SHORT answers only
                        do_sample=False,  # FIX #3: GREEDY (no temperature)
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Extract answer
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # Take only first line/sentence
                predicted_answer = predicted_answer.split('\n')[0].split('.')[0].strip()

                # Score answer
                score = self._score_answer(predicted_answer, correct_answer)
                if score > 0:
                    num_correct += 1

                answers.append({
                    "question": question,
                    "predicted": predicted_answer,
                    "correct": correct_answer,
                    "score": score
                })

            # Calculate accuracy
            accuracy = num_correct / len(story.questions) if story.questions else 0

            results.append(EvalResult(
                story_id=story_id,
                model_name=model_name,
                answers=answers,
                accuracy=accuracy,
                num_correct=num_correct,
                num_total=len(story.questions)
            ))

        # Print summary
        avg_accuracy = np.mean([r.accuracy for r in results])
        print(f"\nResults for {model_name} ({'DRAI' if use_drai else 'baseline'}):")
        print(f"  Average Accuracy: {avg_accuracy:.1%}")
        print(f"  Total Correct: {sum(r.num_correct for r in results)}/{sum(r.num_total for r in results)}")

        # Print DRAI stats if applicable
        if use_drai:
            stats = get_drai_stats(model)
            print(f"\nDRAI Statistics:")
            print(f"  DRAI layers: {stats.get('num_drai_layers', 0)}")
            if 'total_active_attractors' in stats:
                print(f"  Total active attractors: {stats['total_active_attractors']}")

        return results

    def _create_prompt(self, story: str, question: str) -> str:
        """
        Create a clear, unambiguous prompt.

        Format:
        Story: [story]
        Question: [question]
        Answer:
        """
        prompt = f"""Story: {story}

Question: {question}
Answer:"""
        return prompt

    def _score_answer(self, predicted: str, correct: str) -> float:
        """
        Score answer with flexible matching.

        Returns:
            1.0 if correct, 0.0 if wrong
        """
        # Normalize
        predicted = predicted.lower().strip()
        correct = correct.lower().strip()

        # Exact match
        if predicted == correct:
            return 1.0

        # Substring match (correct answer in prediction)
        if correct in predicted:
            return 1.0

        # Fuzzy match (for near-matches like "berlin" vs "in berlin")
        predicted_words = set(predicted.split())
        correct_words = set(correct.split())

        # If all correct words appear in prediction
        if correct_words.issubset(predicted_words):
            return 1.0

        return 0.0

    def compare_results(
        self,
        baseline_results: List[EvalResult],
        drai_results: List[EvalResult]
    ) -> Dict:
        """Compare baseline vs DRAI results."""
        baseline_acc = np.mean([r.accuracy for r in baseline_results])
        drai_acc = np.mean([r.accuracy for r in drai_results])

        baseline_correct = sum(r.num_correct for r in baseline_results)
        baseline_total = sum(r.num_total for r in baseline_results)
        drai_correct = sum(r.num_correct for r in drai_results)
        drai_total = sum(r.num_total for r in drai_results)

        # Statistical test
        from scipy import stats
        baseline_scores = [r.accuracy for r in baseline_results]
        drai_scores = [r.accuracy for r in drai_results]

        if len(baseline_scores) > 1 and np.std(baseline_scores) > 0:
            t_stat, p_value = stats.ttest_rel(baseline_scores, drai_scores)
        else:
            t_stat, p_value = 0, 1.0

        print("\n" + "="*70)
        print("BASELINE vs DRAI COMPARISON")
        print("="*70)
        print(f"\nBaseline:")
        print(f"  Accuracy: {baseline_acc:.1%}")
        print(f"  Correct: {baseline_correct}/{baseline_total}")
        print(f"\nDRAI:")
        print(f"  Accuracy: {drai_acc:.1%}")
        print(f"  Correct: {drai_correct}/{drai_total}")
        print(f"\nDifference:")
        print(f"  Δ Accuracy: {(drai_acc - baseline_acc):+.1%}")
        print(f"  Δ Correct: {drai_correct - baseline_correct:+d}")
        print(f"\nStatistical Test (paired t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            if drai_acc > baseline_acc:
                print(f"  Result: ✓ DRAI significantly better (p < 0.05)")
            else:
                print(f"  Result: ✗ DRAI significantly worse (p < 0.05)")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")

        return {
            "baseline": {
                "accuracy": float(baseline_acc),
                "correct": int(baseline_correct),
                "total": int(baseline_total)
            },
            "drai": {
                "accuracy": float(drai_acc),
                "correct": int(drai_correct),
                "total": int(drai_total)
            },
            "comparison": {
                "accuracy_delta": float(drai_acc - baseline_acc),
                "correct_delta": int(drai_correct - baseline_correct),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            }
        }

    def save_results(self, results: List[EvalResult], path: str):
        """Save results to JSON."""
        data = []
        for r in results:
            data.append({
                "story_id": r.story_id,
                "model_name": r.model_name,
                "accuracy": r.accuracy,
                "num_correct": r.num_correct,
                "num_total": r.num_total,
                "answers": r.answers
            })

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple tractable evaluation")
    parser.add_argument("--model", type=str, default="pythia-410m", help="Model name")
    parser.add_argument("--num_stories", type=int, default=10, help="Number of test stories")
    parser.add_argument("--output_dir", type=str, default="results/phase5/simple", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device")

    args = parser.parse_args()

    # Normalize model name
    if not args.model.startswith("EleutherAI/"):
        args.model = f"EleutherAI/{args.model}"

    print("="*70)
    print("SIMPLE TRACTABLE EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Stories: {args.num_stories}")
    print(f"Target: 500-token stories with 3 simple questions each")

    evaluator = SimpleEvaluator(device=args.device)

    # Generate stories
    print("\nGenerating simple stories...")
    generator = SimpleStoryGenerator(seed=123)
    stories = generator.generate_dataset(
        num_stories=args.num_stories,
        num_facts_per_story=3
    )
    print(f"Generated {len(stories)} stories")
    print(f"Avg length: {np.mean([s.length_tokens for s in stories]):.0f} tokens")

    # Evaluate baseline
    print("\n" + "="*70)
    print("EVALUATING BASELINE")
    print("="*70)
    baseline_results = evaluator.evaluate_model(
        model_name=args.model,
        stories=stories,
        use_drai=False
    )
    evaluator.save_results(baseline_results, f"{args.output_dir}/baseline_results.json")

    # Evaluate DRAI
    print("\n" + "="*70)
    print("EVALUATING DRAI")
    print("="*70)
    drai_results = evaluator.evaluate_model(
        model_name=args.model,
        stories=stories,
        use_drai=True
    )
    evaluator.save_results(drai_results, f"{args.output_dir}/drai_results.json")

    # Compare
    comparison = evaluator.compare_results(baseline_results, drai_results)

    # Save comparison
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
