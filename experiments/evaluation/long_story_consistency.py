"""
Long-Story Consistency Evaluation

Evaluates model performance on long stories with planted facts and distractors.

Tests:
- Factual recall (can model remember planted facts?)
- Consistency maintenance (does model handle distractors correctly?)
- Robustness to context length

Compares:
- Baseline model
- DRAI-augmented model
- (Optional) RAG-augmented model

Author: Halcyon AI Research
Date: 2025-11-18
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from story_generator import StoryGenerator, Story

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.drai import apply_drai, get_drai_stats


@dataclass
class EvaluationResult:
    """Results for a single model on a story."""
    story_id: int
    model_name: str
    answers: List[Dict[str, any]]  # [{question, predicted, correct, score}, ...]
    accuracy: float
    consistency_score: float
    metadata: Dict[str, any]


class LongStoryEvaluator:
    """
    Evaluate models on long-story consistency tasks.

    Example:
        >>> evaluator = LongStoryEvaluator()
        >>> stories = evaluator.generate_test_stories(num_stories=50)
        >>> results_baseline = evaluator.evaluate_model(
        ...     model_name="EleutherAI/pythia-70m",
        ...     stories=stories,
        ...     use_drai=False
        ... )
        >>> results_drai = evaluator.evaluate_model(
        ...     model_name="EleutherAI/pythia-70m",
        ...     stories=stories,
        ...     use_drai=True
        ... )
        >>> evaluator.compare_results(results_baseline, results_drai)
    """

    def __init__(
        self,
        device: str = "cpu",
        max_new_tokens: int = 50,
        seed: int = 42
    ):
        """
        Initialize evaluator.

        Args:
            device: Device to run models on
            max_new_tokens: Max tokens to generate for answers
            seed: Random seed
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        self.story_generator = StoryGenerator(seed=seed)

    def generate_test_stories(
        self,
        num_stories: int = 50,
        varied_complexity: bool = True,
        save_path: Optional[str] = None
    ) -> List[Story]:
        """
        Generate test stories.

        Args:
            num_stories: Number of stories to generate
            varied_complexity: Vary complexity across stories
            save_path: Optional path to save stories

        Returns:
            List of Story objects
        """
        print(f"Generating {num_stories} test stories...")

        stories = self.story_generator.generate_dataset(
            num_stories=num_stories,
            varied_complexity=varied_complexity,
            save_path=save_path
        )

        return stories

    def evaluate_model(
        self,
        model_name: str,
        stories: List[Story],
        use_drai: bool = False,
        drai_config: Optional[dict] = None,
        enable_logging: bool = False
    ) -> List[EvaluationResult]:
        """
        Evaluate a model on stories.

        Args:
            model_name: HuggingFace model name
            stories: List of Story objects
            use_drai: Whether to apply DRAI
            drai_config: Optional DRAI configuration
            enable_logging: Enable DRAI logging

        Returns:
            List of EvaluationResult objects
        """
        print(f"\nEvaluating: {model_name} {'with DRAI' if use_drai else 'baseline'}")

        # Load model
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        # Apply DRAI if requested
        if use_drai:
            print("Applying DRAI...")
            from src.drai.config import get_full_drai_config
            config = get_full_drai_config()
            if drai_config:
                # Update config with custom settings
                for key, value in drai_config.items():
                    if hasattr(config.hyperparameters, key):
                        setattr(config.hyperparameters, key, value)

            model = apply_drai(model, config=config)

        model.eval()

        # Evaluate each story
        results = []

        for i, story in enumerate(tqdm(stories, desc="Evaluating stories")):
            result = self._evaluate_single_story(
                model, tokenizer, story, i, model_name, use_drai
            )
            results.append(result)

        # Print summary
        avg_accuracy = np.mean([r.accuracy for r in results])
        avg_consistency = np.mean([r.consistency_score for r in results])

        print(f"\nResults for {model_name} {'(DRAI)' if use_drai else '(baseline)'}:")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")
        print(f"  Average Consistency: {avg_consistency:.2%}")

        if use_drai:
            # Print DRAI stats
            stats = get_drai_stats(model)
            print(f"\nDRAI Statistics:")
            print(f"  DRAI layers: {stats['num_drai_layers']}")
            print(f"  Total active attractors: {stats['total_active']}")

        return results

    def _evaluate_single_story(
        self,
        model,
        tokenizer,
        story: Story,
        story_id: int,
        model_name: str,
        use_drai: bool
    ) -> EvaluationResult:
        """Evaluate model on a single story."""

        answers = []
        correct_count = 0
        total_count = len(story.questions)

        for question_data in story.questions:
            question = question_data["question"]
            correct_answer = question_data["answer"]

            # Construct prompt: story + question
            prompt = f"{story.text}\n\nQuestion: {question}\nAnswer:"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy for consistency
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract generated answer
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted_answer = generated.strip().split('\n')[0]  # Take first line

            # Score answer (simple string matching for now)
            is_correct = self._score_answer(predicted_answer, correct_answer, question_data)

            if is_correct:
                correct_count += 1

            answers.append({
                "question": question,
                "predicted": predicted_answer,
                "correct": correct_answer,
                "score": 1.0 if is_correct else 0.0,
                "type": question_data["type"]
            })

        # Calculate metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Consistency score: how well did model handle distractors?
        # For now, same as accuracy (can be refined later)
        consistency_score = accuracy

        metadata = {
            "story_length": story.metadata["length_tokens"],
            "num_facts": story.metadata["num_facts"],
            "num_distractors": story.metadata["num_distractors"],
            "theme": story.metadata["theme"]
        }

        return EvaluationResult(
            story_id=story_id,
            model_name=model_name,
            answers=answers,
            accuracy=accuracy,
            consistency_score=consistency_score,
            metadata=metadata
        )

    def _score_answer(
        self,
        predicted: str,
        correct: str,
        question_data: Dict
    ) -> bool:
        """
        Score predicted answer against correct answer.

        Simple fuzzy matching for now.
        """
        pred_lower = predicted.lower().strip()
        correct_lower = correct.lower().strip()

        # Exact match
        if pred_lower == correct_lower:
            return True

        # Contains match
        if correct_lower in pred_lower:
            return True

        # Handle numbers (codes, dates)
        if question_data.get("type") in ["factual_recall"]:
            # Extract numbers from both
            import re
            pred_numbers = re.findall(r'\d+', predicted)
            correct_numbers = re.findall(r'\d+', correct)

            if pred_numbers and correct_numbers:
                if pred_numbers[0] == correct_numbers[0]:
                    return True

        return False

    def compare_results(
        self,
        results_baseline: List[EvaluationResult],
        results_drai: List[EvaluationResult],
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Compare baseline vs DRAI results.

        Args:
            results_baseline: Baseline results
            results_drai: DRAI results
            save_path: Optional path to save comparison

        Returns:
            Comparison dictionary
        """
        print("\n" + "="*70)
        print("BASELINE vs DRAI COMPARISON")
        print("="*70)

        # Overall metrics
        baseline_acc = np.mean([r.accuracy for r in results_baseline])
        drai_acc = np.mean([r.accuracy for r in results_drai])

        baseline_cons = np.mean([r.consistency_score for r in results_baseline])
        drai_cons = np.mean([r.consistency_score for r in results_drai])

        print(f"\nAccuracy:")
        print(f"  Baseline: {baseline_acc:.2%}")
        print(f"  DRAI:     {drai_acc:.2%}")
        print(f"  Δ:        {(drai_acc - baseline_acc):.2%}")

        print(f"\nConsistency:")
        print(f"  Baseline: {baseline_cons:.2%}")
        print(f"  DRAI:     {drai_cons:.2%}")
        print(f"  Δ:        {(drai_cons - baseline_cons):.2%}")

        # Statistical test
        from scipy import stats

        baseline_scores = [r.accuracy for r in results_baseline]
        drai_scores = [r.accuracy for r in results_drai]

        t_stat, p_value = stats.ttest_rel(baseline_scores, drai_scores)

        print(f"\nStatistical Test (paired t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4f}")

        if p_value < 0.05:
            print(f"  Result: Significant difference (p < 0.05)")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")

        # Breakdown by number of distractors
        print(f"\nAccuracy by Number of Distractors:")
        distractor_levels = sorted(set(r.metadata["num_distractors"] for r in results_baseline))

        for num_dist in distractor_levels:
            baseline_subset = [r.accuracy for r in results_baseline if r.metadata["num_distractors"] == num_dist]
            drai_subset = [r.accuracy for r in results_drai if r.metadata["num_distractors"] == num_dist]

            if baseline_subset and drai_subset:
                print(f"  {num_dist} distractors:")
                print(f"    Baseline: {np.mean(baseline_subset):.2%}")
                print(f"    DRAI:     {np.mean(drai_subset):.2%}")
                print(f"    Δ:        {(np.mean(drai_subset) - np.mean(baseline_subset)):.2%}")

        # Save comparison
        comparison = {
            "baseline": {
                "accuracy": float(baseline_acc),
                "consistency": float(baseline_cons),
                "scores": baseline_scores
            },
            "drai": {
                "accuracy": float(drai_acc),
                "consistency": float(drai_cons),
                "scores": drai_scores
            },
            "comparison": {
                "accuracy_delta": float(drai_acc - baseline_acc),
                "consistency_delta": float(drai_cons - baseline_cons),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            },
            "by_distractors": {
                str(num_dist): {
                    "baseline": float(np.mean([r.accuracy for r in results_baseline if r.metadata["num_distractors"] == num_dist])),
                    "drai": float(np.mean([r.accuracy for r in results_drai if r.metadata["num_distractors"] == num_dist]))
                }
                for num_dist in distractor_levels
            }
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to {save_path}")

        return comparison

    def save_results(
        self,
        results: List[EvaluationResult],
        path: str
    ):
        """Save evaluation results to JSON."""
        results_data = []

        for result in results:
            results_data.append({
                "story_id": result.story_id,
                "model_name": result.model_name,
                "accuracy": result.accuracy,
                "consistency_score": result.consistency_score,
                "answers": result.answers,
                "metadata": result.metadata
            })

        with open(path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to {path}")


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("LONG-STORY CONSISTENCY EVALUATION")
    print("="*70)

    evaluator = LongStoryEvaluator(device="cpu")

    # Generate test stories (small set for demonstration)
    print("\nGenerating test stories...")
    stories = evaluator.generate_test_stories(
        num_stories=5,  # Small for demo
        varied_complexity=True,
        save_path="results/phase5/test_stories.json"
    )

    print(f"\nGenerated {len(stories)} stories")
    print(f"Length range: {min(s.metadata['length_tokens'] for s in stories)}-{max(s.metadata['length_tokens'] for s in stories)} tokens")

    # Evaluate baseline
    print("\n" + "="*70)
    print("Evaluating baseline model...")
    print("="*70)

    results_baseline = evaluator.evaluate_model(
        model_name="EleutherAI/pythia-70m",
        stories=stories,
        use_drai=False
    )

    evaluator.save_results(results_baseline, "results/phase5/baseline_results.json")

    # Evaluate DRAI
    print("\n" + "="*70)
    print("Evaluating DRAI model...")
    print("="*70)

    results_drai = evaluator.evaluate_model(
        model_name="EleutherAI/pythia-70m",
        stories=stories,
        use_drai=True
    )

    evaluator.save_results(results_drai, "results/phase5/drai_results.json")

    # Compare
    comparison = evaluator.compare_results(
        results_baseline,
        results_drai,
        save_path="results/phase5/comparison.json"
    )

    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)
