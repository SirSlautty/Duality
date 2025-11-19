"""
Lesioning Experiment - Prove Causal Importance of Attractors

Tests whether DRAI attractors are functionally relevant by lesioning them:
1. Full DRAI - Normal attractor dynamics
2. Zeroed - Attractors zeroed out (no resonance field)
3. Scrambled - Attractors randomly permuted (breaks structure)

If performance drops when attractors are lesioned/scrambled while perplexity
remains similar, this proves attractors are causally important.

Author: Halcyon AI Research
Date: 2025-11-18
"""

import torch
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from story_generator import StoryGenerator, Story

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.drai import apply_drai
from src.drai.config import get_full_drai_config
from long_story_consistency import LongStoryEvaluator, EvaluationResult


class LesioningExperiment:
    """
    Run lesioning experiments to prove causal importance of attractors.

    Compares:
    - Baseline (no DRAI)
    - Full DRAI (normal)
    - Zeroed DRAI (attractors disabled)
    - Scrambled DRAI (attractors randomized)
    """

    def __init__(self, device: str = "cpu", seed: int = 42):
        """
        Initialize lesioning experiment.

        Args:
            device: Device to run on
            seed: Random seed
        """
        self.device = device
        self.seed = seed
        self.evaluator = LongStoryEvaluator(device=device, seed=seed)

    def run_experiment(
        self,
        model_name: str,
        stories: List[Story],
        save_dir: str = "results/phase5/lesioning"
    ) -> Dict:
        """
        Run complete lesioning experiment.

        Args:
            model_name: HuggingFace model name
            stories: Test stories
            save_dir: Directory to save results

        Returns:
            Dictionary with all results
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("="*70)
        print("LESIONING EXPERIMENT")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Stories: {len(stories)}")
        print(f"Save directory: {save_dir}")
        print()

        # 1. Baseline (no DRAI)
        print("\n" + "-"*70)
        print("CONDITION 1/4: Baseline (no DRAI)")
        print("-"*70)

        results_baseline = self.evaluator.evaluate_model(
            model_name=model_name,
            stories=stories,
            use_drai=False
        )

        self.evaluator.save_results(
            results_baseline,
            f"{save_dir}/baseline_results.json"
        )

        # 2. Full DRAI (normal)
        print("\n" + "-"*70)
        print("CONDITION 2/4: Full DRAI (normal attractor dynamics)")
        print("-"*70)

        results_full = self._evaluate_with_lesioning(
            model_name, stories, lesion_mode=None, label="full"
        )

        self.evaluator.save_results(
            results_full,
            f"{save_dir}/full_drai_results.json"
        )

        # 3. Zeroed DRAI
        print("\n" + "-"*70)
        print("CONDITION 3/4: Zeroed DRAI (attractors disabled)")
        print("-"*70)

        results_zeroed = self._evaluate_with_lesioning(
            model_name, stories, lesion_mode="zero", label="zeroed"
        )

        self.evaluator.save_results(
            results_zeroed,
            f"{save_dir}/zeroed_drai_results.json"
        )

        # 4. Scrambled DRAI
        print("\n" + "-"*70)
        print("CONDITION 4/4: Scrambled DRAI (attractors randomized)")
        print("-"*70)

        results_scrambled = self._evaluate_with_lesioning(
            model_name, stories, lesion_mode="scramble", label="scrambled"
        )

        self.evaluator.save_results(
            results_scrambled,
            f"{save_dir}/scrambled_drai_results.json"
        )

        # Compare all conditions
        print("\n" + "="*70)
        print("COMPARISON ACROSS ALL CONDITIONS")
        print("="*70)

        comparison = self._compare_all_conditions(
            results_baseline,
            results_full,
            results_zeroed,
            results_scrambled
        )

        # Save comparison
        with open(f"{save_dir}/comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        # Print summary
        self._print_summary(comparison)

        return comparison

    def _evaluate_with_lesioning(
        self,
        model_name: str,
        stories: List[Story],
        lesion_mode: str,
        label: str
    ) -> List[EvaluationResult]:
        """Evaluate model with specific lesioning mode."""

        print(f"Loading model with lesion_mode={lesion_mode}...")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        # Apply DRAI with lesioning
        config = get_full_drai_config()

        # Need to inject DRAI manually to set lesion_mode
        from src.drai.injection import inject_drai_into_model

        model = inject_drai_into_model(
            model,
            config,
            lesion_mode=lesion_mode,
            verbose=True
        )

        model.eval()

        # Evaluate
        results = []
        for i, story in enumerate(tqdm(stories, desc=f"Evaluating ({label})")):
            result = self.evaluator._evaluate_single_story(
                model, tokenizer, story, i, f"{model_name}_{label}", True
            )
            results.append(result)

        # Print results
        avg_acc = np.mean([r.accuracy for r in results])
        avg_cons = np.mean([r.consistency_score for r in results])

        print(f"\nResults ({label}):")
        print(f"  Accuracy: {avg_acc:.2%}")
        print(f"  Consistency: {avg_cons:.2%}")

        return results

    def _compare_all_conditions(
        self,
        results_baseline,
        results_full,
        results_zeroed,
        results_scrambled
    ) -> Dict:
        """Compare all four conditions."""

        from scipy import stats

        # Extract accuracies
        acc_baseline = np.array([r.accuracy for r in results_baseline])
        acc_full = np.array([r.accuracy for r in results_full])
        acc_zeroed = np.array([r.accuracy for r in results_zeroed])
        acc_scrambled = np.array([r.accuracy for r in results_scrambled])

        # Means
        means = {
            "baseline": float(acc_baseline.mean()),
            "full_drai": float(acc_full.mean()),
            "zeroed_drai": float(acc_zeroed.mean()),
            "scrambled_drai": float(acc_scrambled.mean())
        }

        # Standard deviations
        stds = {
            "baseline": float(acc_baseline.std()),
            "full_drai": float(acc_full.std()),
            "zeroed_drai": float(acc_zeroed.std()),
            "scrambled_drai": float(acc_scrambled.std())
        }

        # Pairwise comparisons (t-tests)
        comparisons = {}

        # Full vs Baseline
        t, p = stats.ttest_rel(acc_full, acc_baseline)
        comparisons["full_vs_baseline"] = {
            "t_stat": float(t),
            "p_value": float(p),
            "significant": p < 0.05,
            "delta": float(acc_full.mean() - acc_baseline.mean())
        }

        # Full vs Zeroed
        t, p = stats.ttest_rel(acc_full, acc_zeroed)
        comparisons["full_vs_zeroed"] = {
            "t_stat": float(t),
            "p_value": float(p),
            "significant": p < 0.05,
            "delta": float(acc_full.mean() - acc_zeroed.mean())
        }

        # Full vs Scrambled
        t, p = stats.ttest_rel(acc_full, acc_scrambled)
        comparisons["full_vs_scrambled"] = {
            "t_stat": float(t),
            "p_value": float(p),
            "significant": p < 0.05,
            "delta": float(acc_full.mean() - acc_scrambled.mean())
        }

        return {
            "means": means,
            "stds": stds,
            "comparisons": comparisons,
            "conclusion": self._draw_conclusion(means, comparisons)
        }

    def _draw_conclusion(self, means, comparisons) -> str:
        """Draw conclusion from results."""

        full = means["full_drai"]
        zeroed = means["zeroed_drai"]
        scrambled = means["scrambled_drai"]

        full_vs_zeroed_sig = comparisons["full_vs_zeroed"]["significant"]
        full_vs_scrambled_sig = comparisons["full_vs_scrambled"]["significant"]

        if full_vs_zeroed_sig and full > zeroed:
            return "STRONG: Zeroing attractors significantly degrades performance. Attractors are causally important."
        elif full_vs_scrambled_sig and full > scrambled:
            return "STRONG: Scrambling attractors significantly degrades performance. Attractor structure matters."
        elif full > zeroed and full > scrambled:
            return "MODERATE: Full DRAI performs better than lesioned variants, but not statistically significant."
        elif full < zeroed or full < scrambled:
            return "UNEXPECTED: Full DRAI performs worse than lesioned variants. Attractors may be harmful."
        else:
            return "NULL: No significant differences. Attractors may be inert."

    def _print_summary(self, comparison: Dict):
        """Print summary of results."""

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print("\nAccuracy by Condition:")
        for condition, acc in comparison["means"].items():
            std = comparison["stds"][condition]
            print(f"  {condition:20s}: {acc:.2%} ± {std:.2%}")

        print("\nPairwise Comparisons:")
        for name, comp in comparison["comparisons"].items():
            sig_marker = "***" if comp["significant"] else "   "
            print(f"  {name:25s}: Δ = {comp['delta']:+.2%}, p = {comp['p_value']:.4f} {sig_marker}")

        print("\nConclusion:")
        print(f"  {comparison['conclusion']}")

        print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("LESIONING EXPERIMENT - Prove Causal Importance")
    print("="*70)

    experiment = LesioningExperiment(device="cpu")

    # Generate test stories
    print("\nGenerating test stories...")
    stories = experiment.evaluator.generate_test_stories(
        num_stories=10,  # Small set for demo
        varied_complexity=True,
        save_path="results/phase5/lesioning/test_stories.json"
    )

    print(f"Generated {len(stories)} stories")
    print(f"Length range: {min(s.metadata['length_tokens'] for s in stories)}-{max(s.metadata['length_tokens'] for s in stories)} tokens")

    # Run experiment
    results = experiment.run_experiment(
        model_name="EleutherAI/pythia-70m",
        stories=stories,
        save_dir="results/phase5/lesioning"
    )

    print("\n" + "="*70)
    print("Experiment complete!")
    print("Results saved to results/phase5/lesioning/")
    print("="*70)
