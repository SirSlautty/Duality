#!/usr/bin/env python3
"""Perplexity evaluation on WikiText-2 for DRAI vs baseline comparison.

This script implements rigorous perplexity evaluation following standard
protocols for language model benchmarking. Results are suitable for
publication in academic papers.

Methodology:
1. Load WikiText-2 test set (standard benchmark)
2. Evaluate baseline model (no DRAI)
3. Evaluate DRAI model (with attractor dynamics)
4. Compute perplexity, bits-per-character, loss statistics
5. Statistical significance testing
6. Save results with complete methodology documentation

Usage:
    python evaluate_perplexity.py --model pythia-70m --output results/
    python evaluate_perplexity.py --model pythia-125m --max-length 512
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.drai.config import DraiConfig, get_full_drai_config
from src.drai.build_drai_neox import build_drai_neox_model


def load_wikitext2(split='test', max_samples=None):
    """Load WikiText-2 dataset.

    Args:
        split: 'test', 'train', or 'validation'
        max_samples: Limit number of samples (for testing)

    Returns:
        List of text samples
    """
    print(f"[Eval] Loading WikiText-2 ({split} split)...")

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Filter empty lines
    texts = [item['text'] for item in dataset if item['text'].strip()]

    if max_samples:
        texts = texts[:max_samples]

    print(f"[Eval] Loaded {len(texts)} samples")

    return texts


def evaluate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    stride: int = 256,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """Evaluate perplexity on a list of texts.

    Uses sliding window approach for long sequences.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of text samples
        max_length: Maximum sequence length
        stride: Sliding window stride
        device: Device to run on
        verbose: Print progress

    Returns:
        Dictionary with perplexity and detailed statistics
    """
    model.eval()
    model.to(device)

    all_losses = []
    all_token_counts = []
    all_seq_losses = []

    if verbose:
        texts = tqdm(texts, desc="Evaluating")

    with torch.no_grad():
        for text in texts:
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors='pt',
                truncation=False,  # We'll handle long sequences
                add_special_tokens=True,
            )

            input_ids = encodings['input_ids'].to(device)
            seq_len = input_ids.size(1)

            # Handle long sequences with sliding window
            if seq_len > max_length:
                # Sliding window
                total_loss = 0
                total_tokens = 0

                for start_idx in range(0, seq_len - 1, stride):
                    end_idx = min(start_idx + max_length, seq_len)
                    chunk = input_ids[:, start_idx:end_idx]

                    # Compute loss
                    outputs = model(chunk, labels=chunk)
                    loss = outputs.loss

                    # Weight by number of tokens
                    n_tokens = chunk.size(1) - 1  # Exclude first token (no loss)
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens

                # Average loss for this sequence
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                all_seq_losses.append(avg_loss)
                all_losses.append(total_loss)
                all_token_counts.append(total_tokens)

            else:
                # Short sequence - process normally
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

                n_tokens = seq_len - 1
                all_seq_losses.append(loss.item())
                all_losses.append(loss.item() * n_tokens)
                all_token_counts.append(n_tokens)

    # Compute overall statistics
    total_loss = sum(all_losses)
    total_tokens = sum(all_token_counts)

    # Perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    # Bits per character (approximate)
    # Assuming avg 4 chars per token (rough estimate)
    bits_per_char = avg_loss / math.log(2) / 4

    # Sequence-level statistics
    seq_losses_array = np.array(all_seq_losses)

    results = {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'bits_per_char': bits_per_char,
        'total_tokens': total_tokens,
        'num_sequences': len(texts),

        # Distribution statistics
        'loss_mean': seq_losses_array.mean(),
        'loss_std': seq_losses_array.std(),
        'loss_median': np.median(seq_losses_array),
        'loss_min': seq_losses_array.min(),
        'loss_max': seq_losses_array.max(),

        # Per-sequence losses (for significance testing)
        'sequence_losses': all_seq_losses,
    }

    return results


def compare_results(baseline_results: Dict, drai_results: Dict) -> Dict:
    """Compare baseline and DRAI results with statistical testing.

    Args:
        baseline_results: Results from baseline model
        drai_results: Results from DRAI model

    Returns:
        Dictionary with comparison statistics
    """
    # Perplexity delta
    ppl_delta = drai_results['perplexity'] - baseline_results['perplexity']
    ppl_percent = (ppl_delta / baseline_results['perplexity']) * 100

    # Loss delta
    loss_delta = drai_results['avg_loss'] - baseline_results['avg_loss']

    # Statistical significance (paired t-test on sequence losses)
    baseline_losses = np.array(baseline_results['sequence_losses'])
    drai_losses = np.array(drai_results['sequence_losses'])

    # Ensure same length (should be from same dataset)
    assert len(baseline_losses) == len(drai_losses), "Mismatched sequence counts"

    # Paired differences
    differences = drai_losses - baseline_losses

    # T-test
    from scipy.stats import ttest_rel

    t_stat, p_value = ttest_rel(baseline_losses, drai_losses)

    # Effect size (Cohen's d)
    mean_diff = differences.mean()
    std_diff = differences.std()
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    # Determine significance level
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'n.s.'

    comparison = {
        'perplexity_delta': ppl_delta,
        'perplexity_percent_change': ppl_percent,
        'loss_delta': loss_delta,

        # Statistical test
        't_statistic': t_stat,
        'p_value': p_value,
        'significance': significance,
        'cohens_d': cohens_d,

        # Interpretation
        'is_significant': p_value < 0.05,
        'is_better': ppl_delta < 0,  # Lower perplexity = better
        'is_worse': ppl_delta > 0,
        'is_neutral': abs(ppl_delta) < 1.0,  # Within 1.0 PPL
    }

    return comparison


def print_results(baseline_results: Dict, drai_results: Dict, comparison: Dict):
    """Print results in a nice format."""
    print("\n" + "=" * 70)
    print("PERPLEXITY EVALUATION RESULTS")
    print("=" * 70)
    print()

    print("BASELINE (No DRAI):")
    print(f"  Perplexity:       {baseline_results['perplexity']:.2f}")
    print(f"  Avg Loss:         {baseline_results['avg_loss']:.4f}")
    print(f"  Bits/Char:        {baseline_results['bits_per_char']:.4f}")
    print(f"  Total Tokens:     {baseline_results['total_tokens']:,}")
    print(f"  Num Sequences:    {baseline_results['num_sequences']:,}")
    print()

    print("WITH DRAI:")
    print(f"  Perplexity:       {drai_results['perplexity']:.2f}")
    print(f"  Avg Loss:         {drai_results['avg_loss']:.4f}")
    print(f"  Bits/Char:        {drai_results['bits_per_char']:.4f}")
    print(f"  Total Tokens:     {drai_results['total_tokens']:,}")
    print(f"  Num Sequences:    {drai_results['num_sequences']:,}")
    print()

    print("COMPARISON:")
    print(f"  PPL Delta:        {comparison['perplexity_delta']:+.2f}")
    print(f"  PPL Change:       {comparison['perplexity_percent_change']:+.2f}%")
    print(f"  Loss Delta:       {comparison['loss_delta']:+.4f}")
    print()

    print("STATISTICAL SIGNIFICANCE:")
    print(f"  t-statistic:      {comparison['t_statistic']:.4f}")
    print(f"  p-value:          {comparison['p_value']:.4f} {comparison['significance']}")
    print(f"  Cohen's d:        {comparison['cohens_d']:.4f}")
    print()

    # Interpretation
    print("INTERPRETATION:")
    if comparison['is_better']:
        print(f"  ✓ DRAI IMPROVES perplexity by {-comparison['perplexity_delta']:.2f}")
    elif comparison['is_neutral']:
        print(f"  ≈ DRAI has MINIMAL IMPACT (within 1.0 PPL)")
    else:
        print(f"  ✗ DRAI INCREASES perplexity by {comparison['perplexity_delta']:.2f}")

    if comparison['is_significant']:
        print(f"  Difference is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Difference is NOT statistically significant")

    print()
    print("=" * 70)


def save_results(
    output_dir: str,
    baseline_results: Dict,
    drai_results: Dict,
    comparison: Dict,
    config: Dict,
):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    # Create comprehensive results dict
    results = {
        'config': config,
        'baseline': convert_to_json_serializable(baseline_results),
        'drai': convert_to_json_serializable(drai_results),
        'comparison': convert_to_json_serializable(comparison),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save to JSON
    output_file = os.path.join(output_dir, 'perplexity_results.json')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[Eval] Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate perplexity on WikiText-2')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m',
                        help='Model name or path')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples (for testing)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=256,
                        help='Sliding window stride')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default='results/perplexity',
                        help='Output directory')
    parser.add_argument('--no-drai', action='store_true',
                        help='Skip DRAI evaluation (baseline only)')

    args = parser.parse_args()

    print("=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70)
    print()
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Max length: {args.max_length}")
    print(f"Stride: {args.stride}")
    print()

    # Load dataset
    texts = load_wikitext2(split='test', max_samples=args.max_samples)

    # Load tokenizer
    print(f"[Eval] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # === BASELINE EVALUATION ===
    print()
    print("[Eval] ===== BASELINE EVALUATION (No DRAI) =====")
    print()

    print("[Eval] Loading baseline model...")
    model_baseline = build_drai_neox_model(
        args.model,
        drai_config=None,  # No DRAI
        torch_dtype=torch.float32,
        device_map=args.device,
        verbose=False,
    )

    print("[Eval] Evaluating baseline...")
    baseline_results = evaluate_perplexity(
        model_baseline,
        tokenizer,
        texts,
        max_length=args.max_length,
        stride=args.stride,
        device=args.device,
    )

    print(f"[Eval] Baseline perplexity: {baseline_results['perplexity']:.2f}")

    # Clean up
    del model_baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.no_drai:
        print("[Eval] Skipping DRAI evaluation (--no-drai flag)")
        return

    # === DRAI EVALUATION ===
    print()
    print("[Eval] ===== DRAI EVALUATION =====")
    print()

    print("[Eval] Loading DRAI model...")
    drai_config = get_full_drai_config()
    model_drai = build_drai_neox_model(
        args.model,
        drai_config=drai_config,
        torch_dtype=torch.float32,
        device_map=args.device,
        verbose=False,
    )

    print("[Eval] Evaluating DRAI...")
    drai_results = evaluate_perplexity(
        model_drai,
        tokenizer,
        texts,
        max_length=args.max_length,
        stride=args.stride,
        device=args.device,
    )

    print(f"[Eval] DRAI perplexity: {drai_results['perplexity']:.2f}")

    # === COMPARISON ===
    print()
    print("[Eval] ===== STATISTICAL COMPARISON =====")
    print()

    comparison = compare_results(baseline_results, drai_results)

    # Print results
    print_results(baseline_results, drai_results, comparison)

    # Save results
    config = {
        'model': args.model,
        'max_length': args.max_length,
        'stride': args.stride,
        'num_samples': len(texts),
        'drai_config': {
            'enabled': True,
            'phase': drai_config.phase,
            'num_drai_heads': drai_config.num_drai_heads,
            'hyperparameters': {
                'max_attractors': drai_config.hyperparameters.max_attractors,
                'coherence_threshold': drai_config.hyperparameters.coherence_threshold,
                'formation_threshold': drai_config.hyperparameters.formation_threshold,
                'decay_rate': drai_config.hyperparameters.decay_rate,
                'ema_momentum': drai_config.hyperparameters.ema_momentum,
            }
        }
    }

    save_results(args.output, baseline_results, drai_results, comparison, config)

    print()
    print("[Eval] Evaluation complete!")


if __name__ == '__main__':
    main()
