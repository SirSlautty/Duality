#!/usr/bin/env python3
"""Create publication-quality perplexity visualization.

This script generates figures for the paper showing perplexity comparison
between baseline and DRAI models.

Usage:
    python plot_perplexity.py --input results/perplexity/perplexity_results.json
    python plot_perplexity.py --output figures/perplexity_comparison.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(input_file: str) -> dict:
    """Load perplexity results from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)


def plot_perplexity_comparison(results: dict, output_file: str):
    """Create perplexity comparison bar chart.

    Args:
        results: Results dictionary from evaluation
        output_file: Path to save figure
    """
    baseline_ppl = results['baseline']['perplexity']
    drai_ppl = results['drai']['perplexity']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars
    x = [0, 1]
    ppls = [baseline_ppl, drai_ppl]
    colors = ['#3498db', '#e74c3c']  # Blue for baseline, red for DRAI
    labels = ['Baseline\n(No DRAI)', 'DRAI\n(Phase 2)']

    bars = ax.bar(x, ppls, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, ppl) in enumerate(zip(bars, ppls)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax.set_title('WikiText-2 Perplexity: Baseline vs DRAI', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(ppls) * 1.2)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add significance annotation
    comparison = results['comparison']
    p_value = comparison['p_value']
    sig = comparison['significance']

    if sig != 'n.s.':
        # Draw bracket
        y_max = max(ppls) * 1.1
        ax.plot([0, 0, 1, 1], [y_max, y_max*1.02, y_max*1.02, y_max], 'k-', linewidth=1.5)
        ax.text(0.5, y_max*1.04, f'p = {p_value:.3f} {sig}',
                ha='center', va='bottom', fontsize=10)
    else:
        # No significance - add note
        ax.text(0.5, max(ppls) * 1.1, 'No significant difference',
                ha='center', va='bottom', fontsize=10, style='italic', color='gray')

    # Tight layout
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Viz] Perplexity comparison saved to: {output_file}")

    plt.close()


def plot_loss_distribution(results: dict, output_file: str):
    """Create loss distribution comparison.

    Args:
        results: Results dictionary from evaluation
        output_file: Path to save figure
    """
    baseline_losses = np.array(results['baseline']['sequence_losses'])
    drai_losses = np.array(results['drai']['sequence_losses'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    bins = np.linspace(min(baseline_losses.min(), drai_losses.min()),
                       max(baseline_losses.max(), drai_losses.max()),
                       50)

    ax.hist(baseline_losses, bins=bins, alpha=0.5, label='Baseline',
            color='#3498db', edgecolor='black', linewidth=0.5)
    ax.hist(drai_losses, bins=bins, alpha=0.5, label='DRAI',
            color='#e74c3c', edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Loss (per sequence)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Loss Distribution: Baseline vs DRAI', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add statistics
    stats_text = (
        f'Baseline: μ={baseline_losses.mean():.3f}, σ={baseline_losses.std():.3f}\n'
        f'DRAI: μ={drai_losses.mean():.3f}, σ={drai_losses.std():.3f}'
    )
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tight layout
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Viz] Loss distribution saved to: {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create perplexity visualizations')
    parser.add_argument('--input', type=str, default='results/perplexity/perplexity_results.json',
                        help='Input results JSON file')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                        help='Output directory for figures')

    args = parser.parse_args()

    print("=" * 70)
    print("PERPLEXITY VISUALIZATION")
    print("=" * 70)
    print()

    # Load results
    print(f"[Viz] Loading results from: {args.input}")
    results = load_results(args.input)

    # Create visualizations
    print("[Viz] Creating perplexity comparison chart...")
    plot_perplexity_comparison(
        results,
        os.path.join(args.output_dir, 'perplexity_comparison.png')
    )

    print("[Viz] Creating loss distribution plot...")
    plot_loss_distribution(
        results,
        os.path.join(args.output_dir, 'loss_distribution.png')
    )

    print()
    print("[Viz] Visualization complete!")
    print(f"[Viz] Figures saved to: {args.output_dir}")
    print()


if __name__ == '__main__':
    main()
