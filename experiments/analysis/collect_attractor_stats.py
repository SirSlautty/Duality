#!/usr/bin/env python3
"""Collect attractor statistics during DRAI inference.

This script runs text generation with DRAI and collects detailed statistics
about attractor formation, evolution, and dynamics.

Usage:
    python collect_attractor_stats.py --model pythia-70m --output results/attractors/
    python collect_attractor_stats.py --num-samples 100 --max-length 256
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.drai.config import get_full_drai_config
from src.drai.build_drai_neox import build_drai_neox_model


def get_drai_layers(model):
    """Extract all DRAI layers from the model."""
    drai_layers = []
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        if hasattr(layer.attention, 'drai'):
            drai_layers.append((layer_idx, layer.attention.drai))
    return drai_layers


def collect_layer_stats(drai_layers: List) -> Dict:
    """Collect statistics from all DRAI layers.

    Args:
        drai_layers: List of (layer_idx, drai_module) tuples

    Returns:
        Dictionary with per-layer statistics
    """
    layer_stats = []

    for layer_idx, drai in drai_layers:
        # Access attractor buffers
        coherence = drai.attractor_coherence.cpu().numpy()
        count = drai.attractor_count.item()

        # Find active attractors (coherence > 0.1)
        active_mask = coherence > 0.1
        num_active = active_mask.sum()

        if num_active > 0:
            active_coherence = coherence[active_mask]

            stats = {
                'layer_idx': layer_idx,
                'attractor_count': count,
                'num_active': int(num_active),
                'avg_coherence': float(active_coherence.mean()),
                'max_coherence': float(active_coherence.max()),
                'min_coherence': float(active_coherence.min()),
                'std_coherence': float(active_coherence.std()),
                'forward_count': drai._forward_count.item(),
                'attractors_created': drai._attractors_created.item(),
                'attractors_reinforced': drai._attractors_reinforced.item(),
                'attractors_decayed': drai._attractors_decayed.item(),
            }
        else:
            stats = {
                'layer_idx': layer_idx,
                'attractor_count': count,
                'num_active': 0,
                'avg_coherence': 0.0,
                'max_coherence': 0.0,
                'min_coherence': 0.0,
                'std_coherence': 0.0,
                'forward_count': drai._forward_count.item(),
                'attractors_created': drai._attractors_created.item(),
                'attractors_reinforced': drai._attractors_reinforced.item(),
                'attractors_decayed': drai._attractors_decayed.item(),
            }

        layer_stats.append(stats)

    return {
        'num_layers': len(drai_layers),
        'layer_stats': layer_stats,
    }


def run_inference_and_collect(model, tokenizer, prompts: List[str], max_length: int = 128) -> Dict:
    """Run inference on prompts and collect final attractor statistics.

    Args:
        model: DRAI-enabled model
        tokenizer: Tokenizer
        prompts: List of prompts
        max_length: Maximum generation length

    Returns:
        Dictionary with statistics before and after inference
    """
    model.eval()

    # Get DRAI layers
    drai_layers = get_drai_layers(model)

    if not drai_layers:
        print("[Attractors] Warning: No DRAI layers found in model!")
        return {'error': 'No DRAI layers found'}

    print(f"[Attractors] Found {len(drai_layers)} DRAI layers")

    # Collect initial stats
    print("[Attractors] Collecting initial statistics...")
    stats_before = collect_layer_stats(drai_layers)

    # Run inference
    print(f"[Attractors] Running inference on {len(prompts)} prompts...")
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = tokenizer(prompt, return_tensors='pt')
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                do_sample=False,
            )

    # Collect final stats
    print("[Attractors] Collecting final statistics...")
    stats_after = collect_layer_stats(drai_layers)

    return {
        'before': stats_before,
        'after': stats_after,
    }


def analyze_stats(stats: Dict) -> Dict:
    """Analyze statistics before and after inference.

    Args:
        stats: Dictionary with 'before' and 'after' statistics

    Returns:
        Dictionary with analysis and comparison
    """
    before = stats['before']
    after = stats['after']

    # Compute changes
    total_attractors_before = sum(layer['num_active'] for layer in before['layer_stats'])
    total_attractors_after = sum(layer['num_active'] for layer in after['layer_stats'])

    total_created = sum(layer['attractors_created'] for layer in after['layer_stats'])
    total_reinforced = sum(layer['attractors_reinforced'] for layer in after['layer_stats'])
    total_decayed = sum(layer['attractors_decayed'] for layer in after['layer_stats'])

    # Per-layer changes
    layer_changes = []
    for before_layer, after_layer in zip(before['layer_stats'], after['layer_stats']):
        layer_changes.append({
            'layer_idx': before_layer['layer_idx'],
            'attractors_delta': after_layer['num_active'] - before_layer['num_active'],
            'coherence_delta': after_layer['avg_coherence'] - before_layer['avg_coherence'],
            'forward_count': after_layer['forward_count'] - before_layer['forward_count'],
            'created': after_layer['attractors_created'] - before_layer['attractors_created'],
            'reinforced': after_layer['attractors_reinforced'] - before_layer['attractors_reinforced'],
            'decayed': after_layer['attractors_decayed'] - before_layer['attractors_decayed'],
        })

    analysis = {
        'total_attractors_before': total_attractors_before,
        'total_attractors_after': total_attractors_after,
        'total_delta': total_attractors_after - total_attractors_before,
        'total_created': total_created,
        'total_reinforced': total_reinforced,
        'total_decayed': total_decayed,
        'layer_changes': layer_changes,
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Collect attractor statistics')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m',
                        help='Model name or path')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of prompts to generate from')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum generation length')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default='results/attractors',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 70)
    print("ATTRACTOR STATISTICS COLLECTION")
    print("=" * 70)
    print()
    print(f"Model: {args.model}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Device: {args.device}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer
    print("[Attractors] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load DRAI model
    print("[Attractors] Loading DRAI model...")
    drai_config = get_full_drai_config()
    model = build_drai_neox_model(
        args.model,
        drai_config=drai_config,
        torch_dtype=torch.float32,
        device_map=args.device,
        verbose=False,
    )

    # Sample prompts
    print("[Attractors] Generating sample prompts...")
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "The meaning of life is",
        "Artificial intelligence will",
        "The future of technology",
        "Deep learning models",
        "Natural language processing",
        "Machine learning algorithms",
        "Neural networks are",
        "The human brain",
        "Computer science is",
        "Python programming",
        "Data science involves",
        "The internet has",
        "Climate change is",
        "Renewable energy sources",
        "Quantum computing will",
        "Space exploration requires",
        "The scientific method",
    ]

    # Limit to num_samples
    prompts = prompts[:args.num_samples]

    # Run inference and collect stats
    print("[Attractors] Running inference and collecting statistics...")
    stats = run_inference_and_collect(model, tokenizer, prompts, args.max_length)

    if 'error' in stats:
        print(f"[Attractors] Error: {stats['error']}")
        return

    # Analyze statistics
    print("[Attractors] Analyzing collected data...")
    analysis = analyze_stats(stats)

    # Print summary
    print()
    print("=" * 70)
    print("ATTRACTOR STATISTICS SUMMARY")
    print("=" * 70)
    print()
    print(f"Total attractors before: {analysis['total_attractors_before']}")
    print(f"Total attractors after:  {analysis['total_attractors_after']}")
    print(f"Net change:              {analysis['total_delta']:+d}")
    print()
    print(f"Attractors created:      {analysis['total_created']}")
    print(f"Attractors reinforced:   {analysis['total_reinforced']}")
    print(f"Attractors decayed:      {analysis['total_decayed']}")
    print()
    print("PER-LAYER CHANGES:")
    print("  Layer | Δ Attractors | Δ Coherence | Created | Reinforced | Decayed")
    print("  " + "-" * 68)
    for change in analysis['layer_changes']:
        print(f"  {change['layer_idx']:5d} | {change['attractors_delta']:12d} | "
              f"{change['coherence_delta']:11.4f} | {change['created']:7d} | "
              f"{change['reinforced']:10d} | {change['decayed']:7d}")
    print()
    print("=" * 70)

    # Save results
    output_file = os.path.join(args.output, 'attractor_statistics.json')

    results = {
        'config': {
            'model': args.model,
            'num_samples': args.num_samples,
            'max_length': args.max_length,
            'drai_config': {
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
        },
        'stats_before': stats['before'],
        'stats_after': stats['after'],
        'analysis': analysis,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[Attractors] Results saved to: {output_file}")
    print()
    print("[Attractors] Collection complete!")


if __name__ == '__main__':
    main()
