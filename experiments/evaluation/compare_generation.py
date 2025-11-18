#!/usr/bin/env python3
"""Qualitative text generation comparison between baseline and DRAI.

This script generates text samples from both baseline and DRAI models
for qualitative comparison in the paper.

Usage:
    python compare_generation.py --model pythia-70m --output results/generation/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.drai.config import get_full_drai_config
from src.drai.build_drai_neox import build_drai_neox_model


def generate_samples(model, tokenizer, prompts: list, max_length: int = 100, num_return: int = 1):
    """Generate text samples from prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_length: Maximum generation length
        num_return: Number of sequences to return per prompt

    Returns:
        List of generated texts
    """
    model.eval()
    generations = []

    with torch.no_grad():
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt')

            # Generate
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=num_return,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode
            texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            generations.append({
                'prompt': prompt,
                'generations': texts,
            })

    return generations


def main():
    parser = argparse.ArgumentParser(description='Compare text generation quality')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m',
                        help='Model name or path')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum generation length')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default='results/generation',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 70)
    print("TEXT GENERATION COMPARISON")
    print("=" * 70)
    print()
    print(f"Model: {args.model}")
    print(f"Max length: {args.max_length}")
    print(f"Device: {args.device}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Sample prompts for comparison
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important discovery in science",
        "Climate change is affecting",
        "The meaning of consciousness is",
    ]

    # Load tokenizer
    print("[Gen] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # === BASELINE GENERATION ===
    print()
    print("[Gen] ===== BASELINE GENERATION (No DRAI) =====")
    print()

    print("[Gen] Loading baseline model...")
    model_baseline = build_drai_neox_model(
        args.model,
        drai_config=None,  # No DRAI
        torch_dtype=torch.float32,
        device_map=args.device,
        verbose=False,
    )

    print("[Gen] Generating samples...")
    baseline_generations = generate_samples(model_baseline, tokenizer, prompts, args.max_length)

    # Clean up
    del model_baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === DRAI GENERATION ===
    print()
    print("[Gen] ===== DRAI GENERATION =====")
    print()

    print("[Gen] Loading DRAI model...")
    drai_config = get_full_drai_config()
    model_drai = build_drai_neox_model(
        args.model,
        drai_config=drai_config,
        torch_dtype=torch.float32,
        device_map=args.device,
        verbose=False,
    )

    print("[Gen] Generating samples...")
    drai_generations = generate_samples(model_drai, tokenizer, prompts, args.max_length)

    # === DISPLAY COMPARISON ===
    print()
    print("=" * 70)
    print("GENERATION COMPARISON")
    print("=" * 70)
    print()

    for i, prompt in enumerate(prompts):
        print(f"\nPROMPT {i+1}: \"{prompt}\"")
        print("-" * 70)

        print("\n[BASELINE]")
        print(baseline_generations[i]['generations'][0])

        print("\n[DRAI]")
        print(drai_generations[i]['generations'][0])

        print()

    # === SAVE RESULTS ===
    output_file = os.path.join(args.output, 'generation_comparison.json')

    results = {
        'config': {
            'model': args.model,
            'max_length': args.max_length,
            'prompts': prompts,
        },
        'baseline': baseline_generations,
        'drai': drai_generations,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[Gen] Results saved to: {output_file}")
    print()
    print("[Gen] Comparison complete!")


if __name__ == '__main__':
    main()
