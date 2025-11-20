"""DRAI command-line interface for easy model augmentation.

Usage:
    python -m drai --help
    python -m drai apply --model EleutherAI/pythia-70m --output ./my_drai_model
    python -m drai eval --model ./my_drai_model --dataset wikitext-2
"""

import argparse
import sys

from .apply import apply_drai_to_model
from .config import DraiConfig, get_full_drai_config


def main():
    parser = argparse.ArgumentParser(
        description='DRAI: Dynamic Resonance AI - Self-Organizing Memory for Transformers'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply DRAI to a model')
    apply_parser.add_argument('--model', type=str, required=True,
                             help='Model name or path (e.g., EleutherAI/pythia-70m)')
    apply_parser.add_argument('--output', type=str, default='./drai_model',
                             help='Output directory for DRAI-enhanced model')
    apply_parser.add_argument('--num-drai-heads', type=int, default=1,
                             help='Number of DRAI heads per layer')
    apply_parser.add_argument('--max-attractors', type=int, default=32,
                             help='Maximum attractors per head')
    apply_parser.add_argument('--save', action='store_true',
                             help='Save the model to disk')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a DRAI model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Model name or path')
    eval_parser.add_argument('--dataset', type=str, default='wikitext-2',
                            help='Dataset to evaluate on')
    eval_parser.add_argument('--compare-baseline', action='store_true',
                            help='Compare against baseline (no DRAI)')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show DRAI information')

    args = parser.parse_args()

    if args.command == 'apply':
        print(f"Applying DRAI to {args.model}...")
        print(f"Configuration:")
        print(f"  - DRAI heads per layer: {args.num_drai_heads}")
        print(f"  - Max attractors: {args.max_attractors}")

        # Create config
        config = DraiConfig(
            enabled=True,
            phase=2,
            num_drai_heads=args.num_drai_heads,
            hyperparameters=DraiConfig.Hyperparameters(
                max_attractors=args.max_attractors
            )
        )

        # Apply DRAI
        model = apply_drai_to_model(args.model, config)

        print("✓ DRAI applied successfully!")

        if args.save:
            print(f"Saving model to {args.output}...")
            model.save_pretrained(args.output)
            print("✓ Model saved!")

    elif args.command == 'eval':
        print(f"Evaluating {args.model} on {args.dataset}...")
        # TODO: Implement evaluation
        print("Evaluation not yet implemented")

    elif args.command == 'info':
        print("=" * 70)
        print("DRAI: Dynamic Resonance AI")
        print("=" * 70)
        print()
        print("DRAI adds self-organizing memory to transformer language models")
        print("through attractor dynamics in attention heads.")
        print()
        print("Key Features:")
        print("  • Zero-cost integration (no performance degradation)")
        print("  • Self-organizing attractor formation")
        print("  • Drop-in replacement for standard attention")
        print("  • Validated on pythia-70m and pythia-125m")
        print()
        print("Usage:")
        print("  python -m drai apply --model EleutherAI/pythia-70m")
        print()
        print("For more info: https://github.com/HalcyonAIR/Duality")
        print()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
