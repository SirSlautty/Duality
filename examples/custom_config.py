#!/usr/bin/env python3
"""DRAI Advanced Configuration Example

This script demonstrates how to customize DRAI configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drai import apply_drai, DraiConfig, get_drai_stats
from transformers import AutoTokenizer

print("=" * 70)
print("DRAI CUSTOM CONFIGURATION")
print("=" * 70)
print()

# Create custom DRAI configuration
print("Creating custom DRAI configuration...")
config = DraiConfig(
    enabled=True,
    phase=2,
    num_drai_heads=2,  # Use 2 DRAI heads per layer
    layer_mode="all",  # Apply to all layers
    hyperparameters=DraiConfig.Hyperparameters(
        max_attractors=64,  # More attractors per head
        coherence_threshold=0.3,
        formation_threshold=0.5,
        decay_rate=0.01,
        ema_momentum=0.9,
    ),
    verbose_logging=True,
)

print("Configuration:")
print(f"  - DRAI heads per layer: {config.num_drai_heads}")
print(f"  - Max attractors: {config.hyperparameters.max_attractors}")
print(f"  - Coherence threshold: {config.hyperparameters.coherence_threshold}")
print(f"  - Formation threshold: {config.hyperparameters.formation_threshold}")
print()

# Apply DRAI with custom config
print("Applying DRAI to model...")
model = apply_drai("EleutherAI/pythia-70m", config=config)

print("✓ DRAI applied with custom configuration!")
print()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# Generate text
prompt = "Self-organizing memory systems"
print(f"Generating from prompt: \"{prompt}\"")

inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(
    inputs['input_ids'],
    max_length=80,
    do_sample=True,
    temperature=0.8,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print()
print("Generated text:")
print(generated_text)
print()

# Check statistics
stats = get_drai_stats(model)

print("DRAI Statistics:")
print(f"  - Total DRAI layers: {stats['num_drai_layers']}")
print(f"  - Active attractors: {stats['total_active']}")
print(f"  - Total created: {stats['total_created']}")
print(f"  - Total reinforced: {stats['total_reinforced']}")
print()

print("Per-layer breakdown:")
for layer_stats in stats['layers']:
    print(f"  Layer {layer_stats['layer_idx']}: "
          f"{layer_stats['active_attractors']} active, "
          f"{layer_stats['attractors_created']} created, "
          f"{layer_stats['attractors_reinforced']} reinforced")

print()
print("=" * 70)
