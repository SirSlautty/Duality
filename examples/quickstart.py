#!/usr/bin/env python3
"""DRAI Quickstart Example

This script demonstrates the simplest way to use DRAI.
"""

# Add parent directory to path for local development
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drai import apply_drai
from transformers import AutoTokenizer

# Step 1: Apply DRAI to a model (that's it!)
print("=" * 70)
print("DRAI QUICKSTART")
print("=" * 70)
print()

print("Step 1: Applying DRAI to pythia-70m...")
model = apply_drai("EleutherAI/pythia-70m")

print("✓ DRAI applied successfully!")
print()

# Step 2: Use the model normally
print("Step 2: Using the model for text generation...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors='pt')

print(f"Prompt: \"{prompt}\"")
print()

# Generate
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated text:")
print(generated_text)
print()

# Step 3: Check DRAI statistics
from drai import get_drai_stats

stats = get_drai_stats(model)

print("Step 3: DRAI Statistics:")
print(f"  - DRAI layers: {stats['num_drai_layers']}")
print(f"  - Active attractors: {stats['total_active']}")
print(f"  - Attractors created: {stats['total_created']}")
print(f"  - Attractors reinforced: {stats['total_reinforced']}")
print()

print("=" * 70)
print("That's all! You now have a model with self-organizing memory.")
print("=" * 70)
