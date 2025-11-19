#!/usr/bin/env python3
"""
DRAI V1 Quickstart Example

The simplest way to add working memory to a transformer.

This demonstrates:
1. One-line DRAI integration
2. Normal model usage
3. Checking attractor statistics
"""

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drai import apply_drai_v1, get_v1_conservative_config, get_drai_v1_stats
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 70)
print("DRAI V1 QUICKSTART")
print("=" * 70)
print()

# Step 1: Load model and apply DRAI
print("[1/3] Loading model and applying DRAI...")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m",
    torch_dtype="auto",
)

# Apply DRAI V1 (one line!)
config = get_v1_conservative_config()
model = apply_drai_v1(model, config=config)

print("  ✓ DRAI V1 applied to pythia-70m")
print(f"  ✓ Configuration: conservative (burn-in={config.hyperparameters.burn_in_threshold})")
print()

# Step 2: Use the model normally
print("[2/3] Generating text...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
tokenizer.pad_token = tokenizer.eos_token

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors='pt')

print(f"  Prompt: \"{prompt}\"")

# Generate
outputs = model.generate(
    inputs['input_ids'],
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Generated: \"{generated_text}\"")
print()

# Step 3: Check DRAI statistics
print("[3/3] Checking DRAI statistics...")
stats = get_drai_v1_stats(model)

print(f"  - DRAI layers: {stats['num_drai_layers']}")
print(f"  - Active attractors: {stats['total_active']}")
print(f"  - Total strength: {stats['total_strength']:.2f}")

if len(stats['layers']) > 0:
    layer_stats = stats['layers'][0]
    print(f"  - Burn-in complete: {'YES' if not layer_stats['burn_in_active'] else 'NO'}")
    print(f"  - Timestep: {int(layer_stats['timestep'])}")

print()
print("=" * 70)
print("Done! Your model now has internal working memory.")
print("=" * 70)
print()
print("What happened:")
print("  1. DRAI added persistent attractors to layer 3")
print("  2. Attractors formed during generation (~10 tokens)")
print("  3. Model now has working memory for context")
print()
print("Try it yourself:")
print("  - Modify the prompt")
print("  - Adjust hyperparameters (see src/drai/config.py)")
print("  - Test on your own models")
print("=" * 70)
