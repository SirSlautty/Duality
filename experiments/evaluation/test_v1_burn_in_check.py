#!/usr/bin/env python3
"""
Quick check: Is burn-in threshold preventing ALL injection?

If attractor strength never exceeds burn_in_threshold during evaluation,
DRAI would be completely passive (no-op)!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_story_generator import SimpleStoryGenerator

from src.drai import apply_drai_v1, get_v1_conservative_config, get_drai_v1_stats

model_name = "EleutherAI/pythia-410m"

# Generate one story
generator = SimpleStoryGenerator(seed=42)
stories = generator.generate_dataset(num_stories=1)
story = stories[0]

# Load model with DRAI
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

config = get_v1_conservative_config()
print(f"Burn-in threshold: {config.hyperparameters.burn_in_threshold}")
print(f"Burn-in mode: {config.hyperparameters.burn_in_mode}\n")

model = apply_drai_v1(model, config=config)

# Run one generation
question = story.questions[0]
prompt = f"""Story: {story.text}

Question: {question['question']}
Answer:"""

print(f"Running generation...")
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=20,
    do_sample=False,
)

# Check DRAI stats
stats = get_drai_v1_stats(model)
print(f"\nDRAI Statistics after generation:")
print(f"  Layers with DRAI: {stats['num_drai_layers']}")
print(f"  Total active attractors: {stats['total_active']}")
print(f"  Total strength: {stats['total_strength']:.2f}")

if len(stats['layers']) > 0:
    layer_stats = stats['layers'][0]
    print(f"\n  Layer {layer_stats['layer_idx']} details:")
    print(f"    Active attractors: {layer_stats['num_alive']}")
    print(f"    Total strength: {layer_stats['total_strength']:.2f}")
    print(f"    Mean strength: {layer_stats['mean_strength']:.2f}")
    print(f"    Max strength: {layer_stats['max_strength']:.2f}")
    print(f"    Burn-in active: {layer_stats['burn_in_active']}")
    print(f"    Burn-in threshold: {layer_stats['burn_in_threshold']:.2f}")
    print(f"    Timesteps: {int(layer_stats['timestep'])}")

    if layer_stats['burn_in_active']:
        print(f"\n  🚨 BURN-IN STILL ACTIVE!")
        print(f"  Total strength ({layer_stats['total_strength']:.2f}) < threshold ({layer_stats['burn_in_threshold']:.2f})")
        print(f"  DRAI IS PASSIVE - NOT INJECTING ANYTHING!")
    else:
        print(f"\n  ✓ Burn-in complete - DRAI is injecting!")
