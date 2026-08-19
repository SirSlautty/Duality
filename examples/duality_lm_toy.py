#!/usr/bin/env python3
"""Run a small DualityLM construction and generation smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duality_lm import ByteTokenizer, DualityLM, DualityLMConfig


def main() -> None:
    torch.manual_seed(7)
    tokenizer = ByteTokenizer()
    config = DualityLMConfig.tiny(vocab_size=tokenizer.vocab_size)
    model = DualityLM(config)

    prompt = "Duality remembers"
    prompt_ids = tokenizer.as_tensor(tokenizer.encode(prompt))
    output = model(prompt_ids, use_cache=True)
    print("logits:", tuple(output.logits.shape))
    print("memory:", model.memory_stats(output.memory_state))

    generated = model.generate(prompt_ids, max_new_tokens=24, do_sample=False)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

