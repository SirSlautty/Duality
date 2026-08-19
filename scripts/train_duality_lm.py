#!/usr/bin/env python3
"""Train a small native DualityLM on a plain-text file.

Example:
    python scripts/train_duality_lm.py --text-file corpus.txt --steps 500

The script is intentionally modest: it is a reproducible bootstrap trainer,
not a distributed pretraining system. Checkpoints contain model weights and
the JSON architecture configuration, never the source corpus.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duality_lm import ByteTokenizer, DualityLM, DualityLMConfig


DEFAULT_CORPUS = """
Duality is a decoder model with a small internal attractor memory.
The memory is bounded, explicit, and updated as tokens move through the model.
Training teaches the projections how to read that state; inference carries it
forward through prompt prefill and cached token generation.
""".strip()


def load_tokens(text_file: Optional[str], tokenizer: ByteTokenizer) -> torch.Tensor:
    """Read a UTF-8 corpus or use the built-in smoke-test corpus."""

    text = Path(text_file).read_text(encoding="utf-8") if text_file else DEFAULT_CORPUS
    return torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=torch.long)


def batches(
    tokens: torch.Tensor,
    sequence_length: int,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield random next-token training windows forever."""

    if tokens.numel() < sequence_length + 1:
        raise ValueError("corpus must contain at least sequence_length + 1 tokens")
    max_start = tokens.numel() - sequence_length - 1
    while True:
        starts = torch.randint(0, max_start + 1, (batch_size,))
        inputs = torch.stack([tokens[start : start + sequence_length] for start in starts])
        labels = torch.stack(
            [tokens[start + 1 : start + sequence_length + 1] for start in starts]
        )
        yield inputs, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("duality_lm_checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.sequence_length <= 1:
        raise ValueError("steps, batch-size, and sequence-length must be positive")

    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()
    config = DualityLMConfig.small(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max(args.sequence_length, 512),
    )
    model = DualityLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    token_data = load_tokens(args.text_file, tokenizer)
    stream = batches(token_data, args.sequence_length, args.batch_size)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, labels = next(stream)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, labels=labels)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            perplexity = math.exp(min(float(output.loss.item()), 20.0))
            print(
                f"step={step:>5} loss={output.loss.item():.4f} "
                f"perplexity={perplexity:.2f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "tokenizer": {"type": "byte", "vocab_size": tokenizer.vocab_size},
    }
    torch.save(checkpoint, args.output)
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()

