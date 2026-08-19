"""Configuration for the native Duality language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class DualityLMConfig:
    """Architecture and memory settings for :class:`DualityLM`.

    The defaults are intentionally small enough for CPU experiments.  They
    describe a real causal language model, not a pretrained checkpoint.  A
    useful model still needs to be trained on a corpus appropriate to its
    intended use.
    """

    vocab_size: int = 259
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: Optional[int] = None
    dropout: float = 0.0

    # ``None`` selects the middle transformer block.  Explicit indices make
    # experiments reproducible and allow memory at multiple depths.
    memory_layers: Optional[Tuple[int, ...]] = None
    memory_slots: int = 16
    memory_match_threshold: float = 0.80
    memory_alpha: float = 0.05
    memory_decay: float = 0.995
    memory_strength_init: float = 0.50
    memory_strength_min: float = 1e-3
    memory_influence_scale: float = 0.15
    memory_temperature: float = 0.20

    tie_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        """Validate settings early so malformed models fail clearly."""

        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "memory_slots": self.memory_slots,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff is not None and (not isinstance(self.d_ff, int) or self.d_ff <= 0):
            raise ValueError("d_ff must be a positive integer when provided")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.memory_match_threshold <= 1.0:
            raise ValueError("memory_match_threshold must be in (0, 1]")
        if not 0.0 < self.memory_alpha <= 1.0:
            raise ValueError("memory_alpha must be in (0, 1]")
        if not 0.0 < self.memory_decay <= 1.0:
            raise ValueError("memory_decay must be in (0, 1]")
        if self.memory_strength_init <= 0.0:
            raise ValueError("memory_strength_init must be > 0")
        if self.memory_strength_min < 0.0:
            raise ValueError("memory_strength_min must be >= 0")
        if self.memory_influence_scale <= 0.0:
            raise ValueError("memory_influence_scale must be > 0")
        if self.memory_temperature <= 0.0:
            raise ValueError("memory_temperature must be > 0")

        token_ids = {
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        for name, value in token_ids.items():
            if not isinstance(value, int) or not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} must be an integer in [0, vocab_size)")

        if self.memory_layers is not None:
            if len(set(self.memory_layers)) != len(self.memory_layers):
                raise ValueError("memory_layers cannot contain duplicates")
            if not all(isinstance(index, int) for index in self.memory_layers):
                raise ValueError("memory_layers must contain integers")
            invalid = [index for index in self.memory_layers if not 0 <= index < self.n_layers]
            if invalid:
                raise ValueError(
                    f"memory layer indices {invalid} are outside [0, {self.n_layers})"
                )

    @property
    def feed_forward_dim(self) -> int:
        """Return the configured or conventional feed-forward width."""

        return self.d_ff or (4 * self.d_model)

    def resolved_memory_layers(self) -> Tuple[int, ...]:
        """Resolve the default memory placement to the middle block."""

        if self.memory_layers is not None:
            return tuple(self.memory_layers)
        return (self.n_layers // 2,)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly configuration dictionary."""

        values = asdict(self)
        if values["memory_layers"] is not None:
            values["memory_layers"] = list(values["memory_layers"])
        return values

    @classmethod
    def tiny(
        cls,
        vocab_size: int = 259,
        max_seq_len: int = 256,
        memory_layers: Optional[Tuple[int, ...]] = None,
    ) -> "DualityLMConfig":
        """Return a fast configuration for tests and CPU smoke runs."""

        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            memory_slots=8,
            memory_layers=memory_layers,
        )

    @classmethod
    def small(
        cls,
        vocab_size: int = 259,
        max_seq_len: int = 512,
        memory_layers: Optional[Tuple[int, ...]] = None,
    ) -> "DualityLMConfig":
        """Return a small, trainable CPU/GPU starting point."""

        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            memory_slots=16,
            memory_layers=memory_layers,
        )
