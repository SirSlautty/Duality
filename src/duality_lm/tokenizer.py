"""Dependency-free byte tokenizer for the first DualityLM experiments."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


class ByteTokenizer:
    """A deterministic UTF-8 byte tokenizer with three special tokens.

    The byte vocabulary makes the model runnable without a tokenizer download
    and guarantees that arbitrary text can be represented.  It is intentionally
    a bootstrap tokenizer; a production checkpoint can later replace it with a
    learned BPE or sentencepiece vocabulary.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = byte_offset + 256

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode UTF-8 text into token IDs."""

        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
        tokens.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """Encode multiple strings."""

        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs, replacing malformed byte sequences safely."""

        raw = bytearray()
        for token_id in token_ids:
            token = int(token_id)
            if token in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                if not skip_special_tokens and token == self.eos_token_id:
                    raw.extend(b"<eos>")
                continue
            if self.byte_offset <= token < self.vocab_size:
                raw.append(token - self.byte_offset)
        return raw.decode("utf-8", errors="replace")

    def batch_decode(self, batch: Iterable[Sequence[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token sequences."""

        return [self.decode(tokens, skip_special_tokens=skip_special_tokens) for tokens in batch]

    def as_tensor(
        self, token_ids: Sequence[int], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Convert a token sequence into a model-ready rank-one tensor."""

        return torch.tensor(token_ids, dtype=torch.long, device=device)
