"""Bounded attractor memory used by the native Duality model.

This is a trainable-model counterpart to the original DRAI V1 engine.  The
state update is deliberately detached from autograd: memory is a recurrent
inference state, while the projections used to read that state remain
trainable.  That keeps backpropagation bounded in long training batches and
makes the same module usable with prompt-prefill plus token-by-token decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class MemoryState:
    """Persistent attractor state for one memory-enabled transformer block."""

    attractors: Tensor
    strengths: Tensor
    timestep: Tensor

    def detach(self) -> "MemoryState":
        """Return a state safe to retain across optimizer steps."""

        return MemoryState(
            attractors=self.attractors.detach(),
            strengths=self.strengths.detach(),
            timestep=self.timestep.detach(),
        )


class DualityMemory(nn.Module):
    """Read and update a compact attractor cloud.

    Parameters
    ----------
    d_model:
        Hidden width of the transformer block.
    slots:
        Maximum number of attractors retained for each batch element.
    match_threshold:
        Cosine similarity above which a query reinforces an existing slot.
    alpha:
        EMA update rate for a matched attractor.
    decay:
        Per-token strength decay for slots that are not reinforced.
    influence_scale:
        Hard upper bound on the memory residual before the learned gate.
    """

    def __init__(
        self,
        d_model: int,
        slots: int = 16,
        match_threshold: float = 0.80,
        alpha: float = 0.05,
        decay: float = 0.995,
        strength_init: float = 0.50,
        strength_min: float = 1e-3,
        influence_scale: float = 0.15,
        temperature: float = 0.20,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        self.match_threshold = match_threshold
        self.alpha = alpha
        self.decay = decay
        self.strength_init = strength_init
        self.strength_min = strength_min
        self.influence_scale = influence_scale
        self.temperature = temperature

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.read_projection = nn.Linear(d_model, d_model, bias=False)

        # Start nearly closed so a newly initialized model behaves close to a
        # normal decoder while the memory pathway learns a useful signal.
        self.influence_gate = nn.Parameter(torch.tensor(-2.0))

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        """Create an empty state for a batch."""

        return MemoryState(
            attractors=torch.zeros(
                batch_size, self.slots, self.d_model, device=device, dtype=dtype
            ),
            strengths=torch.zeros(batch_size, self.slots, device=device, dtype=dtype),
            timestep=torch.zeros(batch_size, device=device, dtype=dtype),
        )

    def _validate_state(self, state: MemoryState, batch_size: int) -> None:
        if state.attractors.ndim != 3:
            raise ValueError("memory attractors must have shape [batch, slots, d_model]")
        expected = (batch_size, self.slots, self.d_model)
        if tuple(state.attractors.shape) != expected:
            raise ValueError(
                f"memory attractors have shape {tuple(state.attractors.shape)}; "
                f"expected {expected}"
            )
        if tuple(state.strengths.shape) != (batch_size, self.slots):
            raise ValueError("memory strengths must have shape [batch, slots]")
        if tuple(state.timestep.shape) != (batch_size,):
            raise ValueError("memory timestep must have shape [batch]")

    def _read(self, query: Tensor, state: MemoryState) -> Tensor:
        """Read the current attractor cloud before the token update."""

        active = state.strengths > self.strength_min
        normalized_query = F.normalize(query, dim=-1, eps=1e-6)
        normalized_attractors = F.normalize(state.attractors, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bsd->bs", normalized_query, normalized_attractors)

        # Mask after temperature scaling with a large finite value.  Using
        # ``torch.finfo(...).min`` before division can underflow to ``-inf``
        # for small temperatures, producing NaNs when every slot is empty.
        scaled_scores = scores / self.temperature
        masked_scores = scaled_scores.masked_fill(~active, -1e4)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = weights * active.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        read = torch.einsum("bs,bsd->bd", weights, state.attractors)
        read = read * (state.strengths.sum(dim=-1, keepdim=True) > self.strength_min)
        return read

    @torch.no_grad()
    def _update_one(self, query: Tensor, state: MemoryState) -> MemoryState:
        """Update a batch state using detached queries.

        The small batch loop keeps slot assignment deterministic and readable;
        the expensive language-model work remains vectorized in the attention
        blocks.  This function never builds an autograd graph.
        """

        attractors = state.attractors.clone()
        strengths = (state.strengths * self.decay).clone()
        timestep = state.timestep + 1.0

        normalized_query = F.normalize(query.detach(), dim=-1, eps=1e-6)
        normalized_attractors = F.normalize(attractors, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bsd->bs", normalized_query, normalized_attractors)
        active = strengths > self.strength_min

        for batch_index in range(query.shape[0]):
            active_indices = torch.nonzero(active[batch_index], as_tuple=False).flatten()
            if active_indices.numel() > 0:
                candidate_scores = scores[batch_index, active_indices]
                best_position = int(torch.argmax(candidate_scores).item())
                best_slot = int(active_indices[best_position].item())
                best_score = float(candidate_scores[best_position].item())
            else:
                best_slot = -1
                best_score = -1.0

            if best_slot >= 0 and best_score >= self.match_threshold:
                updated = (1.0 - self.alpha) * attractors[batch_index, best_slot]
                updated = updated + self.alpha * normalized_query[batch_index]
                attractors[batch_index, best_slot] = F.normalize(updated, dim=-1, eps=1e-6)
                strengths[batch_index, best_slot] = min(
                    1.0, float(strengths[batch_index, best_slot].item()) + self.strength_init
                )
                continue

            free = torch.nonzero(~active[batch_index], as_tuple=False).flatten()
            if free.numel() > 0:
                slot = int(free[0].item())
            else:
                slot = int(torch.argmin(strengths[batch_index]).item())
            attractors[batch_index, slot] = normalized_query[batch_index]
            strengths[batch_index, slot] = self.strength_init

        return MemoryState(attractors=attractors, strengths=strengths, timestep=timestep)

    def forward(
        self,
        hidden_states: Tensor,
        state: Optional[MemoryState] = None,
    ) -> Tuple[Tensor, MemoryState]:
        """Read and update memory for ``hidden_states``.

        Parameters
        ----------
        hidden_states:
            Tensor with shape ``[batch, sequence, d_model]``.
        state:
            State from an earlier call, typically returned during prompt
            prefill or generation.  ``None`` starts an empty cloud.

        Returns
        -------
        influence, new_state:
            A bounded residual with the same shape as ``hidden_states`` and
            the updated detached state.
        """

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, d_model]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.d_model:
            raise ValueError(f"hidden size {hidden_size} does not match d_model {self.d_model}")

        if state is None:
            state = self.init_state(
                batch_size, hidden_states.device, hidden_states.dtype
            )
        else:
            self._validate_state(state, batch_size)
            state = state.detach()
            if state.attractors.device != hidden_states.device:
                state = MemoryState(
                    attractors=state.attractors.to(hidden_states.device),
                    strengths=state.strengths.to(hidden_states.device),
                    timestep=state.timestep.to(hidden_states.device),
                )

        projected_queries = self.query_projection(hidden_states)
        reads = []
        current_state = state
        for token_index in range(sequence_length):
            query = projected_queries[:, token_index]
            read = self._read(query, current_state)
            reads.append(read)
            current_state = self._update_one(query, current_state)

        if reads:
            read_tensor = torch.stack(reads, dim=1)
        else:
            read_tensor = hidden_states.new_zeros(batch_size, 0, hidden_size)

        gate = torch.sigmoid(self.influence_gate)
        influence = torch.tanh(self.read_projection(read_tensor))
        influence = influence * (gate * self.influence_scale)
        return influence, current_state

    def stats(self, state: Optional[MemoryState]) -> Dict[str, float]:
        """Return compact, serializable diagnostics for a memory state."""

        if state is None:
            return {
                "active_slots": 0.0,
                "total_strength": 0.0,
                "mean_strength": 0.0,
                "timestep": 0.0,
            }
        active = state.strengths > self.strength_min
        active_strengths = state.strengths * active.to(state.strengths.dtype)
        return {
            "active_slots": float(active.sum().item()),
            "total_strength": float(active_strengths.sum().item()),
            "mean_strength": float(
                active_strengths.sum().item() / max(int(active.sum().item()), 1)
            ),
            "timestep": float(state.timestep.max().item()),
        }
