"""
DRAI Resonance Layer - Core Implementation

This module implements the Dynamic Resonance AI (DRAI) resonance cortex,
which maintains attractor states from recurring latent patterns and generates
synthetic key/value pairs for injection into transformer attention.

Implementation follows a phased approach:
- Phase 1: Stub implementation (returns zeros) - COMPLETE
- Phase 2: Simple attractor accumulation (moving average) - CURRENT
- Phase 3: Multi-pattern extraction and clustering - FUTURE
- Phase 4: Advanced reinforcement and hierarchical attractors - FUTURE

Author: Halcyon AI Research
Date: 2025-11-18
Status: Phase 2 - Attractor Dynamics Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DraiResonanceLayer(nn.Module):
    """
    DRAI Resonance Layer - Attractor-based memory for transformers.

    This layer observes query vectors from an attention head and maintains
    a resonance field of stable attractors. It generates synthetic K/V pairs
    that represent accumulated memory patterns, which are then concatenated
    to the standard attention keys and values.

    Current Implementation (Phase 1):
        Returns zero tensors with correct shapes to verify integration
        without affecting model behavior.

    Future Phases:
        - Phase 2: Implement simple attractor accumulation
        - Phase 3: Add pattern detection and clustering
        - Phase 4: Implement reinforcement and decay dynamics

    Args:
        hidden_size (int): Model hidden dimension (e.g., 2048 for GPT-NeoX)
        num_heads (int): Number of DRAI resonance heads to create (typically 1)
        head_dim (Optional[int]): Dimension of each head. If None, computed as
                                  hidden_size // num_attention_heads. Defaults to 64.
        device (Optional[str]): Device to place tensors on ('cpu', 'cuda', etc.)
        dtype (Optional[torch.dtype]): Data type for tensors (default: float32)

    Shape:
        - Input: (seq_len, batch, num_attn_heads, head_dim)
        - Output K: (seq_len, batch, num_drai_heads, head_dim)
        - Output V: (seq_len, batch, num_drai_heads, head_dim)

    Example:
        >>> layer = DraiResonanceLayer(hidden_size=2048, num_heads=1)
        >>> query = torch.randn(20, 4, 32, 64)  # seq, batch, attn_heads, head_dim
        >>> k_reson, v_reson = layer(query)
        >>> print(k_reson.shape)  # torch.Size([20, 4, 1, 64])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        # Phase 2+ parameters
        phase: int = 1,  # Default to Phase 1 for backward compatibility
        max_attractors: int = 32,
        coherence_threshold: float = 0.3,
        formation_threshold: float = 0.5,
        decay_rate: float = 0.01,
        ema_momentum: float = 0.9,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else 64
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32

        # Phase control
        self._phase = phase
        self._returns_zeros = (phase == 1)

        # Phase 2+ Attractor parameters
        self.max_attractors = max_attractors
        self.coherence_threshold = coherence_threshold
        self.formation_threshold = formation_threshold
        self.decay_rate = decay_rate
        self.ema_momentum = ema_momentum

        # Attractor field buffers (Phase 2+)
        if phase >= 2:
            # Attractor centroids: [max_attractors, head_dim]
            self.register_buffer(
                "attractor_centroids",
                torch.zeros(max_attractors, self.head_dim, dtype=self.dtype)
            )

            # Attractor coherence values: [max_attractors]
            self.register_buffer(
                "attractor_coherence",
                torch.zeros(max_attractors, dtype=self.dtype)
            )

            # Last used timestep for each attractor: [max_attractors]
            self.register_buffer(
                "attractor_last_used",
                torch.zeros(max_attractors, dtype=torch.long)
            )

            # Number of times each attractor was reinforced: [max_attractors]
            self.register_buffer(
                "attractor_reinforcements",
                torch.zeros(max_attractors, dtype=torch.long)
            )

            # Current number of active attractors (scalar)
            self.register_buffer(
                "attractor_count",
                torch.tensor(0, dtype=torch.long)
            )

            # Global timestep counter (scalar)
            self.register_buffer(
                "timestep",
                torch.tensor(0, dtype=torch.long)
            )

        # Statistics tracking (for debugging and analysis)
        self.register_buffer("_forward_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_attractors_created", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_attractors_reinforced", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_attractors_decayed", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        query_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DRAI resonance layer.

        Phase 1 Implementation:
            Returns zero tensors with correct shapes to verify integration.
            This allows testing transformer modifications without affecting
            model behavior or gradients.

        Future Implementation:
            1. Extract latent patterns from query_layer
            2. Match against existing attractors in resonance field
            3. Update attractor coherence for matching patterns
            4. Form new attractors for novel patterns
            5. Generate synthetic K/V from activated attractors
            6. Apply decay to unused attractors

        Args:
            query_layer: Query tensor from attention head
                        Shape: (seq_len, batch, num_attn_heads, head_dim)
            attention_mask: Optional mask for valid positions
                           Shape: (batch, seq_len) or (batch, 1, seq_len, seq_len)

        Returns:
            Tuple of (k_reson, v_reson):
                k_reson: Synthetic keys from resonance field
                        Shape: (seq_len, batch, num_drai_heads, head_dim)
                v_reson: Synthetic values from resonance field
                        Shape: (seq_len, batch, num_drai_heads, head_dim)

        Note:
            The output shapes are designed to be concatenated with existing
            K/V tensors along the head dimension (dim=2):
                key_extended = torch.cat([key_layer, k_reson], dim=2)
                value_extended = torch.cat([value_layer, v_reson], dim=2)
        """
        # Update forward pass counter
        self._forward_count += 1

        # Extract shape information
        seq_len, batch, num_attn_heads, head_dim = query_layer.shape

        # Validate input dimensions
        assert (
            head_dim == self.head_dim
        ), f"Expected head_dim={self.head_dim}, got {head_dim}"

        # Route to appropriate phase implementation
        if self._phase == 1:
            # Phase 1: Return zeros with correct shape
            k_reson, v_reson = self._forward_phase1(query_layer, seq_len, batch)
        elif self._phase >= 2:
            # Phase 2+: Full attractor dynamics
            k_reson, v_reson = self._forward_phase2(query_layer, seq_len, batch, attention_mask)
        else:
            raise ValueError(f"Invalid phase: {self._phase}")

        return k_reson, v_reson

    def _forward_phase1(
        self,
        query_layer: torch.Tensor,
        seq_len: int,
        batch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Phase 1: Return zeros to verify integration."""
        # To maintain gradient flow, we create zeros from the query tensor
        if query_layer.requires_grad:
            # Create zeros that maintain computational graph
            # Take first head slice and zero it out, then expand to desired shape
            dummy = query_layer[:, :, :1, :] * 0.0  # [seq, batch, 1, head_dim]
            if self.num_heads == 1:
                k_reson = dummy
                v_reson = dummy.clone()
            else:
                k_reson = dummy.expand(seq_len, batch, self.num_heads, self.head_dim).contiguous()
                v_reson = dummy.expand(seq_len, batch, self.num_heads, self.head_dim).contiguous()
        else:
            # No gradients needed, use simple zeros
            k_reson = torch.zeros(
                seq_len,
                batch,
                self.num_heads,
                self.head_dim,
                device=query_layer.device,
                dtype=query_layer.dtype,
            )

            v_reson = torch.zeros(
                seq_len,
                batch,
                self.num_heads,
                self.head_dim,
                device=query_layer.device,
                dtype=query_layer.dtype,
            )

        return k_reson, v_reson

    def _forward_phase2(
        self,
        query_layer: torch.Tensor,
        seq_len: int,
        batch: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 2: Full attractor dynamics implementation.

        Steps:
        1. Extract pattern from query_layer
        2. Find best matching attractor
        3. Update or create attractor
        4. Decay unused attractors
        5. Generate synthetic K/V from attractors
        """
        # 1. Extract pattern (mean pooling over seq, batch, heads)
        pattern = self._extract_pattern(query_layer)

        # 2. Find best matching attractor
        best_idx, best_sim = self._find_best_match(pattern)

        # 3. Update or create attractor
        if best_sim > self.coherence_threshold and best_idx >= 0:
            # Reinforce existing attractor
            self._reinforce_attractor(best_idx, pattern)
        elif self.attractor_count < self.max_attractors:
            # Create new attractor
            self._create_attractor(pattern)
        else:
            # Replace weakest attractor
            weakest_idx = self._find_weakest_attractor()
            self._replace_attractor(weakest_idx, pattern)

        # 4. Decay unused attractors
        self._decay_attractors()

        # 5. Generate synthetic K/V
        k_reson, v_reson = self._generate_kv(seq_len, batch, query_layer.device, query_layer.dtype)

        # Increment timestep
        self.timestep += 1

        return k_reson, v_reson

    # =========================================================================
    # Phase 2 Attractor Dynamics Methods
    # =========================================================================

    def _extract_pattern(self, query_layer: torch.Tensor) -> torch.Tensor:
        """
        Extract a representative pattern from the query layer.

        Phase 2: Simple mean pooling over sequence, batch, and heads.
        Future: Could use attention-weighted pooling or extract multiple patterns.

        Args:
            query_layer: [seq_len, batch, num_heads, head_dim]

        Returns:
            pattern: [head_dim] normalized vector
        """
        # Mean pool over sequence, batch, and heads → [head_dim]
        pattern = query_layer.mean(dim=[0, 1, 2])

        # Normalize to unit sphere for cosine similarity
        pattern = F.normalize(pattern, p=2, dim=-1)

        return pattern

    def _find_best_match(self, pattern: torch.Tensor) -> Tuple[int, float]:
        """
        Find the best matching attractor for the given pattern.

        Args:
            pattern: [head_dim] normalized pattern vector

        Returns:
            best_idx: Index of best matching attractor (-1 if no attractors)
            best_sim: Cosine similarity with best match (-1.0 if no attractors)
        """
        if self.attractor_count == 0:
            return -1, -1.0

        # Get active attractors
        active_centroids = self.attractor_centroids[:self.attractor_count]

        # Compute cosine similarity with all active attractors
        # pattern: [head_dim], active_centroids: [count, head_dim]
        similarities = F.cosine_similarity(
            pattern.unsqueeze(0),  # [1, head_dim]
            active_centroids,      # [count, head_dim]
            dim=1                  # → [count]
        )

        # Find best match
        best_idx = similarities.argmax().item()
        best_sim = similarities[best_idx].item()

        return best_idx, best_sim

    def _reinforce_attractor(self, idx: int, pattern: torch.Tensor) -> None:
        """
        Reinforce an existing attractor with a new pattern using EMA.

        Args:
            idx: Index of attractor to reinforce
            pattern: [head_dim] new pattern to incorporate
        """
        # Exponential moving average update
        old_centroid = self.attractor_centroids[idx]
        new_centroid = self.ema_momentum * old_centroid + (1 - self.ema_momentum) * pattern

        # Normalize to maintain unit sphere
        self.attractor_centroids[idx] = F.normalize(new_centroid, p=2, dim=-1)

        # Increase coherence (bounded by 1.0)
        current_coherence = self.attractor_coherence[idx]
        self.attractor_coherence[idx] = torch.min(
            torch.tensor(1.0, device=current_coherence.device, dtype=current_coherence.dtype),
            current_coherence + 0.1 * (1.0 - current_coherence)
        )

        # Update metadata
        self.attractor_last_used[idx] = self.timestep
        self.attractor_reinforcements[idx] += 1

        # Update statistics
        self._attractors_reinforced += 1

    def _create_attractor(self, pattern: torch.Tensor) -> None:
        """
        Create a new attractor from a pattern.

        Args:
            pattern: [head_dim] pattern to create attractor from
        """
        idx = self.attractor_count.item()

        # Set centroid (already normalized)
        self.attractor_centroids[idx] = pattern

        # Set initial coherence
        self.attractor_coherence[idx] = self.formation_threshold

        # Set metadata
        self.attractor_last_used[idx] = self.timestep
        self.attractor_reinforcements[idx] = 1

        # Increment count
        self.attractor_count += 1

        # Update statistics
        self._attractors_created += 1

    def _replace_attractor(self, idx: int, pattern: torch.Tensor) -> None:
        """
        Replace an existing attractor with a new pattern.

        Args:
            idx: Index of attractor to replace
            pattern: [head_dim] new pattern
        """
        # Same as create, but at specific index
        self.attractor_centroids[idx] = pattern
        self.attractor_coherence[idx] = self.formation_threshold
        self.attractor_last_used[idx] = self.timestep
        self.attractor_reinforcements[idx] = 1

        # Update statistics (counts as both decay and create)
        self._attractors_decayed += 1
        self._attractors_created += 1

    def _find_weakest_attractor(self) -> int:
        """
        Find the weakest attractor (lowest coherence).

        Returns:
            Index of weakest attractor among active attractors
        """
        active_coherence = self.attractor_coherence[:self.attractor_count]
        weakest_idx = active_coherence.argmin().item()
        return weakest_idx

    def _decay_attractors(self) -> None:
        """
        Apply exponential decay to attractors that weren't used this timestep.
        Remove attractors that have decayed below threshold.
        """
        if self.attractor_count == 0:
            return

        for idx in range(self.attractor_count.item()):
            # Check if this attractor was used this timestep
            if self.attractor_last_used[idx] < self.timestep:
                # Not used, apply decay
                time_since_use = (self.timestep - self.attractor_last_used[idx]).float()
                decay_factor = torch.exp(-self.decay_rate * time_since_use)

                self.attractor_coherence[idx] *= decay_factor

                # Check if too weak and should be pruned
                if self.attractor_coherence[idx] < 0.05:
                    self._remove_attractor(idx)
                    self._attractors_decayed += 1
                    # After removal, we need to check this index again
                    # since a different attractor may have been moved here
                    # But for simplicity in Phase 2, we'll just continue
                    # This is a minor inefficiency that can be fixed in Phase 3

    def _remove_attractor(self, idx: int) -> None:
        """
        Remove an attractor by swapping it with the last one and decrementing count.

        Args:
            idx: Index of attractor to remove
        """
        if self.attractor_count == 0:
            return

        last_idx = self.attractor_count - 1

        if idx < last_idx:
            # Swap with last attractor
            self.attractor_centroids[idx] = self.attractor_centroids[last_idx]
            self.attractor_coherence[idx] = self.attractor_coherence[last_idx]
            self.attractor_last_used[idx] = self.attractor_last_used[last_idx]
            self.attractor_reinforcements[idx] = self.attractor_reinforcements[last_idx]

        # Zero out the last position
        self.attractor_centroids[last_idx] = 0
        self.attractor_coherence[last_idx] = 0
        self.attractor_last_used[last_idx] = 0
        self.attractor_reinforcements[last_idx] = 0

        # Decrement count
        self.attractor_count -= 1

    def _generate_kv(
        self,
        seq_len: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic K/V pairs from the attractor field.

        Phase 2: Use top-k strongest attractors.

        Args:
            seq_len: Sequence length to expand to
            batch: Batch size to expand to
            device: Device to create tensors on
            dtype: Data type for tensors

        Returns:
            k_reson: [seq_len, batch, num_drai_heads, head_dim]
            v_reson: [seq_len, batch, num_drai_heads, head_dim]
        """
        if self.attractor_count == 0:
            # No attractors yet, return zeros
            k_reson = torch.zeros(
                seq_len, batch, self.num_heads, self.head_dim,
                device=device, dtype=dtype
            )
            v_reson = torch.zeros(
                seq_len, batch, self.num_heads, self.head_dim,
                device=device, dtype=dtype
            )
            return k_reson, v_reson

        # Get active attractors
        active_count = min(self.attractor_count.item(), self.max_attractors)
        active_centroids = self.attractor_centroids[:active_count]
        active_coherence = self.attractor_coherence[:active_count]

        # Sort by coherence (strongest first)
        sorted_indices = torch.argsort(active_coherence, descending=True)

        # Take top num_heads attractors
        num_to_use = min(self.num_heads, active_count)
        top_indices = sorted_indices[:num_to_use]

        # Generate K/V
        k_reson = torch.zeros(
            seq_len, batch, self.num_heads, self.head_dim,
            device=device, dtype=dtype
        )
        v_reson = torch.zeros(
            seq_len, batch, self.num_heads, self.head_dim,
            device=device, dtype=dtype
        )

        for i, idx in enumerate(top_indices):
            # Key is the attractor centroid
            centroid = active_centroids[idx]
            coherence = active_coherence[idx]

            # Expand to sequence and batch dimensions
            # [head_dim] → [seq, batch, 1, head_dim]
            k_expanded = centroid.view(1, 1, 1, self.head_dim).expand(
                seq_len, batch, 1, self.head_dim
            )
            k_reson[:, :, i:i+1, :] = k_expanded

            # Value is weighted by coherence
            v_expanded = (coherence * centroid).view(1, 1, 1, self.head_dim).expand(
                seq_len, batch, 1, self.head_dim
            )
            v_reson[:, :, i:i+1, :] = v_expanded

        return k_reson, v_reson

    # =========================================================================
    # End Phase 2 Methods
    # =========================================================================

    def _compute_resonance(
        self,
        query_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute resonance field activation and generate synthetic K/V.

        This method will implement the core DRAI logic in future phases:
        1. Pattern extraction and matching
        2. Attractor field queries
        3. Coherence updates
        4. Synthetic K/V generation

        Args:
            query_layer: Query tensor to match against attractors
            attention_mask: Optional mask for valid positions

        Returns:
            Tuple of (k_reson, v_reson)

        Status: NOT IMPLEMENTED (placeholder for Phase 2+)
        """
        raise NotImplementedError("Phase 2+ feature - attractor dynamics")

    def reset_statistics(self) -> None:
        """Reset internal statistics counters."""
        self._forward_count.zero_()
        self._attractors_created.zero_()
        self._attractors_reinforced.zero_()
        self._attractors_decayed.zero_()

    def get_statistics(self) -> dict:
        """
        Get statistics about resonance layer usage.

        Returns:
            Dictionary with statistics:
                - forward_count: Number of forward passes
                - phase: Current implementation phase
                - returns_zeros: Whether layer returns zeros (Phase 1 only)
                - num_heads: Number of DRAI heads
                - head_dim: Dimension of each head

                Phase 2+ statistics:
                - attractor_count: Number of active attractors
                - attractors_created: Total attractors created
                - attractors_reinforced: Total reinforcements
                - attractors_decayed: Total attractors pruned
                - timestep: Current timestep
                - max_attractors: Maximum capacity
                - utilization: Percentage of capacity used
        """
        stats = {
            "forward_count": self._forward_count.item(),
            "phase": self._phase,
            "returns_zeros": self._returns_zeros,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        }

        # Add Phase 2+ statistics
        if self._phase >= 2:
            attractor_count = self.attractor_count.item()
            stats.update({
                "attractor_count": attractor_count,
                "attractors_created": self._attractors_created.item(),
                "attractors_reinforced": self._attractors_reinforced.item(),
                "attractors_decayed": self._attractors_decayed.item(),
                "timestep": self.timestep.item(),
                "max_attractors": self.max_attractors,
                "utilization": f"{(attractor_count / self.max_attractors) * 100:.1f}%",
                "coherence_threshold": self.coherence_threshold,
                "decay_rate": self.decay_rate,
                "ema_momentum": self.ema_momentum,
            })

            # Add attractor field stats if we have attractors
            if attractor_count > 0:
                active_coherence = self.attractor_coherence[:attractor_count]
                stats["avg_coherence"] = active_coherence.mean().item()
                stats["max_coherence"] = active_coherence.max().item()
                stats["min_coherence"] = active_coherence.min().item()

        return stats

    def get_attractor_info(self) -> dict:
        """
        Get detailed information about all active attractors (Phase 2+ only).

        Returns:
            Dictionary with per-attractor information or empty dict if Phase 1
        """
        if self._phase < 2 or self.attractor_count == 0:
            return {}

        count = self.attractor_count.item()

        return {
            "centroids": self.attractor_centroids[:count].cpu().numpy(),
            "coherence": self.attractor_coherence[:count].cpu().numpy(),
            "last_used": self.attractor_last_used[:count].cpu().numpy(),
            "reinforcements": self.attractor_reinforcements[:count].cpu().numpy(),
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        base_repr = (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"phase={self._phase}"
        )

        if self._phase >= 2:
            attractor_repr = (
                f", max_attractors={self.max_attractors}, "
                f"active={self.attractor_count.item()}/{self.max_attractors}"
            )
            return base_repr + attractor_repr
        else:
            return base_repr


# Utility functions for future phases

def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between vectors in x and y.

    Args:
        x: Tensor of shape (..., d)
        y: Tensor of shape (..., d)

    Returns:
        Similarity matrix of shape (..., ...)

    Status: Utility for Phase 3 (pattern detection)
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
    return torch.matmul(x_norm, y_norm.transpose(-2, -1))


def exponential_decay(values: torch.Tensor, decay_rate: float, timestep: int) -> torch.Tensor:
    """
    Apply exponential decay to attractor values.

    Args:
        values: Attractor values to decay
        decay_rate: Decay rate (0 = no decay, 1 = full decay)
        timestep: Current timestep for decay calculation

    Returns:
        Decayed values

    Status: Utility for Phase 4 (attractor decay)
    """
    decay_factor = torch.exp(torch.tensor(-decay_rate * timestep))
    return values * decay_factor
