"""
DRAI Resonance Layer V1 - Production-Ready Attractor Algorithm

This is a clean, production-ready implementation of the DRAI attractor algorithm
that addresses the fundamental issues discovered in Phase 5 gating experiments.

Key improvements over Phase 2:
1. Proper pattern detection: Only updates on good matches (live + score thresholds)
2. Novel pattern handling: Creates new attractors in free slots only
3. Field vector approach: Weighted mean is more stable than individual attractors
4. Soft gating: tanh(total_strength) * max_influence_scale prevents catastrophic influence
5. Conservative deployment: Designed for 1-2 layers, not all layers

This algorithm:
- Doesn't wreck small models (tested on pythia-410m)
- Has real pattern detection / accumulation / decay
- Only injects latent-valid, gated K/V

Algorithm design by Halcyon AI Research
PyTorch implementation: 2025-11-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DraiResonanceLayerV1(nn.Module):
    """
    DRAI Resonance Layer V1 - Safe, production-ready attractor algorithm.

    State tracked per layer:
    - A ∈ [M, d]: Attractor vectors (centroids of recurring patterns)
    - S ∈ [M]: Attractor strengths (accumulated from reinforcement)
    - L ∈ [M]: Last-used timestep (for age-based decay, optional)
    - step: Global timestep counter

    Algorithm flow:
    1. DETECT: Which queries match which attractors?
    2. ACCUMULATE: Update matched attractors (centroid + strength)
    3. CREATE: Form new attractors from novel patterns
    4. DECAY: Global decay of strengths
    5. EVICT: Clear out dead/weak attractors
    6. GENERATE: Create synthetic K/V with safety gating

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of DRAI resonance heads (typically 1)
        head_dim: Dimension of each head
        max_attractors: Fixed bank size (16 for 410M, 32 for larger models)
        theta_match: Cosine similarity threshold for pattern matching (default: 0.8)
        alpha_update: Learning rate for centroid updates (default: 0.05)
        lambda_decay: Global decay factor per step (default: 0.995)
        strength_init: Initial strength for new attractors (default: 0.5)
        strength_min: Minimum strength before eviction (default: 1e-3)
        max_influence_scale: Cap on K/V influence (default: 0.15 for 410M)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        # V1 Hyperparameters
        max_attractors: int = 16,
        theta_match: float = 0.8,
        alpha_update: float = 0.05,
        lambda_decay: float = 0.995,
        strength_init: float = 0.5,
        strength_min: float = 1e-3,
        max_influence_scale: float = 0.15,
        burn_in_threshold: float = 50.0,
        burn_in_mode: str = "strength",
        burn_in_tokens: int = 10,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else 64
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32

        # V1 hyperparameters
        self.M = max_attractors
        self.d = self.head_dim
        self.theta_match = theta_match
        self.alpha_update = alpha_update
        self.lambda_decay = lambda_decay
        self.strength_init = strength_init
        self.strength_min = strength_min
        self.max_influence_scale = max_influence_scale
        self.burn_in_threshold = burn_in_threshold
        self.burn_in_mode = burn_in_mode
        self.burn_in_tokens = burn_in_tokens

        # State buffers (registered as buffers so they move with model)
        # A ∈ [M, d] - attractor vectors
        self.register_buffer(
            "attractor_vectors",
            torch.randn(self.M, self.d, device=self.device, dtype=self.dtype) * 1e-3
        )

        # S ∈ [M] - attractor strengths
        self.register_buffer(
            "attractor_strengths",
            torch.zeros(self.M, device=self.device, dtype=self.dtype)
        )

        # L ∈ [M] - last used timestep
        self.register_buffer(
            "attractor_last_used",
            torch.zeros(self.M, device=self.device, dtype=self.dtype)
        )

        # Global timestep counter
        self.register_buffer(
            "timestep",
            torch.tensor(0.0, device=self.device, dtype=self.dtype)
        )

    def forward(self, query_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process query vectors and return synthetic K/V pairs.

        Args:
            query_layer: [seq_len, batch, num_attn_heads, head_dim]
                        Query vectors from transformer attention

        Returns:
            k_reson: [seq_len, batch, num_drai_heads, head_dim]
            v_reson: [seq_len, batch, num_drai_heads, head_dim]
        """
        seq_len, batch, num_attn_heads, head_dim = query_layer.shape

        # For now, average across attention heads to get single representation
        # [seq_len, batch, head_dim]
        query = query_layer.mean(dim=2)

        # Reshape to [batch, seq_len, head_dim]
        query = query.transpose(0, 1)

        # Get state
        A = self.attractor_vectors
        S = self.attractor_strengths
        L = self.attractor_last_used
        step = self.timestep

        # Flatten queries: [batch, seq_len, d] -> [N, d] where N = batch * seq_len
        B, T, d = query.shape
        q_flat = query.reshape(-1, d)  # [N, d]

        # 1) DETECT: which queries match which attractors?
        match_indices, match_scores, match_mask, novel_mask = \
            self._detect_patterns(q_flat, A, S)

        # 2) ACCUMULATE: update matched attractors
        A, S, L = self._accumulate_patterns(
            q_flat, A, S, L,
            match_indices, match_mask, step
        )

        # 3) CREATE: form new attractors from novel patterns
        A, S, L = self._create_new_attractors(
            q_flat, A, S, L,
            novel_mask, step
        )

        # 4) DECAY: global decay of strengths
        A, S, L = self._apply_decay(A, S, L)

        # 5) EVICT: clear out dead/weak attractors
        A, S, L = self._evict_weak_attractors(A, S, L)

        # Increment step
        step = step + 1.0

        # Update state
        self.attractor_vectors.copy_(A)
        self.attractor_strengths.copy_(S)
        self.attractor_last_used.copy_(L)
        self.timestep.copy_(step)

        # 6) GENERATE: synthetic K/V with safety gating
        k_reson, v_reson = self._generate_synthetic_kv(
            query, A, S
        )

        # Reshape back to expected format: [seq_len, batch, num_drai_heads, head_dim]
        k_reson = k_reson.transpose(0, 1)
        v_reson = v_reson.transpose(0, 1)

        return k_reson, v_reson

    def _detect_patterns(
        self,
        q_flat: torch.Tensor,
        A: torch.Tensor,
        S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect which queries match existing attractors.

        Args:
            q_flat: [N, d] - flattened query vectors
            A: [M, d] - attractor vectors
            S: [M] - attractor strengths

        Returns:
            match_indices: [N] - best attractor index per query
            match_scores: [N] - cosine similarity scores
            match_mask: [N] - True if query reinforces an attractor
            novel_mask: [N] - True if query can seed a new attractor
        """
        eps = 1e-8
        N, d = q_flat.shape
        M = A.shape[0]

        # Normalize queries and attractors for cosine similarity
        q_norm = F.normalize(q_flat, p=2, dim=-1, eps=eps)  # [N, d]
        A_norm = F.normalize(A, p=2, dim=-1, eps=eps)       # [M, d]

        # Compute similarity: [N, M]
        sim = torch.matmul(q_norm, A_norm.T)

        # Best match per query
        match_scores, match_indices = torch.max(sim, dim=-1)  # [N], [N]

        # A query can only match a "live" attractor (S > strength_min)
        S_best = S[match_indices]  # [N]

        live_mask = S_best > self.strength_min
        score_mask = match_scores > self.theta_match

        match_mask = live_mask & score_mask  # [N]
        novel_mask = ~match_mask

        return match_indices, match_scores, match_mask, novel_mask

    def _accumulate_patterns(
        self,
        q_flat: torch.Tensor,
        A: torch.Tensor,
        S: torch.Tensor,
        L: torch.Tensor,
        match_indices: torch.Tensor,
        match_mask: torch.Tensor,
        step: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update matched attractors (centroid + strength).

        Each matching token nudges its attractor toward itself.
        Strengths increase with use.
        Last-used timestamps updated.

        Args:
            q_flat: [N, d] - query vectors
            A: [M, d] - attractor vectors
            S: [M] - attractor strengths
            L: [M] - last used timesteps
            match_indices: [N] - best attractor index per query
            match_mask: [N] - which queries matched
            step: scalar - current timestep

        Returns:
            A_new, S_new, L_new - updated state
        """
        N, d = q_flat.shape
        M = A.shape[0]

        # Only consider matched queries
        matched_q = q_flat * match_mask.unsqueeze(-1)  # [N, d]

        # Count how many times each attractor was hit
        # PyTorch equivalent of JAX segment_sum
        counts = torch.zeros(M, device=A.device, dtype=A.dtype)
        counts.scatter_add_(
            0,
            match_indices,
            match_mask.float()
        )  # [M]

        # Get current attractor values for each query
        A_for_tokens = A[match_indices]  # [N, d]

        # Compute deltas: (q - A[m])
        deltas = matched_q - A_for_tokens  # [N, d]

        # Sum deltas per attractor
        delta_sum = torch.zeros(M, d, device=A.device, dtype=A.dtype)
        delta_sum.scatter_add_(
            0,
            match_indices.unsqueeze(-1).expand(-1, d),
            deltas
        )  # [M, d]

        # Avoid divide-by-zero
        counts_safe = torch.clamp(counts, min=1.0)  # [M]

        # Centroid-like update: A_new = A + α * (mean_delta)
        A_new = A + self.alpha_update * (delta_sum / counts_safe.unsqueeze(-1))

        # Strength update: gentle decay + reinforcement proportional to hits
        gamma = 0.01
        S_new = (1.0 - gamma) * S + counts

        # Last-used: update where counts > 0
        used_mask = counts > 0
        L_new = torch.where(used_mask, step.expand_as(L), L)

        return A_new, S_new, L_new

    def _create_new_attractors(
        self,
        q_flat: torch.Tensor,
        A: torch.Tensor,
        S: torch.Tensor,
        L: torch.Tensor,
        novel_mask: torch.Tensor,
        step: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Seed new attractors from novel patterns in free/weak slots.

        Args:
            q_flat: [N, d] - query vectors
            A: [M, d] - attractor vectors
            S: [M] - attractor strengths
            L: [M] - last used timesteps
            novel_mask: [N] - which queries are novel
            step: scalar - current timestep

        Returns:
            A_new, S_new, L_new - updated state
        """
        M, d = A.shape

        # Get indices of novel queries
        novel_indices = torch.where(novel_mask)[0]  # [K]
        K = novel_indices.shape[0]

        if K == 0:
            return A, S, L

        # Get indices of free/weak slots
        free_mask = S < self.strength_min  # [M]
        free_indices = torch.where(free_mask)[0]  # [F]
        F = free_indices.shape[0]

        if F == 0:
            # No free slots, do nothing (let decay/eviction handle it)
            return A, S, L

        # Assign up to min(K, F) new attractors
        num_assign = min(K, F)

        # Take first num_assign novel tokens and free slots
        assign_novel = novel_indices[:num_assign]
        assign_slots = free_indices[:num_assign]

        # Get novel query vectors
        q_new = q_flat[assign_novel]  # [num_assign, d]

        # Update state
        A_new = A.clone()
        S_new = S.clone()
        L_new = L.clone()

        A_new[assign_slots] = q_new
        S_new[assign_slots] = self.strength_init
        L_new[assign_slots] = step

        return A_new, S_new, L_new

    def _apply_decay(
        self,
        A: torch.Tensor,
        S: torch.Tensor,
        L: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply global decay to attractor strengths.

        Can be extended with age-dependent rules in future versions.
        """
        A_new = A * 1.0  # For now, don't shrink vectors, just strengths
        S_new = S * self.lambda_decay
        return A_new, S_new, L

    def _evict_weak_attractors(
        self,
        A: torch.Tensor,
        S: torch.Tensor,
        L: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Zero out attractors whose strength has fallen below minimum.
        """
        dead_mask = S < self.strength_min  # [M]

        A_new = torch.where(dead_mask.unsqueeze(-1), torch.zeros_like(A), A)
        S_new = torch.where(dead_mask, torch.zeros_like(S), S)
        L_new = torch.where(dead_mask, torch.zeros_like(L), L)

        return A_new, S_new, L_new

    def _generate_synthetic_kv(
        self,
        query: torch.Tensor,
        A: torch.Tensor,
        S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic K/V pairs with safety gating and burn-in threshold.

        Design:
        - Compute global field vector as strength-weighted mean of active attractors
        - Gate by overall strength and number of alive attractors
        - Scale by max_influence_scale (gentle nudge, not bulldozer)
        - BURN-IN THRESHOLD: Don't inject until total_strength >= threshold
        - K_reson == V_reson == field_vector (can add transforms later)

        Args:
            query: [B, T, d] - original query (not used directly, just for shape)
            A: [M, d] - attractor vectors
            S: [M] - attractor strengths

        Returns:
            k_reson: [B, T, 1, d] - synthetic keys
            v_reson: [B, T, 1, d] - synthetic values
        """
        B, T, d = query.shape
        eps = 1e-8

        # Only use "alive" attractors
        alive_mask = S > self.strength_min  # [M]
        num_alive = torch.sum(alive_mask)

        if num_alive == 0:
            # No resonance - return zeros
            zero = torch.zeros((B, T, 1, d), dtype=query.dtype, device=query.device)
            return zero, zero

        # Get alive attractors and their strengths
        S_alive = torch.where(alive_mask, S, torch.zeros_like(S))  # [M]
        A_alive = torch.where(alive_mask.unsqueeze(-1), A, torch.zeros_like(A))  # [M, d]

        total_strength = torch.sum(S_alive) + eps  # scalar

        # BURN-IN GATING: Don't inject until attractors have accumulated enough strength OR tokens
        # This prevents early instability and feedback loops
        burn_in_active = False
        if self.burn_in_mode == "strength":
            # Strength-based: adaptive to actual attractor formation
            burn_in_active = total_strength < self.burn_in_threshold
        elif self.burn_in_mode == "tokens":
            # Token-based: predictable, ignores dynamics
            burn_in_active = self.timestep < self.burn_in_tokens
        else:
            raise ValueError(f"Unknown burn_in_mode: {self.burn_in_mode}")

        if burn_in_active:
            # Attractors still warming up - return zeros (passive mode)
            zero = torch.zeros((B, T, 1, d), dtype=query.dtype, device=query.device)
            return zero, zero

        # Weighted mean field vector
        field_vec = torch.sum(A_alive * S_alive.unsqueeze(-1), dim=0) / total_strength  # [d]

        # Normalize to keep in reasonable norm range
        norm = torch.linalg.norm(field_vec) + eps
        field_unit = field_vec / norm

        # Soft gating: scale influence based on total_strength
        raw_scale = torch.tanh(total_strength)  # ∈ (0, 1)
        scale = self.max_influence_scale * raw_scale  # final influence weight

        field_scaled = scale * field_unit  # [d]

        # Broadcast to [B, T, 1, d]
        field_bt = field_scaled.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, 1, d)

        return field_bt, field_bt

    def get_stats(self) -> dict:
        """
        Get current attractor statistics for monitoring.

        Returns:
            dict with keys:
                - num_alive: Number of active attractors
                - total_strength: Sum of all strengths
                - mean_strength: Average strength of alive attractors
                - max_strength: Maximum strength
                - timestep: Current timestep
                - burn_in_active: Whether burn-in is still active (strength < threshold)
                - burn_in_threshold: The threshold value
                - burn_in_progress: Percentage of burn-in completed (0-100%)
        """
        alive_mask = self.attractor_strengths > self.strength_min
        num_alive = torch.sum(alive_mask).item()
        total_strength = torch.sum(self.attractor_strengths).item()

        if num_alive > 0:
            mean_strength = torch.sum(
                torch.where(alive_mask, self.attractor_strengths, torch.zeros_like(self.attractor_strengths))
            ).item() / num_alive
        else:
            mean_strength = 0.0

        max_strength = torch.max(self.attractor_strengths).item()

        # Burn-in status (depends on mode)
        if self.burn_in_mode == "strength":
            burn_in_active = total_strength < self.burn_in_threshold
            if self.burn_in_threshold > 0:
                burn_in_progress = min(100.0, (total_strength / self.burn_in_threshold) * 100.0)
            else:
                burn_in_progress = 100.0  # Burn-in disabled
        elif self.burn_in_mode == "tokens":
            timestep = self.timestep.item()
            burn_in_active = timestep < self.burn_in_tokens
            if self.burn_in_tokens > 0:
                burn_in_progress = min(100.0, (timestep / self.burn_in_tokens) * 100.0)
            else:
                burn_in_progress = 100.0  # Burn-in disabled
        else:
            burn_in_active = False
            burn_in_progress = 100.0

        return {
            "num_alive": num_alive,
            "total_strength": total_strength,
            "mean_strength": mean_strength,
            "max_strength": max_strength,
            "timestep": self.timestep.item(),
            "burn_in_active": burn_in_active,
            "burn_in_mode": self.burn_in_mode,
            "burn_in_threshold": self.burn_in_threshold,
            "burn_in_tokens": self.burn_in_tokens,
            "burn_in_progress": burn_in_progress,
        }
