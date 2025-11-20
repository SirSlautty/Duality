"""
DRAI Resonance Layer V1 - Production-Ready Attractor Algorithm
(Updated for Plan A: TRUE multi-head K/V output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DraiResonanceLayerV1(nn.Module):
    """
    Safe, production-ready V1 attractor algorithm.

    This variant has been updated so that synthetic K/V pairs are returned with:
        [batch, seq_len, num_drai_heads, head_dim]

    instead of:
        [batch, seq_len, 1, head_dim]

    This enables true K/V concatenation inside attention (Plan A).
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
        self.num_heads = num_heads            # ← now actually used in output
        self.head_dim = head_dim if head_dim is not None else 64

        self.device = device or "cpu"
        self.dtype = dtype or torch.float32

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

        # -------------------------------------------
        # Registered buffers (move with model)
        # -------------------------------------------
        self.register_buffer(
            "attractor_vectors",
            torch.randn(self.M, self.d, device=self.device, dtype=self.dtype) * 1e-3
        )

        self.register_buffer(
            "attractor_strengths",
            torch.zeros(self.M, device=self.device, dtype=self.dtype)
        )

        self.register_buffer(
            "attractor_last_used",
            torch.zeros(self.M, device=self.device, dtype=self.dtype)
        )

        self.register_buffer(
            "timestep",
            torch.tensor(0.0, device=self.device, dtype=self.dtype)
        )

    # -------------------------------------------------------------------------
    # Forward: query → state update → synthetic K/V
    # -------------------------------------------------------------------------
    def forward(self, query_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_layer: [seq_len, batch, num_attn_heads, head_dim]

        Returns:
            k_reson: [seq_len, batch, num_drai_heads, head_dim]
            v_reson: same shape
        """
        seq_len, batch, _, _ = query_layer.shape

        # Collapse attention heads → mean representation
        query = query_layer.mean(dim=2)         # [seq_len, batch, d]
        query = query.transpose(0, 1)           # [batch, seq_len, d]

        A = self.attractor_vectors     # [M, d]
        S = self.attractor_strengths   # [M]
        L = self.attractor_last_used   # [M]
        step = self.timestep

        # Flatten queries
        B, T, d = query.shape
        q_flat = query.reshape(-1, d)  # [N, d]

        # Pattern detection
        match_idx, match_scores, match_mask, novel_mask = \
            self._detect_patterns(q_flat, A, S)

        # Updates
        A, S, L = self._accumulate_patterns(
            q_flat, A, S, L, match_idx, match_mask, step
        )
        A, S, L = self._create_new_attractors(q_flat, A, S, L, novel_mask, step)
        A, S, L = self._apply_decay(A, S, L)
        A, S, L = self._evict_weak_attractors(A, S, L)

        # Persist state
        self.attractor_vectors.copy_(A)
        self.attractor_strengths.copy_(S)
        self.attractor_last_used.copy_(L)
        self.timestep.copy_(step + 1.0)

        # Generate synthetic K/V for Plan A
        k_reson, v_reson = self._generate_synthetic_kv(query, A, S)

        # Shape back to NeoX format
        return k_reson.transpose(0, 1), v_reson.transpose(0, 1)

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------
    def _detect_patterns(self, q_flat, A, S):
        eps = 1e-8

        q_norm = F.normalize(q_flat, p=2, dim=-1, eps=eps)
        A_norm = F.normalize(A, p=2, dim=-1, eps=eps)

        sim = q_norm @ A_norm.T
        match_scores, match_idx = torch.max(sim, dim=-1)

        S_best = S[match_idx]

        match_mask = (S_best > self.strength_min) & (match_scores > self.theta_match)
        novel_mask = ~match_mask
        return match_idx, match_scores, match_mask, novel_mask

    # -------------------------------------------------------------------------
    # Accumulation
    # -------------------------------------------------------------------------
    def _accumulate_patterns(self, q_flat, A, S, L, match_idx, match_mask, step):
        N, d = q_flat.shape
        M = A.shape[0]

        matched_q = q_flat * match_mask.unsqueeze(-1)

        counts = torch.zeros(M, device=A.device, dtype=A.dtype)
        counts.scatter_add_(0, match_idx, match_mask.float())

        A_for_tokens = A[match_idx]
        deltas = matched_q - A_for_tokens

        delta_sum = torch.zeros(M, d, device=A.device, dtype=A.dtype)
        delta_sum.scatter_add_(
            0,
            match_idx.unsqueeze(-1).expand(-1, d),
            deltas
        )

        counts_safe = torch.clamp(counts, 1.0)
        A_new = A + self.alpha_update * (delta_sum / counts_safe.unsqueeze(-1))

        gamma = 0.01
        S_new = (1 - gamma) * S + counts

        used_mask = counts > 0
        L_new = torch.where(used_mask, step.expand_as(L), L)

        return A_new, S_new, L_new

    # -------------------------------------------------------------------------
    # Novel Pattern Creation
    # -------------------------------------------------------------------------
    def _create_new_attractors(self, q_flat, A, S, L, novel_mask, step):
        novel_idx = torch.where(novel_mask)[0]
        if len(novel_idx) == 0:
            return A, S, L

        free_idx = torch.where(S < self.strength_min)[0]
        if len(free_idx) == 0:
            return A, S, L

        k = min(len(novel_idx), len(free_idx))
        assign_nov = novel_idx[:k]
        assign_slots = free_idx[:k]

        q_new = q_flat[assign_nov]

        A_new = A.clone()
        S_new = S.clone()
        L_new = L.clone()

        A_new[assign_slots] = q_new
        S_new[assign_slots] = self.strength_init
        L_new[assign_slots] = step

        return A_new, S_new, L_new

    # -------------------------------------------------------------------------
    # Decay & Eviction
    # -------------------------------------------------------------------------
    def _apply_decay(self, A, S, L):
        return A, S * self.lambda_decay, L

    def _evict_weak_attractors(self, A, S, L):
        dead = S < self.strength_min
        zeroA = torch.zeros_like(A)
        zeroS = torch.zeros_like(S)
        zeroL = torch.zeros_like(L)
        return (
            torch.where(dead.unsqueeze(-1), zeroA, A),
            torch.where(dead, zeroS, S),
            torch.where(dead, zeroL, L),
        )

    # -------------------------------------------------------------------------
    # Synthetic K/V generation (Updated for Plan A)
    # -------------------------------------------------------------------------
    def _generate_synthetic_kv(self, query, A, S):
        """
        Output:
            k_reson: [batch, seq_len, num_heads, d]
            v_reson: [batch, seq_len, num_heads, d]
        """
        B, T, d = query.shape
        eps = 1e-8

        alive = S > self.strength_min
        if torch.sum(alive) == 0:
            zero = torch.zeros((B, T, self.num_heads, d), device=query.device, dtype=query.dtype)
            return zero, zero

        S_alive = torch.where(alive, S, torch.zeros_like(S))
        A_alive = torch.where(alive.unsqueeze(-1), A, torch.zeros_like(A))

        total_strength = torch.sum(S_alive) + eps

        # Burn-in gating
        if self.burn_in_mode == "strength":
            burn = total_strength < self.burn_in_threshold
        elif self.burn_in_mode == "tokens":
            burn = self.timestep < self.burn_in_tokens
        else:
            raise ValueError("Invalid burn_in_mode")

        if burn:
            zero = torch.zeros((B, T, self.num_heads, d), device=query.device, dtype=query.dtype)
            return zero, zero

        # Weighted mean field
        field_vec = torch.sum(A_alive * S_alive.unsqueeze(-1), dim=0) / total_strength
        norm = torch.linalg.norm(field_vec) + eps
        unit = field_vec / norm

        scale = self.max_influence_scale * torch.tanh(total_strength)
        field_scaled = scale * unit

        # Broadcast to all DRAI heads
        out = (
            field_scaled
            .view(1, 1, 1, d)
            .expand(B, T, self.num_heads, d)
        )

        return out, out

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def get_stats(self):
        S = self.attractor_strengths
        alive = S > self.strength_min
        total = torch.sum(S).item()

        if torch.sum(alive) == 0:
            mean = 0.0
        else:
            mean = torch.sum(torch.where(alive, S, torch.zeros_like(S))) / torch.sum(alive)

        return {
            "num_alive": torch.sum(alive).item(),
            "total_strength": total,
            "mean_strength": float(mean),
            "max_strength": torch.max(S).item(),
            "timestep": self.timestep.item(),
        }

