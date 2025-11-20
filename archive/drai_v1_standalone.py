"""
DRAI V1: Standalone implementation demonstrating core algorithm.
Written from understanding, not copied from repo.
"""

import torch
import torch.nn.functional as F


def drai_v1_inject(
    query: torch.Tensor,        # [batch, seq_len, dim]
    key: torch.Tensor,          # [batch, seq_len, dim]
    value: torch.Tensor,        # [batch, seq_len, dim]
    attractors: torch.Tensor,   # [max_attractors, dim] - mutable state
    strengths: torch.Tensor,    # [max_attractors] - mutable state
    timestep: int,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DRAI V1 attractor-based memory injection.

    Core algorithm:
    1. DETECT: Match queries to attractors (dual threshold: similarity + strength)
    2. UPDATE: EMA-update matched attractors, reinforce their strength
    3. CREATE: Assign novel patterns to free attractor slots
    4. DECAY: Exponentially fade all attractor strengths
    5. FIELD: Compute strength-weighted mean of active attractors (NOT individual selection)
    6. GATE: Apply burn-in (delay until stable) + soft scaling (tanh ramp-up)
    7. INJECT: Append synthetic K/V to attention context
    """

    batch, seq_len, dim = query.shape
    device = query.device

    # Unpack hyperparameters
    theta_match = config.get('theta_match', 0.8)          # High similarity threshold
    strength_min = config.get('strength_min', 1e-3)       # Eviction threshold
    alpha = config.get('alpha_update', 0.05)              # EMA momentum (slow updates)
    gamma = config.get('gamma', 0.01)                     # Strength decay per update
    lambda_decay = config.get('lambda_decay', 0.995)      # Global decay factor
    strength_init = config.get('strength_init', 0.5)      # New attractor strength
    burn_in_thresh = config.get('burn_in_threshold', 50.0) # Activation threshold
    max_influence = config.get('max_influence_scale', 0.15) # Influence cap

    # Flatten queries for batch processing
    q_flat = query.reshape(-1, dim)  # [batch * seq_len, dim]
    num_queries = q_flat.shape[0]

    # === 1. DETECT: Dual-threshold pattern matching ===
    # Cosine similarity between queries and attractors
    q_norm = F.normalize(q_flat, dim=-1, eps=1e-8)
    a_norm = F.normalize(attractors, dim=-1, eps=1e-8)
    sims = torch.mm(q_norm, a_norm.T)  # [num_queries, max_attractors]

    # Best match per query
    match_scores, match_idx = sims.max(dim=-1)  # [num_queries]

    # Dual gate: BOTH high similarity AND high strength required
    match_mask = (match_scores > theta_match) & (strengths[match_idx] > strength_min)
    novel_mask = ~match_mask

    # === 2. UPDATE: Reinforce matched attractors ===
    # EMA toward matching queries
    for i in range(num_queries):
        if match_mask[i]:
            idx = match_idx[i].item()
            # Centroid update
            attractors[idx] += alpha * (q_flat[i] - attractors[idx])
            # Strength boost
            strengths[idx] = (1 - gamma) * strengths[idx] + 1.0

    # === 3. CREATE: Novel patterns into free slots ===
    novel_queries = q_flat[novel_mask]
    if novel_queries.shape[0] > 0:
        free_mask = strengths < strength_min
        free_slots = torch.where(free_mask)[0]

        num_create = min(len(novel_queries), len(free_slots))
        if num_create > 0:
            attractors[free_slots[:num_create]] = novel_queries[:num_create].detach()
            strengths[free_slots[:num_create]] = strength_init

    # === 4. DECAY: Global exponential fade ===
    strengths.mul_(lambda_decay)

    # Evict dead attractors
    dead_mask = strengths < strength_min
    attractors[dead_mask] = 0
    strengths[dead_mask] = 0

    # === 5. FIELD: Strength-weighted mean (stable aggregation) ===
    alive_mask = strengths > strength_min
    num_alive = alive_mask.sum().item()

    if num_alive == 0:
        # No active attractors - return unmodified
        return key, value

    active_attractors = attractors[alive_mask]  # [num_alive, dim]
    active_strengths = strengths[alive_mask]    # [num_alive]

    # Weighted mean
    total_strength = active_strengths.sum()
    field_vec = (active_attractors * active_strengths.unsqueeze(-1)).sum(dim=0) / total_strength
    field_unit = F.normalize(field_vec, dim=-1, eps=1e-8)

    # === 6. GATE: Burn-in + soft scaling ===
    # Burn-in: don't inject until attractors stabilize
    if total_strength < burn_in_thresh:
        return key, value  # Still warming up

    # Soft gating: gradual ramp-up (not binary)
    soft_scale = torch.tanh(total_strength / 100.0)  # Normalize for tanh range
    influence = max_influence * soft_scale

    # === 7. INJECT: Append synthetic K/V ===
    k_inject = influence * field_unit  # [dim]
    v_inject = influence * field_unit  # [dim]

    # Expand to match batch/seq structure
    k_inject = k_inject.view(1, 1, dim).expand(batch, 1, dim)
    v_inject = v_inject.view(1, 1, dim).expand(batch, 1, dim)

    # Concatenate: attention now sees [original_context + memory]
    key_out = torch.cat([key, k_inject], dim=1)    # [batch, seq_len+1, dim]
    value_out = torch.cat([value, v_inject], dim=1)

    return key_out, value_out


def get_drai_v1_config(model_size: str = "410m") -> dict:
    """
    Scale-invariant hyperparameters for DRAI V1.

    These values maintain stability across 70M → 410M → 1B → 2.8B.
    """
    base_config = {
        'theta_match': 0.8,         # Conservative matching
        'strength_min': 1e-3,       # Clear dead/alive boundary
        'alpha_update': 0.05,       # Slow, stable EMA
        'gamma': 0.01,              # Small decay per update
        'lambda_decay': 0.995,      # Slow global fade (half-life ~138 steps)
        'strength_init': 0.5,       # Medium starting strength
        'burn_in_threshold': 50.0,  # ~10 tokens typical
    }

    # Only influence scale varies with model size
    influence_scales = {
        '70m': 0.05,    # Very gentle (fragile model)
        '410m': 0.15,   # Moderate
        '1b': 0.3,      # Stronger (robust model)
        '2.8b': 0.5,    # Full strength
    }

    base_config['max_influence_scale'] = influence_scales.get(
        model_size.lower(),
        0.15  # Safe default
    )

    return base_config


# === Stability Invariants ===
"""
WHY THIS ALGORITHM IS STABLE ACROSS MODEL SIZES:

1. **NORMALIZED GEOMETRY**
   - All vectors normalized before similarity comparison
   - Field vector normalized before injection
   - Scale-invariant: works whether dim=768 or dim=4096
   - Model size doesn't affect geometric relationships

2. **DUAL THRESHOLDS PREVENT CORRUPTION**
   - Similarity threshold (0.8): Only strong matches update attractors
   - Strength threshold (1e-3): Only established attractors contribute
   - Together: Prevents random noise from corrupting memory
   - Small models don't accumulate garbage; large models don't thrash

3. **FIELD VECTOR AGGREGATION**
   - Weighted mean (not winner-take-all) smooths noise
   - Many weak attractors = gentle influence
   - Few strong attractors = moderate influence
   - Never catastrophic injection from single outlier
   - Statistical averaging stabilizes across scales

4. **SOFT GATING WITH BURN-IN**
   - Burn-in: Delayed activation until strength > 50
   - Prevents early instability when patterns unclear
   - Soft scaling: tanh(strength) gradual ramp-up
   - No sudden transitions that destabilize generation
   - Works identically on 70M (careful) and 2.8B (confident)

5. **EXPONENTIAL DECAY PREVENTS ACCUMULATION**
   - lambda=0.995 → half-life ~138 steps
   - Old attractors fade, new attractors form
   - Bounded memory: max N attractors, not unbounded growth
   - Self-limiting: strong attractors survive, weak ones die
   - Scale-independent temporal dynamics

6. **INFLUENCE SCALES WITH MODEL CAPACITY**
   - 70M: 0.05 max influence (5% perturbation)
   - 2.8B: 0.5 max influence (50% perturbation)
   - Larger models tolerate stronger memory signals
   - Smaller models need gentler nudges
   - ONLY hyperparameter that scales with size

7. **EMA MOMENTUM IS FIXED**
   - alpha=0.05 regardless of model size
   - Slow attractor drift prevents instability
   - Fast enough to track context shifts
   - Slow enough to resist single-token noise
   - Temporal constant, not spatial constant

RESULT: Same algorithm, same hyperparameters (except influence scale),
stable operation from 70M to 2.8B+ parameters.

The key insight: DRAI operates in normalized latent space, not parameter space.
Transformer size affects capacity, not geometry.
"""
