# Attractor Mathematics for DRAI

**Date:** 2025-11-18
**Status:** Phase 2 Design Specification
**Purpose:** Formal mathematical definition of DRAI attractor dynamics

---

## Overview

This document defines the mathematical formulation for attractor formation, pattern detection, reinforcement, and decay in the Dynamic Resonance AI (DRAI) system. The goal is to create a memory mechanism that:

1. **Detects** recurring patterns in latent space
2. **Forms** stable attractors from these patterns
3. **Reinforces** attractors when patterns repeat
4. **Decays** unused attractors over time
5. **Generates** synthetic K/V pairs from active attractors

---

## Core Concepts

### Attractor Definition

An **attractor** `A_i` is a stable point in latent space representing a recurring pattern. Each attractor consists of:

```
A_i = {
    μ_i ∈ ℝ^d        # Centroid (mean pattern representation)
    σ_i ∈ ℝ          # Coherence strength (how stable/strong)
    t_i ∈ ℕ          # Last activation timestep
    n_i ∈ ℕ          # Number of reinforcements
}
```

Where:
- `μ_i`: The d-dimensional centroid representing the "average" of all patterns that formed this attractor
- `σ_i`: Coherence value in [0, 1] indicating attractor strength
- `t_i`: Timestep when last reinforced (for decay)
- `n_i`: Count of how many times this attractor has been activated

### Attractor Field

The **attractor field** `F` is the collection of all attractors:

```
F = {A_1, A_2, ..., A_k}
```

Where `k` is the current number of attractors (dynamic, grows/shrinks over time).

---

## Phase 2 Implementation: Simple Moving Average

For Phase 2, we implement a simplified but functional attractor system using **exponential moving average (EMA)** for accumulation.

### Parameters

```python
max_attractors: int = 32          # Maximum number of attractors to maintain
head_dim: int = 64                # Dimension of latent vectors
coherence_threshold: float = 0.3  # Minimum similarity to match existing attractor
formation_threshold: float = 0.5  # Minimum coherence to form new attractor
decay_rate: float = 0.01          # Exponential decay per timestep
ema_momentum: float = 0.9         # Momentum for exponential moving average
```

### 1. Pattern Extraction

Given query layer `Q ∈ ℝ^(S×B×H×D)` where:
- S = sequence length
- B = batch size
- H = number of attention heads
- D = head dimension

Extract representative pattern `p`:

```
p = mean(Q, dim=[0, 1, 2])  ∈ ℝ^D
```

We average over sequence, batch, and heads to get a single D-dimensional vector representing the current "thought".

**Alternative (Phase 3+):** Use attention-weighted pooling or extract multiple patterns.

### 2. Pattern Matching

For each existing attractor `A_i`, compute cosine similarity with pattern `p`:

```
sim(p, μ_i) = (p · μ_i) / (||p|| · ||μ_i||)
```

This gives similarity in range [-1, 1], where 1 = identical direction.

Find best matching attractor:

```
i* = argmax_i sim(p, μ_i)
s_max = sim(p, μ_{i*})
```

### 3. Attractor Update or Creation

#### Case A: Strong Match (s_max > coherence_threshold)

**Reinforce existing attractor** using exponential moving average:

```python
# Update centroid with EMA
μ_{i*} ← ema_momentum * μ_{i*} + (1 - ema_momentum) * p

# Normalize to unit sphere (optional but recommended)
μ_{i*} ← μ_{i*} / ||μ_{i*}||

# Increase coherence (bounded by 1.0)
σ_{i*} ← min(1.0, σ_{i*} + 0.1 * (1 - σ_{i*}))

# Update metadata
t_{i*} ← t_current
n_{i*} ← n_{i*} + 1
```

The EMA update means old patterns have weight `ema_momentum` (0.9) and new pattern has weight (0.1), creating smooth evolution.

#### Case B: Weak Match (s_max ≤ coherence_threshold) and Space Available

**Create new attractor** if we haven't reached max_attractors:

```python
if len(F) < max_attractors:
    A_new = {
        μ: p / ||p||,           # Normalized pattern
        σ: formation_threshold,  # Initial coherence
        t: t_current,
        n: 1
    }
    F ← F ∪ {A_new}
```

#### Case C: Weak Match and No Space

**Replace weakest attractor** (lowest coherence):

```python
i_weakest = argmin_i σ_i

A_{i_weakest} ← {
    μ: p / ||p||,
    σ: formation_threshold,
    t: t_current,
    n: 1
}
```

### 4. Attractor Decay

After each forward pass, decay all attractors that weren't activated:

```python
for each A_i in F:
    if t_i < t_current:  # Not activated this step
        Δt = t_current - t_i
        σ_i ← σ_i * exp(-decay_rate * Δt)

        # Remove if too weak
        if σ_i < 0.05:
            F ← F \ {A_i}
```

This implements exponential decay: coherence drops by factor `exp(-decay_rate)` per timestep.

### 5. Synthetic K/V Generation

Generate synthetic keys and values from **top-k strongest** attractors:

```python
# Sort attractors by coherence
sorted_attractors = sort(F, key=λ A: A.σ, reverse=True)

# Take top-k (e.g., k=1 for single DRAI head)
top_k = sorted_attractors[:num_drai_heads]

# Generate K/V for each
for i, A in enumerate(top_k):
    # Key is the attractor centroid
    k_reson[i] = A.μ

    # Value is weighted by coherence (strong attractors contribute more)
    v_reson[i] = A.σ * A.μ
```

**Shape handling:**
```python
# Expand to sequence and batch dimensions
# k_reson: [1, 1, num_heads, head_dim] → [seq, batch, num_heads, head_dim]
k_reson = k_reson.expand(seq_len, batch, num_drai_heads, head_dim)
v_reson = v_reson.expand(seq_len, batch, num_drai_heads, head_dim)
```

---

## Mathematical Properties

### Stability

The EMA update ensures attractors are **stable** - they won't drastically change from a single input:

```
μ_{t+1} = 0.9 * μ_t + 0.1 * p
```

This means it takes ~10 reinforcements for a pattern to significantly shift an attractor.

### Capacity

With `max_attractors = 32` and `head_dim = 64`, the attractor field has:
- **Memory capacity:** 32 distinct patterns
- **Parameter count:** 32 × 64 = 2,048 floats (8 KB at float32)
- **Overhead:** Minimal compared to transformer parameters

### Decay Dynamics

With `decay_rate = 0.01`:
- After 10 steps: coherence × exp(-0.1) ≈ 0.90 × coherence
- After 50 steps: coherence × exp(-0.5) ≈ 0.61 × coherence
- After 100 steps: coherence × exp(-1.0) ≈ 0.37 × coherence
- After 300 steps: coherence × exp(-3.0) ≈ 0.05 × coherence (pruned)

Unused attractors naturally fade over ~300 forward passes.

---

## Implementation Pseudocode

```python
class DraiResonanceLayer(nn.Module):
    def __init__(self, ...):
        # Attractor field (k × head_dim)
        self.register_buffer('attractor_centroids', torch.zeros(max_attractors, head_dim))
        self.register_buffer('attractor_coherence', torch.zeros(max_attractors))
        self.register_buffer('attractor_last_used', torch.zeros(max_attractors, dtype=torch.long))
        self.register_buffer('attractor_count', torch.tensor(0))
        self.register_buffer('timestep', torch.tensor(0))

    def forward(self, query_layer):
        # 1. Extract pattern
        pattern = query_layer.mean(dim=[0, 1, 2])  # [head_dim]
        pattern = F.normalize(pattern, p=2, dim=-1)

        # 2. Find best match
        if self.attractor_count > 0:
            active_centroids = self.attractor_centroids[:self.attractor_count]
            similarities = F.cosine_similarity(
                pattern.unsqueeze(0),
                active_centroids,
                dim=1
            )
            best_idx = similarities.argmax()
            best_sim = similarities[best_idx]
        else:
            best_sim = torch.tensor(-1.0)

        # 3. Update or create attractor
        if best_sim > self.coherence_threshold:
            # Reinforce existing
            self._reinforce_attractor(best_idx, pattern)
        elif self.attractor_count < self.max_attractors:
            # Create new
            self._create_attractor(pattern)
        else:
            # Replace weakest
            weakest_idx = self.attractor_coherence[:self.attractor_count].argmin()
            self._replace_attractor(weakest_idx, pattern)

        # 4. Decay unused attractors
        self._decay_attractors()

        # 5. Generate synthetic K/V
        k_reson, v_reson = self._generate_kv(query_layer.shape)

        self.timestep += 1
        return k_reson, v_reson
```

---

## Advantages of This Approach

1. **Simple:** Easy to implement and understand
2. **Efficient:** O(k) similarity checks, k ≪ sequence length
3. **Stable:** EMA prevents wild oscillations
4. **Self-organizing:** Automatically forms/prunes attractors
5. **Gradient-friendly:** All operations differentiable
6. **Memory-efficient:** Fixed memory footprint

---

## Limitations and Future Work

### Current Limitations (Phase 2)

1. **Single pattern per forward pass:** Only extracts one pattern via mean pooling
2. **No spatial awareness:** Ignores sequence position information
3. **Fixed capacity:** Hard limit on number of attractors
4. **Simple similarity:** Only uses cosine similarity
5. **No clustering:** Each attractor is independent

### Future Enhancements (Phase 3+)

1. **Multi-pattern extraction:** Extract multiple patterns per batch using clustering
2. **Positional encoding:** Incorporate sequence position into attractor matching
3. **Hierarchical attractors:** Parent-child relationships between attractors
4. **Learnable thresholds:** Train coherence_threshold, decay_rate as parameters
5. **Attention-based extraction:** Use attention weights to extract salient patterns
6. **Cross-attractor dynamics:** Allow attractors to influence each other

---

## Validation Metrics

To measure whether attractors are working:

### 1. Attractor Formation Rate
```
formation_rate = new_attractors_created / total_forward_passes
```
Should be high initially, then stabilize.

### 2. Attractor Utilization
```
utilization = active_attractors / max_attractors
```
Should grow to ~50-80% for healthy memory usage.

### 3. Coherence Distribution
Plot histogram of `σ_i` values. Healthy:
- Some high coherence (>0.7): stable, well-reinforced patterns
- Some medium (0.3-0.7): forming patterns
- Some low (<0.3): recent or weak patterns

### 4. Pattern Recall
On repeated sequences, measure:
```
recall = P(attractor_matched | pattern_seen_before)
```
Should be high (>0.8) for well-formed attractors.

### 5. Diversity
Measure average pairwise distance between attractors:
```
diversity = mean_{i≠j} ||μ_i - μ_j||
```
Should be high to indicate attractors represent distinct patterns.

---

## Hyperparameter Tuning Guide

| Parameter | Low Value | High Value | Effect |
|-----------|-----------|------------|--------|
| `max_attractors` | 8-16 | 64-128 | Memory capacity vs overhead |
| `coherence_threshold` | 0.2 | 0.6 | Easier/harder to match existing |
| `formation_threshold` | 0.3 | 0.7 | Initial strength of new attractors |
| `decay_rate` | 0.001 | 0.1 | Faster/slower forgetting |
| `ema_momentum` | 0.8 | 0.99 | Faster/slower attractor updates |

**Recommended starting values:**
- `max_attractors`: 32 (good balance)
- `coherence_threshold`: 0.3 (medium selectivity)
- `formation_threshold`: 0.5 (medium strength)
- `decay_rate`: 0.01 (slow forgetting)
- `ema_momentum`: 0.9 (moderate smoothing)

---

## Phase 2 Implementation Checklist

- [ ] Add attractor field buffers to DraiResonanceLayer
- [ ] Implement pattern extraction (mean pooling)
- [ ] Implement cosine similarity matching
- [ ] Implement EMA update for reinforcement
- [ ] Implement new attractor creation
- [ ] Implement attractor decay
- [ ] Implement synthetic K/V generation
- [ ] Add configuration parameters
- [ ] Create unit tests for each component
- [ ] Create integration test with repeated patterns
- [ ] Measure attractor formation on toy dataset
- [ ] Document results and findings

---

**Next:** Implement these formulas in `src/drai/resonance_layer.py` Phase 2 code.
