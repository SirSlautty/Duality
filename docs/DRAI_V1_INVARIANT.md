# DRAI V1: The Invariant Equation

**Date**: 2025-11-19
**Status**: Core mathematical formulation

---

## The Single Invariant

DRAI V1 can be expressed as a topological morphism on the model's latent manifold:

```
F(q; M_A) = θ(‖S‖) · π_M_A(q̂)
```

**where**:

- **M_A = {(a_i, s_i)}** : attractor manifold state (positions + strengths)
- **π_M_A : ℝ^d → M_A** : soft projection operator onto attractor manifold
- **θ : ℝ → [0, α_max]** : gating function (burn-in + soft scaling)
- **q̂** : normalized query vector

---

## Component Definitions

### Soft Projection Operator

```
π_M_A(q̂) := (∑_i s_i · κ(q̂, â_i) · â_i) / (∑_i s_i · κ(q̂, â_i))
```

**Interpretation**: Weighted mean of attractors, where weights = (strength × similarity)

**Properties**:
- Continuous mapping from query space to attractor manifold
- Preserves manifold structure (no discontinuities)
- Reduces to nearest attractor when one dominates
- Becomes uniform blend when all similar

### Similarity Kernel

```
κ(q̂, â_i) = exp(β · ⟨q̂, â_i⟩)
```

**where**:
- ⟨·,·⟩ : cosine similarity (inner product of normalized vectors)
- β : temperature parameter (controls sharpness)

**Properties**:
- Smooth, differentiable everywhere
- Maps [-1, 1] → (0, ∞) smoothly
- High similarity → high weight
- Prevents division by zero (always positive)

### Gating Function

```
θ(‖S‖) = α_max · tanh(‖S‖/τ) · H(‖S‖ - τ_burn)
```

**where**:
- ‖S‖ = ∑_i s_i : total attractor strength
- τ : scaling constant for tanh
- τ_burn : burn-in threshold
- H : Heaviside step function
- α_max : maximum influence scale

**Properties**:
- Burn-in: θ = 0 when ‖S‖ < τ_burn
- Soft ramp: θ smoothly increases as ‖S‖ grows
- Saturates: θ → α_max as ‖S‖ → ∞
- No discontinuities (except at burn-in boundary)

### Attractor Evolution

```
dM_A/dt = -∇_A E(M_A, Q_t)
```

**where**:
- E(M_A, Q_t) : memory energy functional
- Q_t : query distribution at time t
- ∇_A : gradient with respect to attractor positions

**In practice** (EMA flow):
```
da_i/dt = α · (q_matched - a_i) · δ(i matched)
ds_i/dt = (γ · δ(i matched) - λ) · s_i
```

**Properties**:
- Smooth gradient flow (exponential smoothing)
- Attractors drift toward matching queries
- Strengths reinforce with hits, decay globally
- Bounded evolution (no explosion)

---

## The Invariant Structure

### What Makes This An Invariant

**F is covariant under model transformations**:

1. **Scale invariance**: Operates on normalized vectors (q̂, â)
   - Works whether hidden_dim = 768 or 4096
   - Angles are scale-free

2. **Rotation invariance**: Cosine similarity is rotation-invariant
   - Same algorithm works in any basis
   - Model's learned geometry doesn't break F

3. **Translation invariance**: Projection π preserves relative structure
   - Moving all attractors by constant doesn't change behavior
   - Only relative positions matter

4. **Temporal smoothness**: All components differentiable
   - θ smooth (except burn-in step, but that's deliberate)
   - π smooth (exponential kernel)
   - dM_A/dt smooth (EMA)

### Why This Works Across Model Sizes

**The equation doesn't reference model parameters**:
- No dependence on layer count
- No dependence on parameter count
- No dependence on architecture details

**Only latent geometry matters**:
- Queries live in ℝ^d (normalized)
- Attractors live in ℝ^d (normalized)
- F maps between these spaces

**Model size affects capacity, not geometry**:
- 70M model: Simpler latent structure, needs gentle influence (α_max = 0.05)
- 2.8B model: Richer latent structure, tolerates strong influence (α_max = 0.5)
- **Only α_max scales** - everything else is identical

---

## Invariants Preserved By The Algorithm

### 1. Topological Invariant
**π_M_A is continuous and well-defined**

Preserved by:
- Dual thresholds: Only use attractors with s_i > ε (prevents division by zero)
- Exponential kernel: κ always positive (denominator never zero)
- Field vector normalization: Output always unit length

### 2. Geometric Invariant
**All operations preserve angles**

Preserved by:
- L2 normalization before similarity computation
- Cosine similarity (angle-based, not magnitude-based)
- Field vector renormalization after projection

### 3. Statistical Invariant
**Strengths represent accumulated evidence**

Preserved by:
- EMA updates: s_i integrates match history
- Exponential decay: Old evidence fades smoothly
- Initialization: s_init provides prior

### 4. Temporal Invariant
**Evolution is smooth and bounded**

Preserved by:
- EMA with α < 1: Bounded update rate
- Exponential decay: Strengths never explode
- Soft gating: No discontinuous jumps (except burn-in)

### 5. Thermodynamic Invariant
**θ acts as temperature controlling influence**

Preserved by:
- Burn-in: "Cold" system (θ = 0) until evidence accumulates
- Soft scaling: "Warm up" as ‖S‖ increases
- Saturation: "Hot" system (θ = α_max) when confident

### 6. Energy Invariant
**System minimizes memory energy E**

Preserved by:
- Gradient flow: dM_A/dt = -∇E
- Attractors pulled toward frequent queries
- Unused attractors decay away (free energy minimization)

---

## Phase Transitions In F

The system undergoes phase transitions when invariants are violated:

### Collapse: rank(M_A) → 0

**Condition**: All attractors converge to same point

**Invariant violated**: Topological (π becomes degenerate)

**Signature**:
```
∀i,j : â_i · â_j → 1  (all attractors identical)
π_M_A(q̂) → constant   (projection loses information)
```

**Detection**: Manifold volume → 0 (SVD singular values → 0)

### Consolidation: ∇κ → ∞

**Condition**: Similarity kernel too permissive

**Invariant violated**: Statistical (all queries match all attractors)

**Signature**:
```
∀q̂,i : κ(q̂, â_i) → constant  (kernel saturated)
π_M_A becomes uniform average  (no selectivity)
```

**Detection**: Basin overlap > 3 (queries match many attractors)

### Runaway: ‖S‖ → ∞

**Condition**: Strength accumulation without bound

**Invariant violated**: Thermodynamic (infinite temperature)

**Signature**:
```
s_i → ∞ for all i      (all attractors maximally strong)
θ(‖S‖) → α_max always  (gating saturated)
```

**Detection**: Strength entropy → 1 (uniform distribution)

---

## Connection to Phase 2 Failure

**Phase 2** injected individual attractors (winner-take-all):
```
F_phase2(q; M_A) = θ · a_winner

where: winner = argmax_i κ(q̂, â_i)
```

**Why this failed**:
- Discontinuous at boundaries (∇F undefined when winner changes)
- No statistical averaging (single attractor can be outlier)
- Topologically non-smooth (discrete selection)
- Vulnerable to single corrupted attractor

**V1 fixes this** with soft projection:
```
F_v1(q; M_A) = θ · π_M_A(q̂)  (weighted blend, always smooth)
```

**Result**:
- Continuous everywhere (∇F well-defined)
- Statistical robustness (outliers averaged out)
- Topologically smooth (no discrete jumps)
- Resilient to noise (requires many attractors to fail)

---

## Implementation Notes

### Practical Form

In code, F is computed as:

```python
# 1. Normalize
q_hat = F.normalize(query, dim=-1)
a_hat = F.normalize(attractors, dim=-1)

# 2. Compute kernel (exponential of cosine similarity)
similarities = torch.mm(q_hat, a_hat.T)
kernel = torch.exp(beta * similarities)  # κ(q̂, â_i)

# 3. Weight by strengths
weights = kernel * strengths  # s_i · κ(q̂, â_i)

# 4. Soft projection (weighted mean)
field = (weights.unsqueeze(-1) * a_hat).sum(dim=1) / weights.sum(dim=1, keepdim=True)
field_unit = F.normalize(field, dim=-1)  # π_M_A(q̂)

# 5. Gating
total_strength = strengths.sum()
if total_strength < burn_in_threshold:
    influence = 0.0  # Burn-in
else:
    influence = alpha_max * torch.tanh(total_strength / tau)  # θ(‖S‖)

# 6. Final output
F_out = influence * field_unit  # F(q; M_A)
```

### Hyperparameter Mapping

| Symbol | Code Variable | Typical Value |
|--------|---------------|---------------|
| β | temperature (implicit in kernel) | ~1.0 |
| τ | strength_scale | 100.0 |
| τ_burn | burn_in_threshold | 50.0 |
| α_max | max_influence_scale | 0.05-0.5 (model-dependent) |
| α | alpha_update (EMA) | 0.05 |
| λ | lambda_decay | 0.995 |
| ε | strength_min | 1e-3 |

---

## Theoretical Implications

### DRAI as Riemannian Geometry

F can be viewed as a Riemannian metric tensor on the latent manifold:

```
g_ij(q) = ∂F/∂q_i · ∂F/∂q_j

Geodesics: Flow lines of gradient ∇F
Curvature: How fast attractors pull nearby queries
```

**Interpretation**: DRAI adds **memory-induced curvature** to the flat latent space.

### Connection to Hopfield Networks

Classical Hopfield: discrete attractors, sharp basins
```
H(x) = sgn(∑_i w_i · x_i)  (discrete, binary)
```

DRAI V1: continuous attractors, soft basins
```
F(q) = ∑_i s_i · κ(q, a_i) / Z  (continuous, probabilistic)
```

**DRAI is a soft, differentiable Hopfield network in continuous latent space.**

### Relation to Kernel Methods

F is a **kernel density estimator** with learned centers:
```
π_M_A(q̂) ≈ KDE(q̂; {â_i}, bandwidth=1/β)
```

**Difference**: Attractors evolve (not fixed samples), weights = strengths (not uniform).

---

## Why This Formulation Matters

1. **Unifies V1 design**: All 7 steps flow from preserving F's invariants

2. **Explains stability**: Scale-invariance means same F works across models

3. **Predicts failures**: Phase transitions = invariant violations

4. **Guides diagnostics**: Measure deviation from invariants (manifold volume, etc.)

5. **Suggests interventions**: Restore violated invariants to fix pathology

6. **Enables theory**: F is mathematically analyzable (convergence proofs, stability bounds)

---

## Future Directions

### Learned Kernel

Replace fixed κ with learned similarity:
```
κ_learned(q̂, â_i) = exp(β · φ(q̂)^T W φ(â_i))

where φ : ℝ^d → ℝ^k learned projection
```

**Benefit**: Adapt similarity metric to model's geometry

### Hierarchical Manifolds

Multiple scales of attractors:
```
F(q) = θ_1 · π_M1(q̂) + θ_2 · π_M2(π_M1(q̂)) + ...

M1: token-level attractors
M2: phrase-level attractors
M3: document-level attractors
```

**Benefit**: Multi-scale memory (current DRAI is single-scale)

### Negative Resonance

Add repulsion to field:
```
F(q) = θ_attract · π_M_A(q̂) - θ_repel · π_M_R(q̂)

M_A: positive attractors (remember)
M_R: negative attractors (forget)
```

**Benefit**: Push away from unwanted patterns (adversarial robustness)

### Metacognitive Gating

Let model control θ:
```
θ(q, M_A) = MLP(q, ‖S‖, uncertainty)

Model learns when to trust memory vs when to ignore it
```

**Benefit**: Task-adaptive memory usage

---

## Conclusion

**F(q; M_A) = θ(‖S‖) · π_M_A(q̂)** is the invariant equation of DRAI V1.

It expresses attractor-based memory as:
- A **topological morphism** (soft projection π)
- With **thermodynamic gating** (burn-in + scaling θ)
- On an **evolving manifold** (EMA flow dM_A/dt)
- Using a **smooth kernel** (exponential similarity κ)

**All design choices in V1 exist to preserve the invariants of this equation.**

The elegance is that F is **covariant**: it transforms naturally under model changes, which is why identical hyperparameters (except α_max) work from 70M to 2.8B parameters.

**This is the mathematics of DRAI.**

---

**Status**: Core invariant identified and documented
**Next**: Use F to prove convergence bounds and derive optimal hyperparameters analytically
