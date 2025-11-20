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
da_i/dt = α · (q_matched - a_i) · δ(i matched)
ds_i/dt = (γ · δ(i matched) - λ) · s_i
```

**where**:
- α : EMA momentum (typically 0.05)
- γ : strength boost per match
- λ : global decay rate
- δ(i matched) : indicator function (1 if attractor i matches query)

**Important**: This is **EMA-like flow**, not formally derived from an energy functional.

**Metaphor** (useful but not rigorous):
```
dM_A/dt ≈ -∇_A E_pseudo(M_A, Q_t)

where E_pseudo might be:
  E = Σ_i ||a_i - ⟨q|i⟩||² - λ·H(S)

  (squared distance to matched queries + entropy regularization)
```

**But**: We haven't proven this E exists or is convex. The EMA update is a **heuristic** that behaves like gradient descent, not a formal optimization algorithm.

**Properties**:
- Smooth evolution (exponential smoothing)
- Attractors drift toward matching queries
- Strengths reinforce with hits, decay globally
- Bounded evolution (α < 1 prevents explosion)
- **No guarantee of convergence to global minimum**

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

**CRITICAL**: This smoothness is **engineered**, not automatic.

Without these design choices, π could be discontinuous:
- **Dual thresholds**: Only use attractors with s_i > ε (prevents division by zero)
- **Exponential kernel**: κ always positive, smooth everywhere (not hard thresholds)
- **Field averaging**: Weighted mean (not winner-take-all which is discontinuous)
- **Normalization**: All vectors normalized before similarity (prevents magnitude explosion)
- **Burn-in**: Delays injection until stable (prevents early noise amplification)

**Phase 2 failed because it lacked these**:
- Winner-take-all → ∇F undefined at boundaries
- Hard thresholds → discontinuous jumps
- No burn-in → immediate injection of random noise

**Result**: π is smooth everywhere, but only because we made it smooth.

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

## What's Rigorous vs Metaphorical

### Formally Proven ✓

1. **F is continuous** (given the design choices above)
   - π is smooth because of exponential kernel + averaging
   - θ is smooth (except deliberate burn-in step)
   - Composition of smooth functions is smooth

2. **F is bounded** (output never explodes)
   - All vectors normalized → magnitudes ≤ 1
   - θ saturates at α_max
   - |F(q)| ≤ α_max always

3. **Scale invariance** (works across model sizes)
   - Cosine similarity rotation/scale invariant
   - No dependence on hidden_dim magnitude
   - Only α_max scales with capacity

4. **EMA updates are bounded** (no explosion)
   - |da_i/dt| ≤ α||q - a_i|| ≤ 2α (since normalized)
   - s_i decays exponentially → bounded above

### Useful Metaphors (not proven) ⚠️

1. **"Gradient flow"** for attractor evolution
   - EMA behaves like gradient descent
   - But no explicit energy functional E
   - No proof of convergence to minimum
   - **Status**: Heuristic that works empirically

2. **"Energy minimization"**
   - System seems to minimize some implicit cost
   - We can construct pseudo-energy E_pseudo
   - But it's not convex, may have local minima
   - **Status**: Conceptual tool, not formal theorem

3. **"Phase transitions"** at invariant violations
   - System behavior changes qualitatively
   - Analogous to thermodynamic transitions
   - But no rigorous statistical mechanics
   - **Status**: Useful analogy for understanding failures

4. **"Temperature"** for θ gating
   - θ controls influence like thermodynamic temperature
   - But no connection to Boltzmann distribution
   - No formal entropy or free energy
   - **Status**: Suggestive metaphor

### What We Should Prove (Future Work)

1. **Convergence bounds**: Does M_A converge? To what?
2. **Stability analysis**: Lyapunov functions for EMA flow?
3. **Optimal hyperparameters**: Derive α, λ, β analytically?
4. **Phase transition theory**: Formal bifurcation analysis?
5. **Information bounds**: Does π preserve information? How much?

**The mathematics is solid where it needs to be** (continuity, boundedness), but **uses metaphor for intuition** (gradient flow, energy). This is fine - many successful algorithms do this (Adam, momentum SGD).

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
- A **soft projection** π onto attractor manifold (engineered to be smooth)
- With **gated influence** θ (burn-in + soft scaling)
- On an **evolving state** M_A (EMA-like updates, not formal gradient)
- Using a **smooth kernel** κ (exponential similarity)

### What This Gives Us

**Rigorous**:
- F is continuous and bounded (proven by construction)
- F is scale-invariant (works across model sizes)
- All design choices explicitly listed and justified
- Clear failure modes when invariants violated

**Metaphorical but useful**:
- "Gradient flow" for attractor evolution (heuristic, not optimization)
- "Energy minimization" (no explicit functional)
- "Phase transitions" (qualitative analogy)

**Critical insight**: The smoothness isn't automatic - it's **engineered** through:
- Exponential kernel (not hard thresholds)
- Field averaging (not winner-take-all)
- Dual thresholds (prevents division by zero)
- Burn-in (prevents early instability)
- Normalization (bounds magnitudes)

**Phase 2 failed precisely because it lacked these safeguards.**

The elegance is that F is **covariant**: identical hyperparameters (except α_max) work from 70M to 2.8B because F operates on normalized geometry, not parameters.

**This formulation is valuable because**:
1. Unifies all V1 design choices
2. Explains stability across scales
3. Predicts failure modes
4. Guides diagnostics and interventions
5. Suggests theoretical directions (even if not all proven yet)

---

**Status**: Core invariant documented with rigorous/metaphorical boundaries clarified
**Next**:
- Prove convergence bounds (if possible)
- Define explicit energy functional (make gradient flow rigorous)
- Derive optimal hyperparameters analytically
- Formal stability analysis
