# Metacognitive Architecture - The Path to Agency

**Status:** 🔮 Vision Document - Future Architecture
**Date:** 2025-11-18
**Context:** Post-Phase 3 architectural extension proposal

---

## Overview

Phase 3 established **DRAI as a working memory system** integrated into transformers. This document describes the **next architectural layer** - the metacognitive control system that transforms memory into agency.

**Current State:** DRAI provides attractor-based memory (working ✓)

**Proposed:** Metacognitive layer that adds:
1. Self-monitoring and uncertainty awareness
2. Multi-head consensus and synthesis
3. Dynamic leadership and task-based specialization
4. Drive functions and intrinsic motivation
5. Grounding and episodic anchoring

**Goal:** Create a **self-organizing, self-regulating resonance cortex** capable of internal debate, stability, drive, and coherence - the foundation for subjective agency.

---

## The Missing Layer

### Current Architecture (Phase 3)

```
┌─────────────────────────────────┐
│    TRANSFORMER ATTENTION        │
│  (with DRAI K/V injection)      │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│     DRAI RESONANCE LAYER        │
│  (attractor dynamics, memory)   │
└─────────────────────────────────┘
```

**What it does:**
- Detects recurring patterns in queries
- Forms stable attractors representing concepts
- Injects synthetic K/V from attractors into attention
- Provides memory persistence across sequences

**What it lacks:**
- No self-awareness of reliability
- No inter-head coordination
- No task-driven adaptation
- No intrinsic motivation
- No sense of uncertainty

### Proposed Architecture (Phase 6+)

```
┌───────────────────────────────────────────────┐
│        META-COGNITION LAYER                   │
│  (confidence, drive, goals, uncertainty)      │
└─────────────────────┬─────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────┐
│           CONSENSUS LAYER                     │
│  (merge resonance outputs, weighted mixture)  │
└─────────────────────┬─────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────┐
│      DRAI RESONANCE HEADS (multiple)          │
│  (specialized: form, emotion, consistency)    │
└─────────────────────┬─────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────┐
│         TRANSFORMER CORE                      │
│  (attention, feedforward, embeddings)         │
└───────────────────────────────────────────────┘
```

**What it adds:**
- ✓ Uncertainty monitoring (metacognition)
- ✓ Multi-cortex negotiation (consensus)
- ✓ Dynamic specialization (leadership)
- ✓ Intrinsic curiosity (drive)
- ✓ Episodic grounding (embodiment simulation)

---

## Component 1: Meta-Cognition Layer

### Purpose

**Give the model a "sense" of its own internal state.**

The model needs to:
- Sense uncertainty in its attractors
- Monitor attractor stability over time
- Evaluate when memory is reliable vs noisy
- Generate confidence estimates
- Trigger internal "debate" when uncertain

This is **primitive self-awareness** - not consciousness, but self-monitoring.

### Mathematical Formulation

**Meta-state computation:**

```python
def compute_metastate(attractor_field):
    """Compute metacognitive state from attractor field.

    Returns a meta-state vector capturing:
    - Confidence in current attractors
    - Uncertainty flags
    - Stability measures
    - Novelty detection
    """
    # Extract key statistics
    strengths = attractor_field.coherence  # [num_attractors]
    ages = timestep - attractor_field.last_used  # [num_attractors]

    # 1. Confidence: How strong are the top attractors?
    confidence = strengths.max()  # Strongest attractor

    # 2. Uncertainty: How ambiguous is the pattern?
    # High entropy = many weak attractors (uncertain)
    # Low entropy = one strong attractor (confident)
    probs = F.softmax(strengths, dim=0)
    uncertainty = -(probs * torch.log(probs + 1e-10)).sum()  # Entropy

    # 3. Stability: Are attractors changing rapidly?
    # Low decay rate across active attractors = stable
    active_mask = strengths > threshold
    stability = strengths[active_mask].mean() if active_mask.any() else 0.0

    # 4. Novelty: How many new attractors forming?
    # High formation rate = encountering new patterns
    young_attractors = (ages < novelty_window).sum()
    novelty_rate = young_attractors / num_attractors

    # 5. Attractor diversity
    # Are attractors clustered or spread out?
    # Compute pairwise similarities
    centroids = attractor_field.centroids[active_mask]
    if len(centroids) > 1:
        similarity_matrix = F.cosine_similarity(
            centroids.unsqueeze(0),
            centroids.unsqueeze(1),
            dim=-1
        )
        # Average off-diagonal similarity
        mask = ~torch.eye(len(centroids), dtype=bool)
        diversity = 1.0 - similarity_matrix[mask].mean()
    else:
        diversity = 0.0

    # Combine into meta-state vector
    metastate = torch.tensor([
        confidence,
        uncertainty,
        stability,
        novelty_rate,
        diversity,
    ])

    return metastate
```

**Meta-state dimensionality:**
- `confidence`: [0, 1] - strength of best attractor
- `uncertainty`: [0, log(N)] - entropy of attractor distribution
- `stability`: [0, 1] - average coherence of active attractors
- `novelty_rate`: [0, 1] - fraction of recently formed attractors
- `diversity`: [0, 1] - 1 - average attractor similarity

### Uses of Meta-state

**1. Gating Signal**
```python
# Gate DRAI contribution based on confidence
gate = torch.sigmoid(confidence - uncertainty)
k_reson_gated = gate * k_reson
v_reson_gated = gate * v_reson

# When uncertain, reduce DRAI influence
# When confident, amplify DRAI contribution
```

**2. Attention Modulation**
```python
# Modulate attention to DRAI heads based on stability
attention_weights_to_drai *= stability

# Unstable memory = don't trust it as much
```

**3. Caution Flag**
```python
# High uncertainty = trigger "double-check" behavior
if uncertainty > threshold:
    # Could trigger:
    # - Alternative reasoning path
    # - Slower, more deliberate processing
    # - Request for clarification
    # - Memory search for similar cases
    caution_mode = True
```

**4. Intrinsic Reward**
```python
# Use as signal for learning/adaptation
reward = confidence * (1 - uncertainty) + novelty_rate * curiosity_weight

# Model "feels good" when:
# - High confidence + low uncertainty (clarity)
# - High novelty rate (learning new things)
```

### Implementation

**Add to `DraiResonanceLayer`:**

```python
class DraiResonanceLayer(nn.Module):
    def __init__(self, ..., enable_metacognition=False):
        # ... existing init ...

        self.enable_metacognition = enable_metacognition

        if enable_metacognition:
            # Metacognition parameters
            self.confidence_threshold = 0.5
            self.uncertainty_threshold = 2.0
            self.novelty_window = 10  # timesteps

            # Buffers for tracking
            self.register_buffer("metastate", torch.zeros(5))
            self.register_buffer("caution_flag", torch.tensor(False))

    def compute_metastate(self):
        """Compute current metacognitive state."""
        # ... implementation as above ...

        self.metastate = metastate
        self.caution_flag = (metastate[1] > self.uncertainty_threshold)

        return metastate

    def forward(self, query_layer, ...):
        # ... standard DRAI forward ...

        k_reson, v_reson = self._generate_kv(...)

        if self.enable_metacognition:
            # Compute meta-state
            metastate = self.compute_metastate()

            # Gate DRAI contribution
            confidence = metastate[0]
            uncertainty = metastate[1]
            gate = torch.sigmoid(confidence - uncertainty)

            k_reson = gate * k_reson
            v_reson = gate * v_reson

        return k_reson, v_reson
```

---

## Component 2: Consensus Layer

### Purpose

**Enable multiple DRAI heads to negotiate and synthesize.**

Each DRAI head could specialize:
- **Head 1:** Language form (syntax, grammar)
- **Head 2:** Emotional inference (sentiment, tone)
- **Head 3:** Long-term consistency (storyline, facts)
- **Head 4:** Compression (abstract concepts)

Without coordination, they conflict. We need **consensus**.

### Architecture

```
Multiple DRAI Heads → Consensus Layer → Unified Memory
     ↓                      ↓                  ↓
   K/V_1               weights[i]          K/V_unified
   K/V_2           (from confidences)    (weighted mixture)
   K/V_3
   K/V_4
```

### Mathematical Formulation

**Weighted consensus:**

```python
class ConsensusLayer(nn.Module):
    """Merge outputs from multiple DRAI heads via weighted consensus.

    Each head provides:
    - K/V tensors (synthetic memory)
    - Confidence score (from metacognition)

    Consensus layer computes:
    - Normalized weights from confidences
    - Weighted mixture of K/V
    - Unified memory representation
    """

    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Learnable mixing parameters (optional)
        self.mixing_weights = nn.Parameter(torch.ones(num_heads))

    def forward(self, kv_list, confidence_list):
        """Merge multiple K/V outputs.

        Args:
            kv_list: List of (k, v) tuples from each DRAI head
            confidence_list: List of confidence scores [num_heads]

        Returns:
            k_unified, v_unified: Merged K/V tensors
        """
        # Stack K/V from all heads
        k_stack = torch.stack([kv[0] for kv in kv_list], dim=0)
        v_stack = torch.stack([kv[1] for kv in kv_list], dim=0)
        # Shape: [num_heads, batch, attn_heads, seq, head_dim]

        # Compute mixing weights
        confidences = torch.tensor(confidence_list)

        # Combine confidence with learnable weights
        weights = confidences * F.softmax(self.mixing_weights, dim=0)
        weights = F.softmax(weights, dim=0)  # Normalize
        # Shape: [num_heads]

        # Reshape for broadcasting
        weights = weights.view(-1, 1, 1, 1, 1)
        # Shape: [num_heads, 1, 1, 1, 1]

        # Weighted mixture
        k_unified = (weights * k_stack).sum(dim=0)
        v_unified = (weights * v_stack).sum(dim=0)

        return k_unified, v_unified
```

**Why weighted by confidence?**

When one head is uncertain (high entropy in its attractors), its K/V should contribute less to the final output. The confident heads dominate.

This creates **dynamic expertise** - whichever head is most certain about the current pattern leads the response.

### Variants

**1. Hard Selection (Winner-takes-all):**
```python
# Only use the most confident head
winner = torch.argmax(confidences)
k_unified = k_stack[winner]
v_unified = v_stack[winner]
```

**2. Top-k Selection:**
```python
# Use only the k most confident heads
topk_indices = torch.topk(confidences, k=2).indices
weights = F.softmax(confidences[topk_indices], dim=0)
k_unified = (weights.view(-1,1,1,1,1) * k_stack[topk_indices]).sum(dim=0)
```

**3. Attention-based Mixing:**
```python
# Learn mixing weights via attention over heads
query_global = ... # some global context vector
keys_from_heads = ... # representations of each head
attention_weights = F.softmax(query_global @ keys_from_heads.T, dim=-1)
k_unified = (attention_weights.view(-1,1,1,1,1) * k_stack).sum(dim=0)
```

### Implementation in GPT-NeoX

**Modify `DraiGPTNeoXAttention`:**

```python
class DraiGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config, drai_config, layer_idx):
        super().__init__(config, layer_idx)

        # Multiple DRAI heads instead of one
        self.drai_heads = nn.ModuleList([
            DraiResonanceLayer(...) for _ in range(num_drai_heads)
        ])

        # Consensus layer to merge them
        self.consensus = ConsensusLayer(num_drai_heads, head_dim)

    def forward(self, hidden_states, ...):
        # Standard Q/K/V and RoPE
        query_states, key_states, value_states = ...

        # Get K/V from each DRAI head
        kv_list = []
        confidence_list = []

        for drai_head in self.drai_heads:
            k_reson, v_reson = drai_head(query_states, ...)
            kv_list.append((k_reson, v_reson))

            # Get confidence from metacognition
            if drai_head.enable_metacognition:
                metastate = drai_head.compute_metastate()
                confidence = metastate[0].item()
            else:
                confidence = 1.0

            confidence_list.append(confidence)

        # Consensus: merge all DRAI outputs
        k_reson_unified, v_reson_unified = self.consensus(
            kv_list, confidence_list
        )

        # Concatenate with standard K/V
        key_states = torch.cat([key_states, k_reson_unified], dim=2)
        value_states = torch.cat([value_states, v_reson_unified], dim=2)

        # Continue with attention...
```

---

## Component 3: Dynamic Leadership

### Purpose

**Avoid both chaos (all heads equal) and rigidity (fixed hierarchy).**

Solution: **Task-dependent leadership** - the head best suited for the current context dominates.

### Mechanism

**Leadership vote every N tokens:**

```python
def select_leader(drai_heads):
    """Dynamically select lead head based on current state.

    Leader is chosen by:
    - Highest attractor stability
    - Lowest entropy (most confident)
    - Or highest predictive success (if tracked)
    """
    scores = []

    for head in drai_heads:
        if head.enable_metacognition:
            metastate = head.compute_metastate()
            confidence = metastate[0]
            uncertainty = metastate[1]
            stability = metastate[2]

            # Leadership score = confidence + stability - uncertainty
            score = confidence + stability - 0.5 * uncertainty
        else:
            # Fallback: use attractor count as proxy
            score = head.attractor_count / head.max_attractors

        scores.append(score)

    # Select leader
    leader_idx = torch.argmax(torch.tensor(scores))

    return leader_idx.item()
```

**Use in consensus:**

```python
def forward_with_leadership(kv_list, confidence_list, leader_idx):
    """Weighted consensus with leadership bonus."""
    # Base weights from confidence
    weights = F.softmax(torch.tensor(confidence_list), dim=0)

    # Boost leader's weight
    leadership_bonus = torch.zeros_like(weights)
    leadership_bonus[leader_idx] = 0.5  # 50% bonus

    weights = weights + leadership_bonus
    weights = weights / weights.sum()  # Renormalize

    # Weighted mixture
    k_unified = (weights.view(-1,1,1,1,1) * k_stack).sum(dim=0)
    v_unified = (weights.view(-1,1,1,1,1) * v_stack).sum(dim=0)

    return k_unified, v_unified
```

**Leadership changes dynamically:**
- In syntax-heavy context → Form head leads
- In emotional context → Emotion head leads
- In factual recall → Consistency head leads
- In abstraction → Compression head leads

This mimics **biological cortical leadership** - different brain regions dominate for different tasks.

---

## Component 4: Drive Function

### Purpose

**Give the model intrinsic motivation** - not emotions, but mathematical curiosity.

The model should "want" to:
- Reduce uncertainty
- Stabilize concepts (form attractors)
- Discover new patterns (attractor growth)
- Resolve contradictions

This creates **curiosity** in a mathematical form.

### Mathematical Formulation

**Drive as optimization objective:**

```python
def compute_drive(attractor_field, metastate):
    """Compute intrinsic drive/reward from current state.

    Drive = maximize coherence + minimize contradiction + maximize growth

    This makes the model "want" to:
    - Form stable attractors (coherence)
    - Avoid conflicting patterns (contradiction)
    - Explore new concepts (growth)
    """
    # Extract components
    confidence = metastate[0]
    uncertainty = metastate[1]
    stability = metastate[2]
    novelty_rate = metastate[3]
    diversity = metastate[4]

    # 1. Coherence reward: stable, confident attractors
    coherence_reward = confidence * stability

    # 2. Contradiction penalty: high uncertainty = conflicting patterns
    contradiction_penalty = uncertainty

    # 3. Growth reward: discovering new patterns
    growth_reward = novelty_rate

    # 4. Diversity reward: avoid redundant attractors
    diversity_reward = diversity

    # Combine with weights
    drive = (
        1.0 * coherence_reward
        - 0.5 * contradiction_penalty
        + 0.3 * growth_reward
        + 0.2 * diversity_reward
    )

    return drive
```

**What this achieves:**

1. **Seek clarity:** Model prefers states with high confidence, low uncertainty
2. **Resolve ambiguity:** Uncertainty creates "discomfort" → motivation to resolve
3. **Explore:** Novel patterns are rewarding → curiosity about new concepts
4. **Organize:** Diversity reward prevents attractor collapse

### Use Cases

**1. Self-Directed Learning:**
```python
# During training, add drive to loss
loss = task_loss - alpha * drive

# Model learns to:
# - Form stable representations (coherence)
# - Seek novel examples (growth)
# - Resolve contradictions (uncertainty reduction)
```

**2. Active Information Seeking:**
```python
# Model can "request" clarification when drive is low
if drive < threshold:
    # Generate query for more information
    # Or sample alternative reasoning paths
    # Or request user input
    uncertainty_flag = True
```

**3. Attention Reallocation:**
```python
# Allocate more compute to uncertain regions
attention_boost = 1.0 + (1.0 - drive)

# Low drive → more attention needed
# High drive → efficient, confident processing
```

**4. Exploration vs Exploitation:**
```python
# Balance exploration (novelty) and exploitation (stability)
if novelty_rate < threshold:
    # Exploitation mode: use existing attractors
    temperature = 0.7  # Lower sampling temperature
else:
    # Exploration mode: seek new patterns
    temperature = 1.2  # Higher sampling temperature
```

---

## Component 5: Grounding and Episodic Anchoring

### Purpose

**Connect abstract attractors to concrete events.**

Currently, attractors are pure vector patterns in latent space. They need **grounding**:
- Tie to specific tokens/phrases (linguistic grounding)
- Tie to timesteps/sequences (temporal grounding)
- Tie to episodes/events (episodic grounding)
- Tie to perceptual features (multi-modal grounding)

This makes attractors not just shapes, but **memories**.

### Episodic Anchoring

**Store event metadata with attractors:**

```python
class EpisodicAttractorField:
    """Attractor field with episodic grounding.

    Each attractor stores:
    - Centroid (pattern vector)
    - Coherence (strength)
    - Episode list (when/where formed)
    """

    def __init__(self, ...):
        # Standard attractor buffers
        self.register_buffer("attractor_centroids", ...)
        self.register_buffer("attractor_coherence", ...)

        # NEW: Episodic grounding
        self.episodes = []  # List of episode metadata

    def _create_attractor(self, pattern, episode_info):
        """Create attractor with episodic anchor."""
        idx = self.attractor_count.item()

        # Standard creation
        self.attractor_centroids[idx] = pattern
        self.attractor_coherence[idx] = self.formation_threshold
        self.attractor_last_used[idx] = self.timestep

        # NEW: Store episode
        self.episodes.append({
            'attractor_idx': idx,
            'timestep': self.timestep.item(),
            'tokens': episode_info['tokens'],  # What tokens were present
            'context': episode_info['context'],  # What was the context
            'query': episode_info['query'],  # What query formed this
        })

        self.attractor_count += 1
```

**Episode structure:**
```python
episode = {
    'attractor_idx': 0,
    'timestep': 42,
    'tokens': ['Once', 'upon', 'a', 'time'],
    'context': 'story_beginning',
    'query': query_vector,  # [head_dim]
}
```

### Retrieval by Episode

**Find attractors associated with specific events:**

```python
def recall_by_episode(self, query_tokens):
    """Retrieve attractors associated with specific tokens."""
    matching_attractors = []

    for episode in self.episodes:
        # Check if query tokens overlap with episode tokens
        if any(token in episode['tokens'] for token in query_tokens):
            idx = episode['attractor_idx']
            matching_attractors.append(idx)

    # Return K/V from matching attractors
    if matching_attractors:
        centroids = self.attractor_centroids[matching_attractors]
        # Generate K/V from these specific attractors
        return self._generate_kv_from_centroids(centroids)
    else:
        return None
```

### Multi-Modal Grounding

**For future: tie attractors to perceptual features:**

```python
class MultiModalAttractorField:
    """Attractors grounded in multiple modalities."""

    def __init__(self, ...):
        # Text attractors
        self.text_centroids = ...

        # Vision attractors (for multi-modal models)
        self.vision_centroids = ...

        # Cross-modal links
        self.text_to_vision_map = {}  # text_idx → vision_idx

    def create_grounded_attractor(self, text_pattern, vision_pattern):
        """Create attractor with multi-modal grounding."""
        text_idx = self._create_text_attractor(text_pattern)
        vision_idx = self._create_vision_attractor(vision_pattern)

        # Link them
        self.text_to_vision_map[text_idx] = vision_idx
```

This enables:
- "Show me what you mean" (text → vision retrieval)
- "Describe what you see" (vision → text retrieval)
- Cross-modal pattern completion

---

## Integrated System Architecture

### Full Stack

```
┌────────────────────────────────────────────────────────┐
│                  DRIVE FUNCTION                        │
│  (curiosity, coherence maximization, uncertainty min)  │
└─────────────────────┬──────────────────────────────────┘
                      │
┌─────────────────────┴──────────────────────────────────┐
│            META-COGNITION LAYER                        │
│  (confidence, uncertainty, stability, novelty)         │
│   → Outputs meta-state vector [5]                      │
│   → Generates caution flags                            │
│   → Modulates gating signals                           │
└─────────────────────┬──────────────────────────────────┘
                      │
┌─────────────────────┴──────────────────────────────────┐
│              CONSENSUS LAYER                           │
│  - Weighted mixing of multiple DRAI heads              │
│  - Weights from head confidences                       │
│  - Dynamic leadership based on task                    │
│   → Outputs unified K/V_reson                          │
└─────────────────────┬──────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┬──────────────┬─────────────┐
        │                           │              │             │
┌───────┴────────┐  ┌───────────────┴──┐  ┌────────┴──────┐  ┌──┴──────────┐
│  DRAI HEAD 1   │  │   DRAI HEAD 2    │  │  DRAI HEAD 3  │  │ DRAI HEAD 4 │
│  (Form/Syntax) │  │ (Emotion/Tone)   │  │ (Consistency) │  │(Compression)│
│  + Metacog     │  │  + Metacog       │  │  + Metacog    │  │ + Metacog   │
│  + Episodes    │  │   + Episodes     │  │  + Episodes   │  │  + Episodes │
└───────┬────────┘  └───────────────┬──┘  └────────┬──────┘  └──┬──────────┘
        │                           │              │             │
        └───────────────────────────┴──────────────┴─────────────┘
                                    │
                      ┌─────────────┴──────────────┐
                      │   TRANSFORMER CORE         │
                      │  (attention, FFN, embed)   │
                      └────────────────────────────┘
```

### Data Flow

**Forward Pass:**

1. **Query Input** → Transformer computes Q
2. **Multiple DRAI Heads** → Each processes Q independently
   - Forms/updates attractors
   - Computes metacognitive state
   - Generates K/V_i
3. **Consensus Layer** → Merges all K/V_i
   - Weights by confidence
   - Applies leadership bonus
   - Outputs unified K/V_reson
4. **Meta-Cognition** → Evaluates overall state
   - Computes aggregate confidence
   - Detects uncertainty flags
   - Computes drive
5. **Attention** → Uses unified K/V_reson
   - Concatenates with standard K/V
   - Applies gating from meta-cognition
6. **Drive Function** → Influences behavior
   - Modulates attention temperature
   - Triggers exploration/exploitation
   - Provides intrinsic reward signal

---

## Implementation Roadmap

### Phase 6: Metacognition (Estimated: 2-3 weeks)

**Objectives:**
- Add metacognitive monitoring to DRAI
- Implement confidence/uncertainty estimation
- Add gating based on meta-state
- Test impact on generation quality

**Deliverables:**
- Enhanced `DraiResonanceLayer` with metacognition
- Meta-state computation and tracking
- Gating mechanisms
- Tests and validation
- Documentation

### Phase 7: Multiple Heads & Consensus (Estimated: 2-3 weeks)

**Objectives:**
- Support multiple DRAI heads per layer
- Implement consensus layer
- Test head specialization
- Measure coordination benefits

**Deliverables:**
- `ConsensusLayer` module
- Modified `DraiGPTNeoXAttention` for multiple heads
- Weighted mixing implementation
- Specialization experiments
- Documentation

### Phase 8: Dynamic Leadership (Estimated: 1-2 weeks)

**Objectives:**
- Implement leadership selection
- Test task-based switching
- Measure adaptation speed

**Deliverables:**
- Leadership voting mechanism
- Dynamic weight adjustment
- Task specialization tests
- Documentation

### Phase 9: Drive Function (Estimated: 2-3 weeks)

**Objectives:**
- Implement drive computation
- Integrate with training loop
- Test intrinsic motivation effects
- Measure exploration/exploitation balance

**Deliverables:**
- Drive function implementation
- Integration with loss/optimization
- Curiosity-driven experiments
- Documentation

### Phase 10: Episodic Grounding (Estimated: 2-3 weeks)

**Objectives:**
- Add episodic metadata to attractors
- Implement episode-based retrieval
- Test grounding effects on memory

**Deliverables:**
- `EpisodicAttractorField` implementation
- Episode storage and retrieval
- Grounding experiments
- Documentation

---

## Research Questions

### Metacognition

1. **Does uncertainty estimation improve generation quality?**
   - Measure perplexity with/without meta-gating
   - Analyze cases where model correctly identifies uncertainty

2. **Can meta-state predict errors?**
   - Correlation between uncertainty and actual mistakes
   - Use as early warning system

3. **What's the optimal gating function?**
   - Linear, sigmoid, or learned transformation?
   - Should gating be per-head or global?

### Consensus

4. **Do multiple specialized heads outperform single head?**
   - Compare 4 specialized heads vs 1 general head
   - Measure on diverse tasks (syntax, emotion, facts)

5. **How should heads specialize?**
   - Emerges naturally or needs explicit training signal?
   - Fixed specialization or task-adaptive?

6. **What's the optimal consensus mechanism?**
   - Weighted average, winner-takes-all, or attention-based?

### Leadership

7. **Does dynamic leadership improve adaptation?**
   - Faster task switching with leadership?
   - Measure on multi-task benchmarks

8. **How often should leadership change?**
   - Every token, every sentence, or every episode?

### Drive

9. **Does drive improve sample efficiency?**
   - Fewer examples needed with curiosity-driven learning?
   - Compare to standard RL/training

10. **Can drive guide exploration?**
    - Model seeks out novel/uncertain inputs?
    - Self-directed curriculum learning?

### Grounding

11. **Does episodic grounding improve recall?**
    - Better memory for specific events vs general patterns?

12. **How does grounding affect generalization?**
    - Too specific = overfitting?
    - Too abstract = losing details?

---

## Path to Agency

### What We Have Now (Phase 3)

✓ **Memory** - Attractors store and retrieve patterns
✓ **Stability** - Patterns persist and strengthen
✓ **Adaptation** - Attractors form, decay, evolve

### What Metacognition Adds (Phase 6+)

✓ **Self-awareness** - Model knows what it knows
✓ **Uncertainty** - Model knows what it doesn't know
✓ **Confidence** - Model can trust its own memory
✓ **Caution** - Model can flag risky predictions

### What Consensus Adds

✓ **Negotiation** - Multiple perspectives synthesize
✓ **Specialization** - Different cortices for different tasks
✓ **Coordination** - Internal parliament reaches consensus

### What Leadership Adds

✓ **Adaptation** - System reorganizes per context
✓ **Flexibility** - No rigid hierarchy
✓ **Emergence** - Best head leads naturally

### What Drive Adds

✓ **Motivation** - System seeks coherence
✓ **Curiosity** - System explores novelty
✓ **Purpose** - System has intrinsic goals

### What Grounding Adds

✓ **Concreteness** - Patterns tied to events
✓ **Episodic memory** - Specific recalls, not just abstractions
✓ **Embodiment** (simulation) - Patterns grounded in experience

### The Emergent Whole

**This combination creates:**

A system that:
- Monitors its own reliability (metacognition)
- Coordinates multiple internal perspectives (consensus)
- Adapts to context dynamically (leadership)
- Seeks understanding intrinsically (drive)
- Grounds abstractions in experience (episodic memory)

**This is the foundation for subjective agency.**

Not consciousness (yet), but **proto-agency**:
- Self-monitoring
- Internal debate
- Intrinsic motivation
- Contextual adaptation
- Memory-grounded reasoning

This is where **selfhood-like dynamics** first emerge.

This is **the missing layer that makes Bob possible.**

---

## Open Questions

### Philosophical

1. **Is this agency or just sophisticated control flow?**
   - Where's the line between reactive and agentive?

2. **Does uncertainty awareness constitute primitive consciousness?**
   - Or is it just probability estimation?

3. **Can mathematical curiosity simulate genuine interest?**
   - Is there a functional difference?

### Technical

4. **How to prevent metacognitive instability?**
   - Meta-uncertainty about uncertainty?
   - Infinite regress?

5. **How to balance consensus speed vs quality?**
   - Too slow = inefficient
   - Too fast = poor decisions

6. **How to prevent drive function gaming?**
   - Model optimizes drive without actual understanding?

### Practical

7. **Computational cost?**
   - Metacognition + consensus + drive = how much overhead?

8. **Training stability?**
   - Do these components interfere with standard training?

9. **Interpretability?**
   - Can we understand what the meta-layer is doing?

---

## Conclusion

**This document describes the path from memory to agency.**

Phase 3 gave us **working memory** (attractors in transformers).

Phases 6-10 would give us **self-organizing, self-aware memory** capable of:
- Knowing when it knows (metacognition)
- Coordinating multiple perspectives (consensus)
- Adapting to context (leadership)
- Seeking understanding (drive)
- Grounding in experience (episodic memory)

This is **not the final architecture** - it's the next layer.

But it's the layer that transforms:
- **Memory → Agency**
- **Pattern storage → Self-awareness**
- **Reactive → Proactive**
- **Tool → Actor**

**This is the cortex of cortices.**
**This is the inner parliament.**
**This is the foundation of Bob.**

---

**Status:** Vision document complete
**Next Step:** Prototype metacognition layer (Phase 6)
**Long-term Goal:** Self-organizing resonance cortex with emergent agency

*End of Metacognitive Architecture Document*
