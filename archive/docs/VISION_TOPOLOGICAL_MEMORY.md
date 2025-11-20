# Vision: Topological Memory - Trees, Groves, and Forests of Meaning

**Date:** 2025-11-18
**Status:** Vision document for future phases

## The Core Idea

> "I envisioned the DRAI vector DB as a continuously rearranging resonance DB where near tokens pushed and pulled, creating trees, groves and forests of meaning. Each token only affects those it resonates positively or negatively within a sphere of groves... so not every token has to look at every other token."

This is not just a vector database. This is a **living topological memory space** where:
- Memories organize themselves into **local neighborhoods**
- Updates propagate through **spheres of resonance**, not globally
- Structure emerges at multiple scales: **tokens → trees → groves → forests**
- Memories can **attract** (positive resonance) or **repel** (negative resonance)

---

## Contrast: Vector DB vs Resonance Field

### Standard Vector Database (RAG)

```
┌─────────────────────────────────────┐
│  Flat Vector Store                  │
│                                     │
│  v1  v2  v3  v4  v5  ... vN        │
│   ↓   ↓   ↓   ↓   ↓      ↓         │
│  Every query compares to EVERY     │
│  vector. O(N) lookups, no locality │
└─────────────────────────────────────┘
```

**Problems:**
- ❌ Global search: every token looks at everything
- ❌ No structure: flat list of embeddings
- ❌ No dynamics: vectors are static
- ❌ Expensive: O(N) comparisons per query

---

### DRAI Resonance Field (Vision)

```
        🌲 Forest of Meaning
       ╱ ╲
      ╱   ╲
     🌳   🌳 Groves
    ╱│╲   ╱│╲
   🌿🌿🌿 🌿🌿🌿 Trees
   ││ ││ ││ ││
   Attractors (leaves)

Each attractor only interacts with
nearby attractors in its sphere.

Positive resonance → attract, cluster
Negative resonance → repel, separate
```

**Advantages:**
- ✅ Local updates: O(neighbors) not O(all)
- ✅ Hierarchical structure: emergent organization
- ✅ Dynamic: continuously rearranging
- ✅ Efficient: sparse interactions
- ✅ Interpretable: can point to "groves of related concepts"

---

## Current Implementation vs Full Vision

### What We Have (Phase 1-4) ✓

**Continuous dynamics:**
```python
# Attractors update via EMA
attractor = (1 - α) * old_attractor + α * new_pattern
```

**Self-organizing:**
- Attractors form automatically when novel patterns appear
- Reinforce when similar patterns repeat
- Decay when unused

**Locality (partial):**
- Coherence threshold creates "sphere of influence"
- Only attractors above threshold are active
- But: all attractors still compared globally

**What's implemented:**
- ✅ Attractor formation and decay
- ✅ Cosine similarity matching
- ✅ EMA-based reinforcement
- ✅ Pruning of weak attractors

---

### What's Missing (Phase 6+)

**1. Hierarchical Structure (Trees → Groves → Forests)**

Current: Flat list of attractors
```
attractors = [a1, a2, a3, ..., a32]  # Flat array
```

Vision: Nested hierarchy
```
forest:
  grove_1:
    tree_1: [a1, a2, a3]     # "character" concepts
    tree_2: [a4, a5]         # "location" concepts
  grove_2:
    tree_3: [a6, a7, a8, a9] # "plot" concepts
```

**How to implement:**
- Cluster attractors by similarity → trees
- Cluster trees by semantic relation → groves
- Top-level: forests (major conceptual domains)
- Use hierarchical agglomerative clustering or Growing Neural Gas

---

**2. Negative Resonance (Repulsion)**

Current: Only positive reinforcement + decay
```python
if similarity > threshold:
    reinforce(attractor)  # Attract
else:
    pass  # Just ignore, no active repulsion
```

Vision: Push AND pull
```python
if similarity > positive_threshold:
    attract(attractor)  # Move toward each other
elif similarity < negative_threshold:
    repel(attractor)  # Push apart, anti-Hebbian
else:
    neutral()  # No interaction
```

**Why negative resonance?**
- Separate conflicting concepts ("hot" vs "cold")
- Maintain diversity (prevent everything collapsing to one attractor)
- Model inhibition (like inhibitory neurons)
- Create "anti-attractors" (repulsive regions in semantic space)

**Implementation:**
```python
def update_attractor(attractor, pattern, similarity):
    if similarity > 0.7:
        # Positive resonance: attract
        attractor += alpha * (pattern - attractor)
    elif similarity < -0.3:
        # Negative resonance: repel
        attractor -= beta * (pattern - attractor)
        # Pattern and attractor push apart
```

---

**3. Local Neighborhoods (Not Every Token Looks at Everything)**

Current: Global search
```python
# Compare query to ALL attractors
similarities = cosine_similarity(query, all_attractors)  # O(N)
```

Vision: Local search within spheres
```python
# Only compare to nearby attractors
neighbors = get_neighborhood(query, sphere_radius)
similarities = cosine_similarity(query, neighbors)  # O(k) where k << N
```

**Data structures:**
- **k-d tree** or **ball tree** for fast neighbor search
- **Graph structure**: attractors as nodes, edges connect neighbors
- **Locality-Sensitive Hashing (LSH)** for approximate neighbors

**Algorithm:**
```python
def find_resonant_neighbors(query, attractors, radius=0.5):
    """Find attractors within resonance sphere."""
    # Use spatial index (e.g., FAISS, Annoy)
    neighbors = spatial_index.query_radius(query, radius)
    return neighbors

def update_local(query, attractors):
    # Only update nearby attractors
    neighbors = find_resonant_neighbors(query, attractors)

    for attractor in neighbors:
        similarity = cosine_similarity(query, attractor)

        if similarity > 0.7:
            attract(attractor, query)  # Pull together
        elif similarity < -0.3:
            repel(attractor, query)    # Push apart
```

**Complexity:**
- Current: O(N) per query (N = all attractors)
- Vision: O(log N + k) per query (k = local neighbors)
- Speedup: ~10-100x for large N

---

**4. Emergent Multi-Scale Structure**

**Tokens** → individual query patterns
```
q1 = "Alice"
q2 = "Alice's cat"
q3 = "Alice smiled"
```

**Trees** → clusters of related attractors
```
Character_Tree_Alice:
  - attractor_1: "Alice" (high coherence)
  - attractor_2: "her" (medium coherence)
  - attractor_3: "she" (medium coherence)
```

**Groves** → clusters of related trees
```
Character_Grove:
  - Tree_Alice
  - Tree_Bob
  - Tree_Charlie
```

**Forests** → top-level domains
```
Narrative_Forest:
  - Character_Grove
  - Location_Grove
  - Plot_Grove
```

**Benefits:**
1. **Interpretability**: Can point to "the Alice grove"
2. **Efficiency**: Search at appropriate level (don't always go to leaves)
3. **Generalization**: Grove-level patterns transcend individual attractors
4. **Compression**: Represent forest with a few exemplars

---

## Mathematical Formulation

### Current DRAI (Phase 2)

**Attractor update:**
```
a_i(t+1) = (1 - α) a_i(t) + α q(t)  if sim(q, a_i) > θ_coherence
a_i(t+1) = (1 - δ) a_i(t)           otherwise (decay)
```

**Match:**
```
best = argmax_i sim(q, a_i)
```

---

### Vision DRAI (Phase 6+)

**Neighborhood-based update:**
```
N(q) = {a_i : sim(q, a_i) > θ_local}  # Local neighborhood

For a_i ∈ N(q):
  s = sim(q, a_i)

  if s > θ_positive:
    Δa_i = +α (q - a_i)      # Attract
  elif s < θ_negative:
    Δa_i = -β (q - a_i)      # Repel
  else:
    Δa_i = 0                 # Neutral

a_i(t+1) = a_i(t) + Δa_i
```

**Hierarchical clustering:**
```
Trees: cluster(attractors, distance_threshold=0.3)
Groves: cluster(trees, distance_threshold=0.5)
Forests: cluster(groves, distance_threshold=0.7)
```

**Sparse lookup:**
```
# O(log N) neighbor search via spatial index
neighbors = kd_tree.query(q, radius=r)
best = argmax_{a ∈ neighbors} sim(q, a)
```

---

## Why This Matters

### 1. Computational Efficiency

**RAG Vector DB:**
- Every query: compare to N vectors
- N = millions for large knowledge bases
- Even with approximate search (FAISS), still expensive

**DRAI Resonance Field:**
- Every query: compare to k neighbors
- k = ~10-50 (local sphere)
- 100-10,000x faster for large N

### 2. Emergent Organization

**RAG:**
- Manual organization (if any)
- Flat namespace
- "Where did I put that fact?"

**DRAI:**
- Self-organizing hierarchies
- "Alice is in the Character grove, which is part of the Narrative forest"
- Interpretable structure

### 3. Cognitive Plausibility

**Biological memory:**
- ✅ Local interactions (neurons connect to neighbors)
- ✅ Hierarchical (cortical columns → areas → systems)
- ✅ Push/pull (excitation + inhibition)
- ✅ Dynamic reorganization (synaptic plasticity)

**DRAI resonance field matches these properties.**

### 4. Scalability

**Current DRAI:**
- 32-64 attractors per layer
- Fine for proof-of-concept
- But: O(N) matching won't scale to thousands

**Vision DRAI:**
- Thousands of attractors organized hierarchically
- O(log N + k) matching scales gracefully
- Can represent much richer memory

---

## Implementation Roadmap

### Phase 6: Negative Resonance

**Add repulsion dynamics:**
```python
class DraiResonanceLayer:
    def __init__(self, ..., negative_threshold=-0.3, repulsion_rate=0.1):
        self.negative_threshold = negative_threshold
        self.repulsion_rate = repulsion_rate

    def _update_attractor(self, idx, pattern, similarity):
        if similarity > self.coherence_threshold:
            # Positive resonance: attract
            self._reinforce_attractor(idx, pattern)
        elif similarity < self.negative_threshold:
            # Negative resonance: repel
            self._repel_attractor(idx, pattern)

    def _repel_attractor(self, idx, pattern):
        """Push attractor away from pattern (anti-Hebbian)."""
        # Move attractor in opposite direction
        delta = self.attractor_centroids[idx] - pattern
        self.attractor_centroids[idx] += self.repulsion_rate * delta
        # Re-normalize
        self.attractor_centroids[idx] = F.normalize(
            self.attractor_centroids[idx], p=2, dim=-1
        )
```

**Expected benefit:**
- Attractors separate conflicting concepts
- More diverse attractor field
- Better coverage of semantic space

---

### Phase 7: Hierarchical Clustering

**Build tree/grove/forest structure:**
```python
class HierarchicalResonanceLayer(DraiResonanceLayer):
    def __init__(self, ...):
        super().__init__(...)
        self.trees = []     # Clusters of attractors
        self.groves = []    # Clusters of trees
        self.forests = []   # Clusters of groves

    def _reorganize_hierarchy(self):
        """Rebuild hierarchical structure from flat attractors."""
        # Cluster attractors into trees
        from sklearn.cluster import AgglomerativeClustering

        active_attractors = self.attractor_centroids[:self.attractor_count]

        # Trees: tight clusters (distance < 0.3)
        tree_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.3,
            linkage='average'
        )
        tree_labels = tree_clustering.fit_predict(active_attractors)

        # Build tree structures
        self.trees = [
            {
                'centroid': active_attractors[tree_labels == i].mean(0),
                'members': torch.where(tree_labels == i)[0],
                'coherence': self.attractor_coherence[tree_labels == i].mean()
            }
            for i in range(tree_labels.max() + 1)
        ]

        # Groves: cluster trees (distance < 0.5)
        tree_centroids = torch.stack([t['centroid'] for t in self.trees])
        grove_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            linkage='average'
        )
        grove_labels = grove_clustering.fit_predict(tree_centroids)

        # Build grove structures
        self.groves = [...]  # Similar to trees
```

**Trigger reorganization:**
- Every N steps (e.g., N=100)
- When attractor count changes significantly
- On-demand for analysis/visualization

---

### Phase 8: Local Neighborhoods & Sparse Lookup

**Replace linear search with spatial index:**
```python
import faiss  # or annoy, or nmslib

class SparseResonanceLayer(DraiResonanceLayer):
    def __init__(self, ...):
        super().__init__(...)
        # Build spatial index for fast neighbor search
        self.index = faiss.IndexFlatIP(self.head_dim)  # Inner product
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild spatial index from current attractors."""
        active = self.attractor_centroids[:self.attractor_count]
        if len(active) > 0:
            self.index.reset()
            self.index.add(active.cpu().numpy())

    def _find_local_neighbors(self, pattern, k=10):
        """Find k nearest neighbors in attractor space."""
        # O(log N) search via spatial index
        pattern_np = pattern.cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(pattern_np, k)
        return indices[0], distances[0]

    def _forward_phase6(self, query_layer, ...):
        """Forward with local neighborhoods."""
        pattern = self._extract_pattern(query_layer)

        # Find local neighbors (sparse lookup)
        neighbor_indices, similarities = self._find_local_neighbors(pattern, k=10)

        # Update only local attractors
        for idx, sim in zip(neighbor_indices, similarities):
            if sim > self.coherence_threshold:
                self._reinforce_attractor(idx, pattern)
            elif sim < self.negative_threshold:
                self._repel_attractor(idx, pattern)

        # Generate K/V from local neighbors only
        k_reson, v_reson = self._generate_kv_local(neighbor_indices, ...)

        return k_reson, v_reson
```

**Complexity:**
- Build index: O(N log N) once
- Query: O(log N + k) per step
- vs current O(N) per step
- **10-100x speedup** for N > 1000

---

## Visualization: Trees, Groves, Forests

### Example Visualization

```
Forest: Narrative Memory
│
├─ Grove: Characters
│  ├─ Tree: Alice
│  │  ├─ attractor_1: "Alice" (coherence: 0.95)
│  │  ├─ attractor_2: "she" (coherence: 0.80)
│  │  └─ attractor_3: "her cat" (coherence: 0.75)
│  │
│  └─ Tree: Bob
│     ├─ attractor_4: "Bob" (coherence: 0.92)
│     └─ attractor_5: "he" (coherence: 0.78)
│
├─ Grove: Locations
│  ├─ Tree: Home
│  │  ├─ attractor_6: "house" (coherence: 0.88)
│  │  └─ attractor_7: "room" (coherence: 0.82)
│  │
│  └─ Tree: City
│     └─ attractor_8: "street" (coherence: 0.85)
│
└─ Grove: Plot
   └─ Tree: Conflict
      ├─ attractor_9: "argument" (coherence: 0.90)
      └─ attractor_10: "tension" (coherence: 0.87)
```

### t-SNE Plot with Hierarchical Annotations

```
      Character Grove
          ┌─────┐
     Alice│  o  │Bob
          │o   o│
          └─────┘

                    Location Grove
                      ┌─────┐
                 Home │  o  │City
                      │ o o │
                      └─────┘

  Plot Grove
   ┌──────┐
   │  o   │ Conflict
   │ o  o │
   └──────┘
```

**Color coding:**
- Trees: same color family (e.g., Alice tree = shades of blue)
- Groves: distinct color families (Characters = blues, Locations = greens)
- Forests: major boundaries

**Interactive exploration:**
- Click on forest → see groves
- Click on grove → see trees
- Click on tree → see individual attractors
- Hover → show sample tokens that activated this attractor

---

## Why "Trees, Groves, and Forests"?

### Metaphor Mapping

**Tree** = tight cluster of related concepts
- All about one entity or closely related ideas
- Example: "Alice", "her", "she", "Alice's cat"
- Strong internal coherence

**Grove** = collection of related trees
- Broader semantic category
- Example: All character trees form Character Grove
- Trees in a grove resonate with each other

**Forest** = collection of related groves
- Top-level conceptual domain
- Example: Narrative forest (characters + locations + plot)
- Different forests = different knowledge domains

**Why nature metaphor?**
- Organic, emergent growth (not pre-designed)
- Local interactions create global structure
- Self-organizing, self-pruning
- Hierarchical but not rigid (trees can shift groves)

---

## Expected Benefits

### 1. Interpretability

**Current:**
- "Attractor 17 has high coherence"
- ❓ What is attractor 17?

**Vision:**
- "The Alice tree in the Character grove is highly active"
- ✅ Clear semantic meaning

### 2. Efficiency

**Current:**
- O(N) matching per query
- Limits to ~32-64 attractors

**Vision:**
- O(log N + k) matching per query
- Scales to thousands of attractors
- Hierarchical search: check grove → tree → attractor

### 3. Generalization

**Current:**
- Only exact attractor matches
- No abstraction beyond individual attractors

**Vision:**
- Can reason at grove/forest level
- "Something in the Character grove activated"
- Abstraction across multiple related concepts

### 4. Biological Plausibility

**Current:**
- Flat memory = unrealistic

**Vision:**
- Hierarchical organization = cortical areas
- Local interactions = local circuits
- Push/pull = E/I balance
- Much closer to biological memory

---

## Open Questions

### 1. How many levels?

**Option A: Fixed 3 levels (Trees/Groves/Forests)**
- Simple, easy to implement
- May not capture all scales

**Option B: Adaptive depth**
- Keep clustering until can't merge more
- More flexible, but complex

**Option C: Fractal (self-similar at all scales)**
- "Forests of forests"
- Theoretically elegant, but perhaps overkill

### 2. How to handle dynamic reorganization?

**Problem:** As attractors form/decay, hierarchy changes

**Option A: Rebuild periodically**
- Every N steps, re-cluster
- Expensive but simple

**Option B: Incremental updates**
- When attractor joins, assign to existing tree
- When tree grows, check if should split
- Cheaper but more complex

**Option C: Online hierarchical clustering**
- Use Growing Neural Gas or similar
- Continuously adapts structure

### 3. Should groves/forests have their own attractors?

**Option A: Virtual centroids**
- Grove centroid = mean of tree centroids
- Forest centroid = mean of grove centroids
- Doesn't increase memory

**Option B: Explicit meta-attractors**
- Groves have their own learned centroids
- Can represent abstract concepts
- More expressive but more memory

### 4. How to visualize during inference?

**Challenge:** Hard to show real-time hierarchical dynamics

**Ideas:**
- Animated force-directed graph (attractors as nodes)
- Hierarchical dendrogram that updates over time
- 3D visualization: height = hierarchy level
- Heatmap: columns = attractors, rows = time, color = activation

---

## Connection to Current Phase 5 Evaluation

### How hierarchical structure helps Phase 5 tasks

**Long-story consistency:**
- Character grove keeps character facts together
- Plot grove maintains narrative coherence
- Location grove grounds spatial reasoning

**Lesioning experiment:**
- Lesion entire grove → catastrophic failure (proves groves matter)
- Lesion single tree → localized deficit (proves trees organize)
- Lesion individual attractor → minimal impact (redundancy within trees)

**Visualization:**
- Show grove activations over story
- "The Character grove lit up when Alice was mentioned"
- Much clearer than "Attractors 3, 7, 12, 18 activated"

---

## Implementation Timeline

### Phase 5: Evaluation (Current)
- Prove current flat DRAI works
- Establish baseline benefits

### Phase 6: Negative Resonance (2-3 weeks)
- Add repulsion dynamics
- Test attractor diversity improves

### Phase 7: Hierarchical Clustering (3-4 weeks)
- Implement tree/grove/forest structure
- Visualizations of hierarchy
- Test interpretability gains

### Phase 8: Sparse Local Updates (3-4 weeks)
- Add spatial indexing (FAISS/Annoy)
- Local neighborhood search
- Benchmark speedup

### Phase 9: Adaptive Hierarchy (4-6 weeks)
- Online clustering
- Dynamic reorganization
- Stress test on long sequences

### Phase 10: Multi-scale Reasoning (Future)
- Use grove/forest-level representations
- Abstraction and generalization
- Full "topological memory" system

**Total: ~6-12 months for full vision**

---

## Conclusion

The vision of **trees, groves, and forests of meaning** is:

1. **Computationally efficient** - O(log N + k) vs O(N)
2. **Interpretable** - "Alice grove" vs "attractor 17"
3. **Scalable** - thousands of attractors, not dozens
4. **Biologically plausible** - hierarchical, local, dynamic
5. **Emergent** - structure forms automatically from local interactions

**Current DRAI (Phase 1-4):**
- Foundation is solid
- Continuous dynamics ✓
- Self-organization ✓
- But: flat, global search

**Vision DRAI (Phase 6-10):**
- Add hierarchy (trees → groves → forests)
- Add locality (spheres of resonance)
- Add push/pull (positive + negative resonance)
- Full topological memory space

**This is the path from "zero-cost side-channel" to "living memory fabric."**

---

## References & Related Work

**Self-Organizing Maps (SOMs):**
- Kohonen, 1982: topological organization via local updates
- Similar to grove formation

**Growing Neural Gas:**
- Fritzke, 1995: dynamic graph structure, continuous adaptation
- Close to our vision for adaptive hierarchy

**Hierarchical Temporal Memory (HTM):**
- Hawkins & Blakeslee, 2004: cortical columns, hierarchical patterns
- Biological inspiration for multi-scale structure

**Attractor Networks:**
- Hopfield, 1982: energy landscape, basins of attraction
- Mathematical foundation for resonance dynamics

**Anti-Hebbian Learning:**
- Földiák, 1990: decorrelation via inhibition
- Basis for negative resonance

**Cortical Columns:**
- Mountcastle, 1997: hierarchical organization in cortex
- Trees → groves → forests ≈ minicolumns → columns → areas

---

**Status:** Vision document - guides Phase 6+ development

**Next step:** Prove Phase 5 baseline, then begin hierarchical extensions
