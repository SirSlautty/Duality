# Design Document for Project Duality

## Overview  
Project Duality aims to explore hybrid cognitive architectures by attaching a Dynamic Resonance AI (DRAI) module to an open‑source transformer model. Rather than using a retrieval‑augmented memory system, we want to embed an inseparable memory layer inside the model by dedicating one or two attention heads to a resonance‑based “second brain.” This document outlines the initial design for implementing this hybrid.

## Architecture  
The core of Duality consists of two interacting components:

- **Brain 1: Transformer Core** – a standard open‑source transformer (e.g. GPT‑NeoX) that processes input tokens through embedding, multi‑head attention, and feed‑forward layers.  
- **Brain 2: Resonance Cortex** – a DRAI module that monitors latent vectors from selected attention heads, detects recurring patterns, and forms stable attractor states in a resonance manifold.

The system uses the transformer’s existing attention mixing to integrate outputs from both brains. A synthetic K/V pair representing the resonance attractor is injected into the attention head, allowing the softmax to consider it alongside regular token memories.

## Resonance Module  
The resonance cortex operates in latent space:

- It observes the query (Q), key (K), and value (V) vectors from a dedicated head.  
- It maintains a coherence accumulator that increases when similar latent vectors recur over time.  
- Once a threshold is crossed, a stable attractor forms. This attractor acts as a persistent memory of the concept, independent of token indices.  
- Each attractor emits a synthetic key and value vector (`K_DRAI`, `V_DRAI`) which are fed back into the attention computation.

Concepts that are not reinforced decay over time, allowing the resonance manifold to remain dynamic and self‑organising.

## Integration Points  
To graft the resonance cortex into the transformer:

1. **Select dedicated head(s)**: Choose one or two attention heads for DRAI integration.  
2. **Tap latent vectors**: Extract the Q, K, V tensors from these heads before projection.  
3. **Feed to DRAI**: Pass these tensors to the resonance module to update attractors and generate synthetic `K_DRAI` and `V_DRAI`.  
4. **Inject back**: Append `K_DRAI` and `V_DRAI` to the original K and V matrices for that head.  
5. **Attention mixing**: Allow the softmax attention mechanism to treat the resonance vector as an additional candidate memory.

Because the resonance vector participates naturally in the softmax, the transformer will learn to rely on it when it offers a more coherent signal than standard token memories.

## Data Flow  
1. Input tokens → Embedding layer → Q/K/V for each head.  
2. For dedicated DRAI head(s), send Q/K/V to resonance module.  
3. Resonance module updates attractor states and produces `K_DRAI`, `V_DRAI`.  
4. Return to attention: `[K; K_DRAI]`, `[V; V_DRAI]`.  
5. Softmax over `Q·K^T` includes resonance key; corresponding value contributes to the head output.  
6. Combined head outputs pass through the usual MLP layers to produce logits.

## Future Work  
- Formalise attractor dynamics: decide thresholds, decay rates, and dimensionality of the resonance manifold.  
- Prototype on a small transformer (e.g. GPT‑2 or NanoGPT) before scaling to NeoX.  
- Develop visualisations of attractor birth and manifold warping over time to monitor the emergence of memory.  
- Explore different coupling strategies (passive vs active) and numbers of dedicated heads.  
- Evaluate performance on tasks requiring long‑term memory or concept reinforcement.

This design document is a starting point; expect revisions as we prototype and learn from experiments.
