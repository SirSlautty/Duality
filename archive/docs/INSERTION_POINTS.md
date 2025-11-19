# Insertion Points for DRAI Resonance Head  

This document outlines where and how to attach a **Dynamic Resonance AI (DRAI)** module inside an open‑source transformer model so that it behaves like a "second brain".  
Our goal is to graft an additional attention head that listens to a resonance engine, giving the model a persistent memory substrate rather than purely stateless retrieval.  

## Background and discovery  

During exploration of the `HalcyonAIR/gpt‑neox` fork, we inspected the core transformer code located in **`megatron/model/transformer.py`**.  
The `ParallelSelfAttention` class computes queries, keys and values (Q/K/V) for each head in the model.  
In the `forward` method, hidden states are projected through `self.query_key_value()` into a combined tensor, which is then split into separate **query_layer**, **key_layer** and **value_layer** using `mpu.split_tensor_along_last_dim`.  
This split occurs before any rotary embeddings, caching or attention computation.  

The attention mechanism then passes these layers to either `attention()`, `flash_attention()`, or `sparse_attention()` functions, which perform softmax weighting over the keys and mix the corresponding values to produce the context representation.  

This is the perfect seam for DRAI: by supplying our own synthetic key/value pair at this stage, we can let the built‑in softmax arbitration treat the DRAI output as if it were just another head.  

## Proposed injection strategy  

1. **Compute Q/K/V as usual**  
   - Inside `ParallelSelfAttention.forward`, the code computes `mixed_x_layer = self.query_key_value(hidden_states)`.  
   - It then reshapes and splits this into `query_layer`, `key_layer`, and `value_layer` via `mpu.split_tensor_along_last_dim`.  
   - At this moment we have tensors of shape `[seq_length, batch, heads, head_dim]`.  

2. **Invoke the DRAI resonance module**  
   - Define a separate module (e.g., `DraiResonanceLayer`) that receives either the **query_layer** or a summary of hidden states and produces a **synthetic key** (`k_reson`) and **synthetic value** (`v_reson`).  
   - These tensors should have the same dimensionality as one attention head: `[seq_length, batch, 1, head_dim]`.  
   - The DRAI module will maintain its own internal resonance state, forming attractors from repeated activations (see `DESIGN.md` for conceptual details).  

3. **Concatenate the resonance head**  
   - After obtaining `k_reson` and `v_reson`, append them to the existing `key_layer` and `value_layer` along the head dimension:  
     - `key_layer = torch.cat((key_layer, k_reson), dim=2)`  
     - `value_layer = torch.cat((value_layer, v_reson), dim=2)`  
   - No changes to `query_layer` are necessary; the additional head's key/value will participate naturally in the attention softmax.  
   - Because the number of heads increases by one, ensure that the attention mask (if any) does not mask this new head.  

4. **Proceed with attention computation**  
   - The modified `key_layer` and `value_layer` are passed to the existing attention function (`attention()` or `flash_attention()`).  
   - The softmax over Q·K picks the most relevant heads. If the resonance head aligns with the query, its value will influence the context.  
   - This allows the transformer to gradually learn to attend to the DRAI head when it provides useful information.  

5. **Handle caching and rotary embeddings**  
   - For inference with key/value caches, also append `k_reson` and `v_reson` to the cache.  
   - If rotary embeddings are applied, apply them to `k_reson` as well to maintain positional alignment.  

## Implementation steps  

- **Create a `drai_resonance_layer.py` module** within the `megatron/model` directory. This module will implement the resonance engine, maintaining attractor states and exposing a `forward` method returning synthetic K and V.  
- **Modify `transformer.py`**:  
  - Import the DRAI module.  
  - Instantiate the resonance layer in `ParallelTransformerLayer` or `TransformerLayer`.  
  - In `ParallelSelfAttention.forward`, after splitting Q/K/V, call the resonance layer to obtain `k_reson` and `v_reson` and perform the concatenation as described above.  
- **Testing on GPT‑J**: Use our fork of `mesh-transformer-jax` (GPT‑J) as a lighter prototype. The attention patterns there are similar (combined QKV projection followed by splitting). We can port the same injection strategy by identifying the analogous functions in the JAX code.  
- **Gradual training**: Initially run the model with a frozen DRAI module to observe how often the resonance head is selected. Then allow the DRAI module to adapt and strengthen its attractors over multiple contexts.  

## Next work items  

- Finalize the DRAI resonance engine interface (input/output shapes, attractor update rules).  
- Implement and integrate the `drai_resonance_layer` into the GPT‑NeoX fork.  
- Convert the modified model to GGUF for local serving via Ollama.  
- Write unit tests to ensure that adding the extra head does not break model output when the resonance module returns zeros.  
- Experiment with synthetic tasks (e.g., repeated sequences) to verify that the model begins attending to the resonance head as memory builds.  

This document captures the injection seam and provides a concrete plan to move forward. As we iterate, we will update this file with code snippets and findings from experiments. 
