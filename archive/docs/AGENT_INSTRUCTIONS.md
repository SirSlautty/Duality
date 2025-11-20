# Agent Instructions for Building NeoX Duality Mode  

## Clone repositories  
- Clone your forks of **GPT‑NeoX** (`HalcyonAIR/gpt-neox`) and **GPT‑J** (`HalcyonAIR/mesh-transformer-jax`) into a local folder (e.g., `models/`).  

## Environment setup  
- Create and activate a Python virtual environment or Conda environment.  
- Install NeoX dependencies using `pip install -r requirements/requirements.txt` from the NeoX repo.  
- Install JAX/Haiku and other dependencies following the GPT‑J repository instructions.  

## Fetch weights  
- Run `git lfs install` and then `git lfs pull` within each model directory to fetch pretrained weights.  
- Start with **GPT‑J 6B** weights for fast prototyping; later you can download **GPT‑NeoX 20B** weights.  

## Create the DRAI resonance module  
- Add a new file `megatron/model/drai_resonance_layer.py` in your NeoX fork.  
- Implement a `DraiResonanceLayer` class that maintains latent attractors and exposes a `forward(query_layer)` method returning `(k_reson, v_reson)` shaped like a single head `[seq_len, batch, 1, head_dim]`.  
- Apply positional encodings to `k_reson` if necessary and include decay rules for old attractors.  

## Modify the transformer  
- In `megatron/model/transformer.py`, import and instantiate the resonance layer in the transformer layer constructor.  
- In `ParallelSelfAttention.forward`, after splitting `mixed_x_layer` into `query_layer`, `key_layer` and `value_layer`, call:  
  - `k_reson, v_reson = self.drai_resonance_layer(query_layer)`  
  - Concatenate along the head dimension:  
    - `key_layer  = torch.cat((key_layer,  k_reson),  dim=2)`  
    - `value_layer = torch.cat((value_layer, v_reson), dim=2)`  
- Ensure attention masks and caches handle the extra head.  

## Port the logic to GPT‑J  
- Find the attention implementation in `mesh-transformer-jax` (e.g., in `transformer.py`).  
- Implement the same resonance layer and concatenate synthetic K/V to the existing key and value tensors.  

## Testing  
- Run the modified models with the resonance layer returning zeros and verify outputs match the original models.  
- Provide non-zero resonance outputs and observe changes in attention weights.  
- Add unit tests to check tensor shapes and gradients.  

## Optional: serve via Ollama  
- After modifying NeoX, convert the model to GGUF and configure an Ollama model file to serve it locally.  

## Next steps  
- Fine-tune the hybrid model on tasks requiring long-term memory to train the resonance layer.  
- Document experiments and update design docs accordingly.
