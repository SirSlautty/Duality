# Build Steps for Duality NeoX Mode  

This file provides step-by-step instructions to build a modified GPT-NeoX model with a Dynamic Resonance AI (DRAI) head attached. Use these steps when setting up your development environment and implementing the Duality modifications.  

## 1. Clone repositories and create a branch  

- Clone your fork of gpt-neox:  
  ```bash  
  git clone https://github.com/HalcyonAIR/gpt-neox.git  
  cd gpt-neox  
  ```  
- Create a new branch for Duality experiments:  
  ```bash  
  git checkout -b duality-mode  
  ```  

## 2. Set up your Python environment  

- Create a Python virtual environment and install dependencies:  
  ```bash  
  python3 -m venv .venv  
  source .venv/bin/activate  
  pip install -r requirements/requirements.txt  
  pip install -r requirements/requirements-dev.txt  
  ```  
- Depending on GPU hardware you may need to install a specific version of PyTorch with CUDA support.  

## 3. Create the DRAI resonance module  

- Inside the `megatron/model` directory of gpt-neox, add a new file named `drai_resonance_layer.py`.  
- Implement a class `DraiResonanceLayer` with a `forward` method that takes a query or hidden representation and returns synthetic key and value tensors with shape `[seq_len, batch, 1, head_dim]`.  
- The module should maintain its own state (e.g. using buffers) to accumulate activations and form attractors over time. Refer to `DESIGN.md` for the resonance concepts.  

## 4. Modify the transformer to inject the resonance head  

- Open `megatron/model/transformer.py`.  
- Import your `DraiResonanceLayer` class at the top of the file.  
- In the `ParallelTransformerLayer` (or `TransformerLayer`) constructor, instantiate the resonance layer:  
  ```python  
  self.drai_layer = DraiResonanceLayer(self.hidden_size, 1)  
  ```  
- In `ParallelSelfAttention.forward`, after splitting the combined QKV tensor into `query_layer`, `key_layer`, and `value_layer`, call the resonance layer to produce `k_reson` and `v_reson`:  
  ```python  
  k_reson, v_reson = self.drai_layer(query_layer)  
  key_layer = torch.cat((key_layer, k_reson), dim=2)  
  value_layer = torch.cat((value_layer, v_reson), dim=2)  
  ```  
- Make sure to handle rotary embeddings and KV caching for the new head as described in `INSERTION_POINTS.md`.  

## 5. (Optional) Modify GPT-J for prototyping  

- Clone your fork of `mesh-transformer-jax` and create a branch (similar to gpt-neox).  
- Identify the attention computation in the JAX code and perform an analogous injection of a resonance head. This is optional if you want to test on a smaller model first.  

## 6. Train or fine-tune  

- For quick testing, run inference with the modified model and ensure it still produces sensible outputs when the resonance module returns zeros.  
- Gradually allow the resonance layer to update its internal attractors and observe whether the model learns to use the new head on repeated sequences.  
- Training from scratch may be expensive; start with small-scale fine-tuning on a simple dataset to validate the approach.  

## 7. Convert to GGUF for Ollama (after modifications)  

- Once the model is working locally, convert the PyTorch weights into GGUF format for serving via Ollama using the appropriate conversion scripts (e.g. `convert-pt-to-gguf.py`).  
- Create a custom Ollama model configuration that points to your GGUF file.  

## 8. Commit and push  

- Stage, commit, and push your changes to the `duality-mode` branch:  
  ```bash  
  git add megatron/model/drai_resonance_layer.py megatron/model/transformer.py  
  git commit -m "Integrate DRAI resonance head for Duality mode"  
  git push origin duality-mode  
  ```  

Follow these steps to build and iterate on a GPT-NeoX model with an embedded resonance cortex. Adjust them as necessary for your development environment. 
