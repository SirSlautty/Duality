# Models  
  
This directory stores references and documentation for external open‑source models used in the Duality project. We do not commit model weights or large datasets here; instead, we describe where to obtain them and how to integrate them.  
  
## Target Models  
  
- **GPT-J 6B**  
  - Repository: [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax)  
  - Weights: available on Hugging Face at `EleutherAI/gpt-j-6B`  
  - Advantages: smaller size for rapid prototyping; permissive license; accessible attention code.  
  
- **GPT-NeoX-20B**  
  - Repository: [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)  
  - Weights: available on Hugging Face at `EleutherAI/gpt-neox-20b`  
  - Advantages: full‑featured transformer architecture with modular attention; open weights; our final target for integration.  
  
## Working with models  
  
- Clone or fork the model repositories into this folder via Git submodules or separate clones, rather than copying code manually.  
- Download pretrained weights separately using `git-lfs` and Hugging Face; do not commit weight files to this repository.  
- Convert models to `gguf` format for running with Ollama if desired. Use tools like `transformers` or conversion tools to export.  
- Instrument the attention code to insert the DRAI resonance head according to our design document. Keep modifications in a separate branch or directory for clarity.  
  
This README will evolve as we integrate models and document the exact steps taken.
