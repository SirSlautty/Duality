# Duality  
  
Welcome to **Project Duality** – an experimental research repository exploring hybrid cognitive architectures. The goal is to augment an open‑source transformer model with a second “resonance cortex” based on the Dynamic Resonance AI (DRAI) principles we’ve been discussing.  
  
## Vision  
  
We want to move beyond retrieval‑based memory and build an inseparable memory layer inside the model’s attention heads. By dedicating one or two heads to a DRAI resonance engine, the model will learn to stabilise and reinject latent attractors instead of fetching tokens. This repository will collect prototypes, diagrams, and experiments exploring this new frontier.  
  
## Objectives  
  
- Prototype a simple hook into an open transformer (e.g. GPT‑NeoX) that routes a head’s Q/K/V through a resonance module.  
- Implement the resonance accumulator that detects recurring latent vectors and forms stable attractors.  
- Inject the attractor outputs back into the attention mixing as synthetic K/V pairs.  
- Visualise the evolution of the resonance manifold as new concepts are stabilised.  
- Document findings, pitfalls, and emergent behaviours along the way.  
  
## Repository structure  
  
This repository will grow over time. For now we plan to include:  
  
- `docs/` – design documents, diagrams, and theory notes.  
- `src/` – prototype code for the DRAI resonance layer and model hooks.  
- `experiments/` – notebooks and scripts for running small‑scale tests.  
- `.gitignore` – ignores virtual envs, logs, datasets, and compiled artifacts as discussed.  
- `LICENSE` – Apache‑2.0 license to allow permissive use while protecting contributions.  
  
Feel free to open issues or discussions as the project evolves. This is a research playground — contributions and critiques are welcome. 
