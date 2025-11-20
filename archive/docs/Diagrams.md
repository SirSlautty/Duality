Diagram 1 — High-Level Architecture

Shows the transformer with the resonance cortex grafted in.

flowchart TD
    A[Input Tokens] --> B[Transformer Layer]

    B --> C[Standard Attention Heads]
    B --> D[Resonance Head(s)]

    D --> E[Resonance Module<br/>(Accumulator + Pattern Detector)]
    E --> F[Attractor Field]
    F --> G[Synthetic K/V Injection]

    G --> B

    B --> H[Output Tokens]

Diagram 2 — Internal Flow of the Resonance Module

Shows how Q/K/V get accumulated, processed, and turned into attractors.

flowchart LR
    QKV[Q / K / V] --> ACC[Resonance Accumulator]

    ACC --> PD[Pattern Detector]
    PD -->|New Pattern| NA[Create New Attractor]
    PD -->|Recurrent Pattern| SA[Strengthen Existing Attractor]

    NA --> AF[Attractor Field]
    SA --> AF

    AF --> SKV[Synthetic K/V Output]

Diagram 3 — Attractor Formation Over Time

A conceptual timeline chart showing stabilization.

sequenceDiagram
    participant T0 as Early Phase
    participant T1 as Mid Phase
    participant T2 as Stable Phase

    T0->>T1: Repeated latent patterns detected
    T1->>T2: Clusters stabilize into attractors
    T2->>T2: Attractors influence next-token reasoning

Diagram 4 — Static Heads vs Resonance Heads

Side-by-side anatomy.

flowchart LR
    A[Static Attention Head] --> B[Attend]
    B --> C[Mix]
    C --> D[Forget]

    E[Resonance Head] --> F[Attend]
    F --> G[Accumulate]
    G --> H[Reinforce Patterns]
    H --> I[Inject Synthetic K/V]
    I --> F

Diagram 5 — Memory Flow Through the Model

How attractor output modifies downstream reasoning.

flowchart TD
    A[Input] --> B[Transformer Layers]

    B --> C[Resonance Cortex]
    C --> D[Attractors]
    D --> E[Synthetic K/V Injection]

    E --> B
    B --> F[Output]

Diagram 6 — Insertion Points (matches INSERTION_POINTS.md)

flowchart TD
    A[Input Embeddings] --> B[Pre-Attention Hook]

    B --> C[Attention Module]
    C --> D[Post-Attention Hook]

    D --> E[Feed-Forward Layers]
    E --> F[Inter-Layer Hook]

    F --> G[Next Transformer Block]

Diagram 7 — Duality vs Retrieval-Augmented Systems

flowchart LR
    subgraph RAG[Retrieval-Augmented Generation]
        U1[User Query] --> E1[Embed + Search]
        E1 --> DB1[Vector DB]
        DB1 --> M1[Model]
        M1 --> O1[Output]
    end

    subgraph DUAL[Duality Architecture]
        U2[Input] --> T2[Transformer]
        T2 --> RC2[Resonance Cortex]
        RC2 --> A2[Attractor Field]
        A2 --> SKV2[Synthetic K/V Injection]
        SKV2 --> T2
        T2 --> O2[Output]
    end

Diagram 8 — Memory Tender ↔ Duality Bridge

For future integration.

flowchart LR
    MT[Memory Tender<br/>Symbolic Long-Term Memory] <--> BRIDGE[Resonance–Symbolic Bridge] <--> DU[Duality<br/>Dynamic Short-Term Memory]

Diagram 9 — Resonance Manifold Evolution

Simple conceptual topography view.

flowchart TD
    A[Latent Vectors<br/>(scattered)] --> B[Clusters Form]
    B --> C[Stable Attractors]
    C --> D[Deepened Valleys in the Manifold]
