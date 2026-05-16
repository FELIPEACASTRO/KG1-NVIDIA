# KG1 Solution Architecture - Mermaid

Generated: 2026-05-16

This document shows the main moving parts used to improve KG1 predictions,
with emphasis on the current weak families:

- `bit_manipulation`
- `equation_transform`

Current active execution line: V487 H200 smoke, launched from seed adapter
`felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.

## Main Pieces

```mermaid
graph TB
  U[User and Codex operator] --> R[GitHub branch v230-v226-complementarity]
  R --> L[HF launcher V487]
  L --> P0[Static safety gate]
  L --> P1[HF job preflight gate]
  L --> P2[V485 PEFT metadata round-trip gate]
  L --> P3[V478 objective alignment gate]

  D0[HF dataset repo kg1-nemotron-training] --> D1[V390/V475 train JSONL]
  D0 --> D2[V390/V475 validation JSONL]
  D1 --> H[Hash, rows, family, subcategory audit]
  D2 --> H
  H --> T[Tokenizer and offset-mask contract]

  A0[V290 checkpoint-6 seed adapter] --> A1[PEFT native load]
  A1 --> A2[target_modules check]
  A1 --> A3[target_parameters check]
  A3 --> A4[Nemotron alias check gate_up to up_proj]

  T --> M[Nemotron 3 Nano 30B BF16]
  A4 --> M
  M --> F[Trainable LoRA filter]
  F --> F1[Train q/k/v/o/lm_head LoRA]
  F --> F2[Freeze up/down/in/out experts but keep active]

  F1 --> TR[HF H200 training job]
  F2 --> TR
  TR --> C[Checkpoints 2/4/6/8/10/12]
  C --> W[Weak eval 315-row contract]
  W --> G{Promotion gate}

  G -->|pass| FULL[Official-like full eval]
  FULL --> PKG[Package adapter-only submit]
  PKG --> K[Kaggle submit]

  G -->|fail| FIN[FinOps cancel or stop]
  FIN --> E[Error ledger and roadmap update]
  E --> R

  subgraph Required Gates Before GPU Spend
    P0
    P1
    P2
    P3
  end

  subgraph Submit-Safe Metrics
    W
    FULL
    PKG
  end
```

## Operational Flowchart

```mermaid
flowchart TD
  S([Start new improvement attempt]) --> Q1{Is there new verified signal?}
  Q1 -->|no| STOP1[Do not train more epochs or run LR sweep]
  Q1 -->|yes| BUILD[Build or select dataset/adapter recipe]

  BUILD --> DATA[Audit dataset]
  DATA --> DOK{Hashes, rows, families, subcategories OK?}
  DOK -->|no| FIXDATA[Fix or quarantine dataset]
  FIXDATA --> DATA
  DOK -->|yes| ADAPT[Audit seed adapter]

  ADAPT --> AOK{PEFT config, tensors, target_parameters OK?}
  AOK -->|no| FIXADAPT[Fix loader or block route]
  FIXADAPT --> ADAPT
  AOK -->|yes| OBJ[Run V478 objective alignment]

  OBJ --> OOK{Bit share >= 0.20 and equation share <= 0.80?}
  OOK -->|no| REWEIGHT[Rebalance source/subcategory weights]
  REWEIGHT --> OBJ
  OOK -->|yes| LAUNCH[Launch short HF smoke]

  LAUNCH --> LOGS[Monitor logs every 40 seconds]
  LOGS --> FFAIL{Preflight/runtime failure?}
  FFAIL -->|yes| LEDGER[Record error, fix root cause, relaunch only after gate]
  LEDGER --> BUILD
  FFAIL -->|no| CKPT[Wait for first checkpoint]

  CKPT --> WEAK[Run weak eval]
  WEAK --> WGATE{total > 192 and equation > 56 and bit >= 136 and trunc = 0?}
  WGATE -->|no| CANCEL[Cancel/stop by FinOps]
  CANCEL --> LEDGER
  WGATE -->|yes| FULL[Run official-like full eval]

  FULL --> FGATE{Full eval beats current safe floor?}
  FGATE -->|no| HOLD[Do not package or submit]
  HOLD --> LEDGER
  FGATE -->|yes| PACKAGE[Package immutable adapter-only submission]
  PACKAGE --> SUBMIT[Kaggle submit]
  SUBMIT --> END([Record score and roadmap result])
```

## Sequence Diagram

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant Codex
  participant GitHub
  participant HF as Hugging Face Jobs
  participant Data as HF Dataset Repo
  participant Adapter as Seed Adapter Repo
  participant Train as Training Script
  participant Eval as Weak/Full Eval
  participant Kaggle

  User->>Codex: Request roadmap execution
  Codex->>GitHub: Commit and push launcher/gates
  Codex->>HF: Launch V487 H200 job
  HF->>GitHub: Clone exact commit
  HF->>Train: Compile scripts
  Train->>Data: Download train/validation JSONL
  Data-->>Train: Return files
  Train->>Train: Validate hashes, rows, families, offset masks
  Train->>Adapter: Load checkpoint-6 via PEFT
  Adapter-->>Train: Adapter config and tensors
  Train->>Train: Validate target_modules and target_parameters
  Train->>Train: Apply Nemotron alias matcher
  Train->>Train: Run baseline eval_loss
  Train->>Train: Train short smoke checkpoints
  Train->>HF: Upload checkpoints during training
  Codex->>HF: Read logs every 40 seconds

  alt First checkpoint is submit-safe candidate
    Codex->>Eval: Run weak eval on checkpoint
    Eval-->>Codex: Return total/equation/bit/trunc metrics
    Codex->>Eval: Run official-like full eval
    Eval-->>Codex: Return full score
    Codex->>Kaggle: Submit only if package gate passes
  else First checkpoint fails gate
    Codex->>HF: Cancel/stop job by FinOps
    Codex->>GitHub: Update error ledger and roadmap
  end
```

## Current Problem Boundary

```mermaid
flowchart LR
  A[Verified CPU solver/verifier gains] --> B[Distilled traces/datasets]
  B --> C[LoRA training]
  C --> D[Adapter-only inference]
  D --> E[Weak ACC]

  A -. strong signal .-> A1[Potential equation 60 and bit 146]
  C -. current plateau .-> C1[Adapter-only best safe: total 192, equation 56, bit 136]
  E --> P{Promotion}
  P -->|not yet| N[Need real weak gain, not lower eval_loss]

  style C1 fill:#331a1a,stroke:#cc4444,color:#ffffff
  style N fill:#332a00,stroke:#d6a100,color:#ffffff
```

The core issue is not that we lack CPU-solvable patterns. The issue is
transferring those patterns into an adapter-only submission without relying on
runtime solvers, postprocessors, constrained decoding, or leaked labels.
