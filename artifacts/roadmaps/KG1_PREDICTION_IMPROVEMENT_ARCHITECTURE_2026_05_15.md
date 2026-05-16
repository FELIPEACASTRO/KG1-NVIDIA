# KG1 Prediction Improvement Architecture

Atualizado: 2026-05-15

Objetivo: documentar o fluxo ponta a ponta usado para tentar melhorar a
predicao submit-safe das familias `equation_transform` e `bit_manipulation`,
sem violar o contrato adapter-only do desafio.

## Principio Central

Ganho so conta se aparecer em um adapter/package submetivel. Solver, verifier,
postprocessor, OpenRouter, literatura e scripts auxiliares sao permitidos como
diagnostico, geracao de hipotese, criacao de dataset e validacao, mas nao podem
ser dependencia runtime do submit.

## Arquitetura

```mermaid
flowchart TD
  A[Kaggle Challenge Rules] --> B[Contrato de submissao]
  B --> B1[Adapter-only package]
  B --> B2[Sem postprocessor runtime]
  B --> B3[Sem verifier ou constrained decoding no submit]
  B --> B4[Sem weak/full leakage no treino]

  C[Fontes de dados e conhecimento] --> C1[Public train / competition_train]
  C --> C2[Artefatos Drive / HF / Kaggle]
  C --> C3[Discussões, kernels e literatura]
  C --> C4[OpenRouter / outras IAs]

  C1 --> D[Auditoria de dados]
  C2 --> D
  C3 --> E[Intel externa]
  C4 --> E

  D --> D1[Hashes SHA256]
  D --> D2[Family counts]
  D --> D3[prompt_sha256 e id anti-leakage]
  D --> D4[answer / boxed semantic audit]
  D --> D5[train / val / weak / full separation]

  E --> F[Hipoteses acionaveis]
  F --> F1[Bit: bit-pair / bitsum / stride]
  F --> F2[Equation: DSL / operator / symbolic sequence]
  F --> F3[Preference: hard negatives]
  F --> F4[Target cleanup: final-answer-only]

  F1 --> G[CPU Gate]
  F2 --> G
  F3 --> G
  F4 --> G

  G --> G1[Sem leakage]
  G --> G2[Dataset correto]
  G --> G3[Conteudo correto]
  G --> G4[Targets limpos]
  G --> G5[Comparativo vs baseline]
  G --> G6[FinOps kill-switch]

  G -- falha --> H[Error ledger]
  H --> H1[Diagnostico causal]
  H1 --> H2[Prompt externo para APIs]
  H2 --> F

  G -- passa --> I[Dataset candidato versionado]
  I --> I1[JSONL train / val]
  I --> I2[Manifest]
  I --> I3[HF dataset publish]

  I --> J[Pre-paid Integration Gate]
  J --> J1[Launcher correto]
  J --> J2[HF paths corretos]
  J --> J3[Hashes batem]
  J --> J4[Adapter inicial correto]
  J --> J5[Prompt alinhado ao target]
  J --> J6[H200 <= 1h]
  J --> J7[Checkpoint/eval cedo]

  J -- falha --> H
  J -- passa --> K[HF Job curto]

  K --> K1[Preinstall gate]
  K --> K2[Artifact gate]
  K --> K3[Postinstall gate]
  K --> K4[Tokenization / offset-mask gate]
  K --> K5[Adapter tensor mapping]
  K --> K6[Trainable param ratio]

  K6 --> L[Baseline internal eval]
  L --> M[Checkpoint 3 eval]
  M -- piora ou nao melhora --> N[Cancelar job por FinOps]
  N --> H

  M -- melhora --> O[Checkpoint candidato]
  O --> P[Weak Eval Gate]
  P --> P1[Total > 192/315]
  P --> P2[equation > 56/155; ideal >= 60]
  P --> P3[bit >= 136/160]
  P --> P4[truncation = 0]

  P -- falha --> H
  P -- passa --> Q[Full official-like gate]
  Q --> Q1[Full > 823/947]
  Q --> Q2[Sem regressao relevante por familia]

  Q -- falha --> H
  Q -- passa --> R[Package adapter-only]
  R --> S[Notebook/package release gate]
  S --> T[Kaggle submit]
  T --> U[Leaderboard feedback]
  U --> H
```

## Componentes Obrigatorios

| Camada | Componente | Funcao |
|---|---|---|
| Regras | Contrato adapter-only | Impede que solver/verifier runtime vire submissao invalida |
| Dados | Hash/family/schema gates | Garante que o treino usa exatamente os dados esperados |
| Dados | Anti-leakage | Bloqueia uso de weak/full ou IDs proibidos em treino |
| Hipoteses | Literatura/OpenRouter/Kaggle | Gera ideias, mas nao promove nada sem gate local |
| CPU Gate | DSL/probes/audits | Prova sinal barato antes de GPU |
| Error Ledger | `KG1_ERROR_LEDGER_2026_05_15.md` | Guarda falhas, causas, regras preventivas e prompts externos |
| Integration Gate | `scripts/kg1_pre_paid_job_integration_gate.py` | Valida pecas, dataset, launcher e kill-switch antes de job pago |
| HF Job | H200 curto | Executa apenas smoke com limite de 1 hora |
| FinOps | Checkpoint 3 | Cancela cedo se a metrica interna piorar |
| Weak Gate | promover somente se `>192/315`, `equation>56/155`, `bit>=136/160`, `truncation=0`; alvo operacional `equation>=60/155` | Primeiro criterio real para promover checkpoint |
| Full Gate | package novo exige official-like `>=831/947` e `truncated<=4`; qualquer ranking delta precisa superar o historico `823/947` | Criterio final antes de package/submit |

## Estado Atual Das Linhas Historicas V439/V440

- Dataset V439: final-answer-only, `109` train, `24` validation.
- Train SHA: `bc032da2f7cada19aef295aa91aef6098e03c7b85215e7729f1ddd71b3e5079a`.
- Val SHA: `57321347f9293e9c0f2f17e6c9de1d88f1246fee4154125574b2e60251aee3a6`.
- V438 audit sobre V439: `hf_gpu_allowed_for_same_objective=true`.
- Integration gate local: aprovado, zero findings.
- HF job V440: H200, max `12` steps, checkpoint/eval no step `3`, timeout `3600s`.
- Resultado V440: checkpoint-3 empatou baseline interno `8/24`, equation `7/22`, bit `1/2`; job cancelado por FinOps.
- Status: linha historica fechada; nao repetir `mean_nll` final-answer-only.
- Proxima rota arquitetural ativa: CPU-only, criar dataset limpo novo somente
  depois de gate de simbolos/contradicoes/referencia; nenhum job GPU ativo.
- V447/V461/V463/V464/V468 e adapters V448/V465/V469 estao quarentenados e
  nao podem alimentar treino, eval, package ou submit.

## O Que Nao Promove Submit

- Loss/eval_loss isolado.
- Preference accuracy interna sem weak/full gain.
- Teacher/solver/verifier fora do adapter.
- Postprocessor ou runtime code.
- Dataset sem manifest, hash e gate.
- Mais epochs ou LR sweep sem novo sinal CPU.
