# KG1 V333 Tong Bit Reasoner Gate - 2026-05-13

## Objective

Measure whether the public Tong Hui Kang bit reasoner can safely improve the current KG1 weak baseline before spending GPU budget.

## Inputs

- Tong repo: `https://github.com/tonghuikang/nemotron`
- Pinned commit: `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85`
- Baseline CSV: `hf://felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv`
- Weak row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`

## Results

- Official train bit rows: `1602`.
- Tong reasoner: `1364/1602 = 85.1436%`.
- Current KG1 local bit solver: `1265/1602 = 78.9638%`.
- Tong vs current solver on train: `+157` gains, `58` losses, net `+99`.

Weak V221-contract audit:

- Baseline: `192/315`, `bit=136/160`, `equation=56/155`.
- Tong bit replacement: `192/315`, `bit=136/160`, `equation=56/155`.
- Tong changed two weak bit rows: `+1` gain (`8740ed31`) and `-1` loss (`ef2fe526`).
- Direct replacement gate: **blocked**.
- Teacher-trace signal: **present but small** (`1` weak gain row).

## Decision

Do not deploy Tong bit replacement directly and do not launch an HF GPU job from this signal alone. Use Tong as a verified teacher/fixture source for bit-pair/bitsum/stride traces, then require a label-free confidence rule or a no-loss CPU gate before promotion.

## Artifacts

- Script: `scripts/run_v333_tong_bit_reasoner_gate.py`
- Manifest: `artifacts/v333_tong_bit_reasoner_gate/20260513T171304Z/v333_tong_bit_reasoner_gate_manifest.json`
- Detail CSV: `artifacts/v333_tong_bit_reasoner_gate/20260513T171304Z/v333_tong_bit_reasoner_gate_tong_bit_detail.csv`
