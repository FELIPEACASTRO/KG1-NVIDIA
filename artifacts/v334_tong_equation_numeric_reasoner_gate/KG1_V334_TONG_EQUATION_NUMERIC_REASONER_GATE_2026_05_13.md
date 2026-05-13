# KG1 V334 Tong Equation Numeric Reasoner Gate - 2026-05-13

## Objective

Test whether the public Tong `reasoners/equation_numeric.py` can be used directly as a runtime override for the current weak equation misses.

## Inputs

- Tong repo: `https://github.com/tonghuikang/nemotron`
- Pinned commit: `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85`
- Baseline CSV: `hf://felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv`
- Weak row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`

## Results

- Baseline: `192/315`, `equation=56/155`, `bit=136/160`.
- Direct Tong equation replacement: `192/315`, `equation=56/155`, `bit=136/160`.
- Parsed equation rows: `155/155`.
- Tong trace statuses: `65` ok, `90` no trace.
- On baseline equation misses: `0` gains, `16` wrong candidates.
- Direct replacement gate: **blocked**.

## Decision

Do not use Tong `equation_numeric.py` as a direct postprocessor or training authorization. Its value is as a source of DSL operations. The verified path remains V324/V329-style guarded classes with conflict tracking and no-loss promotion.

## Artifacts

- Script: `scripts/run_v334_tong_equation_numeric_reasoner_gate.py`
- Manifest: `artifacts/v334_tong_equation_numeric_reasoner_gate/20260513T172300Z/v334_tong_equation_numeric_reasoner_gate_manifest.json`
- Detail CSV: `artifacts/v334_tong_equation_numeric_reasoner_gate/20260513T172300Z/v334_tong_equation_numeric_reasoner_gate_tong_equation_detail.csv`
