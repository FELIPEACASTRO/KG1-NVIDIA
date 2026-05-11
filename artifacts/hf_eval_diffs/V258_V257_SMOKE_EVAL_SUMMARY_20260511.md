# V258 V257 Smoke Weak Eval Summary - 2026-05-11

## Result

V258 evaluated the V257 H200 4-step smoke adapters on the V221 315-row weak contract. The best candidate was `v257_checkpoint_4_v221_contract`:

- overall: `192/315` (`60.95%`)
- equation_transform: `56/155` (`36.13%`)
- bit_manipulation: `136/160` (`85.00%`)
- truncated: `1`
- weak gate: **not passed** (`total>=193`, `equation>=60`, `bit>=133`, `trunc<=3`)

## Baseline Comparison

Against V256 HF V226 checkpoint1 (`191/315`, equation `56`, bit `135`, trunc `1`), V258 checkpoint-4 is +1 total, +1 bit, +0 equation, +0 truncation. The single correctness gain is row `4ada9150`, family `bit_manipulation`, expected `01111011`, V256 predicted `01111111`, V258 predicted `01111011`.

Against V255 HF V194 (`191/315`, equation `56`, bit `135`, trunc `1`), the same net family delta holds, but the row-level swaps must be read from `v255_v194_vs_v258_ckpt4_correctness_deltas_20260511.csv`.

## Interpretation

This is the first HF-only V249 smoke signal that improves the operational weak total while preserving the recovered equation score and restoring bit to the historical 136 guardrail. It still does not authorize full eval because equation remains the bottleneck (`56`, target `60`). The next experiment should stay small and equation-targeted; `checkpoint-4` is a reasonable seed candidate, but only behind the same preflight gate and immediate weak eval.

## Artifacts

- V258 job: https://huggingface.co/jobs/felipesp1983/6a0136a1317220dbbd1a77e5
- V258 upload commit: https://huggingface.co/felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke/commit/be437d3f431e0c46998243e573cda53fa68f26c6
- Summary JSON: `artifacts/hf_eval_diffs/v258_v257_smoke_eval_summary_20260511.json`
- V256 delta CSVs: `v256_v226_vs_v258_ckpt4_*_20260511.csv`
- V255 delta CSVs: `v255_v194_vs_v258_ckpt4_*_20260511.csv`
