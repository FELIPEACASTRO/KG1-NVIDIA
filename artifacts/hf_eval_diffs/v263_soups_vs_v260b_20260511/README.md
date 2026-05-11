# V263 adapter soups vs V260B weak eval

Generated: 2026-05-11

Purpose: compare the V263 adapter soup weak predictions against the best V260B
checkpoint-4 weak predictions under the canonical V221 contract.

Inputs:

- V260B best:
  `felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv`
- V263 soups:
  `felipesp1983/kg1-nemotron-lora-v262-adapter-soups/evals/v263-h200-v262-soups-v221contract-eval-20260511T050000Z/eval/.../*_predictions.csv`

Summary:

| Candidate | Aligned rows | Gain vs V260B | Loss vs V260B | Oracle total | Bit oracle | Equation oracle |
|---|---:|---:|---:|---:|---:|---:|
| `v263_soup_v226_050_v257_050` | 315 | 1 | 1 | 193 | 137 | 56 |
| `v263_soup_v226_050_v259_050` | 315 | 0 | 1 | 192 | 136 | 56 |
| `v263_soup_3way` | 315 | 0 | 2 | 192 | 136 | 56 |

Interpretation:

- V263 does not add any new `equation_transform` correct row over V260B.
- The best soup only swaps one bit row: one gain and one loss.
- The row-level oracle reaches total `193`, but only by bit manipulation and still
  stays at `56/155` equation. It does not solve the gate, because the gate needs
  `equation_transform >= 60`.
- Do not spend more H200 on adapter soups without a new preflight proving a path
  to at least `+4` equation rows.

Artifacts:

- `v263_soups_vs_v260b_summary.csv`
- `v263_soups_vs_v260b_correctness_matrix.csv`
