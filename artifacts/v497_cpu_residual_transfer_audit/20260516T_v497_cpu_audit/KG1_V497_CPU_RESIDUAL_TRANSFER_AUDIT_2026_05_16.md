# KG1 V497 CPU Residual Transfer Audit

Generated UTC: `2026-05-16T18:08:04.056674+00:00`

## Decision

- Decision: `do_not_promote_v496_or_repeat_h200_sft`
- Reason: v496_total_projection=191; baseline_total=192; v496_bit_losses=2; v324_cpu_gain=4
- Next action: Implement a CPU teacher/verifier that explains equation gains without bit regression before any paid GPU job.

## Metrics

| Metric | Value |
|---|---:|
| Baseline total correct | 192 |
| Equation miss rows | 99 |
| V324 verified equation gain | 4 |
| V324 projected total | 196 |
| V496 changed rows | 17 |
| V496 verified equation gain | 1 |
| V496 bit loss rows | 2 |
| V496 total projection from diff | 191 |

## Top Residual Equation Clusters

| Cluster | Rows | V324 gains | V496 gains | Unresolved |
|---|---:|---:|---:|---:|
| `equation_symbolic_punct|alen=3|qlen=5|qops_seen=0|ans_subset_q=0|ans_subset_out=0|ans_subseq_q=0` | 11 | 0 | 0 | 11 |
| `equation_symbolic_punct|alen=4|qlen=5|qops_seen=0|ans_subset_q=0|ans_subset_out=0|ans_subseq_q=0` | 11 | 0 | 0 | 11 |
| `equation_symbolic_punct|alen=2|qlen=5|qops_seen=1|ans_subset_q=0|ans_subset_out=1|ans_subseq_q=0` | 9 | 0 | 0 | 9 |
| `equation_symbolic_punct|alen=2|qlen=5|qops_seen=0|ans_subset_q=0|ans_subset_out=1|ans_subseq_q=0` | 8 | 0 | 0 | 8 |
| `equation_symbolic_punct|alen=3|qlen=5|qops_seen=1|ans_subset_q=0|ans_subset_out=0|ans_subseq_q=0` | 8 | 0 | 0 | 8 |
| `equation_symbolic_punct|alen=3|qlen=5|qops_seen=0|ans_subset_q=0|ans_subset_out=1|ans_subseq_q=0` | 7 | 0 | 0 | 7 |
| `equation_symbolic_punct|alen=2|qlen=5|qops_seen=0|ans_subset_q=0|ans_subset_out=0|ans_subseq_q=0` | 6 | 0 | 0 | 6 |
| `equation_numeric_operator|alen=3|qlen=5|qops_seen=0|ans_subset_q=0|ans_subset_out=0|ans_subseq_q=0` | 4 | 0 | 0 | 4 |

## Bit Guardrail Failures

| id | answer | V496 prediction | failure |
|---|---|---|---|
| `8740ed31` | `01101000` | `01111000` | `binary_wrong_value` |
| `59bee375` | `10010101` | `2` | `non_binary_or_wrong_length` |

## Implementation Consequence

Do not launch another H200 SFT run from V475/V390/V326 directly. The next executable step is a CPU teacher/verifier redesign that explains at least four equation misses while preserving the exact bit guardrail before any paid GPU job.
