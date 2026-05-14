# V360 V359 Transfer Failure Audit

Generated: 2026-05-14T10:27:19.186787+00:00

## Decision

- Status: `v360_blocks_more_hf_on_v358_v359`
- HF GPU allowed: `False`
- Reason: V359 checkpoint-2 regressed to 190/315, bit=134/160, truncated=1; V358 did not transfer V357 CPU gains.
- Next action: Build V361 CPU-gated answer-first/boxed-only transfer data or return to equation DSL. Do not launch another HF job from V358/V359 artifacts.

## Measured Evidence

- V357 CPU teacher: `{"accuracy": 0.6793650793650794, "correct": 214, "family": {"bit_manipulation": {"correct": 151, "rows": 160}, "equation_transform": {"correct": 63, "rows": 155}}, "rows": 315}`
- V359 checkpoint-2: `190/315`, equation `56/155`, bit `134/160`, truncated `1`.
- V358 train rules: `15` rules, `1152` rows.
- V358 validation rules: `15` rules, `288` rows.
- V358 preference rows: train `2304`, val `576`, used by V359 launcher `False`.

## Findings

- `blocking` V359 did not transfer the V357 CPU gain.
  Evidence: V357 CPU summary={"accuracy": 0.6793650793650794, "correct": 214, "family": {"bit_manipulation": {"correct": 151, "rows": 160}, "equation_transform": {"correct": 63, "rows": 155}}, "rows": 315}; V359 checkpoint-2 total=190, equation=56, bit=134, truncated=1.
  Action: Do not full-eval, package, submit, or continue V359 checkpoints without a new CPU-gated dataset format.
- `high` The hard-negative preference files were not used by the V359 SFT launcher.
  Evidence: Launcher references the SFT train JSONL, but not the preference train/val JSONL.
  Action: Either train with a real preference objective or remove preference artifacts from the active plan.
- `high` V358 is narrow: it repeats 15 verified rules instead of teaching a broad bit solver.
  Evidence: train unique_rule_slugs=15; val unique_rule_slugs=15.
  Action: Next dataset must either be answer-only replay for exact rules or expand rule coverage before another GPU run.
- `medium` Completion format differs from weak-eval prompt intent.
  Evidence: assistant_only_boxed_rows=0 while all rows include Rule/Check examples/Final answer text.
  Action: Test an answer-first or boxed-only V361 dataset in CPU/token gates before HF.
- `info` V358 passed tokenization, so the failure is not explained by training truncation.
  Evidence: {"train": {"completion_tokens_dropped": 0, "fallback_masks": 0, "family_summary": {"bit_manipulation": {"loss_token_max": 245, "loss_token_min": 236, "loss_token_p50": 243, "rows": 1152, "token_max": 524, "token_p50": 522, "token_p90": 524, "token_p99": 524}}, "loss_token_max": 245, "loss_token_min": 236, "loss_token_p50": 243, "offset_masks": 1152, "prompt_truncated": 0, "prompt_truncation_rate": 0.0, "rows": 1152, "token_max": 524, "token_mean": 520.472, "token_min": 515, "token_p50": 522, "token_p90": 524, "token_p99": 524}, "validation": {"completion_tokens_dropped": 0, "fallback_masks": 0, "family_summary": {"bit_manipulation": {"loss_token_max": 245, "loss_token_min": 236, "loss_token_p50": 243, "rows": 288, "token_max": 524, "token_p50": 522, "token_p90": 524, "token_p99": 524}}, "loss_token_max": 245, "loss_token_min": 236, "loss_token_p50": 243, "offset_masks": 288, "prompt_truncated": 0, "prompt_truncation_rate": 0.0, "rows": 288, "token_max": 524, "token_mean": 520.472, "token_min": 515, "token_p50": 522, "token_p90": 524, "token_p99": 524}}
  Action: Keep token gates, but treat ACC regression as a learning/objective/format issue.
- `blocking` V359 bit-only data cannot improve equation_transform.
  Evidence: V358 train/val family counts are only bit_manipulation; measured equation stayed at 56/155.
  Action: Equation must continue through DSL/verifier/teacher data, not this bit-only SFT route.

## Promotion Rule

No full eval, package, Kaggle submit, or additional HF run is allowed from V358/V359 unless a new CPU-gated dataset explains the failure and preserves the V357 gains before paid training.
