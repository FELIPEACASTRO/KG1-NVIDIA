# KG1 V532 Kaggle Dataset Download Audit

Generated from Kaggle CLI dataset search/download. Raw downloaded files were stored only in a temp directory and deleted after analysis.

- official_train_exists: `True`
- official_train_rows: `9500`
- analyzed_entries: `18`

## Candidate Summary

| Candidate | Key files/signals | Decision |
|---|---|---|
| `konbu17/nemotron-bm-et-with-generated-cot` | train_Bit_Manipulation_with_generated_cot.csv rows=1506 fam={'bit_like': 1506, 'equation_like': 1506} cot=241 overlap=1506 mismatch=0; train_Equation_Transformation_with_generated_cot.csv rows=1466 fam={'equation_like': 1466, 'bit_like': 17} cot=170 overlap=1466 mismatch=0 | P0: inspect for CPU gate dataset/traces |
| `itskshivam/nemotron-equation-candidate-distill-v2` | README.md; bucket_summary.csv rows=5 fam={} cot=5 overlap=0 mismatch=0; clean_distill_bucket.csv rows=232 fam={'equation_like': 232} cot=232 overlap=232 mismatch=0; critique_summary.csv rows=14 fam={'equation_like': 14} cot=2 overlap=0 mismatch=0; equation_candidate_distill_v2_train.csv rows=535 fam={'equation_like': 535} cot=535 overlap=535 mismatch=0; summary.json | P1: equation candidate/routing evidence |
| `itskshivam/nemotron-equation-candidate-critic-v2` | README.md; bucket_summary.csv rows=10 fam={} cot=0 overlap=0 mismatch=0; candidate_pool.csv rows=51338 fam={'equation_like': 51338} cot=9 overlap=51338 mismatch=0; equation_candidate_critic_v2_train.csv rows=2951 fam={'equation_like': 2951} cot=2951 overlap=0 mismatch=0; summary.json | P1: equation candidate/routing evidence |
| `itskshivam/nemotron-equation-candidate-critique-router-v1` | README.md; candidate_pool.csv rows=51334 fam={'equation_like': 51334} cot=9 overlap=51334 mismatch=0; equation_candidate_critique_router_v1_train.csv rows=768 fam={'equation_like': 768} cot=768 overlap=0 mismatch=0; judge_topk_candidates.csv rows=6755 fam={'equation_like': 4115} cot=0 overlap=6755 mismatch=0; summary.json | P1: equation candidate/routing evidence |
| `sohamp13/nemotron-equation-candidate-selection-v2` | README.md; summary.json | P1: equation candidate/routing evidence |
| `furkankesen/equation-solver-swap-v1` | README.md; summary.json | P1: equation candidate/routing evidence |
| `mohammedtanvir/nemotron-reasoning-traces` | members=2 | triage |
| `adityakrishnanmohan/nemotron-synthetic-cot-gpt-clean` | README.md; manifest.json; solver_hard_triad_supplement.csv rows=1800 fam={'bit_like': 600, 'equation_like': 1200} cot=3 overlap=0 mismatch=0; solver_template_supplement.csv rows=720 fam={'bit_like': 120, 'equation_like': 240} cot=0 overlap=0 mismatch=0; synthetic_nemotron_cot_gpt_clean.csv rows=4801 fam={'bit_like': 821, 'equation_like': 1548} cot=1810 overlap=0 mismatch=0 | P0: inspect for CPU gate dataset/traces |
| `tkm2261/nemotron-exp043-bundle` | EXP/EXP043/solver_variants.py; EXP/EXP043/validation_manifest.csv rows=480 fam={'equation_like': 220} cot=0 overlap=480 mismatch=0; input/nemotron-cot-hybrid/exp043-c001-nemotron-gated-crypta-silver-v1/build_summary.json; input/nemotron-cot-hybrid/exp043-c001-nemotron-gated-crypta-silver-v1/train_cot_full.csv rows=8284 fam={'bit_like': 2500, 'equation_like': 4001} cot=31 overlap=6201 mismatch=6 | P0: inspect for CPU gate dataset/traces |
| `alexproger23/nemotron-validation-runtime` | data/reasoning/nemotron_cot_labels/final_Nemotron_training_data.csv rows=9500 fam={'bit_like': 1624, 'equation_like': 7073} cot=2179 overlap=9500 mismatch=7; data/reasoning/nemotron_cot_training_data/train_cot.jsonl rows=9110 fam={'bit_like': 1604, 'equation_like': 9110} cot=9110 overlap=None mismatch=None; data/reasoning/nemotron_cot_training_data/train_cot_full.json | P0: inspect for CPU gate dataset/traces |
| `dgxchen/nemotron-cot-tong` | less_cot.csv rows=6014 fam={'equation_like': 2926, 'bit_like': 1681} cot=6014 overlap=0 mismatch=0 | P0: inspect for CPU gate dataset/traces |
| `kienngx/nemotron-30b-competition-trainingdata-cot-labels` | members=1 | triage |
| `amanux/nemotron-cot-v1` | train_cot.jsonl rows=9500 fam={'bit_like': 1602, 'equation_like': 3157} cot=9500 overlap=None mismatch=None | P0: inspect for CPU gate dataset/traces |
| `damndeepesh/nemotron-cot-labels` | members=1 | triage |
| `samvalladares/huikang-nemotron-artifacts :: bit_manip_3input_synthesized_traces.jsonl` | download failed | skip |
| `samvalladares/huikang-nemotron-artifacts :: bit_manipulation_3input_traces.jsonl` | download failed | skip |
| `samvalladares/huikang-nemotron-artifacts :: corpus.jsonl` | download failed | skip |
| `samvalladares/huikang-nemotron-artifacts :: README.md` | download failed | skip |

## Operational Decision

- Do not run another generic LoRA/SFT from these datasets without CPU weak gate signal.
- Immediate P0 is to convert `konbu17/nemotron-bm-et-with-generated-cot` into a deterministic trace/probe comparison against current weak misses, not train blindly.
- Equation datasets from `itskshivam`, `sohamp13`, and `furkankesen` should feed a CPU candidate-ranking/verifier experiment first; promote only if it finds at least +1 confirmed weak row with zero regression, target +4 equation.
- Huikang artifact selected trace files are useful as bit reference, but only if their prompt/answer contracts align with our official prompts after normalization.

## Kaggle Topics/Comments Refresh

The public Kaggle search pages for:

- `https://www.kaggle.com/search?q=NVIDIA+Nemotron+in%3Atopics`
- `https://www.kaggle.com/search?q=NVIDIA+Nemotron+in%3Acomments`

render as a dynamic app shell, so the auditable path used here was Kaggle's
discussion topic endpoint through `scripts/audit_v512_kaggle_discussions.py`
and the locally collected challenge topic IDs.

Refresh output:

- topic IDs requested: `140`
- topics fetched: `58`
- posts scanned: `357`
- relevant hits: `238`
- summary: `artifacts/v532_kaggle_dataset_search_audit/discussion_refresh/V512_KAGGLE_DISCUSSION_AUDIT_SUMMARY.md`
- raw manifest: `artifacts/v532_kaggle_dataset_search_audit/discussion_refresh/v512_kaggle_discussion_manifest.json`

Highest-signal discussion findings:

1. `690307` confirms the bit path: bit-pair/bitsum/stride plus deterministic
   CoT, not brute-force expression enumeration.
2. `689915` confirms the training objective lesson: min-logprob / masked loss
   and trace design matter more than broad data volume.
3. `697491` and `693260` confirm that better synthetic solver coverage can
   lower leaderboard if the training objective saturates or destroys easy
   behavior.
4. `694556` confirms the equation/symbol family has duplicates, multiple
   plausible candidates, unknown operators, and weak identifiability; it needs a
   candidate/verifier/canonicalization lane.

## Net New Actionable Findings

The useful new material is not another adapter training mix by itself. The
useful material is a set of external candidate/verifier datasets that can be
used to build a CPU-first equation selection gate:

- `itskshivam/nemotron-equation-candidate-critic-v2`
  - `candidate_pool.csv`: `51338` equation candidate rows with
    `verifier_valid`, `verifier_score`, `failure_reason`,
    `best_program_family`, and `competition_match`.
  - `competition_match` is audit-only/quarantined. It must never be exported as
    a selector feature or training signal.
- `itskshivam/nemotron-equation-candidate-critique-router-v1`
  - `candidate_pool.csv`: `51334` rows with `canonicalization_status`,
    `profile_normalized_prediction`, `expression_family`,
    `sympy_parse_success`, and judge top-k assets.
- `sohamp13/nemotron-equation-candidate-selection-v2`
  - `train/eval`: 3-way answer-choice judge dataset with `policy_top_wrong_rate`
    about `0.618`, useful for ranking candidates rather than generating answers
    from scratch.
- `furkankesen/equation-solver-swap-v1`
  - `80` external solver-verified equation rows, using families:
    `concat`, `swap_concat`, `add`, `sub`, `abs_sub`, `mul`,
    `rev_both_add_rev`, `rev_both_abs_sub_rev`, `rev_both_mul_rev`.

For bit, the useful material remains deterministic trace data and solver logic:

- `konbu17/nemotron-bm-et-with-generated-cot`
  - `1506` official-overlap bit rows with answer mismatches `0`.
- `adityakrishnanmohan/nemotron-synthetic-cot-gpt-clean`
  - hard-triad supplement with bit/equation rows, but it must be triaged by CPU
    gate because it is synthetic and not direct proof of weak improvement.
- `samvalladares/huikang-nemotron-artifacts`
  - high-value reference, but Kaggle CLI file-level download failed. Full
    dataset URL for manual download:
    `https://www.kaggle.com/datasets/samvalladares/huikang-nemotron-artifacts`.

## Blockers

- `mohammedtanvir/nemotron-reasoning-traces/train_v14.csv` has official answer
  mismatches and is blocked as positive training data.
- `final_Nemotron_training_data.csv` mirrors show `7` mismatches in the scanned
  official overlap; do not use blindly as gold.
- `tkm2261/nemotron-exp043-bundle/train_cot_full.csv` has `6` mismatches in the
  scanned official overlap; useful as code/reference, not as unfiltered gold.
- Generic broad SFT from these bundles remains blocked until a CPU gate proves
  row-level gains with zero regressions.
- Weak row gains found by any diagnostic scan are not training rows. They may
  only motivate a source-only rule/canonicalization hypothesis after all
  weak/full overlaps are excluded.
