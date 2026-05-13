# KG1 V327 - Kaggle Rules Audit

Date: 2026-05-13

Source acquisition:

- Official Kaggle competition pages were extracted through the authenticated Kaggle API into `competition_pages.json`.
- Kaggle CLI was used to verify competition files, submissions, leaderboard, public notebooks, and Kaggle model listings.
- No notebook was created or modified in this audit, so `scripts/notebook_release_gate.py` is not applicable for this change.

## Official Requirements Confirmed

- Submission format: `submission.zip`.
- Submission payload: a LoRA adapter for `NVIDIA Nemotron-3-Nano-30B`.
- Required adapter file: `adapter_config.json`; package must also include adapter weights.
- Maximum LoRA rank: `32`.
- Official inference path: vLLM.
- Official metric: accuracy after final answer extraction, prioritizing `\boxed{}`.
- Official runtime parameters include `max_tokens=7680`, `temperature=0.0`, `top_p=1.0`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`, and `max_model_len=8192`.
- Competition data files are `train.csv` and `test.csv`; scored test replaces the sample test.
- Submission limit: maximum `5` submissions per day; up to `2` final submissions.
- Prize eligibility requires a public Kaggle notebook and write-up documenting methods, datasets, and techniques.
- External data/tools are allowed only when publicly available/equally accessible at no cost, or otherwise reasonably accessible with minimal cost and reproducible enough for sponsor review.
- Competition data must not be redistributed to non-participants.
- Submissions must not use hand labeling or human prediction of validation/test records.

## Compliance Assessment

Current packageable path is compliant when it is adapter-only:

- V291 package was `submission.zip`, LoRA rank `32`, `adapter_config.json` plus `adapter_model.safetensors`.
- V291/V326 evaluation jobs use official-like vLLM inference settings and boxed-answer prompting.
- HF jobs used for weak/full gates do not submit to Kaggle automatically.
- The V274/V275 solver/postprocessor gain is correctly treated as non-submit-ready unless absorbed into a LoRA adapter, because the official submission accepts an adapter package, not an external runtime solver.
- The current anti-leakage gates by `id`, prompt hash, family counts, tokenization, truncation, and weak/full contracts are necessary and aligned with the rules.

Non-compliant or blocked paths:

- Any CSV prediction file, standalone solver, postprocessor, verifier, GGUF model, or external service endpoint is not a valid Kaggle submission format for this competition.
- Gated/private datasets such as `andy279/*` remain blocked until access, terms, license, and equal-access/reasonableness are resolved.
- External API outputs from paid/closed models should be treated as research guidance unless exact prompts, models, costs, and reproducibility are documented and the data does not contain hidden-test information.
- Do not publish or redistribute Kaggle train/test-derived data outside allowed competition contexts.

## V326 Result Incorporated

V326 weak evaluation completed on HF:

- Job: `https://huggingface.co/jobs/felipesp1983/6a047ae74f7d89ac5e217d81`
- Upload commit: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v326-nemo-a100-equation-bit-replay-v290ckpt6/commit/edcac4df8b02cf7dccba12c62c09f89ca2b50241`
- `checkpoint-2`: `190/315`, `equation_transform=56/155`, `bit_manipulation=134/160`, `truncated=0`.
- `checkpoint-4`: `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=0`.
- `checkpoint-6`: `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=0`.
- `checkpoint-8`: `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=0`.

Decision: reject V326 for full eval, package, and Kaggle submit. It did not absorb the V324/V325 equation solver gain and regressed bit by `-1` against the best adapter-only baseline `192/315`, `equation=56`, `bit=136`.

## Operational Rule

Before any new HF GPU job:

1. CPU gate must show a new verified signal, not just a new weighting recipe.
2. The signal must target `equation_transform>56` while preserving `bit_manipulation>=136`.
3. Any dataset must pass anti-leakage, tokenization/offset-mask, family-count, and weak-contract gates.
4. First checkpoint must be evaluated before continuing spend.
5. No full eval/package/submit unless weak gate improves adapter-only behavior without family regression.
