# Execution Roadmap V216 Public Adapter Triage - 2026-05-07

## Objective

Raise the public score from the protected `0.86` baseline only if a candidate beats the local proxy gates. Do not spend Kaggle submissions on candidates that fail local validation.

## Protected Baseline

- Public score: `0.86`
- Local full proxy: `822/947 = 0.868004`
- Weak-family baseline: `190/315`
- Strong-family baseline: `632/632`
- Best known safe artifact: V194/V199B lineage
- Rejected path: V214, because weak eval was `137/315`, with `55/315` truncations

## Submission Policy

- No automatic Kaggle submission.
- Keep at least one submission slot unused.
- Submit only after human approval.
- Submit only if full gate beats the protected baseline.

## Gates

1. Structure gate:
   - adapter directory exists
   - `adapter_config.json` exists
   - `adapter_model.safetensors` or `adapter_model.bin` exists
   - LoRA rank/target modules are readable

2. Weak screen:
   - Evaluate only `bit_manipulation` and `equation_transform`
   - Must beat `190/315`
   - Preferred threshold: `>=196/315`
   - Any truncation spike is a rejection signal

3. Full 947-row gate:
   - Run only for weak-positive candidates
   - Must beat `822/947`
   - Preferred threshold: `>=828/947`
   - No family regression accepted unless net gain is clearly positive and manually approved

4. Submission gate:
   - Human must review full report, predictions, per-task CSV, and adapter source
   - Human must explicitly approve the Kaggle submission

## Execution Phases

### Phase 0 - Preserve Baseline

Keep V194/V199B unchanged. Do not overwrite or retrain it. It remains the fallback if every public adapter fails.

The V207B notebook is now self-bootstrapping for the baseline gate. If the expected V207A Drive artifacts are missing, it downloads the validated baseline exports from the branch and reconstructs:

- `official_train_seed42_stratified10_val.csv`
- `v194_baseline_predictions.csv`
- `v194_baseline_per_task.csv`
- `v194_baseline_eval_report.json`

This avoids requiring a separate V207A notebook run before public-adapter triage.

The notebook no longer clones the repository at runtime. It creates `/content/kg1`, writes the audited metric scripts directly into that workspace, compiles them, and uses those embedded scripts for all evaluations. The vLLM runtime is pinned to `vllm==0.20.1` by default because that is the version already exercised on the H100 logs with the DeepGEMM safety hotfix.

### Phase 1 - Download Public Adapters

Use only public Kaggle model assets:

- Huikang `default` versions `20` through `27`
- Kienngx variants referenced by public notebooks:
  - `1200samples-cot-1e-5`
  - `1200samples-cot-5e-5`
  - `cot-labels-3000samples`
  - `600-samples-packing-false`
  - `1800s-lora-rank32-false`

The notebook downloads directly to Drive under:

`/content/drive/MyDrive/KG1_PUBLIC_ADAPTERS/{candidate}/adapter`

Stop condition: if Kaggle credentials are missing, add `kaggle.json` to Drive and rerun the download cell.

### Phase 2 - Audit Structures

Run the structure audit before any model evaluation. Reject broken or incompatible adapters without spending H100 time.

### Phase 3 - Weak Screen

Run the weak-family screen. This is the cheapest reliable filter for score movement.

Stop condition: if no candidate beats `190/315`, submit nothing and move to data/training research.

### Phase 4 - Full Gate

Run full 947-row evaluation only for candidates promoted by the weak screen.

Stop condition: if no candidate beats `822/947`, submit nothing.

### Phase 5 - Human Approval

If a candidate beats the full gate, review:

- `v207b_full_gate_results.csv`
- candidate eval report JSON
- candidate predictions CSV
- per-task CSV
- adapter source/ref

Only then approve one Kaggle submission.

## Active Notebook

Notebook:

`notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb`

Exact Colab URL on branch `v207b-external-triage`:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v207b-external-triage/notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb`

The remote branch includes this notebook and roadmap.

## Required Human Interaction

1. Add Kaggle API token if the notebook cannot find it:
   - `/content/drive/MyDrive/kaggle.json`
   - or `/content/drive/MyDrive/KG1_SECRETS/kaggle.json`
2. Run the Colab notebook on H100/A100.
3. Approve or reject submission after full-gate results.
