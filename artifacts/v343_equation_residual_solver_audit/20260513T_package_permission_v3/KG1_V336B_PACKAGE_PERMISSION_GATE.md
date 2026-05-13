# KG1 V336B - Package Permission Gate

Generated at UTC: `2026-05-13T22:22:09.141535+00:00`

## V336A Input

- Integrated weak: `199/315`.
- Equation: `63/155`.
- Bit: `136/160`.
- Losses: `0`.

## Official/Local Evidence

- Official adapter-only requirement confirmed: `True`.
- Local V291 zip adapter-only: `True`.
- Package script rejects prediction postprocessor: `True`.

## Decision

- `solver_verifier_direct_package_blocked_adapter_only_required`
- Reason: Official extracted pages require submission.zip containing a rank<=32 LoRA adapter; the valid local package contains only adapter_config.json and adapter_model.safetensors; the package script rejects prediction_postprocessor, keeps Kaggle submit hard-locked, and the reference package rank is <=32. V336A gain must be transferred into adapter-only behavior before Kaggle submit.
- Next action: Proceed to V337D minimal transfer dataset; do not submit solver/verifier package.
