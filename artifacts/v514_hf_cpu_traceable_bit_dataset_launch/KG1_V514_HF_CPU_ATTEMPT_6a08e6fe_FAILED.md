# V514 HF CPU Attempt 6a08e6fe Failed

- Job: `https://huggingface.co/jobs/felipesp1983/6a08e6fe3308d79117b915bb`
- Commit: `bcfd28b0f0fa399da2d7c36e4a2e43b935e5f0b9`
- Flavor: `cpu-upgrade`
- Result: `ERROR`

## Root Cause

The V514 builder imports `scripts/run_v296_bit_stride_solver_audit.py` to reuse `solve_stride`. That module imports `pandas`, but the HF CPU launcher installed only `huggingface_hub`, `transformers`, and `tokenizers`.

Error:

```text
ModuleNotFoundError: No module named 'pandas'
```

## Fix

Add `pandas>=2.0.0` to the V514 HF CPU launcher dependencies before relaunching.

## Impact

No training, packaging, evaluation, or submit ran. This was a cheap CPU dependency failure only.
