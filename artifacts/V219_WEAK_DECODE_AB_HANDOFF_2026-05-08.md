# V219 weak decode A/B handoff - 2026-05-08

Notebook:

`notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb`

Colab URL after the notebook is pushed to branch `v219-decode-ab`:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v219-decode-ab/notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb`

## Purpose

V218 showed that `--disable-thinking` plus `max_tokens=1024` removed truncation
but collapsed weak accuracy to 18/315. V219 is a weak-only A/B diagnostic that
keeps thinking enabled and tests whether a lower but not tiny generation budget
can reduce truncation without destroying accuracy.

## What V219 runs

- Adapter A: V217 final adapter.
- Adapter B: protected V194 adapter.
- Split: 315 weak rows only (`equation_transform` + `bit_manipulation`).
- Default decode:
  - thinking enabled;
  - `max_tokens=3584`;
  - `max_model_len=8192`;
  - `max_num_seqs=64`;
  - `warmup_rows=0`;
  - boxed final-answer prompt suffix.

## Gates

Full eval remains blocked unless the best weak candidate reaches all gates:

- weak total >= 193;
- `equation_transform` correct >= 60;
- `bit_manipulation` correct >= 133;
- truncated <= 3.

Even if weak passes, full eval is off by default. To explicitly allow full eval:

```python
import os
os.environ["KG1_V219_RUN_FULL_IF_GATE"] = "1"
```

Do not enable this until the weak gate output has been reviewed.

## Validation already run locally

Commands:

```powershell
python -m py_compile scripts\notebook_release_gate.py scripts\build_v219_weak_decode_ab_colab.py scripts\evaluate_lora_adapter.py
python scripts\build_v219_weak_decode_ab_colab.py
python scripts\notebook_release_gate.py notebooks\KG1_V218_DECODE_RESCUE_COLAB.ipynb notebooks\KG1_V219_WEAK_DECODE_AB_COLAB.ipynb --output-json artifacts\notebook_release_gate\v218_v219_report.json
```

Result: gate passed for both V218 and V219.

## Files changed for V219

- `scripts/evaluate_lora_adapter.py`
  - added `--max-model-len`;
  - added `--warmup-rows`;
  - logs `warmup_rows = 0` when explicit warmup is skipped.
- `scripts/build_v219_weak_decode_ab_colab.py`
  - creates the V219 notebook with progress logs in every operational cell.
- `scripts/notebook_release_gate.py`
  - added strict V219 checks so the failed V218 no-thinking decode path cannot be reintroduced.
- `notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb`
  - generated notebook; committed clean with no outputs.

## Human action needed

The notebook URL only works after these files are pushed to GitHub branch
`v219-decode-ab`. The local worktree currently has unrelated dirty/conflicting
files, so push/commit should be done carefully with selected paths only after
the existing unrelated git state is resolved.
