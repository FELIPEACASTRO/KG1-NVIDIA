# V275 HF CPU deployable postprocessor gate

Run date: 2026-05-11

Branch: `v230-v226-complementarity`

HF job: `https://huggingface.co/jobs/felipesp1983/6a01b708aff1cd33e8f338ed`

Status:

- `COMPLETED`
- `cpu-basic`
- `12s` total, `7s` running
- Secret used: `HF_TOKEN`

Validated from job inspect/log tail:

- `weak_gate.pass = true`
- source guard `forbidden_hits = []`
- source guard forbidden terms: `answer`, `correct`, `verify_answer`, `solution`
- postprocessor module SHA256: `f992ba7c4b4d2eb070e09cc59b11aae899872b1f9f18f82bc328fe20f8db4d2d`
- `wrong_on_baseline_misses = 0`

The local V275 manifest in `artifacts/hf_cpu_runs/v275_deployable_postprocessor_gate_20260511T1210Z/` contains the full metric payload:

- postprocessed weak score: `196/315`
- `equation_transform = 60/155`
- `bit_manipulation = 136/160`
- truncation: `0`
- applied rows: `4`
- gains: `4`
- losses: `0`

Decision:

- V275 is ready for one guarded full-eval experiment if the full evaluation CSV can be accessed from HF and the final competition path can carry a postprocessor.
