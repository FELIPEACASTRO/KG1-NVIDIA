# V274 HF CPU numeric operator override audit

Run date: 2026-05-11

Branch: `v230-v226-complementarity`

Local commit used for launch: `73070585d520b872f7cda10c5f1affa0ac1da160`

HF jobs:

- Failed unauthenticated preflight: `https://huggingface.co/jobs/felipesp1983/6a01b42caff1cd33e8f338bf`
  - Status: `ERROR`
  - Reason: the private/gated V259 prediction repo required `HF_TOKEN`.
- Authenticated validation: `https://huggingface.co/jobs/felipesp1983/6a01b475aff1cd33e8f338cf`
  - Status: `COMPLETED`
  - Runtime: `13s` total, `8s` running, `cpu-basic`
  - Secret used: `HF_TOKEN`

Validated metrics from the authenticated job log:

- `weak_gate.pass = true`
- override total: `196/315`
- `equation_transform = 60/155`
- `bit_manipulation = 136/160`
- truncation: `0`
- `wrong_on_baseline_misses = 0`

Decision:

- V274 is reproducible in HF CPU.
- Next step is V275: package the same label-free postprocessor for deployable inference/full-eval gating before any H200 run.
