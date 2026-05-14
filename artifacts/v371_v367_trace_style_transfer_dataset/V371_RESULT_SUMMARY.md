# V371 V367 trace-style transfer dataset

Generated: 2026-05-14

## Result

- Train rows: `1128`.
- Validation rows: `282`.
- Train trace-style assistant rows: `1128/1128`.
- Validation trace-style assistant rows: `282/282`.
- Train boxed-suffix rows: `1128/1128`.
- Validation boxed-suffix rows: `282/282`.
- Train/validation prompt overlap: `0`.

## Decision

Dataset built for CPU/tokenization review only. HF remains blocked until V286 real tokenization passes and the roadmap explicitly accepts a new smoke test.

## Local artifacts

- Manifest: `artifacts\v371_v367_trace_style_transfer_dataset\20260514T_cpu_gate\v371_v367_trace_style_transfer_manifest.json`
- Train: `artifacts\v371_v367_trace_style_transfer_dataset\20260514T_cpu_gate\v371_v367_trace_style_transfer_train.jsonl`
- Validation: `artifacts\v371_v367_trace_style_transfer_dataset\20260514T_cpu_gate\v371_v367_trace_style_transfer_val.jsonl`
