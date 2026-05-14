# V361 Boxed-Only Transfer Dataset Summary

Generated: 2026-05-14

## Status

V361 CPU dataset and real tokenization gate passed.

## Purpose

V360 showed that V359 trained long `Rule/Check examples/Final answer` completions and did not use the preference hard negatives. V361 keeps the verified V358 prompts and answers, but changes the supervised target to exactly one boxed answer:

`\boxed{answer}`

## Dataset

- Train rows: `1152`.
- Validation rows: `288`.
- Family: `bit_manipulation` only.
- Unique rules: `15`.
- Train subcategories: `832` ternary, `320` binary replay.
- Validation subcategories: `208` ternary, `80` binary replay.
- Assistant boxed-only rows: `1152/1152` train, `288/288` validation.
- Assistant length: `16` chars for every row.
- Train/validation prompt overlap: `0`.
- Preference rows regenerated: `2304` train, `576` validation.

## Hashes

- Train SHA256: `be742d7a82bf1c98f33d67bed8903006068c139ab74f798055fcc7d435ffa4db`.
- Validation SHA256: `4c93766e7fae72da14f879177e15c3c6300b7991e4efc8fba4d7fe75d3df5332`.
- Preference train SHA256: `f7f2e11540adbf16bcc93cc8f88a8f3f9c432086df59add4b4132daa821b4b06`.
- Preference validation SHA256: `565a079d523f74cafe76e6e6161fd53f86d64ac90a684215d133b4b435763df3`.

## Tokenization Gate

- Gate: `scripts/run_v286_generic_tokenization_gate.py`.
- Mode: `boxed_only`.
- Real tokenizer: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Prompt truncation: `0`.
- Completion token drops: `0`.
- Fallback masks: `0`.
- Train token max: `286`.
- Validation token max: `286`.
- Loss tokens: `15`.

## Decision

V361 repairs the completion-format issue found by V360. It does not prove adapter-only ACC gain by itself and does not touch `equation_transform`. A future HF smoke may only be a short `max_steps<=2` test with first-checkpoint weak eval and FinOps kill-switch.
