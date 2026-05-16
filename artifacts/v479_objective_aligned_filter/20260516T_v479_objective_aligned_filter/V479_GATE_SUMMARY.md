# V479 Gate Summary

Generated: 2026-05-16

## Purpose

V479 is a CPU-only correction after V478 found that V476 made
`bit_manipulation` only `0.9492%` of the effective weighted objective. V479
filters V475 to V324-evidenced equation rule classes and keeps bit replay as a
real objective component.

## Dataset

| Split | Rows | equation_transform | bit_manipulation |
|---|---:|---:|---:|
| train | 992 | 480 | 512 |
| validation | 248 | 120 | 128 |

Included subcategories:

- `equation_numeric_add_direct`
- `equation_numeric_colon_trailing_zero`
- `equation_numeric_minus_signed`
- `bit_guardrail_replay`

## Gates

| Gate | Result |
|---|---|
| Static safety | passed |
| V478 objective alignment, equal weights | passed |
| V286 real tokenization | passed |

Objective alignment with equal weights:

| Family | Effective train share |
|---|---:|
| bit_manipulation | 51.6129% |
| equation_transform | 48.3871% |

Tokenization:

- train prompt truncation: `0`
- validation prompt truncation: `0`
- train offset masks: `992/992`
- validation offset masks: `248/248`
- max token length: `331`

## Decision

V479 fixes the known V476 objective-alignment bug. It still does not authorize a
submit or full eval by itself. A GPU smoke train would only be justified with a
strict first-checkpoint kill switch:

- weak total must be `>192`
- `equation_transform` must be `>56`
- `bit_manipulation` must stay `>=136`
- `truncated` must stay `0`
