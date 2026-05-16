# V469 vs Previous

## Purpose

V469 replaces the contaminated V465/V466 route with the fixed V468 dataset.

## Comparison

| Item | Previous V465/V466 | New V469 |
|---|---|---|
| Dataset | `v464_v463_numeric_multirule_dataset` | `v468_v464_symbol_fix_dataset` |
| Silent contradiction count | `30/56` equation rows | `0/56` equation rows |
| Train rows | `558` | `558` |
| Validation rows | `138` | `138` |
| Equation train rows | `46` | `46` |
| Bit replay train rows | `512` | `512` |
| Tokenization gate | passed before semantic contradiction gate existed | passed after contradiction gate |
| Source weights | old V464 source | fixed V468 source |
| Promotion gate | weak `>192`, equation `>56`, bit `>=136`, trunc `0` | same |

## FinOps Rule

Run only as a smoke. Evaluate checkpoint-4 first. If checkpoint-4 does not show a deployable path, cancel before spending on later checkpoints.

Block conditions:

- weak total `<=192/315`;
- `equation_transform <=56/155`;
- `bit_manipulation <136/160`;
- `truncated >0`.
