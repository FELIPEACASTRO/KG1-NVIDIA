# V403 Formal Solver Abstain Audit

Baseline: V290 checkpoint-6 weak predictions.

## Result

- Accepted candidates: `112`
- Accepted gains: `2`
- Accepted losses: `0`
- Accepted delta: `2`

## Policy

- Accept only exact byte-global bit rules: global unary, global binary, or exact ternary.
- Reject `CONSENSUS`, `UNSOLVED`, and non-global per-bit fallbacks.
- Reject the old equation v2 parser as a source of weak gains; it abstains/fails on current equation rows.

## Accepted Gain Rows

- `4ada9150`: `01111111` -> `01111011` via `Global binary: output = OR(ROL2(input), SHL4(input))`
- `4c327b55`: `11011110` -> `11011100` via `Global binary: output = XOR(SHL1(input), SHR4(input))`

## Decision

formal_solver_global_bit_signal_found_but_not_adapter_submit_safe

This is a CPU solver signal, not a Kaggle-submitable adapter gain.
