# V420 Symbolic Cryptarithm Aggressive Gate

Generated: 2026-05-15

V420 re-ran the symbolic cryptarithm CPU gate with `max_operator_symbols=4`, `max_solutions_per_assignment=200`, and `solver_time_limit_s=0.03`.

## Result

| Metric | Value |
|---|---:|
| Symbolic equation miss rows audited | `83` |
| Accepted candidates | `1` |
| Accepted ID | `99d6a3b5` |
| New beyond known V409/V414 teacher | `0` |
| Conflicts | `0` |

The accepted ID `99d6a3b5` is already present in the V409/V414 accepted union. Multi-operator cryptarithm variants produced incorrect candidates and are not promotable.

## Decision

No HF job is authorized. V420 confirms that widening the existing cryptarithm gate does not solve the `80` punct-only residual bucket identified by V419.
