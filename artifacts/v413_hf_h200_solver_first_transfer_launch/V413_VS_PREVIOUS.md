# V413 Vs Previous

| Item | Previous | V413 |
|---|---:|---:|
| Transfer dataset | V410, not trained | V410, H200 smoke-ready |
| Train rows | `2320` | `2320` |
| Validation rows | `580` | `580` |
| CPU projection basis | V409 `202/315`, equation `63`, bit `139` | same |
| Max steps | n/a | `4` |
| Save/eval cadence | n/a | every `2` steps |
| GPU | n/a | HF `h200`, cost-gated at `<= $0.09/min` |
| Submit-safe before eval | no | no |

V413 is a short transfer smoke, not a package and not a submit artifact.
Promotion requires weak `total > 192`, `equation_transform > 56`,
`bit_manipulation >= 136`, and `truncated = 0`.
