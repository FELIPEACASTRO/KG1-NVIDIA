# V406 Vs Previous

| Item | Previous V326/V337 style | V406 solver-first transfer |
|---|---:|---:|
| Train rows | `5031` V326 or `1440` V337D | `2064` |
| Val rows | `532` V326 or `340` V337D | `516` |
| Equation numeric synthetic | V325/V390 only | kept V390/V325 `800 train / 200 val` |
| Symbolic cryptarithm | absent in V326 | added V330 `240 train / 60 val` |
| New bit exact-global rules | absent | added V403 OR/XOR rules `512 train / 128 val` |
| Bit replay guardrail | broad V304 or V217 replay | compact V217 `512 train / 128 val` |
| Weak/full rows used for train | `0` | `0` |

V406 is not a submit artifact. It is the smallest responsible adapter-transfer candidate after V405 showed `201/315` CPU solver projection.
