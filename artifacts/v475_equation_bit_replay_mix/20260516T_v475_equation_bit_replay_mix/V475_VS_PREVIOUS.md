# V475 Vs Previous

| Item | Previous active state | V475 CPU gated state |
|---|---:|---:|
| Baseline weak | `192/315` | `192/315` |
| CPU equation projection | V324 historic varied; current roadmap had no clean V475 entry | `56 -> 60` from 4 accepted candidates |
| Train rows | no current mixed dataset | `1312` |
| Validation rows | no current mixed dataset | `328` |
| Equation train/val | V325 current only | `800 / 200` |
| Bit replay train/val | required but not yet combined | `512 / 128` |
| Weak/full rows used for train | `0` | `0` |
| Token gate status | V325-only passed | combined V286 real gate passed: token max `331`, truncation `0`, offset masks complete |

V475 is not a submit artifact. It is the smallest responsible adapter-transfer candidate after the current CPU gate found `+4` equation signal.
