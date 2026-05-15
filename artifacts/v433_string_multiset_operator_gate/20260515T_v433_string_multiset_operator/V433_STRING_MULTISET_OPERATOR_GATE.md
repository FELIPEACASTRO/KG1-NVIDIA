# V433 String Multiset Operator Gate

Generated: `2026-05-15T11:42:24.700474+00:00`

CPU-only gate for operator-local string/multiset transforms in Alice equation rows.

## Comparison

| Metric | Baseline V291/V290 | V433 projection | Delta |
|---|---:|---:|---:|
| Total weak correct | `192/315` | `192/315` | `0` |
| equation_transform | `56/155` | `56/155` | `0` |
| bit_manipulation | `136/160` | `136/160` | `0` |
| Truncated | `0` | `0` | `0` |

## Gate Counts

| Metric | Value |
|---|---:|
| Candidate rows | `9` |
| Ambiguous rows | `6` |
| Ambiguous rows containing answer | `4` |
| Accepted new gains | `0` |
| Conflicts | `0` |

## Accepted Rows

| id | prediction | answer |
|---|---|---|
| none | none | none |

## Ambiguous Correct Candidates

| id | answer | candidate_predictions |
|---|---|---|
| `dea42835` | `['[/` | `'/|'/'/|['/|['/[|['[/` |
| `69aa57b3` | `4472` | `4472|472|4724` |
| `88b43464` | `5311` | `531|5311|5315` |
| `d3b20e29` | `3922` | `3922|3923` |

## Decision

No GPU. The class produced no unique label-free gain; rows with the answer in the candidate set remain ambiguous and cannot be promoted without oracle selection.
