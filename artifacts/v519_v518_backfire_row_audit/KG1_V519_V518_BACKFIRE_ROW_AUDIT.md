# V519 V518 Backfire Row Audit

## Summary

- Baseline V516 label-free: `191/315`, equation `55/155`, bit `136/160`, trunc `0`.
- V518 checkpoint-2: `191/315`, equation `56/155`, bit `135/160`, trunc `0`.
- Net result: no total gain; one equation gain is canceled by one bit regression.
- Decision: block V517/V518 for full/package/submit and use the changed rows as a CPU guard only.

## Changed Rows

| id | family | delta | answer | baseline | V518 | bit diff | note |
|---|---|---:|---|---|---|---|---|
| `518deb39` | `equation_transform` | `1` | `$` | `{` | `$` | `` | true equation gain |
| `b06625c4` | `equation_transform` | `0` | `^&/` | `"$ $]"` | `"$$]"` | `` | prediction changed but still wrong |
| `b0206bb7` | `equation_transform` | `0` | `]$!` | `$){` | `$){}` | `` | prediction changed but still wrong |
| `8740ed31` | `bit_manipulation` | `-1` | `01101000` | `01101000` | `01111000` | `3` | bit regression / F2 backfire |
| `844f826c` | `equation_transform` | `0` | `/"` | ``@>``` | ``*``` | `` | prediction changed but still wrong |
| `5bcb572e` | `equation_transform` | `0` | `-!#` | `"{!"` | `"{!` | `` | prediction changed but still wrong |

## Required Guard

- Any new checkpoint must keep `8740ed31 = 01101000` correct before it can claim an equation gain.
- `518deb39 = $` is useful only if it transfers without bit regression.
- Loss movement alone is not actionable; the weak label-free row-level guard is mandatory.
