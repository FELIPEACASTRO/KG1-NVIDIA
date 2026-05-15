# V444 High-Confidence Reconstructed SFT Versus Previous Runs

## Objective

Run one short H200 smoke train using only reconstructed SFT rows with
`rule_found` or `hypothesis_formed` status. This removes the `rule_unknown`
rows that diluted V397/V398 and keeps the run bounded to four steps.

## Difference Table

| Item | Previous V398 | New V444 |
| --- | --- | --- |
| Dataset | broad reconstructed SFT | high-confidence reconstructed SFT |
| Included statuses | `rule_found`, `hypothesis_formed`, `rule_unknown` | `rule_found`, `hypothesis_formed` only |
| Train rows | 2578 | 1848 |
| Val rows | 264 | 172 |
| Train families | bit/equation mixed | bit/equation mixed |
| Tokenization gate | passed | passed |
| Init adapter | V290 checkpoint-6 | V290 checkpoint-6 |
| Max steps | 4 | 4 |
| LR schedule | `1.5e-8 -> 5.0e-9` | `1.2e-8 -> 4.0e-9` |
| HF timeout | 5400 seconds | 3600 seconds |
| Promotion gate | weak improvement required | weak total > 192, equation > 56, bit >= 136, trunc = 0 |

## Expected Value

This is not a guaranteed gain. It is the most responsible short GPU test after
V443 showed that simple certified string rules produce zero deployable equation
pairs. V444 tests whether cleaner reconstructed CoT supervision can move
`equation_transform` without the `bit_manipulation` regressions seen in broad
SFT attempts.

## Stop Rule

Do not continue this branch unless weak eval beats the locked submit-safe
baseline: `192/315`, `equation_transform=56/155`, `bit_manipulation=136/160`,
`truncated=0`.
