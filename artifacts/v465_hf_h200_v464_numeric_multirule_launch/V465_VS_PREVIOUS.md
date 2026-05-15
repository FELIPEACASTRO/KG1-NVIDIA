# V465 vs Previous

| Item | Previous active transfer smoke | V465 proposal |
|---|---:|---:|
| Route | V448 target-aligned trace SFT | V464 numeric multi-rule hard-negative SFT |
| Train rows | 1164 | 558 |
| Validation rows | 129 | 138 |
| Equation hard negatives from real adapter raw output | 0 in route | 22 train / 4 validation |
| Hard-negative rule classes | 0 in route | 3 |
| Bit replay train rows | included in mixed trace route | 512 explicit guardrail rows |
| Bit replay validation rows | included in mixed trace route | 128 explicit guardrail rows |
| Tokenization status | passed for V447 route | passed for V464 route |
| GPU authorization | only smoke, then weak eval | only smoke, then weak eval |
| Promotion gate | weak total > 192, equation > 56, bit >= 136, trunc = 0 | same |

V465 is more targeted than the previous broad trace route. It is still a smoke
only: the adapter must beat the locked weak gate before any full eval, package,
or Kaggle submit.
