# V255/V256 HF line diff summary

- Repo: `felipesp1983/kg1-strong-adapters-v194-v226`
- V255: `evals/v255-h200-v221contract-v194-20260511T005050Z/eval/hf_v194_protected_v221_contract/v245_hf_weak_hf_v194_protected_v221_contract_predictions.csv`
- V256: `evals/v256-h200-v221contract-v226ckpt1-20260511T0110Z/eval/hf_v226_checkpoint1_v221_contract/v245_hf_weak_hf_v226_checkpoint1_v221_contract_predictions.csv`
- Join key: `id`
- Rows joined: `315`
- Prediction diff rows: `5`
- Correctness diff rows: `2`
- V226 gains over V194: `1`
- V226 losses vs V194: `1`
- Same correctness: `313`
- Truncation diff rows: `0`

## Family delta

| type               |   rows |   v194_correct |   v226_correct |   net_v226_minus_v194 |   v226_gain_rows |   v226_loss_rows |   prediction_diff_rows |
|:-------------------|-------:|---------------:|---------------:|----------------------:|-----------------:|-----------------:|-----------------------:|
| bit_manipulation   |    160 |            135 |            135 |                     0 |                1 |                1 |                      3 |
| equation_transform |    155 |             56 |             56 |                     0 |                0 |                0 |                      2 |

## Correctness delta rows

| id       | type             |   answer_v194 |   prediction_v194 | correct_v194   | truncated_v194   |   answer_v226 |   prediction_v226 | correct_v226   | truncated_v226   |   correct_delta |
|:---------|:-----------------|--------------:|------------------:|:---------------|:-----------------|--------------:|------------------:|:---------------|:-----------------|----------------:|
| 4ef88f92 | bit_manipulation |      01010111 |          01011111 | False          | False            |      01010111 |          01010111 | True           | False            |               1 |
| 8740ed31 | bit_manipulation |      01101000 |          01101000 | True           | False            |      01101000 |          01111000 | False          | False            |              -1 |
