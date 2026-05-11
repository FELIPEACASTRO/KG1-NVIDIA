# Drive vs HF V221-contract diff summary

## drive_v221_v194_vs_hf_v255_v194

- Rows joined: `315`
- Prediction diff rows: `14`
- Correctness diff rows: `5`
- HF gains: `3`
- HF losses: `2`
- Truncation diff rows: `1`

| type               |   rows |   drive_correct |   hf_correct |   net_hf_minus_drive |   hf_gain_rows |   hf_loss_rows |   prediction_diff_rows |   truncated_diff_rows |
|:-------------------|-------:|----------------:|-------------:|---------------------:|---------------:|---------------:|-----------------------:|----------------------:|
| bit_manipulation   |    160 |             136 |          135 |                   -1 |              1 |              2 |                      4 |                     1 |
| equation_transform |    155 |              54 |           56 |                    2 |              2 |              0 |                     10 |                     0 |

## drive_v226_vs_hf_v256_v226

- Rows joined: `315`
- Prediction diff rows: `11`
- Correctness diff rows: `4`
- HF gains: `2`
- HF losses: `2`
- Truncation diff rows: `1`

| type               |   rows |   drive_correct |   hf_correct |   net_hf_minus_drive |   hf_gain_rows |   hf_loss_rows |   prediction_diff_rows |   truncated_diff_rows |
|:-------------------|-------:|----------------:|-------------:|---------------------:|---------------:|---------------:|-----------------------:|----------------------:|
| bit_manipulation   |    160 |             136 |          135 |                   -1 |              1 |              2 |                      3 |                     1 |
| equation_transform |    155 |              55 |           56 |                    1 |              1 |              0 |                      8 |                     0 |

