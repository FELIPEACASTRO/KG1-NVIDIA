# V524 Quota Token Objective Audit

## Decision

- GPU allowed: `False`
- Status: `objective_adjustment_required`
- Reason: Rows and loss-token mass differ materially; paid GPU must use an objective that prevents token-length bias.
- Next action: Before GPU, either enable row/family-normalized loss or build V525 shorter bit traces, then rerun V286/V513/V524.

## Calculation

- Reference gain bit share: `0.741935`
- V523 row bit share: `1.0`
- V523 loss-token bit share: `1.0`
- Loss-token mass: `{"bit_manipulation": 335938}`

## Findings

- `warning` `bit_token_share_above_reference_gain_share`: V523 row bit share=1.000, token bit share=1.000, reference gain bit share=0.742. Use row-normalized loss, family weights, or shorter bit traces before GPU.

## Literature Mapping

- Class-balanced loss supports reweighting when row counts do not reflect useful signal.
- Focal/hard-example ideas support emphasizing hard residual classes only when labels are verified.
- Curriculum learning supports short verified traces before harder residual traces.
- Scaling/mixture laws warn that loss movement can be dominated by mixture/token mass rather than target ACC.
