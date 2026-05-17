# V524 Quota Token Objective Audit

## Decision

- GPU allowed: `False`
- Status: `quota_ok_cpu_only`
- Reason: Rows and loss-token mass differ materially; paid GPU must use an objective that prevents token-length bias.
- Next action: Before GPU, either enable row/family-normalized loss or build V525 shorter bit traces, then rerun V286/V513/V524.

## Calculation

- Reference gain bit share: `0.741935`
- V523 row bit share: `0.688109`
- V523 loss-token bit share: `0.819409`
- Loss-token mass: `{"bit_manipulation": 153908, "equation_transform": 33920}`

## Findings

- `info` `quota_within_reference_gain_band`: row/token shares are within the configured tolerance band

## Literature Mapping

- Class-balanced loss supports reweighting when row counts do not reflect useful signal.
- Focal/hard-example ideas support emphasizing hard residual classes only when labels are verified.
- Curriculum learning supports short verified traces before harder residual traces.
- Scaling/mixture laws warn that loss movement can be dominated by mixture/token mass rather than target ACC.
