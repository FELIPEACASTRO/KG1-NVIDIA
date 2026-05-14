# V395 HF CPU Aggressive Symbolic Gate Results

Generated: 2026-05-14

HF job: `https://huggingface.co/jobs/felipesp1983/6a0647a9e48bea4538b9d78a`

HF artifact dataset: `felipesp1983/kg1-v395-cpu-symbolic-gate-artifacts`

HF artifact path: `v395-hf-cpu-aggressive-symbolic-gate-20260514T220636Z`

## Result

| Check | Result |
|---|---|
| Runtime | HF CPU `cpu-upgrade`, `8 vCPU`, `32 GB`, `US$0.0005/min` |
| V324 aggressive | `6` known numeric gains, `0` conflicts, projected `equation=62/155` |
| V329 wide | `1` known symbolic gain (`99d6a3b5`), `0` conflicts, projected `equation=63/155` |
| V336 integrated | `199/315`, `equation=63/155`, `bit=136/160`, `losses=0` |
| Adapter-only gain | `0` new measured adapter-only gain |
| Decision | Do not launch GPU training from the same signal; V391 already showed this transfer does not move the LoRA. |

The full downloaded result tree was not committed because its nested HF path exceeds Windows filename limits. The canonical complete artifact is the HF dataset path above.
