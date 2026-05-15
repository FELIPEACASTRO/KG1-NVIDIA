# V417 Transfer Blocker Gate

Generated: 2026-05-15T03:48:33.541245+00:00

## Baseline

| Candidate | Total | equation_transform | bit_manipulation | Truncated |
|---|---:|---:|---:|---:|
| V291/V290 checkpoint-6 baseline | `192/315` | `56/155` | `136/160` | `0` |

## V416 Result

| Candidate | Total | equation_transform | bit_manipulation | Truncated | Delta | Decision |
|---|---:|---:|---:|---:|---:|---|
| v416_rawstyle_transfer_checkpoint_2_v221_contract | `190/315` | `56/155` | `134/160` | `1` | `-2 total, 0 eq, -2 bit` | reject |
| v416_rawstyle_transfer_checkpoint_4_v221_contract | `191/315` | `56/155` | `135/160` | `1` | `-1 total, 0 eq, -1 bit` | reject |

## Teacher Versus Adapter Gap

V414 CPU teacher projection reaches `222/315`, `equation=63/155`, `bit=159/160`, but V415 found only `2` adapter hits across `30` V414 gain rows. The transfer gap is still the bottleneck.

## Decision

`hf_gpu_allowed = false`.

New GPU SFT is blocked until a CPU gate proves a materially new adapter/package-level signal. A stronger solver/verifier teacher alone is not enough, because V368, V413, and V416 all failed to transfer it into weak ACC.

## Allowed Next Work

1. CPU-only row-level mining.
2. Formal rule expansion with abstain/no-loss proofs.
3. Adapter/package behavior probes that do not train on weak/full rows.
4. HF GPU only after the pre-GPU gate can plausibly beat `total>192`, `equation>56`, `bit>=136`, `truncated=0`.
