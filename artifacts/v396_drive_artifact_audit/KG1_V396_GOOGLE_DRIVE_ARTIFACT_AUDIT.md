# KG1 V396 Google Drive Artifact Audit

Generated: 2026-05-14

Scope: targeted audit of Google Drive KG1 directories that could change the current ranking plan. The audit deliberately avoided downloading large adapter weights or submission ZIPs. Only folder listings and small JSON/CSV manifests were inspected.

## Executive Decision

No newly discovered Google Drive artifact supersedes the current adapter-only baseline.

- Current adapter-only baseline remains V291/V290 checkpoint-6: weak `192/315`, `equation_transform=56/155`, `bit_manipulation=136/160`, full official-like `823/947`.
- Current CPU solver/verifier ceiling remains diagnostic only: V395 integrated no-loss `199/315`, `equation=63/155`, `bit=136/160`.
- Nothing found in Drive authorizes a new GPU training run by itself.

## Valid Artifacts

| Drive area | Evidence | Use now |
|---|---|---|
| `KG1_NVIDIA_V221` | Candidate registry with 7 weak candidates. Best historical rows: V194 `190/315`, bit `136`, eq `54`; V217 `190/315`, bit `135`, eq `55`; Kienngx `183/315`, bit `128`, eq `55`. | Keep as prediction registry/oracle evidence already consumed by V230. Not a stronger adapter. |
| `KG1_NVIDIA_V226` | Checkpoint sweep: best checkpoint-1 `191/315`, trunc `0`; V230 synthesis records bit `136`, eq `55`. | Valid predecessor baseline and adapter lineage. Already consumed. |
| `KG1_NVIDIA_V230` | Complementarity analysis: row-level oracle can reach `197/315`, but weak gate still fails because equation remains short. | Keep miss packs and router analysis as solver/verifier inputs. Not submit-safe. |
| `KG1_NVIDIA_V231` | Miss-pack mining: `100` equation misses, `24` bit misses, only `2` equation rows and `4` bit rows with correct alternative among candidates. | Useful diagnostic of why adapter complementarity was weak. |
| `KG1_NVIDIA_V232` | Workbench: `100` equation solver workitems and `24` bit guardrail workitems. | Valid work queue; no direct score. |
| `KG1_NVIDIA_V233` | Equation probes: `0` deployable equation overrides, `2` nondeployable oracle rows. | Confirms early parser was insufficient. |
| `KG1_NVIDIA_V234/V235` | External/source triage inventory; V235 requires source/license/token checks before ingest. | Use only as registry of external leads. Do not train directly from it. |
| `KG1_NVIDIA_V236` | Local DSL probes: `0` verified equation overrides, bit guardrail ready. | Confirms DSL needed more expansion. |
| `KG1_NVIDIA_V237` | Prompt-format audit: `68` zero-pair equation rows in sample, parser-specific issue identified. | Parser guidance only. |
| `KG1_NVIDIA_V238` | Alice parser probes: `1` verified row, `0` incorrect. | Single-row signal only; not enough for submit. |
| `KG1_NVIDIA_V199` | `final_submit_doublecheck.json` shows a production-ready historical adapter ZIP with valid root-only layout, rank `32`, target modules including `lm_head`, `12011` tensors, SHA checks. | Valid lineage/package audit artifact. It is historical and does not supersede V291/V290. |

## Rejected Or Non-Actionable Artifacts

| Drive area | Evidence | Decision |
|---|---|---|
| `KG1_NVIDIA_V227/V228/V229` targeted equation micro sweep | V228 triage window regressed badly: only `3-4/160`, truncation `111/160` (`69.375%`). V229 staged fast eval: best V227 adapter `16/315`, eq `9`, bit `7`, trunc `0`. | Hard reject. Do not reuse prompt suffix/training setup. |
| `KG1_PUBLIC_ADAPTERS/huikang_default_v20` | Weak eval log failed at vLLM LoRA load due target module mismatch with Nemotron mixer modules. | Not directly usable. Only revisit if conversion shim is built and gated. |
| `KG1_PUBLIC_ADAPTERS/kienngx_cot_labels_3000samples_adapter` | Weak eval `32/315`, trunc `202` (`64.1%`). | Reject. |
| `Submit/submission.zip` | Only a historical ZIP in Drive root was visible; fetch of raw zip content was intentionally avoided and the file predates current V291 package. | Do not treat as current best. Current best package is local V291 with known SHA. |

## Drive Audit Impact On Roadmap

1. V221/V226/V230-V238 stay useful as evidence and diagnostics, not as new candidates.
2. V227/V228/V229 should remain closed. They explain past regressions and should not be relaunched.
3. V199 is a valid historical package/lineage anchor, but current baseline is still V291/V290.
4. Public adapters in Drive are not drop-in improvements; they need compatibility conversion plus weak/full gates before any use.
5. The next real action remains adapter-only transfer investigation, not another broad training run.

## Next Gate

Before spending GPU, the next job must show one of:

- adapter-only weak `>192/315` with `equation>56`, `bit>=136`, `truncated=0`; or
- full official-like expectation `>=824/947`; or
- a new CPU gate signal not already tested by V391/V395, with trace coverage and no-loss verification.
