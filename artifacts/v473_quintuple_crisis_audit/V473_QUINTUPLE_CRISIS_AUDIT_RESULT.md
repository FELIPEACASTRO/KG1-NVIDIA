# V473 Quintuple Crisis Audit Result

## Scope

V473 rechecked the solution after V472 with focus on silent bugs that can affect
loss, eval, ACC, score promotion, FinOps, data contamination, symbols/operators,
and stale thresholds.

## Findings

| Finding | Status |
|---|---|
| Argparse defaults still allowed weak bit floor `133` and truncation cap `3` in old analysis/gate scripts | Fixed to bit `136`, truncation `0` |
| Static safety gate did not catch those argparse defaults | Fixed with new regex checks and self-test coverage |
| Static safety gate blocked contaminated datasets but not adapters trained from them | Fixed by blocking V448/V465/V469 adapter repo markers |
| V448/V462/V465/V466/V469/V470 launchers could still be manually executed | Replaced by fail-closed archive launchers that raise immediately |
| V447/V464 builders could recreate quarantined datasets manually | Builders now fail closed unless explicitly overridden for forensic work; future work must use a new clean version |
| Package script could validate one full manifest and download another adapter/revision | Package now requires V284 official-like manifest, immutable revision, repo/subfolder match, row contract, official controls, and adapter file hashes |
| V284 eval manifest did not record adapter hashes/revision | Official-like eval now emits `adapter_config_sha256`, `adapter_model_sha256`, `revision`, and `resolved_revision` |
| Eval jobs could return exit code `0` while promotion gate was false | V276/V277/V284 now fail by default on false gate; diagnostic success requires `KG1_ALLOW_FAILED_GATE_EXIT_0=1` |
| Central boxed-answer parser could consume braces after the answer | Parser now uses balanced `\boxed{...}` extraction; regression self-tests cover post-box LaTeX and nested payloads |
| HF training preflight did not lock expected `MAX_LENGTH` | Preflight now supports `KG1_EXPECTED_MAX_LENGTH`; train default is aligned to `8192` |
| Historical notebooks still contain old thresholds | Not edited; repository policy says historical notebooks are not retroactive until changed, and edited notebooks must pass the current notebook release gate |

## Operational Decision

No GPU route is active after this audit. V447, V461, V463, V464, V468 and
adapters derived from V448, V465, and V469 are blocked for training/eval spend.

The only valid next path is CPU-first:

1. build a new clean dataset/version, not a reused quarantined path;
2. prove equation gains in CPU with zero losses and `bit>=136`;
3. pass V286 with forbidden reference CSVs where available;
4. pass static/preflight gates;
5. only then consider a one-checkpoint HF smoke.

## Evidence

- `v473_static_full_repo_prepatch_report.json`: captured stale notebooks and
  executable quarantined launchers before patch.
- `v473_static_scripts_artifacts_postpatch_report.json`: clean scan for
  `scripts`, `src`, and `artifacts` after patch.
- `v473_static_scripts_artifacts_final_report.json`: final clean scan after
  the package/parser/gate hardening pass.
