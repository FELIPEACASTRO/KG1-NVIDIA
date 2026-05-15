# V454 Bit Guardrail Decision

Generated: 2026-05-15T21:34:57.145382+00:00

## Result

| Metric | Value |
|---|---:|
| Submit-safe baseline bit | `136/160` |
| Tong train teacher bit signal | `1364/1602` |
| Best weak CPU teacher bit signal | `159/160` |
| Best adapter-transfer bit after V359/V368 | `136/160` |
| Transfer gap indicator | `23` |
| `hf_gpu_allowed` | `false` |

## Evidence

| Source | Finding | Decision |
|---|---|---|
| V296 stride | train `1201/1602`, gains `154`, losses `218` | lossy diagnostic |
| V333 Tong weak replace | total `192/315`, bit `136/160`, gains `1`, losses `1` | not deployable |
| V366 CPU teacher | total `222/315`, bit `159/160`, losses `0` | teacher only |
| V359 adapter transfer | total `190/315`, bit `134/160`, trunc `1` | rejected |
| V368 adapter transfer | total `191/315`, bit `135/160`, trunc `0` | rejected |

## Decision

Bit teacher signal is strong, but direct weak replacement is lossy and adapter-transfer attempts regressed below bit>=136.

Do not launch another bit-only HF GPU job. Bit remains a guardrail/replay family.
The next active route is equation CPU target audit; bit is included only to prevent regression.
