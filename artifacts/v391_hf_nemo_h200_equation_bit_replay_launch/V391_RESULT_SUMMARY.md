# V391 Result Summary

Data: 2026-05-14

| Candidate | Total weak | equation_transform | bit_manipulation | Truncated | Decisao |
|---|---:|---:|---:|---:|---|
| Baseline adapter-only | `192/315` | `56/155` | `136/160` | `0` | referencia |
| V391 checkpoint-2 | `191/315` | `56/155` | `135/160` | `0` | rejeitado |
| V391 checkpoint-4 | `191/315` | `56/155` | `135/160` | `0` | rejeitado |

Jobs:

- Train: `https://huggingface.co/jobs/felipesp1983/6a0629a83308d79117b8fe00`
- Weak eval: `https://huggingface.co/jobs/felipesp1983/6a0630323308d79117b8fe61`

FinOps:

- O treino H200 completou.
- O weak eval foi cancelado depois dos checkpoints 2 e 4 porque ambos falharam o gate: `total>192`, `equation>56`, `bit>=136`, `truncated=0`.

Conclusao:

V391 confirmou que a projecao CPU V390 (`198/315`, `equation=62`, `bit=136`) nao transferiu para LoRA. Nao rodar full eval, package, Kaggle submit ou outro relaunch H200 desta linha sem novo gate de transferencia adapter-only.
