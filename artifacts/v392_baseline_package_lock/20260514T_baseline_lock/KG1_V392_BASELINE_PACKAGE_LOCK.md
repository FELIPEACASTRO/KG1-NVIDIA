# KG1 V392 Baseline Package Lock

Data: 2026-05-14

## Decisao

O baseline submitavel travado para comparacao imediata e o V291/V290 checkpoint-6 adapter-only package.

Ele corresponde ao submit mais recente do time `Felipe Angelo` no Kaggle CLI:

| Campo | Valor |
|---|---|
| Kaggle team | `Felipe Angelo` |
| Leaderboard exibido | posicao `19` na pagina retornada pelo Kaggle CLI |
| Submission date | `2026-05-11 22:19:17.163000` |
| Description | `V291 V290 checkpoint-6 adapter-only full823 trunc1 official-like gate` |
| Public score | `0.86` |
| Private score | vazio/no CLI |

## Package Travado

| Campo | Valor |
|---|---|
| Package manifest | `artifacts/v291_submission_package/v291_h200_checkpoint6_823_20260511T212028Z/package_manifest.json` |
| Submission zip | `artifacts/v291_submission_package/v291_h200_checkpoint6_823_20260511T212028Z/submission.zip` |
| Zip SHA256 | `293b414f316330db7ac12c4f3001e7796b0a087ed5dd86af6e13d98620b43433` |
| Adapter repo | `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke` |
| Adapter subfolder | `checkpoint-6` |
| Adapter model SHA256 | `0a7b6144231d9358ae73a5e57d8778b32be1520fa47e3041414b3e025aaa1aa1` |
| Adapter config SHA256 | `a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d` |

## Metricas Locais

| Eval | Total | equation_transform | bit_manipulation | Truncated | Status |
|---|---:|---:|---:|---:|---|
| V290 weak checkpoint-6 | `192/315` | `56/155` | `136/160` | `0` | melhor weak adapter-only |
| V291 full official-like checkpoint-6 | `823/947` | `56/155` | `135/160` | `1` | package criado; public score `0.86` |
| V391 checkpoint-2 | `191/315` | `56/155` | `135/160` | `0` | rejeitado |
| V391 checkpoint-4 | `191/315` | `56/155` | `135/160` | `0` | rejeitado |

## Implicacao Para o Roadmap

- O objetivo de hoje nao e "recriar 0.86"; isso ja existe no V291/V290 checkpoint-6.
- Para subir no ranking, um novo candidato precisa bater este pacote em full official-like ou pelo menos mostrar weak gain real antes de full.
- O alvo operacional minimo para novo weak e `total > 192`, `equation > 56`, `bit >= 136`, `truncated=0`.
- A rota V390/V391 esta bloqueada porque perdeu `1` bit e nao moveu equation.
- O proximo experimento permitido e V393: sweep sem treino do melhor adapter/package.
