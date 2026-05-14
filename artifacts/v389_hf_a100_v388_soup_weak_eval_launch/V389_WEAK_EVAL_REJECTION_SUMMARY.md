# V389 Weak Eval Rejection Summary

Data: 2026-05-14

Job HF: https://huggingface.co/jobs/felipesp1983/6a061dbde48bea4538b9d4b6

Resultado: rejeitado. Nenhum soup V291/V382 passou o weak promotion gate.

Gate de promocao:

- `correct > 192/315`
- `equation_transform > 56/155`
- `bit_manipulation >= 136/160`
- `truncated = 0`

Resultados medidos:

| Candidato | Total weak | equation_transform | bit_manipulation | truncated | Decisao |
|---|---:|---:|---:|---:|---|
| `v388_soup_v291_095_v382_005` | `191/315` | `56/155` | `135/160` | `2` | rejeitado |
| `v388_soup_v291_090_v382_010` | `190/315` | `56/155` | `134/160` | `1` | rejeitado |
| `v388_soup_v291_105_v382_neg005` | `191/315` | `56/155` | `135/160` | `1` | rejeitado |

Conclusao:

- A combinacao linear simples entre V291 e V382 nao melhorou `equation_transform`.
- A combinacao regrediu `bit_manipulation` e/ou truncation em todos os candidatos.
- Nenhum candidato deve ser promovido para full official-like eval.
- Esta linha nao deve ser repetida sem novo CPU gate que demonstre ganho independente antes do gasto em GPU.

