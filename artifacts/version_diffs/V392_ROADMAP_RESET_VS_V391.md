# V392 Roadmap Reset vs V391

Atualizado: 2026-05-14

| Item | V391 | V392 decisao | Delta / motivo |
|---|---|---|---|
| Linha ativa | Treino H200 com dataset V390 equation+bit replay | Pausar LoRA direto | V391 nao transferiu a projecao CPU |
| Melhor weak medido | `191/315` | baseline volta a `192/315` | V391 ficou `-1` total vs melhor adapter-only |
| `equation_transform` | `56/155` | meta continua `>56`, minimo para full `>=60` | V391 nao saiu do teto |
| `bit_manipulation` | `135/160` | guardrail `>=136/160` | V391 perdeu `-1` bit |
| Truncation | `0` | manter `0` | truncation nao foi o problema do V391 |
| Criterio de promocao | weak checkpoints 2/4/6/8/10/12 | sweep sem treino + baseline lock antes de GPU | Mais treino sem prova de transferencia esta bloqueado |
| FinOps | eval cancelado apos checkpoints 2/4 ruins | manter cancelamento automatico quando nao puder bater gate | evita gastar em checkpoints sem chance objetiva |

Conclusao: V392 nao e uma nova receita de treino. E um reset operacional para buscar ganho submetivel mais rapido: travar o melhor package historico, fazer sweep sem treino do melhor adapter/package, e so voltar a GPU se houver prova de transferencia adapter-only.
