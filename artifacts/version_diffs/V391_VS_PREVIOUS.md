# V391 vs Versao Anterior

Data: 2026-05-14

Regra aplicada: toda nova versao deve trazer um quadro comparando contra a versao anterior.

| Item | Versao anterior V390 | V391 | Delta / decisao |
|---|---:|---:|---|
| Objetivo | treinar V390 em A100 | treinar mesma receita em H200 | troca somente de runtime/hardware |
| Dataset | V390 equation+bit replay `5031/532` | mesmo dataset | sem mudanca de dados |
| CPU signal | `equation 56 -> 62`, weak `192 -> 198` projetado | mesmo sinal | preservado |
| Hardware | `a100-large` | `h200` | necessario por CUDA 13 |
| Runtime gate | bloqueado: CUDA 13 em A100 | esperado passar em H200 | FinOps evita repeticao do erro |
| Max steps | `12` | `12` | sem mudanca |
| Promocao | nao treinou | depende weak eval | so promover se `total>192`, `equation>56`, `bit>=136`, `truncated=0` |

Conclusao: V391 nao e uma nova hipotese de dados; e o relancamento correto da V390 em hardware compativel depois do gate barrar A100/CUDA 13.
