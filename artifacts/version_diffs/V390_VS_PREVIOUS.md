# V390 vs Versao Anterior

Data: 2026-05-14

Regra aplicada: toda nova versao deve trazer um quadro comparando contra a versao anterior.

| Item | Versao anterior | V390 | Delta / decisao |
|---|---:|---:|---|
| Linha anterior avaliada | V388/V389 adapter soups V291/V382 | V390 CPU gate + dataset mix | Troca de direcao: parar soup linear e voltar para sinal CPU verificavel |
| Weak medido adapter-only | melhor V389 `191/315` | ainda nao medido em adapter; CPU projection `198/315` | V390 ainda precisa HF weak eval antes de qualquer full/submit |
| `equation_transform` | V389 `56/155` | CPU projection `62/155` | `+6` em CPU gate, sem conflitos |
| `bit_manipulation` | V389 `135/160` melhor | guardrail CPU `136/160`; dataset mix inclui bit replay | `+1` vs V389 medido, igual ao baseline operacional |
| Truncation | V389 `1-2` | CPU gate `0`; tokenization gate `0` | remove o problema de truncation da linha V389 |
| Dataset train | V389 sem treino novo, apenas soup | `5031` rows: `4231` bit + `800` equation | Mix pequeno e focado |
| Dataset validation | V389 sem treino novo, apenas soup | `532` rows: `332` bit + `200` equation | Valida bit replay + equation narrow |
| Tokenization | n/a | max train `749`, max val `748`, offset masks `5563/5563` | passa gate com `max_length=1024` |
| Promocao | V389 rejeitado | bloqueado ate HF weak eval | so promover se `total>192`, `equation>56`, `bit>=136`, `truncated=0` |

Conclusao: V390 e a primeira linha apos V389 com sinal novo objetivo: `+6` equation no CPU gate. Ainda nao e submit-safe; precisa gerar adapter e passar weak/full gate.

