# V448 vs Previous

## Objetivo

V448 testa se o dataset V447 clean, derivado do gate V446/Tong target-alignment,
consegue transferir traces submit-safe para o adapter V290 checkpoint-6 sem
perder a familia `bit_manipulation`.

## Comparativo

| Item | V290/V291 baseline | V444 ultimo H200 | V448 medido |
|---|---:|---:|---:|
| Weak total medido | `192/315` | `190/315` | `190/315` |
| `equation_transform` | `56/155` | `56/155` | `56/155` |
| `bit_manipulation` | `136/160` | `134/160` | `134/160` |
| `truncated` | `0` | `1` | `1` |
| Dataset de treino | adapter existente | high-conf reconstructed amplo | V447 clean target-aligned |
| Train rows | n/a | `1848` | `1164` |
| Val rows | n/a | `172` | `129` |
| Contradicoes `boxed` | n/a | nao bloqueadas neste builder | `0`, bloqueio obrigatorio |
| GPU recipe | n/a | H200, 4 steps | H200, 6 steps |
| Criterio de promocao | referencia | reprovou | `total>192`, `equation>56`, `bit>=136`, `truncated=0` |

## Por Que V448 E Diferente

- Usa somente traces que passaram por anti-leakage, contrato weak, source
  inventory e gate de tokenizacao oficial.
- Remove exemplos cujo ultimo `\boxed{}` contradizia a resposta oficial.
- Nao repete SFT amplo nem LR sweep generico.
- Mantem timeout de 1 hora e kill-switch por FinOps.

## Decisao Esperada

V448 foi executado. O primeiro weak eval util (`checkpoint-3`) reprovou:
`190/315`, `equation=56/155`, `bit=134/160`, `truncated=1`. O job foi
cancelado por FinOps antes de avaliar `checkpoint-6`.

Decisao: nao repetir V448 com mais steps, epochs, LR sweep ou H200 maior.
Voltar para CPU transfer-debug, DSL v2 e metric/parser audit antes de qualquer
novo job pago.
