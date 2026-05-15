# V452 Equation DSL v2 Certified Builder Result

Data: 2026-05-15

## Objetivo

Testar em CPU se a expansao de DSL/certificador consegue transformar os erros
reais V439 em pares treinaveis para `equation_transform`, sem usar weak/full e
sem abrir GPU.

## Entrada

| Item | Valor |
|---|---:|
| Source rows V439 | `133` |
| Equation rows | `120` |
| Bit rows ignorados | `13` |

## Resultado

| Item | Valor |
|---|---:|
| Audit rows | `133` |
| Candidate rows | `7` |
| Certified pair rows | `2` |
| Train pairs | `1` |
| Val pairs | `1` |
| Independent modes | `1` |
| `hf_gpu_allowed` | `false` |

Classes auditadas:

| Classe | Candidatos | Corretos | Incorretos | Promovidos |
|---|---:|---:|---:|---:|
| `v274_guarded_numeric_minus_direct_negative_restore_sign` | `2` | `2` | `0` | `2` |
| `numeric_v2_global_abs_diff_revop0_revres0` | `1` | `0` | `1` | `0` |
| `numeric_v2_global_add_minus1_revop0_revres0` | `1` | `0` | `1` | `0` |
| `numeric_v2_global_add_revop0_revres0` | `1` | `0` | `1` | `0` |
| `numeric_v2_global_max_mod_min_revop0_revres0` | `1` | `0` | `1` | `0` |
| `numeric_v2_global_mul_revop1_revres1` | `1` | `0` | `1` | `0` |

## Decisao

V452 bloqueia GPU.

Motivo: `pairs=2`, `val_pairs=1`, `modes=1`. Isso nao atinge o minimo para
treino pago e confirma que a expansao numerica ingenua gera candidatos
aparentemente plausiveis, mas incorretos contra o label publico.

## Interpretacao

1. O unico sinal treinavel seguro encontrado em V439 e a restauracao de sinal
   negativo ja coberta pelo V274.
2. As regras numericas globais com LOO continuam perigosas porque conseguem
   explicar exemplos locais, mas falham no target.
3. O problema ativo nao e parser/ACC; e ausencia de regra transferivel suficiente
   para o adapter-only em `equation_transform`.

## Proxima Acao

Nao abrir H200 a partir de V452. O caminho agora e:

1. usar V452 como bloqueio formal para treino DSL v2 insuficiente;
2. focar em mineracao CPU de notebooks publicos/source autorizado para obter
   regra nova que produza pelo menos quatro modos independentes no-loss;
3. se nao houver novo sinal, manter submit-safe V291/V290 e nao gastar GPU.
