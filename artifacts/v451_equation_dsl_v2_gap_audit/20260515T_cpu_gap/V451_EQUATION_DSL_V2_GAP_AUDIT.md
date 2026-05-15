# V451 Equation DSL v2 Gap Audit

Data: 2026-05-15

## Objetivo

Definir, com base em artefatos locais ja medidos, onde a proxima DSL de
`equation_transform` precisa agir. Esta auditoria nao treina, nao usa GPU, nao
faz submit e nao promove candidato.

## Evidencia Existente

### V324

Artefato:

- `artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v324_on_v290_checkpoint6/v324_equation_expanded_solver_manifest.json`

Resultado:

| Item | Valor |
|---|---:|
| Baseline equation | `56/155` |
| Equation misses auditados | `99` |
| Accepted no-loss candidates | `6` |
| Projected equation | `62/155` |
| Projected weak | `198/315` |
| Conflicts | `0` |
| Bit guardrail baseline | `136/160` |

IDs aceitos V324:

- `274def88`
- `528ec0d8`
- `7688e06e`
- `c5b058d6`
- `d1bd7478`
- `fb623471`

Classes que geraram sinal:

| Rule class | Verified candidates |
|---|---:|
| `v274_guarded_numeric_minus_signed_opposite_sign_guarded` | `2` |
| `v274_guarded_numeric_minus_direct_negative_restore_sign` | `2` |
| `v299_same_operator_unique_numeric_dsl` | `1` |
| `v274_guarded_numeric_colon_absdiff_restore_trailing_zero` | `1` |
| `v274_guarded_numeric_add_direct_over_model_add_variant` | `1` |

Interpretacao: existe sinal CPU real em equation, mas ele ainda e solver-only.
Ele nao virou adapter gain em V391/V413/V444/V448.

### V443

Artefato:

- `artifacts/v443_certified_equation_pair_builder/20260515T_v443_certified_equation_pair_builder/v443_certified_equation_pair_builder_manifest.json`

Resultado:

| Item | Valor |
|---|---:|
| Probes public-train do adapter | `133` |
| Equation rows | `120` |
| Bit rows | `13` |
| Certified pair rows | `0` |
| Status | `cpu_pairs_insufficient_no_gpu` |

Reason counts:

| Reason | Count |
|---|---:|
| `no_unique_certified_rule` | `120` |
| `not_equation` | `13` |

Interpretacao: o certificador atual e conservador demais ou incompleto para os
erros reais do adapter. Ele nao gera pares treinaveis, entao nao libera GPU.

## Gap Tecnico

O problema nao e "falta de mais treino"; e falta de uma representacao que
transforme regra correta em comportamento adapter-only.

O gap especifico:

1. V324 consegue resolver alguns misses weak usando DSL numerica guardada.
2. V443 nao consegue certificar regra unica para os erros reais coletados do
   adapter.
3. V448 mostrou que traces limpos, mesmo com target alignment, nao transferem.
4. Logo, V452 precisa ampliar a DSL/certificador antes de qualquer GPU.

## Requisitos Para V452

V452 deve ser CPU-only e deve atacar os `120` equation rows de V443 que caem em
`no_unique_certified_rule`.

Implementar primeiro:

1. Parser mais geral para `equation_transform`:
   - nao assumir sempre `len(lhs)==5`;
   - suportar operadores em posicao variavel;
   - preservar tokens de pontuacao;
   - separar numeric, symbolic e mixed.
2. DSL numerica inspirada em `tonghuikang/nemotron/reasoners/equation_numeric.py`:
   - concat;
   - reverse concat;
   - soma/subtracao/multiplicacao;
   - `+1/-1`;
   - abs diff;
   - div/mod;
   - reverse operand/result;
   - prefix/suffix;
   - digitos/determinante.
3. DSL simbolica para pontuacao:
   - slot mapping por posicao;
   - operator-index template;
   - char transducer;
   - keep/delete/replace selected chars;
   - renaming stability obrigatoria.
4. Certificacao:
   - regra unica;
   - Leave-One-Out;
   - renaming stability;
   - MDL/tiebreak congelado antes de olhar answer;
   - abstain em empate.

## Gate De Promocao V452

V452 so libera novo GPU se produzir:

- pelo menos `+4` equation no CPU gate estrito;
- `0` conflitos;
- `bit>=136` preservado;
- dataset sem weak/full leakage;
- pares ou traces curtos com alvo final emitivel pelo adapter;
- tokenization/offset-mask gate antes de HF.

Se V452 produzir `0` pares ou apenas repetir os `6` V324 sem novo mecanismo de
transferencia, nao abrir H200.
