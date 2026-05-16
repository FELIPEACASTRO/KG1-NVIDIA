# KG1 V505 Label-Free Candidate Revalidation

Data: 2026-05-16

## Objetivo

Revalidar artefatos locais de predicao sem gastar GPU e sem usar `answer`
conhecido para decidir ACC. A regra usada foi:

- se ha `raw_output`, recomputar `prediction` com `extract_final_answer(raw_output)`;
- se nao ha `raw_output`, tratar o CSV como reference-only e nunca como
  candidato adapter-only submittable;
- comparar `stored_correct` contra `label_free_correct`.

## Resultado

| Item | Valor |
|---|---:|
| CSVs candidatos varridos | 30 |
| CSVs weak315 revalidados | 22 |
| Weak315 com `raw_output` de adapter | 8 |
| Weak315 reference-only | 14 |
| Candidatos adapter-only promocionais | 0 |

Melhor `raw_output` adapter:

- arquivo: `artifacts/v333_tong_bit_reasoner_gate/20260513T171304Z/v333_tong_bit_reasoner_gate_tong_bit_replace_predictions.csv`;
- label-free: `191/315`;
- `equation_transform=55/155`;
- `bit_manipulation=136/160`;
- `truncated=0`;
- stored anterior: `192/315`, delta `-1`.

Melhor reference-only:

- arquivo: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv`;
- label-free/stored: `222/315`;
- `equation_transform=63/155`;
- `bit_manipulation=159/160`;
- `truncated=0`;
- decisao: nao submittable como adapter-only porque nao contem `raw_output` de
  modelo.

## Achado Principal

O plateau mistura duas realidades:

1. O adapter bruto, quando medido label-free, nao passou de `191/315` nos
   artefatos locais revalidados.
2. Os solvers/postprocessors CPU conseguem numeros muito melhores, mas esses
   CSVs nao sao comportamento do adapter e nao podem ser promovidos diretamente.

## Decisao

Nao abrir novo H200 baseado em CSV reference-only. O proximo ganho precisa vir
de uma dessas rotas:

1. transformar os acertos solver/postprocessor em dataset/trace que o LoRA
   aprenda sem regredir bit;
2. provar que o pacote permitido aceita o mesmo comportamento sem runtime
   solver/postprocessor;
3. encontrar um candidato adapter `raw_output` com `>192/315`,
   `equation>=60`, `bit>=136`, `trunc=0`.

Artefatos:

- `v505_label_free_candidate_revalidation_manifest.json`;
- `v505_label_free_revalidation_summary.csv`;
- `v505_label_free_diff_rows.csv`.
