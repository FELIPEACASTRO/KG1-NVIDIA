# KG1 Error Ledger

Atualizado: 2026-05-15

Este ledger registra erros, evidencia, impacto e regra preventiva. Ele deve ser
usado antes de qualquer novo job HF/Kaggle para evitar repeticao de custo sem
ganho.

## Estado Base

| Metrica | Melhor adapter-only submit-safe |
|---|---:|
| weak total | 192/315 |
| equation_transform | 56/155 |
| bit_manipulation | 136/160 |
| truncation | 0 |

## Erros Confirmados

### E001 - V435E misto contaminou preference

Evidencia:

- V435E antigo tinha `200` pares, sendo `133` hard negatives semanticos e `67`
  format-only negatives.
- V435F recheck bloqueou o dataset antigo: `format_negatives_absent_for_preference=false`.
- V436 sobre o dataset misto piorou no primeiro checkpoint: baseline `6/40`,
  step 4 `5/40`; equation `4/22 -> 3/22`.

Impacto:

- O objetivo mean-NLL recebeu rejeitados que nao eram semanticamente errados.
- GPU foi gasto em sinal misturado que nao poderia melhorar ACC de forma
  confiavel.

Regra preventiva:

- `scripts/kg1_static_safety_gate.py` bloqueia launchers ativos com
  `ALLOW_FORMAT_NEGATIVES` e flags `--include-format-negatives` /
  `--allow-format-negatives`.
- Qualquer dataset preference para GPU deve conter somente
  `hard_negative_adapter_exact_wrong`.

Status: corrigido e bloqueado.

### E002 - V436B hard-negative-only ainda piorou o objetivo

Evidencia:

- Job: `https://huggingface.co/jobs/felipesp1983/6a073c74e48bea4538b9e652`.
- Dataset hard-negative-only: `133` pares; equation `120`, bit `13`.
- Gates remotos OK: H200, hashes, tokenizacao sem truncation, adapter V290
  checkpoint-6 mapeado `12011/12011`.
- Baseline preference eval: `6/24`; equation `4/22`; bit `2/2`.
- Checkpoint-3: `4/24`; equation `2/22`; bit `2/2`.

Impacto:

- Mesmo sem negativos de formato, o primeiro checkpoint moveu equation na
  direcao errada.
- Nao houve candidato para weak/full eval, package ou submit.

Regra preventiva:

- Nao abrir novo H200/A100 com preference mean-NLL direto sobre V435E ate V438
  ou sucessor provar novo objetivo em CPU.
- Kill-switch: se o primeiro checkpoint nao melhorar a metrica hard-negative,
  cancelar imediatamente.

Status: cancelado por FinOps; linha bloqueada.

### E003 - Chosen completion ensina texto errado para transferencia

Evidencia V438:

- V438 usou extracao/normalizacao oficial de `\\boxed{}`.
- `answer_box_mismatch_rows=0` e `rejected_box_mismatch_rows=0`; os labels estao
  semanticamente corretos.
- `chosen_mentions_adapter_prediction_rows=123/133`.
- `chosen_mentions_public_train_label_audit_rows=133/133`.
- Chosen e maior que rejected: media `34.08` tokens vs `26.80`, ratio `1.2767`.

Impacto:

- O target correto repete o candidato errado dentro do proprio texto.
- O modelo pode aprender a mencionar auditoria/erro em vez de aprender uma
  resposta final curta e submit-safe.
- Esse padrao explica por que o preference smoke piorou equation mesmo com
  labels corretos.

Regra preventiva:

- Novos targets nao devem mencionar `public-train label audit`, `frozen adapter`
  ou a resposta errada dentro do `chosen`.
- Proxima rota deve usar answer-only ou final-answer-only equalizado:
  `Final answer: \\boxed{ANSWER}` contra `Final answer: \\boxed{WRONG}`.
- Antes de GPU, um gate deve comprovar:
  `chosen_mentions_adapter_prediction_rows=0`,
  `chosen_mentions_public_train_label_audit_rows=0`,
  `answer_box_mismatch_rows=0`,
  `rejected_box_mismatch_rows=0`.

Status: aberto; proxima acao tecnica.

### E003 Fix Candidate - V439 final-answer-only

Evidencia:

- V439 gerou targets equalizados:
  - chosen: `Final answer: \boxed{ANSWER}`.
  - rejected: `Final answer: \boxed{ADAPTER_WRONG}`.
- V438 audit sobre V439:
  - `answer_box_mismatch_rows=0`.
  - `rejected_box_mismatch_rows=0`.
  - `chosen_mentions_adapter_prediction_rows=0`.
  - `chosen_mentions_public_train_label_audit_rows=0`.
  - `chosen_tokens_mean=4.83`.
  - `rejected_tokens_mean=4.80`.

Decisao:

- V439 remove a contaminacao textual de E003.
- V439 ainda nao e ganho de ACC; ele apenas libera um smoke curto com
  kill-switch no primeiro checkpoint, se publicado em HF e se todos os hashes
  forem fixados.

Status: candidato limpo para proximo gate.

### E004 - Loss/eval_loss nao representa ACC das familias

Evidencia:

- Varias execucoes reduziram loss/eval_loss sem melhorar
  `equation_transform` alem de `56/155`.
- V436B teve metricas internas suficientes para abortar antes de weak/full:
  checkpoint-3 piorou preference, apesar de o treino estar numericamente
  saudavel.

Impacto:

- Otimizar loss isolado levou a gasto sem ganho de ranking.

Regra preventiva:

- `eval_loss`, `train_loss` e preference accuracy interna nao promovem submit.
- Essas metricas so servem para matar job cedo ou selecionar qual checkpoint
  merece weak gate.

Status: regra permanente.

## Prompt Externo

Prompt consolidado para OpenRouter/outras APIs:

- `artifacts/openrouter/KG1_V438_ERROR_LEDGER_EXTERNAL_API_PROMPT_2026_05_15.md`

Uso esperado:

1. Perguntar a modelos externos como corrigir E003 sem violar adapter-only.
2. Exigir propostas testaveis em CPU antes de GPU.
3. Rejeitar respostas que recomendem broad SFT, mais epochs, weak/full leakage,
   runtime verifier, postprocessor ou submit com script.
