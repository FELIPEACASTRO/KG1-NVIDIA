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

### E005 - Auditoria de integracao antes de job pago

Evidencia:

- V435E misto passou por treino pago antes de ficar claro que havia `67`
  negativos apenas de formato.
- V436B passou gates remotos, mas mostrou que o objetivo ainda estava
  estruturalmente desalinhado.
- O usuario formalizou a regra: antes de rodar script/job/notebook, validar
  dataset correto, conteudo do dataset, pecas, componentes e integracoes.

Impacto:

- Sem auditoria integrada, um launcher pode apontar para path correto mas ainda
  carregar manifest antigo, dataset com target contaminado, kill-switch ausente
  ou comparacao incompleta contra baseline.

Regra preventiva:

- Todo job pago ou notebook operacional novo/alterado deve passar por
  `scripts/kg1_pre_paid_job_integration_gate.py` quando houver dataset/launcher
  envolvidos.
- Esse gate precisa aprovar launcher, hashes, row counts, targets, flags de
  leakage, audit manifest, prompt de sistema alinhado ao target, H200
  timeout/cost gate e primeiro checkpoint/eval.

Status: implementado para a linha V440/V439.

### E006 - Final-answer-only mean-NLL preference nao trouxe sinal

Evidencia:

- V439 removeu a contaminacao textual de E003:
  `chosen_mentions_adapter_prediction_rows=0` e
  `chosen_mentions_public_train_label_audit_rows=0`.
- V440 passou integration gate local/remoto, hashes, tokenizacao, adapter load e
  H200 gates.
- Baseline interno V439 validation: `8/24`, equation `7/22`, bit `1/2`.
- Checkpoint-3 V440: `8/24`, equation `7/22`, bit `1/2`.

Impacto:

- Corrigir o template era necessario, mas nao suficiente para gerar ganho.
- Mais steps/epochs na mesma formulacao `mean_nll` nao tem evidencia de ganho e
  tende a gastar GPU sem melhorar weak/full.

Regra preventiva:

- Nao repetir V440 ou variantes triviais de LR/epoch sobre o mesmo objetivo.
- Proximo job pago precisa mudar uma destas duas coisas:
  1. objetivo: comparar somente boxed payload/logit do answer, ou outra perda
     que nao seja mean-NLL da sequencia inteira;
  2. dado: novo CPU solver/DSL encontra pares adicionais com sinal diferente e
     passa gate sem regressao.

Status: V440 cancelado por FinOps; sem weak/full/package/submit.

### E007 - Boxed-payload preference tambem nao trouxe sinal

Evidencia:

- V441 usou o mesmo dataset V439 final-answer-only, mas mudou o score de
  preferencia para tokens dentro do payload final `\boxed{...}`.
- V441 passou static gate, integration gate, tokenizacao, mask de payload,
  adapter load e H200 gates.
- Baseline interno V441 validation: `7/24`, equation `6/22`, bit `1/2`.
- Checkpoint-3 V441: `7/24`, equation `6/22`, bit `1/2`.
- O job `felipesp1983/6a075046e48bea4538b9e7d3` foi cancelado por FinOps.

Impacto:

- Trocar mean-NLL da sequencia inteira por mean-NLL apenas do payload boxed
  era uma hipotese tecnicamente correta, mas nao produziu movimento medido.
- Repetir V439/V440/V441 com mais steps, LR diferente ou H200 maior nao tem
  evidencia para superar `192/315`.
- O problema restante nao e so formato de resposta; falta dado/objetivo com
  regra label-free certificada que transfira para o adapter.

Regra preventiva:

- Bloquear novos jobs GPU baseados apenas no V435E/V439 hard-negative dataset.
- Exigir CPU gate com regra unica, MDL/LOO/renaming ou certificado equivalente
  antes de qualquer novo treino pago.
- Se o dataset nao tiver certificado de regra congelada antes da resposta
  correta, ele pode servir para diagnostico, mas nao para novo job de ranking.

Status: V441 cancelado por FinOps; sem weak/full/package/submit.

### E008 - Regras simples certificadas nao geraram pares V443

Evidencia:

- V443 auditou `133` linhas diagnosticas V439/V435E.
- `120` eram linhas de equation.
- O builder tentou transformacoes diretas de string, reverse concat,
  slot/global maps, LOO e renaming stability.
- Resultado: `0` candidatos certificados e `0` pares certificados.
- Debug sem os gates estritos mostrou candidatos brutos em muitas linhas, mas
  `raw_correct=0`; portanto o problema nao era apenas excesso de rigor do gate.

Impacto:

- Nao ha base para novo job pago de preference/mean-NLL sobre V439/V435E.
- O ganho de equation exige DSL/solver mais expressivo ou dado supervisionado
  mais limpo, nao mais steps na mesma formulacao.

Regra preventiva:

- Se um dataset de pares nao tiver certificado `rule_frozen_before_answer`,
  LOO/renaming ou evidencia equivalente, ele nao pode justificar H200.
- Quando o builder retornar `0` pares certificados, registrar a falha e mudar a
  classe de hipotese, nao repetir LR/epoch.

Status: V443 fechado; V444 high-confidence SFT e o unico smoke GPU permitido.

### E009 - Truncar assistant text antes da tokenizacao pode quebrar offset mask

Evidencia:

- A primeira montagem V444 com `max_assistant_chars=9000` falhou no gate:
  `assistant text not found in rendered chat`.
- A causa foi truncar do lado errado do texto, deixando o assistant com
  whitespace/prefixo que nao casava com o chat template.
- Rebuild com `max_assistant_chars=14000` passou:
  prompt truncation `0`, completion truncation `0`, offset masks completas.

Impacto:

- Um dataset pode parecer correto em JSONL e ainda falhar no mapeamento de loss
  por offset mask.

Regra preventiva:

- Todo dataset novo/alterado precisa passar tokenization/offset-mask gate antes
  de qualquer HF job.
- Se houver truncamento manual de assistant text, o dataset deve ser refeito ou
  o truncamento deve preservar exatamente o sufixo esperado pelo template.

Status: regra aplicada em V444; dataset final passou o gate.

### E010 - V444 high-confidence SFT regrediu no weak gate

Evidencia:

- V444 treinou em H200 por quatro steps usando o dataset
  `v444_sft_reconstructed_high_conf`, derivado de `sft_reconstructed.jsonl`
  com apenas `rule_found` e `hypothesis_formed`.
- O dataset passou hashes, tokenization gate, prompt truncation `0`,
  completion truncation `0` e offset-mask gate.
- O weak eval avaliou o primeiro checkpoint (`checkpoint-2`) no contrato V221.
- Resultado do checkpoint-2:
  - total weak `190/315`;
  - `equation_transform=56/155`;
  - `bit_manipulation=134/160`;
  - `truncated=1`.
- Baseline submit-safe permanece `192/315`, `equation=56/155`,
  `bit=136/160`, `truncated=0`.
- Job de eval `felipesp1983/6a075f1a3308d79117b907ff` foi cancelado antes de
  avaliar checkpoint-4.

Impacto:

- Filtrar `rule_unknown` foi necessario para limpar o dataset, mas nao foi
  suficiente para transferir ganho para o adapter.
- O primeiro checkpoint perdeu bit e truncou, exatamente as regresssoes que o
  gate permanente deve bloquear.
- Repetir V444 com mais steps, H200 maior, LR sweep ou epoch sweep nao tem
  base tecnica.

Regra preventiva:

- Nenhum novo job pago pode usar V444 high-confidence SFT ou variante trivial
  dele sem um novo CPU gate que mostre sinal material antes.
- Se um primeiro checkpoint nao mantiver `bit>=136` e `truncated=0`, cancelar
  imediatamente antes de avaliar checkpoints seguintes.
- A proxima tentativa deve mudar a classe de evidencia para raw-output
  prediction/parse audit ou solver label-free certificado, nao apenas trocar
  hiperparametros.

Status: V444 weak eval cancelado por FinOps; sem full/package/submit.

### E011 - Metrica permissiva podia supercontar bit em diagnosticos

Evidencia:

- `scripts/analyze_eval_predictions.py` usava `answers_equivalent` para a coluna
  `official_correct`.
- `answers_equivalent` aplica tolerancia numerica quando ambos os valores podem
  ser convertidos para numero. Isso faz strings binarias diferentes como
  `11100011` e `11100010` parecerem corretas por proximidade numerica.
- O scorer de weak/full submit-safe usa `verify_answer`, que trata qualquer
  gabarito `[01]+` como exato. Esse caminho principal estava correto.
- O V449 audit confirmou o risco em artefatos locais:
  - baseline/prediction V405: estrito `192/315`, permissivo `206/315`, inflacao
    de `+14` em `bit_manipulation`;
  - V405 integrated: estrito `201/315`, permissivo `213/315`, inflacao de `+12`;
  - V414 projection: estrito `222/315`, permissivo `223/315`, inflacao de `+1`.

Impacto:

- Nao houve evidencia de submit indevido por esse erro, porque os gates de
  promocao usam `verify_answer`.
- O risco era analitico: diagnosticos e comparacoes poderiam sugerir ganho de
  bit inexistente.

Regra preventiva:

- ACC de promocao, `official_correct`, weak/full gate e qualquer decisao de
  submit devem usar `src.competition_utils.verify_answer`.
- `answers_equivalent` fica restrito a diagnostico exploratorio sem decisao de
  promocao.
- `scripts/kg1_static_safety_gate.py` agora bloqueia `official_correct` com
  `answers_equivalent`.
- `scripts/audit_v449_acc_metric_integrity.py` deve ser usado quando houver
  duvida sobre ACC, metric parity ou divergencia simbolica.

Status: corrigido em `scripts/analyze_eval_predictions.py`; gate e auditor V449
adicionados.

## Prompt Externo

Prompt consolidado para OpenRouter/outras APIs:

- `artifacts/openrouter/KG1_V438_ERROR_LEDGER_EXTERNAL_API_PROMPT_2026_05_15.md`

Uso esperado:

1. Perguntar a modelos externos como corrigir E003 sem violar adapter-only.
2. Exigir propostas testaveis em CPU antes de GPU.
3. Rejeitar respostas que recomendem broad SFT, mais epochs, weak/full leakage,
   runtime verifier, postprocessor ou submit com script.
