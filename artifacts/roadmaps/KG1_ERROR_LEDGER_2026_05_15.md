# KG1 Error Ledger

Atualizado: 2026-05-16

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

### E036 - Load manual de adapter com `target_parameters` podia mascarar regressao PEFT

Evidencia:

- Os arquivos OpenRouter de 16/05/2026 convergiram que a linha V480/V391
  precisava provar continuidade PEFT antes de novo gasto.
- A documentacao PEFT exige config correta quando o adapter original usa
  `target_parameters`; injecao por `state_dict` nao basta para esse caso.
- O launcher V391 ainda exportava `INIT_ADAPTER_LOAD_MODE='manual'` enquanto
  carregava `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`.

Impacto:

- Um treino poderia carregar/inicializar LoRA em namespace estrutural diferente
  do adapter baseline, reduzir `eval_loss` e ainda nao transferir ACC.
- Isso explica um modo de falha compativel com o plateau: loss mexe, mas
  `equation_transform`/`bit_manipulation` nao melhoram de forma submit-safe.

Regra preventiva:

- `scripts/hf_job_train_v90.py` agora usa `INIT_ADAPTER_LOAD_MODE=peft` por
  padrao.
- `scripts/hf_job_preflight_gate.py` bloqueia `INIT_ADAPTER_LOAD_MODE=manual`
  quando o init adapter tem `target_parameters`.
- `scripts/kg1_static_safety_gate.py` bloqueia launchers ativos que combinam
  MoE `LORA_TARGET_PARAMETERS` com `INIT_ADAPTER_LOAD_MODE='manual'`.
- Proxima etapa obrigatoria: gate CPU round-trip V485 antes de qualquer novo
  job pago.

Status: mitigado por gate; V485 seed metadata gate aprovado para V290
checkpoint-6. Proximo job ainda precisa passar o V485 embutido no launcher
antes de qualquer treino.

### E058 - V510/V511 bit replay answer-only nao transfere ACC

Evidencia:

- V511 executou tecnicamente correto no H200: dataset HF correto, MoE
  `gate_up/down` treinaveis, `lm_head` congelado, checkpoint salvo e upload
  completo.
- Mesmo assim o loss local piorou levemente: `2.8125 -> 2.8128`, sem sinal
  para weak eval.
- V513 auditou o dataset ativo V510 e encontrou `742/742` linhas de
  `bit_manipulation` no estilo `Final answer: \boxed{...}` sem CoT ou termos
  de regra; p50 de assistant bit = `3` palavras, `0` bit traces.

Impacto:

- O modelo recebe pressao para memorizar respostas binaria curtas, mas nao para
  reproduzir o algoritmo bit-pair/bitsum/stride descrito nas discussions.
- Isso explica o padrao recorrente: loss pode mexer, mas `bit_manipulation`
  perde guardrail ou nao sobe de forma submit-safe.

Regra preventiva:

- Bloquear novo GPU se o dataset ativo tiver share alto de bit answer-only.
- Exigir no minimo `32` bit traces deterministicas curtas antes de qualquer
  novo treino que envolva bit guardrail.
- Rerodar V513 apos qualquer alteracao de dataset/trace; so promover se
  `bit_trace_rows>=32`, `bit_answer_only_share<=0.05`, truncation zero e
  objective/FinOps gates passarem.

Status: mitigado estruturalmente pelo V514/V515, mas ainda aberto ate weak eval
provar que nao houve regressao. V514 converteu `581/742` linhas bit para traces
verificadas, descartou `161` bit sem prova, passou tokenization real e passou
V513 recheck com `0` blockers. HF CPU reproduction tambem passou no job
`felipesp1983/6a08e9ad3308d79117b91609`. V515 recuperou mais `8` bit rows
residuais com `fullbyte_unique_prediction` e passou V286/V513 localmente e no
HF CPU job `felipesp1983/6a08edcf3308d79117b9167f`. GPU continua bloqueada ate
objective/pre-paid gate e smoke minimo.

### E059 - V514 HF CPU launcher sem dependencia transitiva do stride solver

Evidencia:

- HF CPU job `felipesp1983/6a08e6fe3308d79117b915bb` clonou o commit correto
  `bcfd28b0f0fa399da2d7c36e4a2e43b935e5f0b9` e falhou antes dos gates.
- `scripts/build_v514_traceable_bit_v510_dataset.py` importa
  `solve_stride` de `scripts/run_v296_bit_stride_solver_audit.py`.
- O script V296 importa `pandas`, mas o launcher V514 instalava apenas
  `huggingface_hub`, `transformers` e `tokenizers`.

Impacto:

- Nenhum treino, eval, package ou submit rodou.
- Falha barata de dependencia no HF CPU, nao falha de dado ou metrica.

Regra preventiva:

- Launcher CPU/GPU que importa scripts auxiliares deve instalar tambem as
  dependencias transitivas desses scripts, ou mover a funcao reutilizada para
  modulo sem dependencia pesada.
- Antes de relancar, `py_compile` e static gate continuam obrigatorios; o
  log da tentativa falha deve ficar versionado para nao repetir o erro.

Status: corrigido no launcher V514 com `pandas>=2.0.0`; validado pelo job HF
CPU `felipesp1983/6a08e9ad3308d79117b91609`.

### E060 - V514 HF CPU tokenizer gate sem `jinja2`

Evidencia:

- HF CPU job `felipesp1983/6a08e83de48bea4538ba0468` clonou o commit correto
  `158183ed80b141ad97f067b0524c35c4a55f18d3`.
- V514 build reproduziu os counts locais no HF CPU: train bit `466`
  convertidas e `143` descartadas; val bit `115` convertidas e `18`
  descartadas.
- O tokenizer oficial baixou, mas `tokenizer.apply_chat_template` falhou
  porque `jinja2` nao estava instalado.

Impacto:

- Nenhum treino, eval, package ou submit rodou.
- O dataset V514 esta reprodutivel; a falha foi no ambiente minimo do
  tokenization gate.

Regra preventiva:

- Todo launcher que usa tokenizer real com `apply_chat_template` deve instalar
  `jinja2`.
- O HF CPU reproduction precisa executar builder, tokenizer real e V513 no
  mesmo ambiente antes de liberar qualquer GPU.

Status: corrigido no launcher V514 com `jinja2>=3.1.0`; validado pelo job HF
CPU `felipesp1983/6a08e9ad3308d79117b91609`.

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
- `scripts/audit_v449_acc_metric_integrity.py --self-test` agora valida os
  casos built-in sem depender de CSV externo.

Status: corrigido em `scripts/analyze_eval_predictions.py`; gate e auditor V449
adicionados.

### E012 - V448 clean trace SFT repetiu a regressao V444

Evidencia:

- V448 treinou em H200 usando o dataset V447 clean:
  - `1164` train rows;
  - `129` validation rows;
  - `0` `boxed` mismatch;
  - tokenization gate com `prompt_truncation_rate=0.0`,
    `completion_tokens_dropped=0`, `fallback_masks=0` e offset masks completas.
- O weak eval V221-contract avaliou `checkpoint-3`.
- Resultado do checkpoint-3:
  - total weak `190/315`;
  - `equation_transform=56/155`;
  - `bit_manipulation=134/160`;
  - `truncated=1`.
- Baseline submit-safe permanece `192/315`, `equation=56/155`,
  `bit=136/160`, `truncated=0`.
- O job `felipesp1983/6a07832e3308d79117b90a27` foi cancelado por FinOps antes
  de avaliar `checkpoint-6`.

Impacto:

- A limpeza V447 corrigiu contradicoes de `boxed` e validou tokenizacao, mas nao
  resolveu o problema central de transferencia para adapter-only.
- O adapter continuou preso no teto `equation=56` e perdeu `2` acertos em bit.
- O padrao e o mesmo de V444, portanto a classe "target-aligned trace SFT" nao
  pode ser repetida como se fosse nova evidencia.

Regra preventiva:

- Nao repetir V448 com mais epochs, mais steps, LR sweep, H200 maior ou outro
  checkpoint do mesmo dataset.
- Nova GPU so pode ser aberta se um gate CPU provar uma classe de evidencia
  diferente: resposta curta emitivel pelo adapter, DSL label-free com abstain ou
  parser/metric fix que altere ACC estrita.
- Se o primeiro checkpoint de qualquer rota futura repetir `equation=56`,
  `bit<136` ou `truncated>0`, cancelar imediatamente.

Status: V448 weak eval cancelado por FinOps; sem full/package/submit.

### E013 - DSL numerica global com LOO ainda gera falsos positivos em V452

Evidencia:

- V452 auditou `133` probes V439 public-train, incluindo `120` rows de
  `equation_transform`.
- A expansao numerica v2 encontrou `5` candidatos globais com LOO, mas todos
  eram incorretos contra o label publico:
  - `numeric_v2_global_abs_diff_revop0_revres0`;
  - `numeric_v2_global_add_minus1_revop0_revres0`;
  - `numeric_v2_global_add_revop0_revres0`;
  - `numeric_v2_global_max_mod_min_revop0_revres0`;
  - `numeric_v2_global_mul_revop1_revres1`.
- O unico sinal seguro foi `v274_guarded_numeric_minus_direct_negative_restore_sign`:
  `2` candidatos, `2` corretos, `0` incorretos.
- Resultado final V452: `2` pares certificados, `1` modo independente,
  `hf_gpu_allowed=false`.

Impacto:

- Explicar exemplos locais com uma regra numerica nao e suficiente para criar
  treino. Em `equation_transform`, regras globais plausiveis podem falhar no
  target mesmo com LOO.
- Abrir H200 a partir de V452 seria gasto sem base tecnica: o dataset tem poucos
  pares e pouca diversidade.

Regra preventiva:

- Toda nova DSL de equation deve reportar candidatos incorretos por classe e
  bloquear a classe inteira se houver qualquer conflito.
- O gate minimo para GPU exige pelo menos quatro modos independentes no-loss,
  nao apenas quatro linhas ou quatro candidatos.
- Regras numericas globais so podem promover dataset se passarem por classe
  safe, row-conflict gate e auditoria posterior com `verify_answer`.

Status: V452 bloqueou GPU corretamente; proxima rota volta para mineracao CPU
de regras novas.

### E014 - V459 detail sem prompt bloqueou o builder V460

Evidencia:

- A primeira execucao de `scripts/build_v460_numeric_one_rule_micro_dataset.py`
  falhou com `KeyError: 'prompt'`.
- Causa: `scripts/build_v459_v458_numeric_hard_negative_audit.py` preservava
  `raw_output`, predicao e hashes, mas nao escrevia a coluna `prompt` no
  detail CSV.
- O erro foi pego antes de qualquer HF GPU ou upload de dataset.

Impacto:

- Nenhum treino, submit ou package foi criado com o artefato incompleto.
- O impacto foi limitado ao CPU builder V460.

Regra preventiva:

- Todo artefato intermediario que possa alimentar dataset precisa carregar
  explicitamente `id`, `family`, `prompt`, `answer` quando permitido,
  `prediction`, hashes de prompt, decode config e identidade do adapter.
- Builders subsequentes devem falhar cedo se `prompt` ou `answer` permitido
  estiver ausente.
- Rerodar self-test, `py_compile` e `kg1_static_safety_gate.py` apos corrigir.

Status: corrigido. V459 agora inclui `prompt`, V459 foi rerodado, V460 foi
construido e o tokenization gate V286 passou com truncation `0`.

### E015 - V465 preflight bloqueou subcategoria de bit herdada do V217

Evidencia:

- O primeiro V465 launch chegou apenas ao `hf_job_preflight_gate.py --phase artifacts`.
- O job falhou antes de treino com:
  `RuntimeError: KG1_REQUIRED_TRAIN_SUBCATEGORIES missing from train: ['bit_guardrail_replay']`.
- Causa: o builder V464 escrevia `subcategory=bit_guardrail_replay` no topo da
  linha, mas preservava `metadata.subcategory`/`metadata.subtype` herdados do
  V217. O preflight do job prioriza `metadata.subcategory`, logo contou varias
  linhas de bit como `bit_manipulation` ou `unknown`.

Impacto:

- Nenhum treino, checkpoint, package, full eval ou submit foi criado.
- O preflight funcionou corretamente e evitou gastar H200 em dataset com schema
  inconsistente.

Regra preventiva:

- Todo builder que reusa replay rows precisa sobrescrever tambem
  `metadata.subcategory` e `metadata.subtype`, nao apenas a coluna top-level.
- O debug do launcher deve validar os mesmos `KG1_REQUIRED_*` que o job remoto
  usa, e qualquer mismatch bloqueia launch.

Status: linha V464/V465 encerrada. O builder V464 agora e fail-closed por
padrao, launch/upload/eval V465/V466 foram arquivados, e nenhum rerun dessa
rota pode ser usado sem nova versao limpa.

### E016 - V464 trace rejeitava candidato igual ao gabarito

Evidencia:

- Auditoria em crisis mode encontrou traces `equation_transform` do V464 em
  que o texto dizia `candidate 'X' is rejected`, mas `X` era igual ao
  `answer` pelo `verify_answer`.
- Contagem no V464 antigo:
  - train: `24/46` linhas equation com candidato rejeitado igual ao gabarito;
  - validation: `6/10` linhas equation com candidato rejeitado igual ao
    gabarito.
- Exemplo: `37:67` com answer `30`; a trace dizia que o candidato `'30'` era
  rejeitado e terminava com `\boxed{30}`.
- V466 avaliou V465 treinado sobre esse dataset contaminado e nao obteve ganho:
  checkpoint-4 `189/315`, checkpoint-8 `192/315`, `equation=56`, `truncated=1`.

Impacto:

- V465/V466 nao podem ser usados como evidencia positiva de treino.
- O erro explica um padrao observado: `eval_loss` pode cair sem melhorar ACC,
  porque a supervisao textual estava semanticamente contraditoria.
- O job V466 foi cancelado por FinOps antes de gastar em checkpoint-12/16/final.

Regra preventiva:

- Builders de trace precisam gravar `metadata.rejected_candidate` e
  `metadata.rejected_candidate_source`.
- Nenhuma linha pode dizer que rejeita candidato que verifica igual ao
  gabarito.
- O gate V286 agora falha se `metadata.rejected_candidate` ou o texto
  `candidate 'X' is rejected` bater com o `answer`.
- `hf_job_train_v90.py --self-test` agora e um self-test real; argumentos
  desconhecidos nao podem iniciar treino por engano.

Status: V468 corrigiu a contradicao `rejected_candidate == answer`, mas depois
foi bloqueado em V472/V473 por seed exata de referencia full. V468/V469/V470
ficam encerrados para treino/eval GPU; nenhuma rota GPU esta liberada a partir
dessa familia de artefatos.

### E017 - Evaluator podia gerar accuracy falso quando `answer` estava ausente

Evidencia:

- A auditoria de crise identificou que `evaluate_lora_adapter.py` e
  `evaluate_lora_adapters_batch.py` aceitavam CSV de solucao sem coluna
  `answer`.
- Nesse caso, o fluxo podia preencher `correct=False` e produzir
  `accuracy=0.0000`, que parece resultado de modelo mas e erro de entrada.

Impacto:

- Um job/notebook poderia parecer ter ACC horrivel por modelo, quando na verdade
  a avaliacao estava sem gabarito.
- Isso afeta diretamente decisoes de FinOps, checkpoint e roadmap.

Regra preventiva:

- Qualquer eval weak/full/local precisa falhar cedo se `answer` nao existir no
  CSV de solucao ou no merge final.
- O avaliador tambem precisa falhar se `len(outputs) != len(prompts)`.

Status: corrigido em `scripts/evaluate_lora_adapter.py` e
`scripts/evaluate_lora_adapters_batch.py`.

### E018 - Truncation podia remover tokens supervisionados

Evidencia:

- `hf_job_train_v90.py` fazia left truncation quando `len(full_ids) >
  MAX_LENGTH`.
- Antes do V471, o job contava truncation/prompt truncation, mas nao bloqueava
  explicitamente o caso em que o overflow cortava tokens com `loss_mask > 0`.

Impacto:

- O treino poderia otimizar loss sobre completion amputada.
- Isso ajuda a explicar situacoes em que loss cai, mas ACC de familia nao sobe.

Regra preventiva:

- Se qualquer token supervisionado seria removido por truncation, o treino deve
  abortar antes de GPU.

Status: corrigido em `scripts/hf_job_train_v90.py`.

### E019 - Weak promotion gate nao era bloqueante no job de weak eval

Evidencia:

- V470 avaliou checkpoint ruim e o launch manifest original ficou com status
  operacional `RUNNING`, exigindo auditoria manual posterior para registrar
  rejeicao.
- O resultado terminal foi `190/315`, `equation=56`, `bit=134`,
  `truncated=1`, abaixo do piso submit-safe.

Impacto:

- Jobs ruins podiam terminar como execucao tecnica bem-sucedida, sem quebrar a
  pipeline.
- Isso aumenta risco de gastar GPU em rotas que ja falharam no primeiro
  checkpoint.

Regra preventiva:

- Weak eval precisa registrar e, quando configurado, impor gate:
  `total>=193`, `equation>=57`, `bit>=136`, `truncated=0`.

Status: corrigido em `scripts/hf_job_weak_eval_v245.py`; launcher V470 novo ja
define `KG1_ENFORCE_WEAK_PROMOTION_GATE=1`.

### E020 - Resposta simbolica podia parecer boxed, mas nao ser extraivel pelo metric path

Evidencia:

- Datasets com braces/backslashes podem passar por verificacao visual de
  `\boxed{...}` e ainda falhar se `extract_final_answer` nao recuperar o mesmo
  valor.
- V447 foi auditado e os arquivos existentes verificam corretamente, mas a
  lacuna existia no gate generico.

Impacto:

- Datasets poderiam treinar a forma textual errada; loss cairia sem melhorar
  ACC.

Regra preventiva:

- O gate de tokenizacao deve executar `extract_final_answer(assistant_text)` e
  validar o resultado com `verify_answer(answer, extracted)`.
- Builders de trace so devem manter/apendar final answer quando o resultado
  final passa pela mesma rota de metrica.

Status: corrigido em `scripts/run_v286_generic_tokenization_gate.py` e
`scripts/build_v447_v446_trace_dataset.py`.

### E021 - Builders historicos mantinham gates fracos obsoletos

Evidencia:

- Quadruple check V472 encontrou builders de Colab V217-V231/V244/V245 ainda com:
  - `WEAK_BIT_MIN_FOR_FULL = 133`;
  - `WEAK_MAX_TRUNC_FOR_FULL = 3`.
- Isso nao afetava notebooks ja versionados diretamente, mas poderia recriar
  notebook ou manifest com criterio de promocao inferior ao piso submit-safe
  atual.

Impacto:

- Risco de full eval/submission ser liberado em rota com regressao de bit ou
  truncation.

Regra preventiva:

- Builders foram atualizados para `WEAK_BIT_MIN_FOR_FULL = 136` e
  `WEAK_MAX_TRUNC_FOR_FULL = 0`.
- `kg1_static_safety_gate.py` agora falha se detectar esses valores antigos em
  arquivo novo/alterado.

Status: corrigido em V472.

### E023 - V468 ainda herdava uma seed exata de referencia full

Evidencia:

- Auditoria independente V472 encontrou o prompt/answer `63-19 -> -55` no
  JSONL de treino V468, antes da remocao operacional V473.
- A mesma combinacao bate com a referencia full local `v291_full_predictions`
  no id `7688e06e`.
- A origem rastreada foi a rota V461/V463.

Impacto:

- Mesmo com a contradicao `rejected_candidate == answer` corrigida, V468 nao e
  limpo para treino porque contem contaminacao de referencia full.

Regra preventiva:

- `hf_job_preflight_gate.py` bloqueia V461/V463/V464/V468 por marcador.
- `run_v286_generic_tokenization_gate.py` agora tambem bloqueia esses
  marcadores e aceita CSVs de referencia proibida para falhar por overlap de
  `id`, `prompt` ou `prompt+answer`.

Status: corrigido em V472. V468/V469/V470 ficam encerrados para GPU.

### E024 - V447 incluia traces `hypothesis_formed` contraditorios

Evidencia:

- Auditoria V472 contou `141` traces `hypothesis_formed` em V447.
- Nesses casos o raciocinio podia conter um `\boxed{...}` interno diferente e,
  depois, apendar a resposta oficial correta.

Impacto:

- O treino pode aprender uma trajetoria contraditoria: loss cai, mas o padrao
  de decisao nao melhora ACC.

Regra preventiva:

- `run_v446_tong_source_target_alignment_gate.py` nao aceita mais
  `hypothesis_formed`.
- `build_v447_v446_trace_dataset.py` descarta qualquer status diferente de
  `rule_found` nessa rota.

Status: corrigido em V472. O dataset V447 existente fica quarantined.

### E025 - Postprocessor de eval rodava antes de truncation existir

Evidencia:

- `evaluate_lora_adapter.py` e `evaluate_lora_adapters_batch.py` aplicavam o
  postprocessor antes de popular `truncated`.
- V274 abstain em linhas truncadas, mas nao recebia essa informacao.

Impacto:

- Linha finalizada por `length` poderia receber override e contaminar ACC de
  weak/full eval.

Regra preventiva:

- Eval agora popula `truncated` e `truncated_bool` antes de qualquer
  postprocessor.

Status: corrigido em V472.

### E026 - CSV de eval podia perder zeros a esquerda e ids corretos

Evidencia:

- Leituras via `pd.read_csv` sem `dtype=str` ainda existiam em eval/analyze.

Impacto:

- Respostas como `03`, ids numericos e campos vazios poderiam ser coercedidos,
  gerando ACC falso ou comparacao errada.

Regra preventiva:

- CSVs de eval/analyze agora usam `dtype=str, keep_default_na=False`.
- `id`/`row_id` e campos criticos vazios passam a falhar explicitamente.

Status: corrigido em V472.

### E027 - V300 bit full-byte podia escolher programa ambiguo

Evidencia:

- O solver full-byte retornava o primeiro programa compativel com exemplos.

Impacto:

- Em prompts subdeterminados, programas diferentes poderiam bater nos exemplos
  e divergir na query, criando override regressivo.

Regra preventiva:

- V300 agora coleta todos os programas compativeis e so aplica quando a
  predicao da query e unica.

Status: corrigido em V472.

### E022 - V464 contaminado precisava de bloqueio operacional, nao so documentacao

Evidencia:

- Data integrity scan V472 confirmou novamente que V464 antigo contem
  `24` linhas train e `6` linhas validation onde o rejected candidate verifica
  igual ao gabarito.
- V468 corrigiu parcialmente a geracao, mas V464 continuava rastreado ate a
  remocao operacional V473.

Impacto:

- Antes de V473, um launch HF manual poderia apontar para o dataset V464 antigo
  e repetir a rota contaminada.

Regra preventiva:

- `hf_job_preflight_gate.py` agora bloqueia qualquer training identity que
  contenha `v464_v463_numeric_multirule_dataset`.
- A mensagem instrui criar dataset posterior limpo; V468 tambem esta
  quarentenado por overlap de referencia.

Status: corrigido em V472.

Atualizacao V473: os JSONL/manifests rastreados de V447/V464/V468 foram
removidos da arvore ativa; o historico fica registrado no ledger e no relatorio
`V473_QUARANTINED_ARTIFACT_REMOVAL.md`.

### E028 - Launchers e defaults antigos podiam reabrir rotas quarentenadas

Evidencia:

- Scan V473 em `scripts`, `src` e `artifacts` encontrou defaults antigos em
  argparse: `--weak-bit-min=133` e `--weak-trunc-max=3`.
- Launchers V448/V462/V465/V466/V469/V470 ainda eram executaveis e apontavam
  para datasets/adapters derivados de V447/V461/V464/V468.

Impacto:

- Mesmo com o roadmap correto, um comando manual poderia relancar uma rota
  sabidamente contaminada ou promover com threshold frouxo.
- Isso cria risco direto de gasto HF e ACC falso.

Regra preventiva:

- Defaults antigos foram atualizados para bit `136` e truncation `0`.
- Launchers das rotas quarentenadas agora falham imediatamente com
  `RuntimeError` fail-closed.
- `kg1_static_safety_gate.py` agora bloqueia defaults argparse antigos e
  adapter repos derivados de V448/V465/V469.

Status: corrigido em V473.

### E029 - Package podia desamarrar manifest avaliado do adapter baixado

Evidencia:

- Auditoria V473 apontou que `scripts/package_hf_adapter_submission.py`
  validava `full_candidate_gate`, mas baixava `repo/subfolder` sem exigir
  revision imutavel nem comparar hashes avaliados do adapter.
- O default antigo `--min-full-correct=823` ficava abaixo do piso operacional
  atual de package novo.

Impacto:

- Um manifest bom poderia autorizar empacotar outro checkpoint ou a ponta
  `latest` do mesmo repo.
- Isso podia gerar submit com adapter diferente do que foi medido.

Regra preventiva:

- Package novo exige manifest official-like V284, controles oficiais
  (`max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, prompt suffix
  oficial e sem postprocessor), row contract full esperado, repo/subfolder
  iguais ao adapter avaliado, revision/resolved_revision imutavel e hashes de
  `adapter_config.json`/`adapter_model.safetensors`.
- Default de package passa a `--min-full-correct=831`; reempacotar V291
  historico exige decisao explicita fora da rota nova.

Status: corrigido em V473; static safety gate agora protege esses snippets.

### E030 - Metric parser de `\boxed{}` podia consumir texto depois da resposta

Evidencia:

- `extract_boxed_answers` procurava o ultimo `}` antes do proximo `\boxed{}`.
  Em saidas como `\boxed{42}. \text{done}`, o payload podia virar
  `42}. \text{done`, produzindo falso negativo de ACC.

Impacto:

- ACC, error mining e gates podiam marcar resposta correta como errada quando
  havia texto LaTeX depois da caixa final.

Regra preventiva:

- Parser central agora e balanceado: fecha no par correspondente de `\boxed{`,
  preserva payloads aninhados como `\frac{1}{2}` e ignora braces posteriores.
- Self-tests adicionados cobrem texto pos-box e payload LaTeX aninhado.

Status: corrigido em V473.

### E031 - Gates de eval podiam sair com sucesso apesar de gate falso

Evidencia:

- Jobs V276/V277/V284 escreviam `full_candidate_gate=false` ou
  `weak_gate_pass=false`, mas `main()` retornava `0`.

Impacto:

- Automacao externa poderia interpretar job verde como promovivel.

Regra preventiva:

- Esses jobs agora falham por padrao quando o gate falha. Modo diagnostico
  precisa ser opt-in explicito com `KG1_ALLOW_FAILED_GATE_EXIT_0=1`.

Status: corrigido em V473.

### E032 - Parser/eval podiam subextrair respostas simbolicas com braces literais

Evidencia:

- Auditoria V474 mostrou que a correcao balanceada de V473 resolvia texto
  LaTeX apos a resposta, mas podia subextrair payloads simbolicos como
  `\boxed{?}}`, `\boxed{{17}` e respostas com `{`, `}` ou `\` literais.
- Isso criava falso negativo de ACC em datasets com respostas de
  `text_encryption`/simbolicas e podia invalidar o tokenization gate para
  datasets corretos que usassem escape adequado.

Impacto:

- Loss podia parecer saudavel, mas a metrica de eval podia marcar correto como
  errado no parser.
- Novos datasets poderiam ser aprovados com boxed target concatenado de forma
  insegura, ou rejeitados quando escapados corretamente.

Regra preventiva:

- `extract_final_answer_for_expected` usa o answer conhecido para desambiguar
  payloads boxed no caminho de eval.
- `evaluate_lora_adapter.py` e `evaluate_lora_adapters_batch.py` usam esse
  helper quando a coluna `answer` existe.
- `run_v286_generic_tokenization_gate.py` usa `box_answer(answer)` para modos
  `boxed_*` e valida com extracao expected-aware.
- `kg1_static_safety_gate.py` passa a exigir esses snippets.

Status: corrigido em V474.

### E033 - Flags antigas V461/V463 podiam reabrir rota quarentenada

Evidencia:

- V461 ainda gravava `hf_raw_probe_allowed=true` em builder/manifest.
- V463 ainda gravava `v464_dataset_build_allowed=true` em builder/manifest.
- Essas flags contradiziam a decisao V473 de quarentenar V464/V468 e adapters
  derivados.

Impacto:

- Um operador ou launcher manual poderia interpretar os artefatos antigos como
  autorizacao para novo raw probe/dataset/GPU, reabrindo a rota contaminada.

Regra preventiva:

- V461 agora e fail-closed: `hf_raw_probe_allowed=false` e
  `quarantined_after_v473=true`.
- V463 agora e fail-closed: `v464_dataset_build_allowed=false` e a condicao
  `route_not_quarantined_after_v473=false` fica no manifest.
- A rota so pode voltar em V475+ com pack novo, fonte isolada e gates de
  contradicao/referencia proibida.

Status: corrigido em V474.

### E034 - Package podia aceitar manifest sem controle official-like completo

Evidencia:

- `package_hf_adapter_submission.py` validava max tokens/model len/modelo e
  hashes de adapter, mas nao exigia `official_like_control_gate` nem `strict`.
- Manifest antigo com `KG1_OFFICIAL_LIKE_STRICT=0` ou sem
  `gpu_memory_utilization=0.85` poderia passar se outros campos batessem.

Impacto:

- Risco de package submit-safe ser criado a partir de full eval que nao seguiu
  exatamente o contrato oficial-like.

Regra preventiva:

- V284 passa a gravar `official_like_control_gate` no manifest final.
- O package agora exige `repo_commit`, `strict=true`, max tokens/model len/seqs,
  `gpu_memory_utilization=0.85`, ausencia de postprocessor e hashes/revision do
  adapter.

Status: corrigido em V474.

### E035 - Treino incremental podia perder `target_parameters` do adapter inicial

Evidencia:

- V480 usou o adapter inicial
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.
- O preflight remoto registrou que esse adapter inicial tem
  `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]`.
- O proprio log de treino V480 registrou `LoRA target_parameters: disabled` e
  `Require LoRA target-parameter match: False`.
- Os checkpoints V480 foram salvos com `adapter_config.json` de `1149` bytes e
  `target_parameters=null`, enquanto o seed V290 tem `1210` bytes e preserva os
  dois `target_parameters`.
- V481 confirmou que os checkpoints V480 nao eram submit-safe: melhor ponto
  observado `191/315`, equation `57/155`, bit `134/160`, trunc `1`.

Impacto:

- O job podia aparentar carregar `12011` tensores e treinar normalmente, mas a
  configuracao PEFT salva nao preservava a receita declarada da linhagem V290.
- Isso criava um bug silencioso: loss/eval_loss se moviam, porem a transferencia
  para ACC weak regredia bit e truncation.

Regra preventiva:

- `hf_job_preflight_gate.py` agora compara `target_modules` e
  `target_parameters` do adapter inicial contra o ambiente do job quando
  `KG1_STRICT_INIT_ADAPTER_CONFIG=1`.
- Se o adapter inicial tem `target_parameters`, o preflight exige
  `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1`.
- `kg1_static_safety_gate.py` bloqueia launchers ativos que zerem
  `LORA_TARGET_PARAMETERS` ou desliguem a verificacao para `mlp.experts.*`.
- O launcher base V391 agora injeta
  `KG1_LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`.

Status: corrigido em V482; proximo job pago deve confirmar no log que
`target_parameter_lora_tensors` nao esta vazio antes de qualquer weak eval.

Atualizacao V483: painel OpenRouter (`openai/gpt-5.3-codex`,
`anthropic/claude-opus-4.7`, `google/gemini-3.1-pro-preview`,
`qwen/qwen3-max-thinking`, `deepseek/deepseek-v4-pro`) confirmou o mesmo
diagnostico operacional: o proximo passo correto e um CPU preflight de
round-trip PEFT nativo, preferindo `PeftModel.from_pretrained(...,
is_trainable=True)`, e comparando `adapter_config.json`, keys, shapes, dtypes,
coverage de `mlp.experts.*` e parametros trainable antes de qualquer GPU.

### E036 - Peso de equation dominava o objetivo efetivo antes do treino

Evidencia:

- O job H200 `felipesp1983/6a086ffce48bea4538b9fc0f` foi lancado com V391 e
  parado pelo gate V478 antes de qualquer treino util.
- A combinacao `equation_numeric_* = 10.00` gerou `bit_manipulation` com apenas
  `0.135975` do objetivo efetivo de treino e `equation_transform` com
  `0.864025`.
- Isso violou os limites `min_bit_effective_share=0.20` e
  `max_equation_effective_share=0.80`.

Impacto:

- O treino parecia agressivo para equation, mas na pratica deixava bit sem
  pressao suficiente e aumentava risco de repetir o padrao `equation=57`,
  `bit=135`, que nao e submit-safe.

Regra preventiva:

- V478 permanece obrigatorio antes de qualquer HF GPU job.
- V486 usa `equation_numeric_* = 6.00`, que passa o gate com
  `bit_manipulation=0.207788` e `equation_transform=0.792212` no objetivo
  efetivo de treino.
- A promocao continua dependendo de weak micro-ACC: `total>192`,
  `equation>56`, `bit>=136`, `truncated=0`.

Status: corrigido para o proximo smoke V486; V391 ficou registrado como
rejeicao FinOps correta.

### E037 - Filtro de treino nao aplicava alias Nemotron para `target_parameters`

Evidencia:

- O job H200 V486 `felipesp1983/6a0872193308d79117b910a1` passou V485, V478,
  postinstall, tokenizacao e carregou o adapter inicial via PEFT.
- Antes do primeiro optimizer step, `scripts/hf_job_train_v90.py` falhou em
  `apply_trainable_lora_module_filter`:
  `LORA_TARGET_PARAMETERS were configured but no matching LoRA tensors were found:
  mlp.experts.gate_up_proj, mlp.experts.down_proj`.
- A causa foi que V485 aceitava corretamente o alias estrutural
  `mlp.experts.gate_up_proj -> mixer.experts.<id>.up_proj`, mas o script de
  treino ainda fazia busca literal por nome.

Impacto:

- Job pago podia passar gates de metadata e falhar somente apos baixar modelo e
  adapter, desperdicando tempo de H200 antes de qualquer step.

Regra preventiva:

- `hf_job_train_v90.py` agora usa `target_parameter_name_matches`, com a mesma
  regra de alias da V485.
- O self-test do treino cobre `gate_up_proj -> up_proj` e `down_proj`.
- `kg1_static_safety_gate.py` passa a exigir a presenca desse matcher no script
  de treino.

Status: corrigido para V487.

### E038 - V487 corrigiu `target_parameters`, mas V488 ainda regrediu bit/truncation

Evidencia:

- O treino H200 V487 `felipesp1983/6a0876033308d79117b910bf` completou com
  `target_parameter_lora_tensors=5934/5934` e checkpoints
  `2/4/6/8/10/12/final`.
- O melhor checkpoint por `eval_loss` foi o checkpoint-10 (`eval_loss=1.3519`).
- A weak eval focada V488
  `felipesp1983/6a087d493308d79117b91108` no checkpoint-10 retornou
  `191/315`, `equation_transform=57/155`, `bit_manipulation=134/160` e
  `truncated=1`.
- O gate de promocao bloqueou por `correct_lt_193`, `bit_lt_136` e
  `truncated_gt_0`.

Impacto:

- O bug de continuidade PEFT/alias era real e foi corrigido, mas nao era o
  unico gargalo.
- `eval_loss` melhor e `target_parameters` ativos ainda nao garantem ganho
  submit-safe.
- Repetir o mesmo SFT ou varrer todos os checkpoints em H200 tem baixa chance
  de retorno sem novo sinal CPU, pois o ganho de equation veio acompanhado de
  regressao de bit e truncation.

Regra preventiva:

- Depois de qualquer correcao estrutural de treino, a primeira weak eval deve
  manter `bit>=136` e `truncated=0`; se falhar, bloquear novo job pago na mesma
  receita.
- Antes de novo H200, baixar apenas artefatos pequenos de predictions/report e
  fazer diff por linha contra V291/V290/V477 para localizar regressao real.
- Candidate order deve ser focado por evidencia previa; nao varrer todos os
  checkpoints em H200 se o tempo projetado ultrapassar a janela FinOps.

Status: V488 bloqueado; proxima acao e diff CPU de predictions e gate de
regressao antes de novo treino pago.

### E039 - `target_parameters` carregados nao significavam `target_parameters` treinados

Evidencia:

- A auditoria V489 comparou V488 checkpoint-10 contra o baseline V290/V291 e
  confirmou que a metrica estrita estava correta: V488 teve `191/315`,
  `equation_transform=57/155`, `bit_manipulation=134/160`, `truncated=1`.
- O diff linha a linha encontrou um ganho real em equation (`518deb39`) e duas
  regressoes reais de bit (`8740ed31`, `59bee375`), sendo uma por truncation.
- O manifesto V487 registrava `target_parameters` na config, mas nao registrava
  contadores de `target_parameter_trainable_lora_tensors`.
- A receita V487 usava
  `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj` e
  `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head`; portanto os
  LoRA de `up_proj/down_proj` podiam estar ativos no forward, mas congelados
  no treino.

Impacto:

- O resultado podia ser interpretado incorretamente como "MoE target_parameters
  foram treinados", quando o que estava provado era apenas "MoE
  target_parameters foram carregados e ficaram ativos".
- Isso e um F2/silent-observability bug: nao corrompe a metrica, mas pode levar
  a nova rodada de GPU com a hipotese errada.

Regra preventiva:

- `scripts/hf_job_train_v90.py` agora grava
  `target_parameter_trainable_lora_tensors`,
  `target_parameter_trainable_lora_params`,
  `target_parameters_trainability_mode` e
  `trainable_parameter_report_after_filter` no manifesto final.
- `scripts/kg1_static_safety_gate.py` exige esses campos de observabilidade no
  script de treino.
- Launchers com MoE `target_parameters` e allowlist `TRAINABLE_LORA_MODULES`
  precisam declarar explicitamente
  `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` ou `1`.
- O launcher V487 foi marcado como `0`, porque aquela receita era
  frozen-active, nao treino de `up_proj/down_proj`.

Status: corrigido em V489. Proximo job pago so pode ser interpretado se o
manifesto declarar `target_parameters_trainability_mode`.

### E040 - Expected-aware extraction podia escolher boxed anterior

Evidencia:

- A auditoria independente apontou que `extract_final_answer_for_expected`
  verificava todos os `\boxed{}` do texto e podia escolher um payload anterior
  se ele batesse com o gabarito.
- Isso nao afetou o V488 por essa via, mas e leakage de avaliacao: o correto e
  usar o gabarito apenas para desambiguar payload simbolico no ultimo boxed.
- No V488, a diferenca real entre extracao simples e expected-aware e uma linha
  simbolica (`4bb8c6cd`), em que o ultimo boxed contem `]}\!` e a extracao
  simples parava em `]`.

Impacto:

- Um run poderia parecer melhor no weak gate se o modelo colocasse a resposta
  correta no raciocinio e uma resposta final errada no ultimo boxed.

Regra preventiva:

- `src/competition_utils.py` agora restringe a expected-aware extraction ao
  ultimo `\boxed{}` e valida o payload com `verify_answer`.
- `scripts/audit_v449_acc_metric_integrity.py` agora possui auditoria raw
  `simple_extracted` vs `expected_aware_extracted`.
- O self-test cobre o caso em que o boxed anterior esta correto e o final esta
  errado; o final errado deve permanecer errado.

Status: corrigido em V489.

### E041 - Static gate anulava snippets criticos por chave duplicada

Evidencia:

- `scripts/kg1_static_safety_gate.py` tinha duas entradas
  `scripts/hf_job_train_v90.py` em `CRITICAL_SNIPPETS`.
- Em dict Python, a segunda entrada sobrescreve a primeira; com isso checks de
  alias Nemotron, trainability de target_parameters e manifesto do filtro
  LoRA nao eram de fato exigidos.

Impacto:

- O gate podia retornar `ok=true` mesmo se um refactor removesse checks
  criticos de F2/target-parameter observability.

Regra preventiva:

- As entradas foram consolidadas.
- O static gate agora exige explicitamente:
  `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE`,
  `target_parameter_trainable_lora_tensors`,
  `target_parameters_trainability_mode`,
  `trainable_lora_module_filter` e
  `expected-aware uses strict verifier`.

Status: corrigido em V489.

### E042 - Preflight SFT nao registrava flags gate/weak/full por linha

Evidencia:

- A auditoria independente apontou que `hf_job_preflight_gate.py` validava
  JSONL, IDs e assistant messages, mas nao registrava os flags
  `gate_rows_used_for_training`, `weak_gate_rows_used_for_training` e
  `full_gate_rows_used_for_training`.
- O gate de preferencia ja fazia esse tipo de check; a rota SFT ainda nao.

Impacto:

- Um dataset futuro poderia carregar metadados de contaminacao sem o preflight
  SFT bloquear antes de GPU.

Regra preventiva:

- `hf_job_preflight_gate.py` agora conta flags presentes/ausentes e falha se
  qualquer flag presente nao for `false`.
- `kg1_static_safety_gate.py` exige essa protecao.

Status: corrigido em V489.

### E043 - V475 SFT transfer ganhou equation mas regrediu bit

Evidencia:

- V495 treinou tecnicamente correto no dataset V475:
  `target_parameters` MoE treinaveis, `up_proj/down_proj` ativos no treino,
  `lm_head` congelado e `ANSWER_SPAN_LOSS_WEIGHT=1.0`.
- V496 weak eval retornou `191/315`, `equation_transform=57/155`,
  `bit_manipulation=134/160`, `truncated=1`.
- Diff estrito vs V290 checkpoint-6:
  - ganho `518deb39` em equation;
  - perdas `8740ed31` e `59bee375` em bit;
  - `59bee375` virou saida nao-binaria `2`.
- O extra expected-aware `4bb8c6cd` ja existia no baseline V290 e nao e
  aprendizado novo do adapter.

Impacto:

- A hipotese "V475 CPU projection 196/315 vai transferir diretamente para LoRA"
  falhou na primeira avaliacao submit-safe.
- Repetir H200 SFT amplo com o mesmo tipo de dado tende a trocar erro por erro
  e nao melhora ranking.

Regra preventiva:

- Bloquear novos H200 SFT V475/V390/V326 sem novo sinal CPU.
- Exigir projecao CPU com `equation>=60`, `bit>=136`, `trunc=0` antes de novo
  treino pago.
- Todo candidato deve passar bit guardrail que rejeita nao-binario e protege
  rows equivalentes a `8740ed31` e `59bee375`.

Status: aberto; proxima acao e CPU residual audit/teacher antes de GPU.

### E044 - H200 weak eval demora por completion tokens, nao por hardware

Evidencia:

- V496 gerou `1,504,306` completion tokens para 315 rows.
- V496 generation elapsed: `516.9s`; V290 checkpoint-6 baseline: `470.2s`.
- O run usa settings official-like: thinking ligado, `max_tokens=7680`,
  `max_model_len=8192`, `max_num_seqs=64`.

Impacto:

- A H200 parece "lenta", mas o custo real vem do protocolo de avaliacao longo.
- Rodar weak full para cada tentativa desperdiça budget quando o candidato
  ainda nao tem sinal CPU forte.

Regra preventiva:

- Avaliacoes com `disable_thinking`, `max_tokens` menor ou subset de rows sao
  diagnostico-only e precisam ser re-baselined contra V290 antes de conclusao.
- Weak eval official-like so roda para finalistas que passaram CPU gate e
  guardrail de bit.

Status: regra ativa de FinOps/performance.

### E045 - CPU solver gain nao transfere automaticamente para LoRA

Evidencia:

- V497 CPU residual transfer audit:
  - baseline V290 checkpoint-6: `192/315`;
  - V324 CPU solver projection: `196/315`, com +4 em equation;
  - V496 adapter transfer: `191/315`, com apenas +1 equation e -2 bit.
- Os quatro ganhos V324 que nao transferiram foram:
  `7688e06e`, `274def88`, `d1bd7478`, `c5b058d6`.
- O unico ganho V496 foi `518deb39`, que e simbolico/pontuacao e nao veio da
  regra numeric V324.

Impacto:

- O problema atual nao e o calculo de ACC nem um bug simples de loss.
- O problema e transformar uma regra CPU correta em comportamento gerativo
  estavel do adapter sem quebrar bit.

Regra preventiva:

- Nenhum novo H200 SFT deve rodar apenas porque existe projecao CPU.
- Primeiro criar trace/teacher curto, deterministico, com hard negatives e
  guardrail de bit, e provar em CPU que o pacote preserva `bit>=136`,
  `trunc=0`, `total>192`.

Status: mitigacao em andamento; V498 foi criado, passou tokenization/leakage
gate real e V499 local debug. A prova final ainda depende de weak eval do
checkpoint V499.

### E046 - Manifest decision schema mismatch no uploader

Evidencia:

- O primeiro dry-run de `upload_v498_dataset_to_hf.py` falhou porque esperava
  `decision == "tokenization_gate_passed"`.
- O manifesto real V286 usa `decision.status == "tokenization_gate_passed"`.
- Hashes/linhas estavam corretos; a falha era na leitura do schema do gate.

Impacto:

- Sem essa correcao, um dataset valido seria bloqueado por falso negativo.
- A correcao nao relaxa o gate: continua exigindo status
  `tokenization_gate_passed`.

Regra preventiva:

- Scripts que consomem manifests devem ler o schema real do produtor e, quando
  possivel, cobrir o campo em dry-run antes de upload/job pago.

Status: corrigido no uploader V499; dry-run e upload passaram.

### E047 - V498/V499 rota autorizada somente como smoke curto

Evidencia:

- V498: 1712/428 linhas, zero overlap com weak/full, token max 331, truncation
  zero e offset masks completos.
- V499 local debug baixou o dataset do HF commit
  `c7e27fd39c598dd23cb25481f567787bdff50820`, conferiu hashes, passou V478 e
  validou H200 a `0.083333/min`.
- O objetivo efetivo ficou `bit=0.390244` e `equation=0.609756`.

Impacto:

- A rota V498 ataca exatamente os 4 ganhos numeric V324 nao transferidos, mas
  ainda nao prova ACC submit-safe.
- So ha justificativa para um H200 smoke de 2 steps, nao treino amplo.

Regra preventiva:

- Promover somente se weak `total>192`, `equation>=60`, `bit>=136`, `trunc=0`.
- Se repetir `equation=57` com perda de bit/truncation, cancelar por FinOps e
  voltar para CPU.

Status: fechado como bloqueado. V499 executou, mas final eval regrediu e
answer-span ficou inativo; nao rodar weak/full/package/submit.

### E048 - ACC promocional podia usar extracao label-aware

Evidencia:

- `extract_final_answer_for_expected()` usa o answer esperado para desambiguar
  `\boxed{}`.
- `evaluate_lora_adapter.py` e `evaluate_lora_adapters_batch.py` usavam esse
  helper para gerar a coluna `prediction` quando `questions-csv` continha
  `answer`.
- Em teste simples, `expected=30` e `raw=\boxed{30 wrong}` poderia virar
  `30`, inflando ACC local.

Impacto:

- Um ganho poderia ser melhoria do parser com label, nao comportamento
  submit-safe do adapter.
- Isso explica divergencias entre valores vistos em logs e valores
  submit-safe.

Regra preventiva:

- `prediction` promocional deve ser sempre
  `extract_final_answer(raw_output)`.
- Expected-aware fica apenas como `label_aware_debug_prediction`.
- Static gate bloqueia `prediction = extract_final_answer_for_expected(...)`.

Status: corrigido em V504.

### E049 - V501 answer-span trainable-MoE bloqueado por final eval regression

Evidencia:

- V501 ativou answer-span weighting: train `1712` exemplos, `15197` tokens;
  validation `428` exemplos.
- MoE `mlp.experts.gate_up_proj` e `mlp.experts.down_proj` ficaram treinaveis.
- `lm_head` ficou congelado.
- Baseline eval loss `1.9919`; final eval loss `1.9923`.

Impacto:

- F2 trainable e answer-span ativo nao sao suficientes para ganho.
- Weak eval caro em V501 seria desperdicio de FinOps.

Regra preventiva:

- V501 adapter/repo entra na lista de bloqueio.
- Nao abrir novo H200 a partir de V498/V501 sem novo sinal CPU label-free.

Status: bloqueado por V503/V504.

### E050 - CSVs reference-only pareciam ganhos submittable

Evidencia:

- V505 varreu `30` CSVs locais de predicao e revalidou `22` conjuntos weak315.
- O melhor conjunto com `raw_output` de adapter ficou em `191/315`,
  `equation=55/155`, `bit=136/160`, `trunc=0`.
- O melhor conjunto reference-only, `v366_integrated_predictions.csv`, chegou a
  `222/315`, `equation=63/155`, `bit=159/160`, mas nao possui `raw_output` de
  modelo e representa solver/postprocessor/projecao CPU.
- O baseline historico `192/315` tinha pelo menos um overcount simbolico
  (`4bb8c6cd`, answer `]}\!`) causado por parse com label/escape, nao por
  resposta label-free limpa.
- V506 comparou melhor adapter raw e melhor reference-only: existem `31`
  targets (`23` bit, `8` equation), com `0` reference-loss risk. Eles sao
  inventario de transferencia, nao ganho submittable.

Impacto:

- Parte da sensacao de "temos ganho mas o adapter nao aprende" vinha de
  misturar duas coisas diferentes: sinal CPU reference-only e comportamento
  adapter-only submit-safe.
- Nenhum CSV sem `raw_output` pode justificar full eval, package ou submit.

Regra preventiva:

- V505 classifica `has_raw_output=false` como
  `not_adapter_only_reference_solver_or_postprocessor`.
- Static/pre-paid gates agora exigem `KG1_CRISIS_MODE_BACKFIRE_GUARD=1` para
  launchers pagos, forçando revisao de F2/backfire, metrica label-free,
  bloqueios de artefatos e FinOps antes de gastar GPU.
- O proximo experimento precisa converter o sinal solver em comportamento do
  adapter ou em pacote permitido; nao promover CSV reference-only.
- O arquivo `v506_reference_gain_targets.csv` vira fonte de debug/transferencia
  controlada; qualquer uso em treino precisa de anti-leakage, tokenization gate
  e weak eval raw-output label-free.

Status: aberto como regra permanente; nenhum novo submit-safe gain encontrado
em V505.

### E051 - V274 expected-aware overcount nao e ganho submit-safe

Evidencia:

- V507 aplicou V274 somente depois de `extract_final_answer(raw_output)`.
- O melhor adapter raw label-free sobe de `191/315` para `195/315`, com
  `equation=59/155`, `bit=136/160`, `0` perdas.
- O valor historico `196/315`, `equation=60/155` dependia de uma linha
  simbolica (`4bb8c6cd`, answer `]}\!`) em que a predicao armazenada/esperada
  preservava payload escapado que o parser label-free nao extrai de forma
  submit-safe.

Impacto:

- `196/equation=60` nao pode ser tratado como candidato de submit adapter-only.
- O alvo real sem violar parser/label e `195/equation=59/bit=136`, ainda como
  postprocessor diagnostico.

Regra preventiva:

- Toda promocao precisa diff por `raw_output`, `simple_extracted`,
  `label_aware_debug_prediction` e `prediction` label-free.
- Linhas simbolicas com braces/backslashes entram em quarantine ate existir
  convencao symbol-safe validada pela metrica label-free.

Status: regra ativa; nenhum package/submit autorizado.

### E052 - V439 tem respostas simbolicas escapadas que nao roundtripam

Evidencia:

- V509 auditou `v439_final_answer_only_pairs_train/val`.
- Resultado: train `109` rows com `26` `assistant_answer_mismatch`; val `24`
  rows com `5` mismatches.
- Exemplos incluem answers literais com `\\`, `\{` e `\}`. O assistant renderiza
  esses caracteres dentro de `\boxed{}`, mas a extracao label-free retorna
  payload escapado ou truncado, diferente do `answer`.
- V440/V441 ja tinham falhado em converter V439 em ganho adapter-only.

Impacto:

- V439 pode baixar loss ensinando strings que a metrica nao extrai como a
  resposta oficial.
- Treinar V439 novamente sem reconstruir a renderizacao simbolica e repetir um
  bug silencioso.

Regra preventiva:

- V439 fica fora do dataset ativo.
- So pode voltar como novo dataset se passar roundtrip:
  `answer == extract_final_answer(assistant)` para 100% das linhas, sem
  expected-aware.

Status: excluido do V510.

### E053 - V443 e dataset vazio, nao fonte de treino

Evidencia:

- O manifesto V443 registra `candidate_rows=0`, `certified_pair_rows=0`,
  `train_pairs=0`, `val_pairs=0`, `hf_gpu_allowed=false`.
- Os arquivos JSONL tem SHA256 vazio (`e3b0...b855`).

Impacto:

- Qualquer launcher apontando para V443 pode passar por caminhos sem dado real
  ou gastar validacao em input nulo.

Regra preventiva:

- V443 fica fora do dataset ativo.
- Builder CPU certificado precisa gerar pares reais antes de qualquer treino.

Status: excluido do V510.

### E054 - Pool de treino disperso causava reuso de linhas historicas falhas

Evidencia:

- V509 auditou 20 arquivos de dataset recentes/historicos.
- V510 criou um pool unico e rastreavel com apenas V498, V475 e V460:
  `2627` train, `637` val.
- V510 removeu `543` duplicados de train e `155` de val por
  `prompt+answer`.
- O reaudit V509 do V510 passou com `blocked_dataset_count=0`.
- O tokenization gate local com tokenizer oficial passou: `offset_masks=2627/637`,
  token max `331`, sem truncation e sem overlap weak/full.

Impacto:

- Reduz risco de misturar datasets antigos que ja falharam transferencia
  (V390/V406/V410/V416), datasets unboxed antigos (V293), V439 escapado ou
  V443 vazio.

Regra preventiva:

- Todo novo job deve apontar explicitamente para V510 ou versao posterior com
  manifesto equivalente.
- Dataset novo so entra se passar V509, tokenization gate e pre-paid gate.

Status: V510 criado, tokenizado localmente, enviado ao HF dataset repo no commit
`40e71a686d9970c3c842d26dcf89200fc4990a51`, e aprovado no pre-paid gate/debug
do launcher V511. Proximo passo permitido: commit/push do launcher V511 e smoke
H200 de 2 steps com kill-switch, nao treino longo.

### E055 - Launcher novo podia omitir contrato literal de trainability MoE

Evidencia:

- O primeiro `kg1_static_safety_gate.py` no V511 falhou antes de qualquer GPU:
  `target_parameter_trainability_not_explicit`.
- Depois de declarar `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`, o gate
  apontou outro risco: `TRAINABLE_LORA_MODULES` nao estava visivel em formato
  literal para provar `up_proj/down_proj`.
- O codigo remoto herdado do V493 exportava esses valores, mas o launcher V511
  precisava carregar o contrato explicitamente para impedir drift silencioso
  entre manifesto, env local e shell remoto.

Impacto:

- Sem essa regra, seria possivel relancar um smoke com `target_parameters`
  carregados, mas nao treinaveis, repetindo a classe de erro F2
  frozen-active.

Regra preventiva:

- Todo launcher promocional com MoE deve conter, em formato auditavel:
  `export TRAINABLE_LORA_MODULES="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"`
  e `export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`.
- O `job_env` tambem deve registrar `KG1_TRAINABLE_LORA_MODULES` e
  `KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE`.
- Static gate e pre-paid gate devem rodar antes do upload/launch pago.

Status: corrigido no V511; `py_compile`, static gate, pre-paid integration gate
e launcher debug passaram.

### E056 - V511 V510 canonical smoke roda, mas nao traz sinal de transferencia

Evidencia:

- HF job `felipesp1983/6a08dc43e48bea4538ba02ce` completou.
- V510 dataset carregado do HF commit
  `40e71a686d9970c3c842d26dcf89200fc4990a51`.
- MoE `target_parameters` ficaram treinaveis:
  `mlp.experts.gate_up_proj=5934` e `mlp.experts.down_proj=5934` trainable
  LoRA tensors.
- `lm_head` ficou congelado.
- Checkpoint-2 e final adapter foram enviados para
  `felipesp1983/kg1-nemotron-lora-v511-nemo-h200-v510-canonical-v290ckpt6`.
- Eval local piorou levemente: `2.8125 -> 2.8128`.

Impacto:

- A infraestrutura H200/V510/PEFT nao e mais o gargalo principal.
- O dataset/objetivo V510 nao demonstrou transferencia suficiente para justificar
  weak/full eval pago.
- Rodar mais steps/epochs nessa mesma rota seria custo sem evidencia.

Regra preventiva:

- Candidato com loss local pior no smoke de 2 steps nao recebe weak/full eval
  pago, salvo novo gate barato independente.
- Antes de outro H200, exigir CPU learnability gate: trace curto, sem duplicate
  CoT conflitante, offset-mask completo e base logprob comparado ao baseline.

Status: bloqueado; sem package/submit.

### E057 - Discussions Kaggle reforcam que solver correto nao basta

Evidencia:

- V512 auditou `140` topicos e `586` posts do Kaggle via API.
- Discussion `690307` confirma que o ganho de bit veio de CoT deterministica
  bit-pair/bitsum/stride.
- Discussion `689915` confirma que a solucao vencedora otimizava SFT para
  minimo logprob com traces deterministas.
- Discussion `697491` mostra que dataset solver-correct pode reduzir LB por
  complexidade de trace, oversampling, duplicate-CoT e format clash.
- Discussion `694710` reforca prompt-loss masking/pretokenized response mask.
- Discussion `698293` mostra solver simbolico gold-conditioned como oracle
  forte, mas nao submit-safe e nao automaticamente transferivel.

Impacto:

- O problema atual e learnability/transferencia para adapter, nao falta de
  mais dados solver-correct.
- Treino amplo com dataset maior deve continuar bloqueado ate passar gates de
  duplicidade, comprimento, logprob e formato.

Regra preventiva:

- Novo dataset so entra se registrar:
  - duplicate-CoT conflitante `0`;
  - trace max preferencial `<1300` tokens ou justificativa explicita;
  - logprob/CE por subfamilia comparado ao baseline;
  - response-mask/offset-mask completo;
  - projection CPU sem perda de bit e sem expected-aware extraction.

Status: regra ativa; V512 gerou
`artifacts/v512_kaggle_discussions_audit/V512_KAGGLE_DISCUSSION_AUDIT_SUMMARY.md`.

## Prompt Externo

Prompt consolidado para OpenRouter/outras APIs:

- `artifacts/openrouter/KG1_V438_ERROR_LEDGER_EXTERNAL_API_PROMPT_2026_05_15.md`

Uso esperado:

1. Perguntar a modelos externos como corrigir E003 sem violar adapter-only.
2. Exigir propostas testaveis em CPU antes de GPU.
3. Rejeitar respostas que recomendem broad SFT, mais epochs, weak/full leakage,
   runtime verifier, postprocessor ou submit com script.
