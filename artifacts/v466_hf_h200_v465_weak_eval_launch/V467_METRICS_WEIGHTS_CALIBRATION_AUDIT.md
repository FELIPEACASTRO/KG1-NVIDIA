# V467 Metrics, Weights, Calibration Audit

Auditoria cirurgica das pecas que afetam `loss`, `ACC`, weak score e decisao
de promocao para as familias `bit_manipulation` e `equation_transform`.

## Veredito

Nao encontrei desalinhamento critico entre dataset, pesos, loss, metricas e
gates. As pecas estao sincronizadas para medir o V465 corretamente no V466 weak
eval.

Encontrei um gap de observabilidade: o manifest final de treino gravava
`baseline_eval_loss`, `final_eval_loss` e `best_eval_loss`, mas nao gravava
historico por step/checkpoint. Isso nao muda ACC nem score, mas atrapalha
debug quando o loss cai e a ACC nao sobe. Corrigido em
`scripts/hf_job_train_v90.py` para futuros jobs.

## Matriz de Sincronizacao

| Area | Estado | Evidencia | Risco residual |
|---|---|---|---|
| Dataset V464 | sincronizado | train/val hashes no manifest V464, no launch V465 e no manifest final V465 | baixo |
| Tokenizacao | sincronizada | V286: `prompt_truncated=0`, `fallback_masks=0`, offset masks em todas linhas | baixo |
| Loss mask | sincronizada | `ANSWER_SPAN_LOSS_WEIGHT=5.0`, min weighted tokens `1000`, sem fallback mask | baixo |
| Sampling/pesos | aplicado | manifest final V465 registra `weighted_replacement`, source/subcategory weights planejados | medio: perda de bit ainda so aparece no weak eval |
| LoRA trainability | aplicado | trainable modules `q,k,v,o,lm_head`; adapter config preserva os 9 target modules do init | baixo |
| Eval runner | sincronizado | V466 usa V245 weak bridge, 315 rows, contract hash, 5 adapters V465 | baixo |
| ACC metric | sincronizada | V449 `metric_path_ok`, `verify_answer` e binary exact mantidos | baixo |
| Prompt de eval | alinhado | V466 replica o padrao V290/V448: suffix boxed, thinking nao desativado por flag | medio: prompt ainda pode gerar saidas longas |
| Promocao | bloqueante | full/submit so se weak `total>192`, `equation>56`, `bit>=136`, `truncated=0` | baixo |

## Pesos e Calibragem Efetiva

Contagem bruta do train V464:

- `bit_manipulation`: `512/558`
- `equation_transform`: `46/558`

Pesos configurados:

- source `v464_v463_numeric_multirule_dataset=2.50`
- source `v217_bit_replay_guardrail=0.75`
- subcategory `bit_guardrail_replay=0.65`
- subcategory `v274_guarded_numeric_add_direct_over_model_add_variant=4.00`
- subcategory `v274_guarded_numeric_colon_absdiff_restore_trailing_zero=0.75`
- subcategory `v274_guarded_numeric_minus_direct_negative_restore_sign=4.00`
- subcategory `v274_guarded_numeric_minus_signed_opposite_sign_guarded=3.50`

Share ponderado aproximado usado pelo sampler:

- `equation_transform`: `57.53%`
- `bit_manipulation`: `42.47%`

Isso e intencionalmente mais agressivo para equation do que a contagem bruta.
A calibragem esta coerente com o objetivo de sair de `equation=56`, mas cria
risco real de regressao em bit. Por isso o gate `bit>=136` precisa continuar
bloqueante.

## Regressao de Impacto do Gap Corrigido

Alteracao feita:

- adicionar `step_loss_history`;
- adicionar `eval_history`;
- adicionar `checkpoint_history`;
- adicionar `sampling_report` no manifest final de treino.

Impacto esperado:

- nao muda dataset;
- nao muda loss;
- nao muda optimizer;
- nao muda sampling;
- nao muda checkpoint salvo;
- nao muda parser;
- nao muda ACC;
- nao muda weak/full gate;
- apenas melhora rastreabilidade dos proximos jobs.

O V466 em execucao usa o commit anterior (`b2ab7a4`) e portanto nao e alterado
por essa melhoria de observabilidade.

## Pontos que Continuam Bloqueados

1. Loss nao e criterio de promocao.
   O V465 teve loss saudavel (`baseline=1.8248`, `best=1.8234`, `final=1.8239`),
   mas isso nao prova ganho de ACC.

2. ACC so fica conhecido apos o V466 weak eval.
   O V466 avalia `checkpoint-4`, `checkpoint-8`, `checkpoint-12`,
   `checkpoint-16` e `final` contra o contrato weak V221.

3. Full eval/package/Kaggle submit permanecem bloqueados.
   So liberar se algum checkpoint passar:
   `total>192`, `equation>56`, `bit>=136`, `truncated=0`.

## Decisao Operacional

Continuar o V466 ate sair o resultado, desde que fique abaixo de 1 hora e sem
erro fatal. Se todos os checkpoints ficarem `<=192` total, `equation<=56`,
`bit<136` ou `truncated>0`, rejeitar V465 e nao gastar full eval.

