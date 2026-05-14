# KG1 Score Improvement Roadmap

Atualizado: 2026-05-14

Este arquivo agora e o roadmap ativo e limpo. O historico detalhado anterior foi arquivado em `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V379_CLEANUP_2026_05_14.md`.

## Objetivo

Melhorar `equation_transform` e `bit_manipulation` sem regressao no adapter-only submit.

Meta minima para gastar HF:

- Manter `bit_manipulation >= 136/160`.
- Sair do teto adapter-only `equation_transform=56/155`.
- Primeiro checkpoint deve manter `total >= 192/315`, `truncated=0`.
- Se qualquer job nao puder mais bater o gate, cancelar por FinOps.

## Regra Operacional Agressiva 2026-05-14

Precisamos buscar subida no ranking ainda hoje, `2026-05-14`. O plano deve ser agressivo, mas responsavel: gastar HF somente quando existe chance objetiva de gerar submit adapter-only melhor que o baseline.

- Prioridade maxima: ganho adapter-only submetivel medido em weak/full gate, nao `eval_loss` isolado.
- Concluir o V382 apenas como smoke curto ja em andamento; escolher checkpoint por ACC weak, nao por loss.
- Rodar V383 weak-eval sweep, mas aplicar corte FinOps apos `checkpoint-6`: se `checkpoint-2`, `checkpoint-4` e `checkpoint-6` nao baterem `total > 192`, `equation_transform > 56`, `bit_manipulation >= 136` e `truncated=0`, cancelar `checkpoint-8/10` e encerrar esta linha.
- Se qualquer checkpoint V383 bater `total > 192`, `equation_transform > 56`, `bit_manipulation >= 136` e `truncated=0`, promover para full/official-like eval e package gate no mesmo dia.
- Se nenhum checkpoint V383 bater o baseline, voltar imediatamente para CPU gates que gerem novo sinal independente; nenhum novo HF training pode iniciar sem ganho CPU medido antes.
- Caminho de hoje apos V383 negativo: gerar candidato submit-safe a partir do melhor adapter-only conhecido, sem solver/postprocessor runtime, e so substituir respostas se houver prova local reproduzivel de que a resposta vem do proprio adapter/package permitido.
- Proibido submit de postprocessor/verifier/teacher-only. Submit hoje so pode vir de adapter/package que passe os gates.

## Estado Atual Medido

| Estado | Total weak | equation_transform | bit_manipulation | Status |
|---|---:|---:|---:|---|
| Melhor adapter-only | `192/315` | `56/155` | `136/160` | baseline real para submit |
| V274/V275 postprocessor/verifier | `196/315` | `60/155` | `136/160` | nao e adapter-only; usar como teacher/diagnostico |
| V366/V336 CPU teacher/verifier | `222/315` | `63/155` | `159/160` | nao submit-safe; melhor teacher CPU |
| V372 HF trace-style smoke | `191/315` | `56/155` | `135/160` | rejeitado; nao continuar |
| V375 equation residual clustering | `92` eq misses restantes | `82 symbolic_punct`, `10 numeric_operator` | n/a | diagnostico |
| V378 solver parquet sobre V375 | `82/92` cobertos | `79/82` corretos | n/a | melhor novo sinal |
| V380 oracle diagnostic | `301/315` | `142/155` | `159/160` | nao submit-safe; usa resposta/teacher |
| V380 reexecuted teacher | `292/315` | `133/155` | `159/160` | dataset teacher permitido; HF ainda bloqueado |
| V380 strict independent | `222/315` | `63/155` | `159/160` | ganho real independente `+0`; nao submeter |
| V381 filtered teacher dataset | n/a | `840` eq sintéticas | `280` bit replay | passou dataset + tokenization gate real; pronto para micro-train HF |
| V382/V383 V381 teacher smoke | `191/315` melhor parcial | `56/155` | `135/160` | rejeitado; checkpoints 2/4/6 nao bateram baseline; V383 cancelado por FinOps |
| V384 V382 V221 prompt weak eval | em execucao | em execucao | em execucao | mede os mesmos ckpt4/6 do V382 com prompt suffix historico V221 e thinking habilitado |

Conclusao: `eval_loss` baixo nao e criterio de promocao. O criterio e ACC por familia no weak/full gate.

Resultado V383 em `2026-05-14`: checkpoint-2 = `190/315`, `equation=56`, `bit=134`, `truncated=1`; checkpoint-4 = `191/315`, `equation=56`, `bit=135`, `truncated=1`; checkpoint-6 = `190/315`, `equation=56`, `bit=134`, `truncated=1`. Como nenhum dos tres podia superar o baseline `192/315`, `equation=56`, `bit=136`, `truncated=0`, o job foi cancelado antes de avaliar `checkpoint-8/10`.

Auditoria V385 de medicao ACC em `2026-05-14`: o weak scorer atual esta correto para comparar candidatos adapter-only. O CSV weak validado pelo proprio runner tem `315` rows, `160` bit, `155` equation, SHA `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6` e contrato `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`. O merge e `one_to_one` por `id`, a familia vem do CSV de solucao, `bit_manipulation` usa igualdade binaria exata e truncation vem de `finish_reason == "length"`. Gap encontrado: V383 usou sufixo curto e e diagnostico; V384 e a comparacao correta contra o prompt historico V221.

## Fontes Auditadas

### Fonte ativa 1 - `solver_results.parquet`

Arquivo original: `C:\Users\davis\Downloads\nemotron_dataset_final\solver_results.parquet`

Status: o parquet bruto nao deve ser tratado como dependencia operacional atual. A evidencia versionada e reproduzivel no repo esta nos CSVs derivados V378/V380.

Evidencia:

- `823` rows de `equation_transform`.
- `800/823` corretas pelo scorer do projeto.
- `741/823` rows tem `conditioned_on_answer=True`; sao evidencia de reparo, nao prova independente.
- `82/823` rows nao foram condicionadas na resposta; este e o subconjunto preferencial para promocao de regra.
- Cobre `82/92` residuos V375.
- `79/82` corretos nesses residuos.
- Categorias: `arithmetic`, `little_endian`, `mixed_concat`, `pure_concat`, `mixed_concat_little_endian`, `query_unseen_concat`.
- V380 reexecutou os `solver_ops` no prompt: `70/79` ganhos reproduzidos como teacher (`36 arithmetic`, `27 little_endian`, `7 mixed_concat`).
- V380 encontrou `0` ganhos strict-independent. Portanto o sinal ainda nao desbloqueia submit nem treino HF direto.

Uso permitido no plano:

- Gerar candidatos CPU para os residuos de `equation_transform`, priorizando rows nao condicionadas.
- Usar `solver_ops`, `solver_mapping`, `solver_category` como regra/teacher somente depois de reexecucao independente no prompt.
- Nunca usar bruto: ha `23` erros.

### Fonte ativa 2 - `filtered_merged_dataset.csv`

Arquivo: `C:\Users\davis\Downloads\nemotron_dataset_final\kaggle_logprob\results\filtered_merged_dataset.csv`

Evidencia:

- `8703` rows.
- `7044` IDs unicos.
- `1659` duplicatas/reweighting.
- `821` rows sao duplicatas exatas.
- Labels `8703/8703` corretos.
- CoT `8691/8703` correto.
- `equation_transform`: `1499` IDs unicos, `2438` rows, CoT `2428/2438`.
- `bit_manipulation`: `1354` IDs unicos, `1754` rows, CoT `1752/1754`.
- Cobre `92/92` residuos V375; `91/92` CoT correto.

Uso permitido no plano:

- Selecionar um melhor trace por ID, filtrado por resposta correta, loss, tamanho e tokenizer.
- Gerar dataset curto de transferencia para LoRA depois que o solver gate CPU passar.
- Nao usar bruto por causa de duplicatas exatas, duplicatas por ID/reweighting e traces longos.

### Fonte ativa 3 - `sft_train_full_9500.jsonl`

Arquivo: `C:\Users\davis\Downloads\nemotron_dataset_final\sft_train_full_9500.jsonl`

Evidencia:

- `9500/9500` correto.
- `9500` IDs unicos.
- Cobertura completa das seis familias.
- `9500/9500` rows tem multiplos spans `\boxed{}`.
- `364` rows tem resposta declarada errada antes do `\boxed{}` final corrigido: `238` em `bit_manipulation`, `126` em `equation_transform`.
- `173` respostas de `equation_transform` contem chaves literais; exigem escaping/extrator oficial.

Uso permitido no plano:

- Fallback de trace correto por ID.
- Comparar formato/prompt/template.
- Nao fazer SFT amplo.
- Se usado, limpar spans intermediarios e manter exatamente um final answer validado.

## Fontes Diagnosticas

| Fonte | Decisao |
|---|---|
| `kaggle_sft_data/dataset_generated.csv` | labels corretos, CoT `9197/9500`; usar para comparacao/hard negatives, nao como treino bruto |
| `kaggle_trajectories/nemotron_traj.csv` | geracao direta so `4542/9500`; nunca usar como label, apenas hard-negative/confidence |
| `sft_train_converted.jsonl` | duplicado pelo `filtered_merged_dataset.csv`; `6923` rows com think-tags malformados; usar so para referencia de formato, nunca bruto |
| `sft_train_reconstructed.jsonl` | contem `8463` rows sinteticas/desconhecidas; fora do plano ativo |
| `sft_reconstructed.jsonl` | `9500/9500`, mas supersedido por `sft_train_full_9500` e logprob filtered |
| `nemotron_hacker_dataset` | subconjunto duplicado do `nemotron_dataset_final`; fora do plano ativo |

## Gaps Corrigidos no Double Check

- `nemotron_dataset_final` tem `13` arquivos, `509113679` bytes.
- `nemotron_hacker_dataset` tem `7` arquivos, `401052223` bytes.
- Arquivos comuns entre os diretorios: `6`.
- Hash mismatches entre arquivos comuns: `0`.
- `nemotron_dataset_final` e o superset ativo.
- `competition_train.csv` tem `9500` rows reais; contagem por linha fisica e invalida porque prompts tem quebras de linha.
- `competition_test.csv` tem `3` rows sample e `3/3` IDs/prompts aparecem no train (`00066667`, `000b53cf`, `00189f6a`); nao e eval.
- O pacote final tem overlap com V217 train em `1476` prompts: `654` bit, `246` equation e `144` em cada familia facil.
- O pacote final tem overlap com V217 val em `103` prompts: `30` bit, `9` equation e `16` em cada familia facil.
- Qualquer dataset futuro precisa filtrar `id`, `prompt_sha256`, prompt normalizado e split V217 val antes de treino ou validacao.
- O relatorio menciona `tong_with_logprob.csv` e `yours_with_logprob.csv`, mas esses arquivos nao existem em nenhum dos dois diretorios auditados. Nao contam como evidencia.

## Triple Check dos Anexos 2026-05-14

Arquivos analisados:

- `Dataset andy279_nemotron-reasoning-challenge - Relatorio Completo de Extracao.md`.
- `Relatorio de Extracao_ Dataset andy279_nemotron-reasoning-challenge.md`.
- `Nemotron Reasoning Challenge - SFT Data.md`.
- `Relatorio_ Dataset andy279_nemotron-reasoning-challenge.md`.
- `competition_train.csv`.

Achados que ficam no plano:

- `competition_train.csv` anexado e identico ao `competition_train.csv` do pacote final: SHA256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`, `9500` rows, `9500` IDs unicos, `0` duplicatas.
- Contagem oficial por familia no train: `bit_manipulation=1602`, `equation_transform=1555`, `gravity_constant=1597`, `numeral_system=1576`, `text_encryption=1576`, `unit_conversion=1594`.
- Os anexos descrevem o SFT original `andy279` como `49290` exemplos de treino / `7200` puzzles unicos e validacao com `1165` exemplos / `1123` puzzles.
- Os anexos citam `399` transformations nao resolvidas no split de validacao original. Isto reforca que `equation_transform` e gargalo de solver/DSL/verificacao, nao de treino generico.
- Os anexos citam forte sinal original para as familias alvo: `17285` traces de bit, `10741` transformation, `1602` solver-guided bit e `1101` solver-guided transformation.
- Esses arquivos SFT originais nao estao disponiveis localmente; portanto nao entram como fonte ativa de treino. Se acesso for aprovado depois, entram apenas por novo gate de hash, resposta, duplicata, overlap e tokenizacao.
- A frase dos relatorios sobre ter `100% dos dados essenciais` nao e aceita como fato operacional: os proprios anexos dizem que o original tem `49290` train e `1165` validation, enquanto os dados locais cobrem `17963`/`9500` derivados e nao incluem a validacao original.
- O README SFT descreve uma regra de qualidade que agora e obrigatoria no V381: limpar artefatos LaTeX dentro de `\boxed{}`, reextrair a resposta final, recomputar corretude pelo scorer e manter somente tentativas corretas.
- O claim de `competition_test.csv` com `34` puzzles foi contradito pela auditoria local: o arquivo auditado tem `3` rows e as `3` aparecem no train. Continua proibido como eval.
- Os anexos citam raw traces multi-attempt (`all_traces_merged.jsonl`, `solver_bit_manipulation_traces_merged.jsonl`, `solver_transformation_traces_merged.jsonl`), mas esses arquivos nao existem nos diretorios locais auditados. So entram no plano se forem adquiridos e auditados em novo gate.
- Um dos relatorios contem padrao de token HF; relatórios brutos nao devem ser versionados. Apenas metadados redigidos entram no repo.

Ganho medido novo desses anexos:

- Adapter-only: `+0`.
- CPU teacher: `+0`.
- Ganho esperado: indireto e condicional. A utilidade real e reduzir erro no V381/V382; ainda nao autoriza HF nem submit.

## Roadmap Ativo

### Step 1 - V380 CPU equation solver candidate patch

Status: concluido em CPU.

Objetivo: transformar `solver_results.parquet`/derivados V378 em candidatos controlados para os `92` residuos V375.

Entrada:

- `solver_results.parquet`.
- `v378_v375_solver_coverage.csv`.
- `v378_v375_residual_trace_solver_coverage.csv`.
- `v366_integrated_predictions.csv`.

Regras:

- Usar somente os `79` residuos V375 com `solver_metric_correct=True`.
- Separar evidencias condicionadas e nao condicionadas; promover regra somente se ela reexecutar sem consultar `answer`.
- Separar por `solver_category`.
- Gerar `old_prediction`, `new_prediction`, `answer`, `category`, `solver_ops`, `solver_mapping`.
- Simular patch em CPU.
- Bloquear qualquer categoria que tenha perda.

Saida esperada:

- Manifest criado em `artifacts/v380_solver_results_patch_gate/20260514T_cpu_gate/v380_solver_results_patch_gate_manifest.json`.
- Resultado diagnostico oracle: `301/315`, `equation=142/155`, `bit=159/160`, ganho `+79`; nao submit-safe.
- Resultado reexecuted teacher: `292/315`, `equation=133/155`, `bit=159/160`, ganho `+70`; pode alimentar V381 como teacher limpo.
- Resultado strict-independent: `222/315`, `equation=63/155`, `bit=159/160`, ganho `+0`; nao desbloqueia HF nem submit.
- Decisao: `teacher_signal_only_no_hf_unlock`.

### Step 2 - V381 filtered trace/tokenization gate

Status: concluido em CPU.

Objetivo: construir dataset pequeno para LoRA a partir dos `70` candidatos V380 reexecuted-teacher, sem tratar oracle/answer-conditioned como regra submetivel.

Entrada:

- Melhor trace por ID de `filtered_merged_dataset.csv` ou `sft_train_full_9500.jsonl`.
- Candidatos aprovados no V380.

Regras:

- Um trace por ID.
- Sem duplicata/reweighting.
- Sem overlap com V217 val.
- Sem `sft_train_converted` bruto: `6923` rows tem `<think>`/`</think>` malformado.
- Sem `sft_train_full_9500` bruto: limpar spans intermediarios, respostas declaradas erradas e chaves literais antes de tokenizar.
- Limpar artefatos LaTeX dentro de `\boxed{}`.
- Reextrair a resposta final depois da limpeza.
- Recomputar corretude pelo scorer do projeto.
- Manter somente tentativas corretas.
- Resposta final validada pelo scorer.
- Prompt/template oficial preservado.
- Offset-mask correto.
- Truncation `0`.
- Separar equation target e bit replay.
- Bloquear HF se o dataset nao provar pelo menos um dos dois:
  - nova regra independentemente reexecutavel com `+4` equation e `0` perdas; ou
  - dataset teacher limpo com cobertura superior ao V304, sem overlap proibido e com bit replay suficiente para manter `bit>=136`.

Saida esperada:

- JSONL pequeno, auditado.
- Tokenization manifest.
- Nenhum HF se `bit` replay nao estiver presente ou se truncation > 0.
- Nenhum submit; este step so prepara gate para treino.

Resultado V381:

- Script: `scripts/build_v381_filtered_teacher_dataset.py`.
- Manifest: `artifacts/v381_filtered_teacher_dataset/20260514T_cpu_gate/v381_filtered_teacher_dataset_manifest.json`.
- Dataset: `909` train rows e `211` validation rows.
- Train: `669` `equation_transform` sinteticas derivadas de V380 reexecuted-teacher + `240` bit replay V217.
- Validation: `171` `equation_transform` sinteticas + `40` bit replay V217.
- Prototipos originais V380: `70` rows em `v381_teacher_prototypes_not_for_training.jsonl`; marcados como nao-treino.
- Exact weak overlap com V366: `0` em train e `0` em validation.
- Overlap train vs V217 val: `0`.
- Tokenization gate real Nemotron passou:
  - `completion_tokens_dropped=0`.
  - `fallback_masks=0`.
  - `prompt_truncation_rate=0.0`.
  - `train_token_max=327`.
  - `val_token_max=327`.
  - `offset_masks=909/909` train e `211/211` validation.
- Decisao: V382 HF micro-train esta permitido como smoke responsavel, mas nao autoriza submit.

### Step 3 - V382 HF micro-train com kill-switch

Status: V382 treinado; V383 weak eval curto rejeitado; V384 V221 prompt weak eval em execucao.

Objetivo: testar se o sinal V381 transfere para adapter-only sem destruir bit.

Regras:

- Comecar com treino curto.
- A100 preferencial por custo; H200/H100 so se houver razao tecnica.
- Primeiro checkpoint deve bater:
  - `total >= 192/315`.
  - `equation >= 56/155`.
  - `bit >= 136/160`.
  - `truncated=0`.
- Se falhar, cancelar.
- Se passar, avaliar checkpoint seguinte.
- Kill-switch adicional V381:
  - cancelar se validation loss cair mas weak ACC ficar em `equation<=56` e `bit<136`;
  - cancelar se o primeiro checkpoint tiver `bit<136`;
  - cancelar se o custo estimado exceder o budget restante sem chance de bater `192/315`.

Saida esperada:

- Adapter-only candidato.
- Weak eval comparavel ao baseline.

### Step 4 - V383 full eval/package/submit gate

Objetivo: submeter somente se houver ganho real adapter-only.

Regras:

- Full eval somente se weak passar.
- Package somente adapter-only, sem solver/postprocessor proibido.
- Kaggle submit somente com ganho medido.
- Registrar diferenca por familia contra o submit 0.86/ranking 19 quando o nome/arquivo exato estiver confirmado.

## Regras Permanentes

- Nenhum HF sem CPU gate com sinal novo.
- Nenhum submit sem ganho medido.
- Nenhuma decisao por `eval_loss`; decisao por ACC e truncation.
- Jobs HF em execucao devem ser verificados a cada aproximadamente `40s`.
- Se o job nao puder mais bater o gate, cancelar por FinOps.
- Nao copiar datasets grandes para o repo.
- Artefatos versionados devem ser pequenos: manifests, CSVs de auditoria, scripts.
- Todo `.ipynb` novo ou alterado deve passar `scripts/notebook_release_gate.py`.

## Itens Removidos do Plano Ativo

| Item | Motivo |
|---|---|
| Mais epochs/LR sem novo CPU gate | ja reduziu loss sem melhorar ACC |
| Broad SFT em todos os traces | risco de regressao; historicamente nao transferiu |
| `nemotron_traj.csv` como label | so `4542/9500` correto |
| `sft_train_reconstructed.jsonl` completo | contem rows sinteticas/desconhecidas |
| `sft_train_converted.jsonl` bruto | duplicatas/reweighting |
| `sft_train_full_9500.jsonl` bruto | multiplos boxed spans e `364` respostas declaradas erradas antes do final corrigido |
| `competition_test.csv` como eval | `3/3` rows aparecem no train |
| Claims de `100% dos dados essenciais` | contradizem a ausencia local de `sft_train.jsonl`, `sft_val.jsonl` e validacao original |
| Raw traces citados nos relatorios | arquivos nao existem nos diretorios locais auditados |
| `dataset_generated.csv` bruto | CoT errado em `303` rows |
| `problems.jsonl` bruto | apenas `8333/9500` correto |
| Tong bit direct replacement | V374 caiu para `bit=136` e teve perdas contra V366 |
| HF V371/V372 trace-style | checkpoint-1 `191/315`, `bit=135` |
| Prompt/thinking variants amplas | regressao severa |
| Adapter soups | nao moveu equation |
| Web/API buscas genericas | so retornam ao plano se virarem regra, dataset ou gate verificavel |

## Proxima Acao Unica

Concluir V384. Se qualquer checkpoint bater `total>192`, `equation>56`, `bit>=136` e `truncated=0`, promover para full/official-like eval e package gate. Se V384 nao bater, encerrar a linha V381/V382 e voltar para CPU gate independente com novo sinal verificavel.

Nao rodar novo HF training antes de novo CPU gate com ganho de ACC medido.
