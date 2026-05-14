# KG1 NVIDIA - Roadmap ativo de melhoria por familia

Atualizado em: 2026-05-14

Este e o roadmap ativo. Ele deve conter apenas o que ainda sera usado para tentar melhorar `bit_manipulation` e `equation_transform`.

Historico completo e rotas antigas ficam fora do plano ativo:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10_FULL_HISTORY_2026_05_13.md`
- historico Git dos commits posteriores a 2026-05-13.

## Verdade operacional

| Estado | Weak / Full | Equation | Bit | Uso atual |
|---|---:|---:|---:|---|
| Melhor adapter-only weak | `192/315` | `56/155` | `136/160` | baseline operacional LoRA |
| Melhor full adapter-only conhecido/packageado | `823/947` | `56/155` | `135/160` | referencia do score conhecido `0.86` |
| Melhor CPU solver/verifier V343 | `199/315` | `63/155` | `136/160` | teacher/verifier, nao submit direto |
| V350 CPU residual no-loss gate | `201/315` | `63/155` | `138/160` | novo teacher CPU: `+2` bit, `0` perdas |
| V351 V350 bit transfer dataset | `640/160` train/val | nao altera equation | baseado nos `2` acertos V350 | dataset minimo aprovado por gate real |
| V352 checkpoint-2 adapter-only | `191/315` | `56/155` | `135/160` | rejeitado; nao transferiu V350 e regrediu bit |
| V354 V352 transfer failure audit | `-10` vs V350 | `-7` vs V350 | `-3` vs V350 | prova de falha de transferencia LoRA |
| V355 CPU residual gate | `201/315` | `63/155` | `138/160` | bloqueado; novas classes tiveram perdas/ambiguidade |
| V356 equation conflict tiebreaker | sem ganho | `0/2` conflitos resolvidos | nao aplica | bloqueado; operador so aparece na query |
| V357 bit global ternary CPU gate | `214/315` | `63/155` | `151/160` | novo teacher CPU: `+13` bit, `0` perdas |
| V358 V357 bit ternary transfer dataset | `1152/288` train/val | nao altera equation | baseado em `13` ternary + `2` replay | dataset aprovado por gate real |
| V359 checkpoint-2 adapter-only | `190/315` | `56/155` | `134/160` | rejeitado; truncation `1`, weak eval cancelado por FinOps |
| V360 V359 transfer audit | bloqueia HF | nao altera equation | nao altera bit | causa provavel: SFT usou 15 regras, formato longo e nao usou preference hard negatives |
| V361 boxed-only transfer dataset | `1152/288` train/val | nao altera equation | baseado nas mesmas `15` regras | corrige formato; V286 real gate passou com `boxed_only` |
| V365 bit residual boolean grammar gate | `214/315` | `63/155` | `151/160` | bloqueado; `73` mudancas candidatas, `0` ganhos, `66` perdas candidatas |
| V366 bit full-byte ternary op gate | `222/315` | `63/155` | `159/160` | teacher CPU aprovado: `+8` bit sobre V357, `0` perdas aceitas |
| V367 V366 transfer dataset | `1128/282` train/val | nao altera equation | `768/192` linhas novas V366 + replay | tokenization real aprovado, boxed-only, `0` truncation |
| V368 checkpoint-1 adapter-only | `191/315` | `56/155` | `135/160` | rejeitado; V367/V366 nao transferiu e regrediu bit |
| V369 V368 transfer audit | `0/8` V366 ganhos transferidos | nao altera equation | `1` ganho, `2` perdas vs baseline | bloqueia mais HF nessa rota |
| Residual depois do V366 | mapa pronto | `92` misses | `1` miss | fila residual CPU |
| V349 Kaggle discussions | `140/140` topicos | reforca ambiguidade/DSL | reforca bit full-byte/bit-pair/3-input | guia do proximo CPU gate |
| Double check compilacoes 2026-05-14 | `3` arquivos | sem novo ganho medido | sem novo ganho medido | classifica links; nao libera HF |

Conclusao honesta:

- Ha ganho real em CPU solver/verifier: `192 -> 222` weak, com `equation_transform 56 -> 63` e `bit_manipulation 136 -> 159`, sempre com `0` perdas aceitas no weak diagnostic.
- O ganho novo comprovado de V357/V366 e forte em bit, mas ainda e teacher CPU. Ainda nao ha ganho adapter-only novo. V338B, V341, V344, V346, V352, V359, V362 e V368 falharam em transferir ganhos para LoRA.
- `eval_loss` menor nao significa ACC maior. Promocao agora e somente por ACC weak/full.
- HF GPU continua bloqueado para as rotas V358/V359/V361/V362/V367/V368. Qualquer HF posterior exige uma nova evidencia CPU que explique a falha de transferencia e passe gates reais antes do smoke.

## Metas

Para liberar novo teacher CPU:

- `losses=0`.
- `bit_manipulation >=159/160` se mexer em bit.
- `equation_transform >63/155` ou `bit_manipulation >159/160` no weak diagnostic.
- Manifest com `id`, `family`, `old_prediction`, `new_prediction`, `rule_class`, `candidate_count`, `conflict_count`, `accepted/rejected`, `reason`.

Para liberar HF GPU:

- Novo teacher CPU aprovado pelo gate acima.
- Dataset novo, diferente de V337D/V344/V346, com hard positives e hard negatives gerados a partir do novo teacher.
- Anti-leakage por `id` e `prompt_sha256`.
- Tokenization/offset-mask gate aprovado.
- Kill-switch no primeiro checkpoint: cancelar se `total<=192`, `equation<=56`, `bit<136` ou truncation regredir.

Para submeter ao Kaggle:

- Candidato adapter-only.
- Weak `>192/315`, `equation_transform>56/155`, `bit_manipulation>=136/160`, truncation nao regressiva.
- Full official-like `>823/947`.
- Package LoRA valido, rank `<=32`, sem solver/verifier/postprocessor externo.
- Sem novo submit por expectativa; somente com ganho medido.

## Evidencias que continuam no plano

| Evidencia | Impacto real | Como sera usada |
|---|---|---|
| V336B package gate | Submissao deve ser adapter-only LoRA; solver/verifier direto bloqueado | impede submit com postprocessor |
| V343 CPU solver/verifier | `199/315`, `equation=63`, `bit=136`, `7` ganhos, `0` perdas | baseline teacher a superar |
| V350 CPU residual no-loss gate | `201/315`, `equation=63`, `bit=138`, `2` ganhos bit, `0` perdas | novo teacher CPU e unica fonte do V351 |
| V351 V350 bit transfer dataset | `640` train, `160` val, `0` overlap por `id`/`prompt_sha256`, V286 real gate aprovado | libera somente smoke V352, nao treino longo |
| V352 checkpoint-2 weak eval | `191/315`, `equation=56`, `bit=135`, truncation `0` | rejeitado; bloqueia full/package/submit |
| V354 transfer failure audit | V352 transferiu `0/2` acertos aceitos do V350; `-10` total vs V350 | prova que a rota bit-transfer direta nao deve continuar |
| V355 CPU residual gate | bit stride/current solver e equation conflicts testados; `0` candidatos aceitos | bloqueia HF; classes novas nao sao seguras |
| V356 conflict tiebreaker | `2` conflitos cryptarithm auditados; ambos tinham operador presente so na query | bloqueia cherry-pick dos 2 acertos aparentes |
| V357 bit global ternary CPU gate | `214/315`, `equation=63`, `bit=151`, `13` ganhos, `0` perdas | novo teacher CPU para bit; libera somente V358/V359 smoke |
| V358 V357 bit ternary dataset | `1152` train, `288` val, `0` overlap por `id`/`prompt_sha256`, V286 real gate aprovado | liberou somente smoke V359 |
| V359 checkpoint-2 weak eval | `190/315`, `equation=56`, `bit=134`, truncation `1` | rejeitado; prova que V358 nao transferiu para adapter-only |
| V360 transfer failure audit | V358 tem `15` regras, `1152/288` SFT rows, preference `2304/576` nao usado pelo launcher, `0` boxed-only completions | bloqueia mais HF nessa rota; exige V361 answer-first/boxed-only ou voltar para DSL equation |
| V361 boxed-only transfer dataset | `1152` train, `288` val, `2304/576` preference rows, `0` train/val prompt overlap, real tokenizer `token_max=286`, `0` truncation, `0` fallback masks | unica correcao de formato permitida; ainda nao libera submit nem full |
| V365 bit residual boolean grammar | `73` mudancas candidatas sobre V357, `0` ganhos, `66` perdas candidatas | bloqueia gramatica per-bit livre; proxima rota bit precisa ser bit-pair/bitsum/stride restrito |
| V366 full-byte ternary `CHO`/`MAJ3` | `222/315`, `equation=63`, `bit=159`, `8` ganhos aceitos, `0` perdas aceitas | novo teacher CPU; base do V367 transfer dataset |
| V367 transfer dataset | `1128/282`, `23` regras, `0` prompt overlap, tokenizer real `token_max=285`, `0` truncation | liberou somente HF smoke V368 |
| V368 checkpoint-1 weak eval | `191/315`, `equation=56`, `bit=135`, truncation `0` | rejeitado; bloqueia full/package/submit |
| V369 transfer failure audit | V368 transferiu `0/8` ganhos V366; mudou `10` linhas vs baseline: `1` ganho, `2` perdas, `7` neutras | prova que V367/V368 bit-only SFT nao deve continuar |
| V348 residual audit | `92` equation misses, `24` bit misses | fila unica do proximo CPU gate |
| V349 discussion `689915` | tokens simples, cobertura de operacoes raras, min-logprob | orientar formato de traces futuros |
| V349 discussion `688461` | taxonomia reversa e gramatica booleana dinamica | orientar DSL CPU |
| V349 discussion `690307` | bit-pair/bitsum/stride | implementar/verificar bit gate |
| V349 discussion `690756` | bit full-byte vs bit-pair e limite 3-input | testar as duas leituras e fallback 3-input limitado |
| V349 discussion `685886` | bit delta, plausibility filter e verificacao por hipotese | orientar trace bit se houver novo teacher |
| V349 discussions `684192`, `694556` | operador ausente e multiplos candidatos simbolicos | exigir abstain por ambiguidade |
| V349 discussions `688277`, `689877` | `equation_transform` continua sem regra simples; operadores podem aparecer ausentes nos exemplos | operador ausente so entra se taxonomia provar regra sem conflito |
| V349 discussion `698293` | oracle simbolico gold-conditioned | usar so como taxonomia/fixture, nunca como inferencia |
| V349 discussion `693260` | synthetic CoT alto pode piorar LB | reforca kill-switch por ACC |
| V349 discussion `697491` | dataset deterministicamente melhor pode piorar LB por formato, duplicacao de traces e saturacao de gradiente | exigir dataset minimo, sem trace duplicado, com hard negatives e ACC gate |
| V349 discussion `687798` | bit e exact-string | manter scorer exato e testes boxed/brace |
| Compilacoes externas 2026-05-14 | listam notebooks/datasets/papers de KD, Nemotron, CoT, Puzzle-KD, Cascade e solvers publicos | nao entram como acao; so sobrevivem as regras/taxonomias ja gateadas acima |

## Roadmap ativo

### 1. V350 - CPU residual no-loss gate

Objetivo: procurar ganho novo sem GPU usando apenas os residuos V348.

Status: concluido e aprovado.

Resultado medido:

- V343 baseline: `199/315`, `equation=63/155`, `bit=136/160`.
- V350 integrado: `201/315`, `equation=63/155`, `bit=138/160`.
- Ganhos: `2`.
- Perdas: `0`.
- IDs aceitos:
  - `4ada9150`: `01111111 -> 01111011`, regra `output=OR(ROL2(input),SHL4(input))`.
  - `4c327b55`: `11011110 -> 11011100`, regra `output=XOR(SHL1(input),SHR4(input))`.
- Artefatos: `artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/`.

Entradas obrigatorias:

- `artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_equation_residuals.csv`
- `artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_bit_residuals.csv`
- Manifest V343 como teacher atual.
- Manifest/triage V349 como fonte de regras e restricoes.

Equation rules a testar:

- symbolic/punctuation residual com contagem de candidatos;
- numeric/operator residual ainda nao coberto por V343;
- operador alvo ausente nos exemplos: aceitar apenas se a regra for inferivel por taxonomia sem conflito;
- multiplos programas simbolicos compativeis: `abstain`;
- `candidate_count` baixo e `conflict_count=0` obrigatorios.

Bit rules a testar:

- full-byte unary transform;
- per-output-bit bit-pair/bitsum/stride;
- bounded 3-input fallback apenas nos `24` misses residuais;
- aceitar somente se a mudanca for no-loss contra o weak diagnostic.

Saida obrigatoria:

- `v350_candidate_rules.csv`;
- `v350_candidate_decisions.csv`;
- `v350_no_loss_gate_manifest.json`;
- resumo por familia, por `rule_class` e por motivo de rejeicao.

Gate de promocao:

- passar somente se `losses=0`;
- `equation_transform>63` ou `bit_manipulation>136`;
- se nao passar, nao rodar HF.

Decisao: passou para V351 por `bit_manipulation 136 -> 138` com `0` perdas.

### 2. V351 - Dataset minimo somente se V350 passar

Objetivo: transformar novo ganho CPU em dataset de transferencia diferente dos que falharam.

Status: concluido e aprovado no gate estrutural/tokenizacao real.

Resultado medido:

- Train: `640` linhas, todas `bit_manipulation`.
- Validation: `160` linhas, todas `bit_manipulation`.
- Train SHA256: `be8192036a570711d0858620aaeae1b0736e86588e4494cbdb2c85e8f8dcd5ed`.
- Val SHA256: `8e928e38a691f41c42ea4080c1227e053031f243cbf30dd4a1a07a98e5907f93`.
- `id_overlap=0`.
- `prompt_sha256_overlap=0`.
- `train_val_prompt_overlap=0`.
- V286 real tokenization gate: aprovado, `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `fallback_masks=0`.
- HF dataset upload: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/439f99442729bfce14432a865abb8c260a637d24`.
- Artefatos: `artifacts/v351_v350_bit_transfer_dataset/20260514T_cpu_gate/`.

Permitido:

- hard positives: adapter erra, V350 acerta;
- hard negatives: casos parecidos onde a regra deve abster;
- replay minimo para preservar `bit>=136`;
- resposta final curta e trace deterministico somente quando necessario.

Proibido:

- repetir V337D/V344/V346 como estao;
- SFT amplo;
- mais epochs/LR sem novo sinal;
- usar weak/full rows como treino direto;
- dataset externo bruto com overlap ou licenca/uso duvidoso.

Gate:

- `id_overlap=0`;
- `prompt_sha256_overlap=0`;
- hashes e family counts fixos;
- tokenization/offset-mask aprovado;
- manifest de origem por linha.

### 3. V352 - HF smoke curto somente se V351 passar

Objetivo: testar se o novo teacher transfere para adapter-only com gasto minimo.

Status: concluido e rejeitado.

Preflight local aprovado:

- HF flavor: `a100-large`.
- Custo observado: `0.041667 USD/min`, abaixo do gate `0.05`.
- Imagem: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`.
- Dataset HF baixado e hash conferido.
- Init adapter: `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.
- Max steps: `8`.
- LR: `3.0e-8 -> 8.0e-9`.
- Trainable modules: `q_proj,k_proj,v_proj,o_proj,lm_head`.
- Artefatos: `artifacts/v352_hf_a100_v351_bit_transfer_launch/`.
- Job HF treino A100: `https://huggingface.co/jobs/felipesp1983/6a0520b13308d79117b8f393`.
- Decisao FinOps aplicada: treino cancelado apos upload completo de `checkpoint-2`; nao gastar nos steps 4/6/8 antes do weak eval.
- Weak eval checkpoint-2: `https://huggingface.co/jobs/felipesp1983/6a0524423308d79117b8f3a1`.
- Resultado weak checkpoint-2: `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, truncation `0`.
- Artefatos do eval: `artifacts/v352_hf_a100_v351_bit_transfer_launch/eval_v352_checkpoint2/`.
- Auditoria V354: `artifacts/v354_v352_transfer_failure_audit/20260514T_cpu_audit/`.
- Diagnostico V354: `0/2` acertos V350 aceitos foram transferidos; V352 ficou `-10` total vs V350 (`-7` equation, `-3` bit).

Configuracao:

- A100 como padrao FinOps; H200 so se tecnicamente necessario.
- Primeiro checkpoint cedo.
- Weak eval imediato.
- Logs acompanhados a cada aproximadamente `40s`.

Kill-switch aplicado:

- cancelar se `bit<136`;
- cancelar se `equation=56`;
- cancelar se `total<=192`;
- cancelar se truncation regredir;
- cancelar se o primeiro checkpoint nao puder mais bater o gate.

Decisao: V352 falhou no kill-switch. Nao continuar checkpoints 4/6/8, nao fazer full eval, nao packagear e nao submeter.

### 4. V353 - Full eval, package e submit

Status: bloqueado porque V352 nao passou weak gate.

Passos:

1. Full eval official-like.
2. Comparar contra `823/947`.
3. Validar package adapter-only rank `<=32`.
4. Gerar manifest com diff por familia.
5. Submeter ao Kaggle somente se o ganho full for medido.

### 5. V355 - Proximo CPU gate residual, sem HF

Objetivo: encontrar ganho novo antes de qualquer GPU. A prioridade volta para solver/verifier CPU, porque V352 provou que repetir LoRA sobre ganho pequeno nao transfere.

Status: concluido e bloqueado.

Resultado medido:

- Baseline V350: `201/315`, `equation=63/155`, `bit=138/160`.
- V355 integrado: `201/315`, `equation=63/155`, `bit=138/160`.
- Candidatos auditados: `49`.
- Candidatos aceitos: `0`.
- Bit stride: rejeitado; nao teve ganhos e gerou perdas.
- Bit solver atual: rejeitado; a classe com 1 ganho tambem tinha 6 perdas.
- Equation cryptarithm conflicts: `2` ganhos potenciais, mas ambos dependem de operador ausente/ambiguidade; continuam `abstain`.
- Artefatos: `artifacts/v355_cpu_residual_gate/20260514T_cpu_gate/`.

Entradas:

- `artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_integrated_predictions.csv`.
- `artifacts/v354_v352_transfer_failure_audit/20260514T_cpu_audit/v354_v352_transfer_failure_manifest.json`.
- `artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/`.
- Regras/taxonomias V349 ja filtradas.

Foco tecnico:

- `equation_transform`: atacar primeiro os `92` misses residuais com DSL expandida, mas aceitar so regra no-loss com `candidate_count` baixo e `conflict_count=0`.
- `bit_manipulation`: completar bit-pair/bitsum/stride e bounded 3-input em CPU, sem treino, buscando ultrapassar `138/160`.
- Gerar `accepted/rejected` por linha, com motivo de rejeicao. Ambiguidade vira `abstain`, nao chute.

Gate:

- Promover somente se `losses=0` e `equation>63` ou `bit>138`.
- Se nao houver ganho CPU novo, nao rodar HF.
- Se houver ganho CPU novo, construir dataset com replay forte do adapter `192/315`, nao repetir V351 puro.

Decisao: V355 nao passou. Nao ha autorizacao para HF.

### 6. V356 - Ambiguidade equation sem weak-label cherry-pick

Objetivo: atacar apenas os residuos que parecem resolviveis, mas foram rejeitados por ambiguidade em V350/V355.

Status: concluido e bloqueado.

Resultado medido:

- Conflitos auditados: `2`.
- Desempates label-free encontrados: `0`.
- Motivo nos 2 casos: o operador decisivo aparece somente na query, nunca nos exemplos.
- Artefatos: `artifacts/v356_equation_conflict_tiebreaker/20260514T_cpu_audit/`.

Escopo permitido:

- estudar os `2` casos de `symbolic_cryptarithm_multi_operator_digits_*` com nova regra de desempate label-free;
- aceitar somente se a regra tambem rejeitar os candidatos incorretos sem olhar a resposta weak;
- manter `conflict_count=0` antes de qualquer promocao.

Escopo proibido:

- escolher `add`/`mul` porque bate no weak label;
- usar operador ausente sem evidencia nos exemplos;
- promover conflito como treino LoRA.

Gate:

- Se nao existir desempate label-free, manter `abstain`.
- Se o desempate existir, rodar contra todos os `92` equation misses e exigir `losses=0`, `equation>63`.
- Sem HF ate passar.

Decisao: V356 nao passou. Nao promover os 2 acertos aparentes de equation.

### 7. V357 - Bit global ternary CPU gate

Objetivo: testar uma familia nova de regras bit que ainda nao havia sido falsificada: expressoes globais ternarias full-byte com transformacoes `ROL`, `SHL`, `SHR` e operadores booleanos.

Status: concluido e aprovado.

Resultado medido:

- Baseline V350: `201/315`, `equation=63/155`, `bit=138/160`.
- V357 integrado: `214/315`, `equation=63/155`, `bit=151/160`.
- Ganhos: `13`.
- Perdas: `0`.
- IDs aceitos: `e6d2a064`, `0e70c867`, `05ca617c`, `a6704625`, `78d02fc5`, `55d834d1`, `4ef88f92`, `202af98d`, `3ace787f`, `3a7dd604`, `06120e47`, `e1f3ffbb`, `82ae858c`.
- Artefatos: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/`.

Regra de uso:

- V357 nao e submit direto, porque package Kaggle precisa continuar adapter-only.
- V357 e teacher/verifier para gerar dataset sintetico controlado.
- Nao misturar V357 com rows weak/full como treino direto.

Decisao: passou para V358 por `bit_manipulation 138 -> 151` com `0` perdas.

### 8. V358 - Dataset de transferencia V357

Objetivo: converter os `13` ganhos V357 em dataset pequeno, verificavel e diferente da rota V351/V352 que falhou.

Status: concluido e aprovado no gate estrutural/tokenizacao real.

Resultado medido:

- Train: `1152` linhas, todas `bit_manipulation`.
- Validation: `288` linhas, todas `bit_manipulation`.
- Subcategorias train: `832` ternary, `320` binary replay.
- Subcategorias val: `208` ternary, `80` binary replay.
- Train SHA256: `6881308c7e46167ea8752513dd6e986d14b39f04f661dbac8d9ed18d189f1a05`.
- Val SHA256: `d92f4bdf2e622be958ae09353bf3965d2a23e1e6fcea95fbd77c8bcbdf0b6b47`.
- `id_overlap_with_reference=0`.
- `prompt_sha256_overlap_with_reference=0`.
- `train_val_prompt_overlap=0`.
- V286 real tokenization gate: aprovado, tokenizer real, `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `fallback_masks=0`.
- Artefatos: `artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate/`.

Permitido:

- usar somente regras verificadas de V357 e replay V350;
- treino smoke curto em HF;
- hard negatives/preferencias apenas se o launcher os usar explicitamente e com gate.

Proibido:

- treino longo antes do primeiro weak eval;
- continuar se o primeiro checkpoint nao superar adapter-only `192/315`;
- usar `eval_loss` como criterio de promocao.

Decisao historica: liberou upload HF e V359 smoke curto, sem liberar full/package/submit.

### 9. V359 - HF smoke V358 bit ternary

Objetivo: verificar se o ganho V357 transfere para LoRA adapter-only. Esta e a primeira tentativa com sinal CPU grande o bastante para justificar GPU apos a falha V352.

Status: concluido e rejeitado.

Configuracao obrigatoria:

- A100 como padrao FinOps.
- Init adapter: melhor adapter-only conhecido, preferencialmente V290 checkpoint-6 usado nos smokes anteriores.
- Dataset HF: V358 depois de upload com hashes fixos.
- Primeiro checkpoint cedo, weak eval imediato.
- Logs acompanhados a cada aproximadamente `40s`.

Kill-switch:

- cancelar se `bit<136`;
- cancelar se `equation<=56`;
- cancelar se `total<=192`;
- cancelar se truncation regredir;
- cancelar se custo/tempo ficar fora do esperado.

Promocao:

- somente se weak adapter-only `>192/315`, `bit>=136`, `equation>56` ou `bit` subir sem derrubar total;
- se passar weak, rodar full official-like;
- se full `>823/947`, packagear e so entao considerar Kaggle submit.

Resultado medido:

- Train job HF A100: `https://huggingface.co/jobs/felipesp1983/6a0598743308d79117b8f539`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v359-nemo-a100-v358-bit-ternary-v290ckpt6`.
- Checkpoints completos: `checkpoint-2`, `checkpoint-4`, `final`.
- Weak eval HF H200: `https://huggingface.co/jobs/felipesp1983/6a059efce48bea4538b9c865`.
- Checkpoint-2: `190/315`, `equation_transform=56/155`, `bit_manipulation=134/160`, truncation `1`.
- Decisao FinOps: weak eval cancelado apos checkpoint-2, porque o primeiro candidato ficou abaixo de `192/315`, abaixo de `bit>=136` e com truncation regressivo.

Decisao: V359 nao libera full eval, package ou Kaggle submit. Nao continuar checkpoint-4/final sem nova evidencia independente, porque o primeiro checkpoint ja falhou no gate.

### 10. V360 - Auditoria pos-falha de transferencia e nova rota CPU

Objetivo: parar de repetir SFT bit puro e descobrir por que o ganho CPU V357 nao vira LoRA.

Status: concluido.

Escopo:

- comparar V359 checkpoint-2 contra a familia bit do baseline assim que houver predicoes completas; se nao houver artefato completo, usar apenas o resumo e nao gastar outro H200 so para auditoria;
- identificar se a falha veio de formato de trace, truncation, resposta final, excesso de tokens ou incapacidade de memorizar regra global ternaria;
- voltar para CPU gate, nao para outro HF imediato;
- testar uma rota mais objetiva: traces ainda mais curtos, resposta first-token mais simples, ou dataset de classificacao de regra antes de gerar resposta.

Gate:

- nenhum novo HF enquanto nao houver um novo dataset que explique a falha V359;
- qualquer V360/V361 deve provar em CPU que o formato novo preserva os `13` ganhos V357 e nao reintroduz truncation;
- proximo HF so pode ser `max_steps<=2` ate primeiro weak eval.

Resultado medido:

- Manifesto: `artifacts/v360_v359_transfer_failure_audit/20260514T_cpu_audit/v360_v359_transfer_failure_audit_manifest.json`.
- Relatorio: `artifacts/v360_v359_transfer_failure_audit/20260514T_cpu_audit/v360_v359_transfer_failure_audit_report.md`.
- Decisao: `v360_blocks_more_hf_on_v358_v359`.
- Evidencia 1: V357 CPU teacher tinha `214/315`, equation `63/155`, bit `151/160`; V359 checkpoint-2 caiu para `190/315`, equation `56/155`, bit `134/160`, truncation `1`.
- Evidencia 2: V358 tinha apenas `15` regras unicas, repetidas em `1152` treino e `288` validacao.
- Evidencia 3: os arquivos preference/hard-negative existem (`2304/576` rows), mas o launcher V359 usou SFT e nao referenciou `preferences_train`.
- Evidencia 4: `assistant_only_boxed_rows=0`; todas as completions ensinam `Rule`, `Check examples` e `Final answer`, enquanto o weak eval forca resposta direta sem explicacao.
- Evidencia 5: tokenization passou com `0` prompt truncation e `0` completion drops, entao a falha nao e truncation de treino; e problema de objetivo/formato/cobertura.

Decisao: nao rodar mais HF com V358/V359. O proximo passo deve ser V361 em CPU: dataset answer-first/boxed-only ou retorno para DSL/verifier de equation. Se houver outro HF, deve ser apenas apos gate CPU e com kill-switch no primeiro checkpoint.

### 11. V361 - Boxed-only transfer dataset

Objetivo: testar a hipotese mais concreta do V360 sem gastar GPU: o V359 treinou completions longas e desalinhadas com o weak eval. V361 usa o mesmo professor V357/V358, mas faz a completion ser exatamente `\boxed{answer}`.

Status: concluido em CPU.

Resultado:

- Script: `scripts/build_v361_v357_boxed_only_transfer_dataset.py`.
- Manifesto: `artifacts/v361_v357_boxed_only_transfer_dataset/20260514T_cpu_gate/v361_v357_boxed_only_transfer_manifest.json`.
- Resumo: `artifacts/v361_v357_boxed_only_transfer_dataset/V361_RESULT_SUMMARY.md`.
- Train/validation: `1152/288`.
- Preference rows: `2304/576`, com hard negative one-bit-flip e negative sem box.
- `assistant_boxed_only_rows`: `1152/1152` train e `288/288` validation.
- Assistant length: `16` chars em todas as linhas.
- Train/validation prompt overlap: `0`.
- Hash train: `be742d7a82bf1c98f33d67bed8903006068c139ab74f798055fcc7d435ffa4db`.
- Hash validation: `4c93766e7fae72da14f879177e15c3c6300b7991e4efc8fba4d7fe75d3df5332`.
- V286 real tokenization gate: `tokenization_gate_passed`, mode `boxed_only`, `token_max=286`, loss tokens `15`, `0` truncation, `0` completion drops, `0` fallback masks.

Limite honesto:

- V361 corrige formato, mas ainda usa as mesmas `15` regras; nao prova ganho adapter-only.
- V361 nao melhora `equation_transform`; esta familia continua dependendo de DSL/verifier.
- Um HF baseado em V361, se rodar, deve ser apenas smoke `max_steps<=2`, A100, com weak eval imediato e cancelamento automatico se `total<=192`, `bit<136`, `equation<=56` ou truncation regressivo.

### 12. V362 - HF smoke V361 boxed-only

Objetivo: testar se o reparo de formato V361 transfere para LoRA adapter-only sem gastar em treino longo.

Status: concluido e rejeitado.

Configuracao:

- Train job HF A100: `https://huggingface.co/jobs/felipesp1983/6a05a6dd3308d79117b8f574`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v362-nemo-a100-v361-boxed-only-v290ckpt6`.
- Dataset: V361 boxed-only, `1152/288`, hashes fixos.
- Init adapter: V290 checkpoint-6.
- Max steps: `2`.
- Checkpoints enviados: `checkpoint-1`, `checkpoint-2`.
- A100 cancelado apos checkpoint-2 por FinOps; final eval/upload nao era necessario.

Weak eval checkpoint-1:

- Job HF H200: `https://huggingface.co/jobs/felipesp1983/6a05aa47e48bea4538b9c8dc`.
- Output commit: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v362-nemo-a100-v361-boxed-only-v290ckpt6/commit/ed701d36d47ad2d80694d552b777a65b70e49a88`.
- Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Resultado: `190/315`, `equation_transform=56/155`, `bit_manipulation=134/160`, truncation `1`.
- Comparacao contra gate adapter-only: baseline `192/315`, equation `56/155`, bit `136/160`, truncation `0`.

Decisao: rejeitado. V362 regrediu `-2` no total, `-2` em bit e introduziu truncation. Nao libera full eval, package, Kaggle submit, nem weak eval de `checkpoint-2`/`final`.

Impacto no roadmap:

- A hipotese "completion boxed-only resolve transferencia" foi falsificada no primeiro checkpoint.
- Nao gastar mais GPU em V361/V362 bit-only sem novo CPU gate.
- Proxima rota ativa volta para CPU: equation DSL/verifier nos `99` misses e expansao real do algoritmo bit-pair/bitsum/stride antes de qualquer SFT novo.

### 13. V363 - Equation residual operator-support gate

Objetivo: executar o proximo passo CPU-only apos V362, sem GPU, testando se os residuos de `equation_transform` ainda tinham ganho label-free por operador visto nos exemplos ou por prior numerico aprendido no train publico sem vazamento.

Status: concluido e bloqueado.

Resultado medido:

- Script: `scripts/analyze_v363_equation_residual_operator_support.py`.
- Manifesto: `artifacts/v363_equation_residual_operator_support/20260514T_cpu_gate/v363_equation_residual_operator_support_manifest.json`.
- Input: `artifacts/v355_cpu_residual_gate/20260514T_cpu_gate/v355_integrated_predictions.csv`.
- Train publico usado somente para prior numerico com exclusao dos `315` IDs weak.
- Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Estado de entrada: `201/315`, `equation_transform=63/155`, `bit_manipulation=138/160`.
- Estado V363: `201/315`, `equation_transform=63/155`, `bit_manipulation=138/160`.
- Ganhos aceitos: `0`.
- Perdas integradas: `0`.
- Candidatos aceitos: `0`.

Mapa dos residuos de `equation_transform`:

- `92` misses restantes.
- `10` numeric operator: todos com operador da query ausente nos exemplos; exigem prior/modelo, nao solver derivado dos exemplos.
- `70` symbolic punctuation: operador da query aparece nos exemplos, mas a DSL same-op nao encontrou candidato unico sem perda.
- `12` symbolic punctuation: operador da query ausente nos exemplos, sem desempate label-free.

Teste de prior numerico:

- V363 aprendeu priors no train publico excluindo IDs weak.
- Priors encontrados: `6`.
- Priors que geraram mudancas foram rejeitados por perdas:
  - `numeric_operator_prior_45_sub_ab`: `8` mudancas, `0` ganhos, `7` perdas.
  - `numeric_operator_prior_123_concat_ab`: `3` mudancas, `0` ganhos, `3` perdas.
  - `numeric_operator_prior_92_abs_diff`: `1` mudanca, `0` ganhos, `1` perda.
  - `numeric_operator_prior_93_abs_diff`: `1` mudanca, `0` ganhos, `1` perda.

Decisao: V363 bloqueia HF. Mais treino sobre os datasets V351/V358/V361 ou prior numerico publico nao e justificavel. O proximo trabalho util precisa ser uma nova familia de programa simbolico para os `70` casos same-op ambiguos ou um desempate label-free real para os casos query-only.

### 14. V364 - Symbolic pair-table gate

Objetivo: testar uma nova familia simbolica para os residuos same-op de `equation_transform`: tabelas por par de posicoes dos operandos (`L0/L1/R0/R1`) para cada caractere de saida.

Status: concluido e bloqueado.

Resultado medido:

- Script: `scripts/analyze_v364_symbolic_pair_table_gate.py`.
- Manifesto: `artifacts/v364_symbolic_pair_table_gate/20260514T_cpu_gate/v364_symbolic_pair_table_gate_manifest.json`.
- Input: `artifacts/v363_equation_residual_operator_support/20260514T_cpu_gate/v363_integrated_predictions.csv`.
- Estado de entrada: `201/315`, `equation_transform=63/155`, `bit_manipulation=138/160`.
- Estado V364: `201/315`, `equation_transform=63/155`, `bit_manipulation=138/160`.
- Candidate changes: `12`.
- Ganhos aceitos: `0`.
- Perdas integradas: `0`.

Resumo das regras testadas:

- `symbolic_pair_table_len_2`: `2` mudancas, `0` ganhos, `0` perdas.
- `symbolic_pair_table_len_3`: `2` mudancas, `0` ganhos, `0` perdas.
- `symbolic_pair_table_len_4`: `8` mudancas, `0` ganhos, `2` perdas.

Decisao: V364 bloqueia HF. A hipotese pair-table nao explica os residuos e nao deve virar dataset de treino.

### 15. V365 - Bit residual boolean grammar gate

Objetivo: testar se uma gramatica booleana per-output-bit mais ampla conseguiria resolver os `9` residuos de `bit_manipulation` restantes depois do V357, sem destruir linhas que o teacher V357 ja acertava.

Status: concluido e bloqueado.

Resultado medido:

- Script: `scripts/analyze_v365_bit_residual_boolean_grammar_gate.py`.
- Manifesto: `artifacts/v365_bit_residual_boolean_grammar_gate/20260514T_cpu_gate/v365_bit_residual_boolean_grammar_gate_manifest.json`.
- Input: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_integrated_predictions.csv`.
- Estado de entrada V357: `214/315`, `equation_transform=63/155`, `bit_manipulation=151/160`.
- Residuos de bit antes do gate: `9`.
- Candidate changes: `73`.
- Candidate gains: `0`.
- Candidate losses: `66`.
- Candidatos aceitos: `0`.
- Estado V365: `214/315`, `equation_transform=63/155`, `bit_manipulation=151/160`.

Leitura tecnica:

- A gramatica booleana per-bit livre encaixa muitas assinaturas dos exemplos, mas nao generaliza para a query.
- Ela altera linhas corretas do V357 e nao resolve nenhum dos `9` misses restantes.
- Esta rota nao deve virar SFT, preference dataset, adapter soup, nem HF job.

Decisao: V365 bloqueia HF. A proxima rota de bit precisa ser mais restrita e estrutural: bit-pair/bitsum/stride ou solver full-byte com prova de classe `0` perdas.

### 16. V366 - Bit full-byte ternary operator gate

Objetivo: testar uma rota mais restrita que V365: uma unica expressao ternaria full-byte precisa explicar todos os exemplos, e a promocao e feita por familia de operador ternario, nao por ID.

Status: concluido e aprovado como teacher CPU.

Resultado medido:

- Script: `scripts/analyze_v366_bit_fullbyte_ternary_op_gate.py`.
- Manifesto: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_bit_fullbyte_ternary_op_gate_manifest.json`.
- Input: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_integrated_predictions.csv`.
- Estado de entrada V357: `214/315`, `equation_transform=63/155`, `bit_manipulation=151/160`.
- Candidate changes: `9`.
- Candidate gains: `8`.
- Candidate losses: `1`.
- Regras aceitas:
  - `bit_fullbyte_ternary_op_CHO`: `4` ganhos, `0` perdas.
  - `bit_fullbyte_ternary_op_MAJ3`: `4` ganhos, `0` perdas.
- Regra rejeitada:
  - `bit_fullbyte_ternary_op_AND_OR`: `0` ganhos, `1` perda.
- Estado V366: `222/315`, `equation_transform=63/155`, `bit_manipulation=159/160`.
- Perdas aceitas: `0`.

IDs aceitos:

- `1abaffca`
- `b8722d19`
- `7192535b`
- `1a7c8520`
- `a6192d29`
- `048cc279`
- `b8aa3072`
- `5ba26f21`

Decisao: V366 e o melhor teacher CPU atual. O proximo passo e V367: dataset de transferencia pequeno com `CHO`/`MAJ3`, replay forte de bit, hard negatives e gate de tokenizacao. Sem HF antes desses checks.

### 17. V367 - V366 bit transfer dataset

Objetivo: converter o teacher V366 em um dataset adapter-only tecnicamente treinavel, corrigindo duas falhas anteriores: sinal novo diluido e completion longa/desalinhada. V367 usa completion `boxed_only`, prioriza as `8` regras novas V366 e mantem replay reduzido V357/V350.

Status: concluido e aprovado pelo tokenization gate real.

Resultado medido:

- Script: `scripts/build_v367_v366_bit_ternary_transfer_dataset.py`.
- Manifesto: `artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_manifest.json`.
- Tokenization gate: `artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json`.
- Train rows: `1128`.
- Validation rows: `282`.
- V366 new rows: `768` train, `192` validation.
- V357 replay rows: `312` train, `78` validation.
- V350 replay rows: `48` train, `12` validation.
- Unique rules: `23`.
- Train/validation prompt overlap: `0`.
- Weak reference id overlap: `0`.
- Weak reference prompt hash overlap: `0`.
- Assistant boxed-only rows: `1128/1128` train, `282/282` validation.
- Real tokenizer: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16@cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Token max: `285`.
- Loss tokens: `15`.
- Prompt truncation rate: `0.0`.
- Completion tokens dropped: `0`.
- Offset masks: `1128/1128` train, `282/282` validation.
- Fallback masks: `0`.

Decisao: V367 liberou somente o HF smoke curto V368. O primeiro checkpoint foi avaliado imediatamente conforme o kill-switch. Como V368 falhou, V367 nao deve ser usado para outro HF job sem novo sinal CPU independente.

### 18. V368 - HF smoke V367/V366 bit ternary

Objetivo: testar se o teacher CPU V366 (`222/315`, bit `159/160`) transferia para LoRA adapter-only usando o dataset V367 boxed-only.

Status: concluido e rejeitado.

Configuracao:

- Train job HF A100: `https://huggingface.co/jobs/felipesp1983/6a05bad7e48bea4538b9c997`.
- Weak eval checkpoint-1 HF H200: `https://huggingface.co/jobs/felipesp1983/6a05be653308d79117b8f5ce`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v368-nemo-a100-v367-bit-ternary-v290ckpt6`.
- Eval artifact commit: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v368-nemo-a100-v367-bit-ternary-v290ckpt6/commit/ffbbbb3a77de65cbd87eb71a6ec9b1516507da68`.
- Dataset: V367, `1128/282`, boxed-only, `0` truncation no tokenizer gate.
- Init adapter: V290 checkpoint-6.
- Max steps: `2`; checkpoint-1 avaliado como primeiro kill-switch.

Resultado medido:

- Checkpoint-1 weak: `191/315`.
- `equation_transform=56/155`.
- `bit_manipulation=135/160`.
- Truncation: `0`.
- Baseline adapter-only: `192/315`, equation `56/155`, bit `136/160`.
- Delta: `-1` total, `0` equation, `-1` bit.
- Artefatos locais: `artifacts/v368_hf_a100_v367_bit_ternary_launch/`.

Decisao: rejeitado. Nao rodar checkpoint-2 weak eval, nao rodar full eval, nao packagear e nao submeter.

### 19. V369 - Auditoria de falha de transferencia V368

Objetivo: medir se V368 aprendeu algum dos `8` ganhos aceitos do teacher V366 e separar ganhos reais de regressao.

Status: concluido e bloqueado.

Resultado medido:

- Script: `scripts/analyze_v369_v368_transfer_failure_audit.py`.
- Manifesto: `artifacts/v369_v368_transfer_failure_audit/20260514T_cpu_audit/v369_v368_transfer_failure_manifest.json`.
- Baseline adapter-only: `192/315`, equation `56/155`, bit `136/160`.
- V366 CPU teacher: `222/315`, equation `63/155`, bit `159/160`.
- V368 checkpoint-1: `191/315`, equation `56/155`, bit `135/160`.
- Ganhos V366 aceitos testados: `8`.
- Ganhos V366 transferidos para V368: `0/8`.
- V368 mudou `10` linhas contra baseline: `1` ganho, `2` perdas, `7` mudancas neutras.
- Unico ganho novo V368: `4ef88f92`.
- Perdas V368: `8740ed31`, `59bee375`.

Leitura tecnica:

- O problema nao e falta de evidencia no teacher CPU; V366 tinha sinal forte e no-loss.
- O problema e transferencia para LoRA com SFT bit-only: a adapter continua respondendo como baseline nos `8` IDs que V366 acertou.
- Mais epochs/checkpoints nessa rota nao sao justificaveis porque o primeiro checkpoint ja violou total e bit.

Decisao: bloquear V367/V368 bit-only SFT. A proxima acao deve ser CPU-only e precisa gerar um novo tipo de sinal, nao apenas repetir V367 com mais treino.

## Removido do plano ativo

Estes itens nao devem ser reexecutados como acao principal. So podem voltar se um novo CPU gate provar uma razao nova.

| Item/rota | Decisao |
|---|---|
| Solver/verifier direto no package | bloqueado pelo V336B; package deve ser adapter-only |
| V337D/V344/V346 datasets como estao | nao transferiram os ganhos para LoRA; usar apenas como historico/taxonomia |
| V338B SFT minimal transfer | falhou: `190/315`, bit `134` |
| V341 clean preference | falhou: `190/315`, bit `134`; preferencia estava saturada |
| V344 preference/abstain | sem ganho: `192/315`, `0` ganhos |
| V346 answer exact-match | regressao: `191/315`, bit `135` |
| V351/V352 bit-transfer direto | V352 checkpoint-2 falhou: `191/315`, `equation=56`, `bit=135`, transferiu `0/2` acertos V350 |
| V355 bit stride/current solver direto | bloqueado: stride teve perdas; solver atual teve 1 ganho mas 6 perdas na melhor classe |
| V355 equation conflict cherry-pick | bloqueado: 2 acertos potenciais dependem de conflitos sem desempate label-free |
| V356 query-only operator conflicts | bloqueado: os 2 acertos aparentes de equation exigiriam escolher operador que nao aparece nos exemplos |
| Mais HF sobre V351/V352 | removido; V352 ja falhou e V358 substitui a rota com sinal CPU maior |
| Mais HF sobre V358/V359 sem auditoria | removido; V359 checkpoint-2 caiu para `190/315`, bit `134`, truncation `1` |
| Mais HF sobre V358/V359 apos V360 | removido; V360 mostrou 15 regras estreitas, hard negatives nao usados e formato de completion desalinhado |
| HF longo sobre V361 | removido; V361 e apenas reparo de formato e nao justifica treino longo |
| Mais HF sobre V361/V362 boxed-only | removido; V362 checkpoint-1 caiu para `190/315`, equation `56`, bit `134`, truncation `1` |
| V362 checkpoint-2/final weak eval | bloqueado por FinOps; checkpoint-1 ja violou total, bit e truncation |
| V363 public-train numeric operator priors | bloqueado; causaram perdas e `0` ganhos |
| V363 same-operator symbolic DSL atual | bloqueado; nao gerou candidato unico no-loss |
| V364 symbolic pair-table | bloqueado; `12` mudancas, `0` ganhos, com perdas em `len_4` |
| V365 bit per-output boolean grammar | bloqueado; `73` mudancas candidatas, `0` ganhos, `66` perdas candidatas |
| Mais HF sobre V367/V368 bit-only | removido; V368 checkpoint-1 caiu para `191/315`, bit `135`, e V369 mostrou `0/8` ganhos V366 transferidos |
| V368 checkpoint-2/final weak eval | bloqueado por FinOps; checkpoint-1 ja violou total e bit |
| Checkpoints restantes V346 4/6 | nao avaliar sem novo sinal independente |
| Checkpoints restantes V352 4/6/8 | bloqueados por FinOps; checkpoint-2 ja caiu abaixo do gate |
| V359 checkpoint-4/final weak eval | cancelado por FinOps; checkpoint-2 ja violou total, bit e truncation |
| Mais epochs/LR/checkpoints sem CPU gate | removido por FinOps |
| `eval_loss` como criterio de promocao | removido; ACC e o criterio |
| Adapter soups | testado; nao moveu equation |
| Prompt/thinking variants amplas | regressao severa |
| Public external adapters | abaixo do baseline |
| Raw `kienngx` / `konbu17` / `furkankesen` datasets | overlap/flags/risco; usar so taxonomia validada |
| Public Kaggle SFT notebooks e CoT datasets listados nas compilacoes | sem ganho medido novo sobre nosso gate; nao reexecutar como rota ativa |
| Generic knowledge distillation notebooks/papers | metodologia generica, sem evidencia em `bit_manipulation`/`equation_transform` do KG1 |
| Nemotron Cascade / OpenReasoning / OpenMath / general HF reasoning datasets | P2 metodologico; nao usar em treino sem novo gate V350/V351 e anti-leakage |
| TruthGuard/metacognition/calibration generica | nao ataca o erro exato de `bit`/`equation` e nao gera adapter-only KG1 |
| Equation solvers genericos de imagem/CNN/SymPy fora do desafio | dominio diferente; nao usar no plano ativo |
| ReasoningGym/Alice direto | drift severo; usar so fixtures/probes se gateado |
| GGUF/Spaces/API externo | nao e submit LoRA adapter-only |
| GRPO | caro e sem policy forte nova; P2 fora do plano ativo |
| OpenRouter/destilacao como acao propria | conclusao ja incorporada; nao e rota executavel |
| Buscas web genericas | so entram se produzirem regra, dataset ou gate verificavel |

## Regras permanentes

- Roadmap ativo so aceita item com impacto medido ou acao concreta.
- Toda evidencia externa deve ser classificada como `ganho medido`, `taxonomia/teacher`, `bloqueado por overlap`, `rejeitado por gate` ou `P2 metodologico`.
- Enquanto job HF estiver rodando, verificar logs a cada aproximadamente `40s`.
- Se o job nao puder mais bater o gate, cancelar. A decisao FinOps correta e cancelar e nao gastar.
- Nenhum notebook alterado pode ser entregue sem `scripts/notebook_release_gate.py`.
- Nenhum Kaggle submit sem weak/full gain medido.

## Proxima acao unica

V370 CPU-only: abandonar mais SFT bit-only sobre V367/V368 e procurar um novo sinal transferivel antes de qualquer HF. A prioridade e uma das duas rotas, nesta ordem:

1. Diagnosticar representacao solver-to-adapter: usar V369 para criar probes de formato ainda mais diretos nos `8` ganhos V366 e nos `2` losses V368, mas sem treino GPU. Se nao houver uma nova representacao verificavel, nao rodar HF.
2. Voltar para `equation_transform`: testar uma nova DSL simbolica diferente de V363/V364 nos `92` misses residuais, com `candidate_count`, `conflict_count`, `losses=0` e ganho acima de `63/155`.

Full/package/submit continuam bloqueados ate um adapter-only bater o baseline weak/full medido.
