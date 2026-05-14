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
| Residual depois do V357 | mapa pronto | `92` misses | `9` misses | fila residual CPU |
| V349 Kaggle discussions | `140/140` topicos | reforca ambiguidade/DSL | reforca bit full-byte/bit-pair/3-input | guia do proximo CPU gate |
| Double check compilacoes 2026-05-14 | `3` arquivos | sem novo ganho medido | sem novo ganho medido | classifica links; nao libera HF |

Conclusao honesta:

- Ha ganho real em CPU solver/verifier: `192 -> 214` weak, com `equation_transform 56 -> 63` e `bit_manipulation 136 -> 151`, sempre com `0` perdas no weak diagnostic.
- O ganho novo comprovado de V357 e forte em bit, mas ainda e teacher CPU. Ainda nao ha ganho adapter-only novo. V338B, V341, V344, V346 e V352 falharam em transferir ganhos menores para LoRA.
- `eval_loss` menor nao significa ACC maior. Promocao agora e somente por ACC weak/full.
- HF GPU volta a ficar permitido apenas para um smoke V359 curto, porque V357 passou e V358 foi gateado. Full/package/submit continuam bloqueados ate ACC weak/full adapter-only melhorar.

## Metas

Para liberar novo teacher CPU:

- `losses=0`.
- `bit_manipulation >=136/160`.
- `equation_transform >63/155` ou `bit_manipulation >138/160` no weak diagnostic.
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
| V358 V357 bit ternary dataset | `1152` train, `288` val, `0` overlap por `id`/`prompt_sha256`, V286 real gate aprovado | unica fonte atual para smoke V359 |
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

Decisao: libera upload HF e V359 smoke curto, nao libera full/package/submit.

### 9. V359 - HF smoke V358 bit ternary

Objetivo: verificar se o ganho V357 transfere para LoRA adapter-only. Esta e a primeira tentativa com sinal CPU grande o bastante para justificar GPU apos a falha V352.

Status: proxima acao.

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
| Checkpoints restantes V346 4/6 | nao avaliar sem novo sinal independente |
| Checkpoints restantes V352 4/6/8 | bloqueados por FinOps; checkpoint-2 ja caiu abaixo do gate |
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

Upload do dataset V358 para HF, criar/validar launcher V359 e rodar apenas o smoke curto com kill-switch. Full/package/submit continuam bloqueados ate ganho adapter-only medido.
