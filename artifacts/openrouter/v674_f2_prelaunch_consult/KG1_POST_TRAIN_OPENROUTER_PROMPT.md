# KG1 post-training crisis consult

You are an external ML/MLOps/code-review panel for the KG1 NVIDIA Nemotron Model Reasoning Challenge solution.
Return only actionable findings. Do not invent files, metrics, or leaderboard results.
If evidence is insufficient, say exactly which local artifact or metric is missing.

## Hard rules
- Kaggle score is final-answer accuracy, not eval_loss alone.
- False gains are forbidden: no promotion without label-free extraction, verify_answer, zero truncation, and protected-row backfire guards.
- Distinguish bad decoding from adapter weights pushing the model toward wrong answers.
- Loss must be cross-entropy on masked assistant/answer tokens and must stay aligned with accuracy gates.
- If row_loss_weight is used in train, validation eval_loss must use the same row-weight contract.
- No new paid GPU run should be recommended unless CPU/gate evidence predicts an accuracy gain and protects current correct rows.
- Prefer A100-large. H200 is allowed only when A100 cannot run the stack or memory requirement is objectively proven.

## Required response format
1. Verdict: proceed / block / needs artifact.
2. Top 5 concrete bugs or gaps, each tied to evidence in the prompt.
3. Exact next experiment that is cheapest and most likely to improve weak ACC.
4. Parameters to change or freeze, with values.
5. Gates that must pass before another paid GPU job.
6. Anything in the current plan that should be deleted because it is noise.


## Current run metadata
- run_id: `v674-f2-prelaunch-rowloss-adapter-save`
- generated_at_utc: `2026-05-19T19:19:25.711489+00:00`
- Current observed plateau: weak ACC has not exceeded the deployable baseline. V664 reached only 192/315.
- Best actionable weak target remains at least 196/315 without protected-row regression, with bit >= 136 and equation >= 60.
- V664 weak result: total 192/315, bit 136/160, equation 56/155, truncated 0, boxed_rate 1.0, but completions were extremely long and protected bit row 8740ed31 backfired.
- V664 training moved only q_proj/v_proj LoRA tensors from V290 checkpoint-6; non-q/v tensors were unchanged.
- V664 train loss decreased in 2 steps, but the generation behavior stayed long and unsafe. Loss movement alone is not acceptable evidence.


## Roadmap
Path: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v284_official_gate_worktree\artifacts\roadmaps\KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md`

```text
# KG1 Score Improvement Roadmap

Atualizado: 2026-05-19 19:15 UTC. V672/V673 substitui o plano ativo V671. A decisao
de hoje vem de cinco fontes: consulta OpenRouter V670, evidencias CPU locais
V541/V612, sinais no-loss V350/V366, auditoria completa das discussions Kaggle
V671 e consenso OpenRouter V672. Objetivo de hoje continua: sair de `192/315` mantendo
`bit_manipulation>=136/160` e levando `equation_transform>=60/155`, sem
backfire protegido, sem ganho falso e sem treino cego.

Artefatos novos:

- Prompt V670:
  `artifacts/openrouter/v670_today_family_gain_consult/KG1_V670_TODAY_FAMILY_GAIN_PROMPT.md`;
- respostas OpenRouter:
  `artifacts/openrouter/v670_today_family_gain_consult/openrouter_responses.md`;
- retry util do GPT-5.5:
  `artifacts/openrouter/v670_today_family_gain_consult/openrouter_gpt55_retry.md`;
- consenso V670:
  `artifacts/openrouter/v670_today_family_gain_consult/KG1_V670_TODAY_FAMILY_GAIN_CONSENSUS.md`;
- V612 V664 vs V290:
  `artifacts/v670_today_family_gain_plan/v670_v664_vs_v290_failure_taxonomy/`;
- V541 miss-map V290 baseline:
  `artifacts/v670_today_family_gain_plan/v670_v290_baseline_missmap/`;
- auditoria Kaggle V671:
  `artifacts/v671_kaggle_discussions_audit/KG1_V671_KAGGLE_DISCUSSIONS_AUDIT.md`;
- indice completo Kaggle V671:
  `artifacts/v671_kaggle_discussions_audit/KG1_V671_KAGGLE_DISCUSSIONS_FULL_INDEX.md`;
- corpus completo Kaggle V671:
  `artifacts/v671_kaggle_discussions_audit/KG1_V671_KAGGLE_DISCUSSIONS_ALL_TEXT.md`;
- prompt OpenRouter V672:
  `artifacts/openrouter/v672_today_family_gain_from_kaggle_discussions/KG1_V672_OPENROUTER_TODAY_GAIN_PROMPT.md`;
- respostas OpenRouter V672:
  `artifacts/openrouter/v672_today_family_gain_from_kaggle_discussions/openrouter_responses.md`;
- consenso V672:
  `artifacts/openrouter/v672_today_family_gain_from_kaggle_discussions/KG1_V672_OPENROUTER_TODAY_GAIN_CONSENSUS.md`;
- ledger V672 dos 36 misses residuais:
  `artifacts/v672_residual_miss_ledger/20260519T173138Z/KG1_V672_RESIDUAL_MISS_LEDGER.md`;
- dataset V673 guardado equation+bit:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T173945Z/v673_guarded_equation_bit_transfer_manifest.json`;
- gate V286 toy de V673:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T173945Z/tokenization_gate_toy/v286_generic_tokenization_gate_manifest.json`.

Atualizacao F2/backfire pre-job, 2026-05-19 18:31 UTC:

- Regra operacional consolidada: H200 esta bloqueada. Novos jobs devem usar
  somente `a100-large`; no check HF feito nesta rodada nao havia job ativo.
- Job A100 V673 `6a0ca9f52dc5b1243da501c2` falhou antes do treino, portanto
  nao gerou adapter novo nem ACC. A causa foi contrato incorreto:
  `gate_up_proj` foi exigido como `REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS`,
  mas `gate_up_proj` e `down_proj` de MoE entram pelo contrato PEFT
  `LORA_TARGET_PARAMETERS`, nao como substring literal de modulo LoRA comum.
- Correcao aplicada no launcher V673:
  `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`,
  `TRAINABLE_LORA_NAME_SUBSTRINGS=''`,
  `REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`,
  mantendo `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`
  e `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`. Assim o gate de MoE continua
  forte, mas sem confundir nome de modulo com target parameter.
- Static gate detectou outro bug silencioso: install remoto usava PEFT/Transformers
  sem pinagem. Corrigido para `transformers==4.57.6` e `peft==0.19.1`, evitando
  regressao tipo V595 por mudanca invisivel de API PEFT.
- Script `hf_job_train_v90.py` corrigido para aplicar `row_loss_weight` tambem
  na validation. Antes, o treino media uma distribuicao ponderada e o eval_loss
  media a distribuicao fisica; isso podia criar decisao errada de best loss e
  mascarar relacao loss/ACC. O manifesto agora declara
  `validation_ignores_row_weight=False`.
- Mascara de loss agora para apos o primeiro EOS (`LOSS_MASK_STOP_AFTER_EOS=1`),
  para reduzir aprendizado de tokens extras pos-resposta e evitar drift de
  formato/decoding.
- Gates reexecutados depois das correcoes:
  `v673_static_safety_gate_after_f2_backfire_fix.json` OK,
  `v673_pre_paid_job_integration_gate_after_f2_backfire_fix.json` OK,
  `v673_v478_objective_alignment_after_lora_fix.json` OK,
  `v673_v619_surface_gate_recheck_after_lora_fix.json` OK,
  `hf_job_train_v90.py --self-test` OK e `py_compile` OK.
- Dry-run local com tokenizer real Nemotron:
  `artifacts/v673_hf_a100_launch/local_tokenize_dry_run_after_f2_fix/dry_run_model_recipe_report.json`.
  Resultado: train `720/720`, validation `180/180`, `truncated=0`,
  `prompt_truncated=0`, `fallback_masks=0`, `offset_masks=720/180`,
  validation `use_row_loss_weight=True`.
- Proximo passo permitido: commitar/pushar as correcoes e relancar somente
  `a100-large` curto V673. Primeiro checkpoint precisa de weak eval antes de
  qualquer continuacao ou submit.

Atualizacao F2/backfire pre-relaunch, 2026-05-19 19:15 UTC:

- O job A100 V673 `6a0cada33aba298b21d14304` foi cancelado antes de promover
  qualquer resultado porque o log PEFT indicou
  `save_embedding_layers=True` automaticamente. Isto era risco real de
  "adapter-only falso": o pacote poderia salvar embeddings/lm_head junto do
  adapter. Correcao: `scripts/hf_job_train_v90.py` agora salva checkpoints e
  final via `save_adapter_only(..., save_embedding_layers=SAVE_EMBEDDING_LAYERS)`,
  com default e launcher `SAVE_EMBEDDING_LAYERS=0`.
- Foi encontrado bug silencioso no objetivo ponderado: com
  `MICRO_BATCH_SIZE=1`, `LOSS_NORMALIZATION_MODE=example_mean` e reducao antiga
  por soma dos pesos, `row_loss_weight` se cancelava matematicamente em cada
  microbatch. Isso podia explicar perda/ACC desalinhados e falsa sensacao de
  calibragem. Correcao: `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`; em
  single-example microbatch o peso agora escala o loss de verdade. O self-test
  do treino prova que `weight=2` dobra o loss nesse caso.
- A validation continua usando `row_loss_weight` quando o treino usa
  `row_loss_weight`; a metrica de loss volta a medir a mesma distribuicao que
  o otimizador. Isto nao garante ACC, mas remove um erro de medicao que podia
  escolher checkpoint por loss errado.
- Thresholds promocionais atualizados e travados:
  `total>=196/315`, `bit>=136/160`, `equation>=60/155`,
  `truncated=0`, `no_box_fallback=0`, `boxed_rate=1.0`,
  protected backfire `0`. O piso antigo `equation=59` foi removido dos gates
  ativos.
- A100-only continua regra operacional. O abort de memoria foi ajustado para
  `ABORT_MAX_RESERVED_GIB=78`, porque o run anterior passou de `72 GiB` sem
  indicar estouro real; H200 permanece fora do plano.
- O dataset V673 foi regenerado porque o metadata dizia
  `completion_format=boxed_only`, mas o assistant alvo era trace curto +
  `Final answer: \boxed{...}`. O metadata ativo agora e
  `trace_plus_final_boxed` em `100%` das linhas.
- Dataset ativo V673:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T190246Z/`;
  train `720` linhas, sha256
  `69f76195e2a004de5c01c919038210da0987b67476911ca706e7ba9b4160477f`;
  validation `180` linhas, sha256
  `df2d44e334de65cb91da935768db93f4727f700edd762dd9fd6d48b3d5d8d14b`.
  Upload HF:
  `felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts`,
  path `v673-guarded-equation-bit-transfer-20260519T190246Z`,
  commit `f729f85dded2bc0a680b85059b60f2e267ae4c6e`.
- Gates limpos para o relaunch V673:
  - V286 tokenizer real:
    `tokenization_gate_passed`, `prompt_truncated=0`,
    `fallback_masks=0`, `completion_tokens_dropped=0`,
    token max `335`;
  - V509 dataset integrity:
    `datasets_pass_integrity_audit`, `blocked_dataset_count=0`;
  - V513 learnability:
    `passed_cpu_structure_only`, `blocker=0`, `warning=0`;
  - V478 objective:
    `hf_gpu_allowed=true`, bit effective share `0.148936`,
    equation effective share `0.851064` dentro dos limites V673;
  - EOS/loss-mask:
    final loss EOS rate `1.0`, sem linhas sem loss;
  - static active gate:
    `artifacts/v673_hf_a100_launch/v673_static_safety_gate_active_after_dryrun_tokenization_fix.json`,
    `ok=true`, `findings=[]`;
  - pre-paid integration:
    `artifacts/v673_hf_a100_launch/v673_pre_paid_job_integration_gate_after_dryrun_tokenization_fix.json`,
    `ok=true`, `findings=[]`;
  - V666 stack:
    `artifacts/v673_hf_a100_launch/v673_v666_cpu_gate_stack_after_rowloss_adapter_save_fix.json`,
    `gpu_allowed=true`, `blockers=[]`.
- O dry-run estruturado do script de treino agora grava contadores
  `tokenization.train/validation` com `prompt_truncated`,
  `fallback_masks`, `completion_tokens_dropped`, `offset_masks` e resumo de
  `row_loss_weight`; isso evita depender de leitura manual de log.
- Proxima acao permitida: commitar e pushar estas correcoes antes de qualquer
  relaunch, porque o launcher usa o `HEAD` remoto como `EXPECTED_COMMIT`.
  Depois, relancar somente `a100-large` V673 curto (`max_steps=20`,
  `save/eval=10`). O checkpoint-10 precisa passar weak eval e protected-row
  guard antes de qualquer continuacao ou submit.

Consenso OpenRouter V670: nao fazer novo treino cego. Todos os modelos uteis
convergiram em `needs one cheap diagnostic` antes de GPU. V664 fica congelado
como rota promocional porque reduziu loss, mas nao melhorou weak ACC, manteve
geracao longa e regrediu `8740ed31`. O pai seguro para hoje e V290
`checkpoint-6`, nao V664.

Evidencia local executada apos a consulta:

- V612 sobre V664 `checkpoint-2` contra V290 baseline: `blocked`,
  `192/315`, `bit=136/160`, `equation=56/155`, `avg_completion_tokens=4772`,
  `p99=7350`, `protected_backfire` em `8740ed31`. Em bit, `160/160` linhas
  ficaram `>256` tokens e `160/160` ficaram `>1000`; isto prova que V664 e uma
  rota de runaway/weight drift, nao apenas problema de parser.
- V541 sobre V290 baseline passou e mapeou os `123` misses atuais:
  `24` bit residual, `12` equation numeric e `87` equation symbolic
  punctuation. Implicacao pratica: para ganhar ainda hoje, mirar primeiro os
  `12` numeric equation e os `24` bit residuals; tentar resolver os `87`
  symbolic punctuation em um unico treino de hoje e risco alto.
- V350/V366 continuam sendo o maior sinal de ganho, mas nao submit-safe:
  V350 deu `bit=138`, `equation=61`; V366 integrado deu `208/315`,
  `bit=147`, `equation=61`, com `9` ganhos bit aceitos e `0` perdas aceitas.
  Isto autoriza gerar dataset/validacao nao-weak a partir das classes de regra,
  nao usar labels weak diretamente como alvo de treino.

Evidencia Kaggle V671 executada:

- Coleta completa via endpoint interno do frontend Kaggle:
  `200/200` topicos, `1224` itens entre topicos e comentarios, `0` falhas
  finais. O corpus foi salvo para auditoria e reprocessamento local.
- Metrica/parser: `\boxed{}` nao e detalhe. O gate precisa exigir exatamente um
  boxed final, sem brace extra, sem resposta vazia, sem texto pos-boxed e com
  testes para binarios com zeros a esquerda, respostas contendo `}` real e
  respostas com `}}` extra. A avaliacao official-like citada usa
  `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64` e temperatura `0`.
- Bit manipulation: nosso `136/160` corresponde ao patamar publico de 85% e
  nao prova breakout. O proximo ganho precisa vir de auditar os `24` misses de
  bit V541 contra regras tipo ROT/SHR/SHL, matching por bit/stride,
  MAJ/CHO/fullbyte e sinais V350/V366. Nao gastar GPU para "ensinar bit" sem
  provar qual regra falta por linha.
- Equation: separar `equation_numeric` de `equation_symbolic`. Para hoje, atacar
  os `12` misses numeric com 4 transformacoes, operadores frequentes, 32 ops e
  caso missing-op/absolute-difference. Symbolic/gold-conditioned fica somente
  como taxonomia; nao usar como alvo direto.
- Loss/learnability: loss medio baixo pode ser ilusao. Novo dataset so passa se
  tiver mascara completion-only, row-level answer-token NLL, min-logprob por
  trace, zero fallback mask, zero completion drop, zero truncation e ACC
  label-free. Hard-category oversampling acima de `3x` fica bloqueado sem prova
  de nao regressao.
- LoRA/submission: DoRA removido do plano curto. Conversoes/SVD/interpolacoes
  precisam medir drift token-a-token antes de qualquer submit. O contrato
  adapter-only deve validar base Nemotron, `adapter_config.json`,
  `adapter_model.safetensors`, rank/alpha, target modules e smoke de geracao.

Consenso OpenRouter V672 apos double check Kaggle:

- Modelos consultados: `openai/gpt-5.5`, `anthropic/claude-sonnet-4.6`,
  `google/gemini-3.1-pro-preview`, `deepseek/deepseek-v4-pro` e
  `qwen/qwen3.6-max-preview`.
- Todos concordaram que nao ha tema critico faltando nas discussions visiveis
  baixadas; a cobertura confirmada e `200/200` topicos, `1024` comentarios e
  `1224` itens totais.
- Maioria forte (`GPT-5.5`, `Claude`, `Qwen`) prioriza `equation_numeric`
  porque a meta de hoje e levar `equation` de `56` para `>=60` e ha apenas `12`
  misses numeric audit

[...TRUNCATED_BY_PROMPT_BUILDER_MIDDLE...]


- parametros:
  `MAX_STEPS=6`, `SAVE_EVERY_STEPS=2`, `EVAL_EVERY_STEPS=2`,
  `LR=5e-7 -> 1e-7`, `MAX_LENGTH=2048`, `example_mean`,
  `USE_ROW_LOSS_WEIGHT=1`;
- LoRA:
  `r=32`, `alpha=32`, modules
  `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`, target_parameters MoE
  `mlp.experts.gate_up_proj,mlp.experts.down_proj`;
- runtime confirmado nos logs:
  `trainable=869,318,656`, `all=32,466,091,456`, `trainable%=2.6776`,
  abaixo do teto `3.5%`;
- tokenizacao remota confirmou `0` truncation, `0` prompt tokens dropped,
  `0` skipped no-loss, `0` fallback masks;
- checkpoints publicados: `checkpoint-2`, `checkpoint-4`, `checkpoint-6`;
- loss remoto:
  - baseline eval: `5.8567`;
  - checkpoint-2 eval: `5.8392`;
  - checkpoint-4 eval: `5.8282`;
  - checkpoint-6 eval: `5.8231`;
- decisao: a queda de loss nao refletiu ACC no primeiro weak eval e nao
  autoriza promocao.

Weak eval V661 checkpoint-2:

- launcher:
  `artifacts/v284_official_gate_worktree/artifacts/v661_hf_h200_weak_eval_launch/launch_v661_hf_weak_eval_checkpoints.py`;
- job:
  `https://huggingface.co/jobs/felipesp1983/6a0bca4de7940de6ee6cf368`;
- output:
  `evals/v661-h200-answerfirst-shorttrace-weak-20260519T022510Z`;
- commit upload:
  `d3948d7542b4adf55402dbc4e3855bbc1b162ae0`;
- contrato: official-like, label-free, `max_tokens=7680`,
  prompt suffix oficial `\boxed{}`, protected rows `8740ed31`,
  `59bee375`, `55d834d1`;
- resultado:
  - total `191/315` (`0.606349`);
  - `bit_manipulation=135/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - `no_box_fallback_rows=1`;
  - `boxed_rate=0.996825`;
  - `starts_boxed_rows=0`, `starts_boxed_rate=0.0`;
  - `starts_final_answer_boxed_rows=0`;
  - `avg_completion_tokens=4775.43`, `max_completion_tokens=7680`,
    total completion tokens `1,504,259`;
  - `first_boxed_correct=191` e `label_aware_debug_correct=191`,
    logo `label-aware - label-free = 0`;
  - protected-row guard falhou:
    - `8740ed31`: baseline `01101000`, candidate `01111000`;
    - `59bee375`: baseline `10010101`, candidate `2`, truncado;
    - `55d834d1`: baseline `10111111`, candidate `10111111`,
      missing required gain contra expected `00111111`;
  - gate promocional bloqueou por `correct_lt_196`, `equation_lt_60`,
    `bit_lt_136`, `truncated_gt_0`, `no_box_fallback_gt_0`,
    `boxed_rate_lt_1.0` e `protected_row_backfire_guard_failed`.

Diagnostico V661:

- o problema nao foi extractor nem ganho label-aware: `first_boxed_correct`,
  label-free e label-aware ficaram iguais;
- o adapter nao aprendeu o contrato answer-first apesar do target local:
  `starts_boxed_rate=0.0`;
- a falha dominante e comportamento/decoding do adapter: ele gera raciocinio
  muito longo antes do boxed, consome tokens demais, causa truncation e mexe em
  protected rows;
- checkpoints posteriores (`4` e `6`) so podem ser avaliados depois de nova
  decisao de rota, porque o checkpoint-2 ja mostrou backfire real e violacao
  de output policy.

Correcoes cirurgicas feitas durante o check:

- V660 inicial tinha system prompt contraditorio herdado de V653
  (`verify it briefly`) enquanto o target era answer-only. V659 agora bloqueia
  esse padrao com `system_prompt_conflicts_answer_only`;
- V286 nao tinha modo para answer-first trace; foi adicionado
  `boxed_prefix`;
- V661 inicial ainda tinha template repetido em bit/equation; o trace agora
  inclui query em palavras para bit e clausula de query para equation, removendo
  o bloqueio `same_normalized_trace_template_multiple_answers`.

Consenso OpenRouter V662:

- prompt:
  `artifacts/v284_official_gate_worktree/artifacts/openrouter/v662_v661_failure_consult/KG1_V662_OPENROUTER_V661_FAILURE_PROMPT.md`;
- consenso:
  `artifacts/v284_official_gate_worktree/artifacts/openrouter/v662_v661_failure_consult/KG1_V662_OPENROUTER_CONSENSUS.md`;
- decisao: congelar V661; `checkpoint-4` e `checkpoint-6` nao devem receber
  full weak porque o `checkpoint-2` ja falsificou a rota por ACC, truncation,
  no-box fallback, `starts_boxed=0` e protected backfire;
- causa raiz consolidada:
  - target local answer-first nao transferiu para a inferencia official-like;
  - LoRA ampla com MLP/MoE causou drift antes de ganho em equation;
  - validation/loss nao estava alinhada ao objetivo de promocao;
  - extractor nao e a causa: label-free, first-boxed e label-aware ficaram
    todos em `191`;
- rota ativa: `V662 attention-only boxed-EOS no-trace`;
- contrato V662:
  - assistant target exatamente `\boxed{answer}` + EOS;
  - sem trace, sem explicacao, sem segundo boxed, sem texto apos a chave final;
  - payload byte-equal ao `answer`;
  - token-level first supervised token inicia `\boxed`;
  - EOS supervisionado imediatamente apos boxed em `100%` das linhas;
  - train objective `bit=0.40`, `equation=0.60`;
  - holdout nao-weak balanceado com `>=300` linhas e `>=120` equation;
- LoRA inicial:
  - `r=16`, `alpha=16`;
  - target modules `q_proj,k_proj,v_proj,o_proj`;
  - target_parameters vazio;
  - sem MLP/MoE;
  - LR `5e-7` constante ate `checkpoint-2`;
  - `max_steps=2` para primeiro gate.

Proxima decisao tecnica:

- implementar os gates V662 antes de qualquer novo job pago;
- nao usar loss-only como sinal de ganho;
- nao usar weak labels em treino, pseudo-labeling ou selecao;
- rodar full weak somente se o checkpoint-2 passar holdout nao-weak,
  protected smoke, starts-boxed e completion-token gates.

## Bloqueadores Permanentes

Cancelar ou reprovar se qualquer item ocorrer:

- CUDA/GPU fora do contrato;
- dataset hash diferente;
- weak overlap, weak label aware selection ou train/val overlap invalido;
- prompt duplicado contraditorio;
- prompt OpenRouter/local divergente quando uma consulta for usada para guiar
  plano;
- treino finalizado ou falho sem prompt/consulta pos-treino OpenRouter quando
  `OPENROUTER_API_KEY` estiver disponivel;
- template SFT/inferencia divergente;
- EOS correto ausente da loss mask;
- token supervisionado apos EOS com peso positivo;
- boxed payload nao byte-equal ao `answer`;
- `finding_counts.warning > 0` em gate promocional;
- V659 com `status != passed`;
- V659 `starts_boxed_required_failed > 0` para rota answer-first;
- V659 `first_box_word_idx_gt_limit > 0` para rota boxed-early;
- template parity SFT vs official-like com mismatch `>0`;
- EOS apos boxed ausente ou fora da loss mask;
- target V662 com texto apos a chave final do boxed;
- assistant target com mais de um boxed utilizavel;
- assistant target token p95 `>16` ou max `>24` na rota boxed-only;
- validation loss sem row weights alinhados quando usada como seletor;
- V478/launcher sem `--require-validation-row-loss-weight` quando
  `--require-row-loss-weight` estiver ativo;
- trainable parameter fraction `>1.0%` na fase V662;
- holdout nao-weak com `<300` linhas ou `<120` equation;
- holdout nao-weak com qualquer overlap weak por id, prompt hash ou
  prompt+answer hash;
- checkpoint-2 holdout `starts_boxed_rate <0.95`;
- checkpoint-2 holdout avg completion tokens `>64` ou p95 `>128`;
- checkpoint-2 holdout bit pior que init adapter por mais de `1` linha;
- checkpoint-2 holdout equation gain menor que `2` linhas;
- `blocked_dataset_count > 0`;
- truncation > `0`;
- completion tokens dropped > `0`;
- fallback masks > `0`;
- resposta sem `\boxed{}` em smoke de output policy;
- extractor usa fallback numerico para promover;
- protected rows com backfire;
- raw output correto mas extractor/config erra;
- raw output errado por adapter drift;
- row-loss melhora mas ACC nao melhora no holdout nao-weak;
- label-aware e label-free divergem em gate promocional;
- bit abaixo de `136/160`;
- equation abaixo de `60/155` em promocao;
- total weak `<196/315` em promocao;
- checkpoint ausente;
- HF LFS preflight falha;
- repo de saida contem artefato antigo ou incompleto;
- job ultrapassa tempo/custo sem checkpoint promocional.

## Itens Fora Do Plano

Nao executar como caminho principal:

- broad SFT antigo V390/V475/V510/V515/V536/V551/V560;
- V660 answer-only como rota GPU;
- usar solver/verifier/postprocessor em runtime como se fosse adapter-only;
- usar `train.csv` completo;
- treinar nos `315` weak rows;
- V630 raw boxed-only como treino promocional;
- promover por loss-only;
- strict no-think/short prompt;
- candidate pools externos como gold direto;
- regex boxed ingenua;
- `\boxed{{{answer}}}` por interpolacao crua;
- packing/multipack sem gate de EOD/position_ids/attention mask;
- datasets externos com upsample alto sem dedup;
- A100 com imagem NeMo CUDA13 sem validacao especifica;
- `AND_OR` de V366;
- continuar V646/V647 como base de treino;
- usar `max_tokens` menor como solucao isolada;
- weak flip rows como treino direto;
- aumentar epochs no mesmo dataset V643 sem output-policy gate;
- lancar V652;
- relancar V653 como rota promocional;
- avaliar V661 `checkpoint-4` ou `checkpoint-6` em full weak como proxima
  acao;
- treinar MLP/MoE target_parameters em nova rota de micro-resgate sem prova
  nao-weak de preservacao;
- lancar V662 boxed-only como rota GPU;
- relancar V663 thinking-trace ou avaliar V663 `checkpoint-4`/`checkpoint-6`
  em full weak sem um novo mecanismo e gates nao-weak que expliquem a reversao
  do backfire;
- relancar/continuar V664 como rota promocional sem passar os gates V666;
- H200 para smoke quando `a100-large` ainda for tecnicamente viavel;
- mudar `r=32/alpha=32` para `r=64/alpha=64` em continuacao do adapter V290;
- usar validation bit-heavy como seletor de checkpoint;
- usar reasoning-only/content vazio como evidencia final;
- aceitar resposta OpenRouter `in_progress` como contrato sem validacao local;
- adotar modulo LoRA contraditorio sugerido por uma IA sem ablation local;
- adapter por familia/composicao sem prova submit-safe;
- selecionar checkpoint por weak smoke;
- usar URLs/search do export OpenRouter como contrato sem validacao
  independente;
- usar simbolo Unicode vindo de resposta de IA diretamente em codigo, threshold
  ou configuracao sem normalizacao;
- temperatura `>0` para diagnostico promocional official-like;
- attention visualization antes dos gates de formato/loss/ACC;
- synthetic/equation externo como gold sem dedupe, weak exclusion,
  byte-equality, origem auditada e gate de contradicao.

## Proxima Acao Executavel

1. Manter congeladas V653/V660/V661/V662/V663/V664 como rotas ativas:
   - V653 falhou por output policy/backfire;
   - V660 answer-only foi bloqueado por learnability;
   - V661 checkpoint-2 caiu para `191/315`, `bit=135`, `equation=56`,
     `starts_boxed=0/315`, `truncated=1`, `no_box_fallback=1` e protected
     backfire;
   - V662 boxed-only foi superado pelo achado de paridade `enable_thinking`.
   - V663 checkpoint-2 caiu para `190/315`, `bit=135`, `equation=55`,
     `starts_boxed=0/315`, `truncated=1`, `no_box_fallback=1`, protected
     backfire e net `-2` vs baseline.
   - V664 checkpoint-2 ficou em `192/315`, `bit=136`, `equation=56`, com
     protected backfire e geracao longa.
2. Rodar somente gates locais V666 antes de qualquer novo GPU:
   - V478 com `--use-row-loss-weight --require-row-loss-weight
     --require-validation-row-loss-weight`;
   - V286 com `0` truncation, `0` dropped completion tokens, `0` fallback masks
     e EOS-stop mask;
   - `scripts/audit_loss_mask_eos_contract.py` com `final_loss_eos_rate=1.0`;
   - protected-row smoke para `8740ed31`, `59bee375`, `55d834d1`;
   - length gate: `avg_completion_tokens<=128`, `max_completion_tokens<=512`;
   - static/pre-paid gates com findings/warnings `0`;
   - pre-paid gate deve receber `--v666-cpu-gate-report-json` apontando para
     relatorio V666 `gpu_allowed`; relatorio `gpu_blocked` bloqueia qualquer
     gasto de HF.
   - nao gastar mais analise em parser/extractor para V664: o diagnostico V666
     provou `0` mismatch de corretude. A proxima hipotese precisa atacar
     diretamente geracao longa e backfire real do adapter.
3. Depois de cada treino concluido ou falho:
   - executar `scripts/kg1_post_train_openrouter_consult.py` com o manifest,
     resumo de falha, roadmap e resultados atuais;
   - salvar prompt, raw results, responses e manifest em `artifacts/openrouter`;
   - classificar cada sugestao como consenso, hipotese, hardening ou rejeitada
     antes de alterar o roadmap.
4. Se todos os gates V666 passarem, abrir apenas `a100-large` como primeira
   tentativa paga; H200 so entra se A100 for tecnicamente impossivel e o gate de
   custo/preflight registrar essa necessidade.
5. Promover somente se:
   - total `>=196/315`;
   - bit `>=136/160`;
   - equation `>=60/155`;
   - truncation `0`;
   - no-box fallback `0`;
   - boxed rate `1.0`;
   - protected backfire `0`;
   - `label-aware - label-free == 0`.

## Criterio De Submit

Submit ao Kaggle somente se:

- adapter-only gerar ganho weak real;
- pacote contem adapter/config correto;
- eval oficial-like confirma ganho sem fallback;
- gates de hash, prompt, max_tokens, LoRA contract, parser e protected rows
  passam;
- o resultado nao depende de solver runtime nem de weak labels.

```


## Training or launch manifest
Path: `artifacts\v673_hf_a100_launch\v673_pre_paid_job_integration_gate_after_dryrun_tokenization_fix.json`

```text
{
  "dataset_schema": "sft",
  "expected_eval_output_contract": "",
  "findings": [],
  "launcher": {
    "contains_expected_flavor": true,
    "contains_timeout_3600": true,
    "declared_dataset_schema": "sft",
    "decoding_vs_adapter_drift_gate": {
      "observed": {
        "EVAL_EVERY_STEPS": 10.0,
        "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
        "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "deferred_post_checkpoint",
        "KG1_EXPECTED_MAX_STEPS": 20.0,
        "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
        "MAX_STEPS": 20.0,
        "SAVE_EVERY_STEPS": 10.0
      },
      "required": {
        "checkpoint_every_steps_lte": 10,
        "first_checkpoint_weak_eval_required": true,
        "max_steps_lte": 20,
        "mode": "deferred_post_checkpoint",
        "purpose": "allow one tiny smoke when V568 can only be measured after a new checkpoint exists",
        "v618_surface_route": true
      }
    },
    "eval_prompt_requires_boxed_only_line": false,
    "expected_abort_max_reserved_gib": 78,
    "expected_data_repo": "felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts",
    "expected_flavor": "a100-large",
    "expected_loss_normalization_mode": "example_mean",
    "expected_max_length": 1024,
    "launcher": "artifacts\\v673_hf_a100_launch\\launch_v673_hf_a100_guarded_eqbit.py",
    "require_row_loss_weight": true,
    "residual_first_gpu_gate": {
      "observed": {
        "KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS": "passed",
        "KG1_CPU_EXTRACTOR_PARITY_STATUS": "passed",
        "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": 1.0,
        "KG1_CPU_SIMULATED_BIT_CORRECT": 136.0,
        "KG1_CPU_SIMULATED_EQUATION_CORRECT": 60.0,
        "KG1_CPU_SIMULATED_LOST_BIT_ROWS": 0.0,
        "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": 0.0,
        "KG1_CPU_SIMULATED_LOST_ROWS": 0.0,
        "KG1_CPU_SIMULATED_TOTAL_CORRECT": 196.0,
        "KG1_CPU_SIMULATION_USES_WEAK_LABELS": "0",
        "KG1_EXPECTED_TRUNCATED": "0",
        "KG1_MAX_TOKEN_HEADROOM_RATIO": 0.327,
        "KG1_PROMPT_TEMPLATE_PARITY_STATUS": "passed",
        "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101,55d834d1=00111111",
        "KG1_RESIDUAL_FIRST_GATE": "1",
        "KG1_STALE_PREDICTION_PARITY_STATUS": "passed",
        "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": "passed",
        "KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE": "0",
        "KG1_V540_EXTRACTION_GATE_STATUS": "passed",
        "KG1_V541_FLIP_LEDGER_STATUS": "passed",
        "KG1_V541_MISSMAP_GATE_STATUS": "passed",
        "KG1_WEAK_LABEL_AWARE_SELECTION": "0"
      },
      "required": {
        "cpu_extractor_parity_status": "passed",
        "cpu_miss_classification_coverage_min": 0.7,
        "cpu_simulated_bit_min": 136,
        "cpu_simulated_equation_min": 60,
        "cpu_simulated_lost_bit_rows_max": 0,
        "cpu_simulated_lost_equation_rows_max": 0,
        "cpu_simulated_lost_rows_max": 0,
        "cpu_simulated_total_min": 196,
        "cpu_simulation_uses_weak_labels": "0",
        "expected_truncated": 0,
        "max_token_headroom_ratio_max": 0.9,
        "prompt_template_parity_status": "passed",
        "protected_rows": [
          "8740ed31=01101000",
          "59bee375=10010101",
          "55d834d1=00111111"
        ],
        "stale_prediction_parity_status": "passed",
        "v516_parser_current_baseline_status": "passed",
        "v536_val_stats_as_weak_evidence": "0",
        "v540_extraction_gate_status": "passed",
        "v541_flip_ledger_status": "passed",
        "v541_missmap_gate_status": "passed",
        "weak_label_aware_selection": "0"
      }
    },
    "row_loss_weight_flag_counts": {
      "--require-row-loss-weight": 5,
      "--require-validation-row-loss-weight": 5,
      "--use-row-loss-weight": 5
    },
    "v666_cpu_gate_launcher_contract": {
      "declared_report": "artifacts/v673_hf_a100_launch/v673_v666_cpu_gate_stack_after_rowloss_adapter_save_fix.json",
      "observed_status": "passed",
      "required_status": "passed"
    }
  },
  "learnability_manifest": {
    "finding_counts": {
      "blocker": 0,
      "info": 2,
      "warning": 0
    },
    "hf_gpu_allowed": true,
    "manifest": "artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v513_learnability\\v513_trace_learnability_gate_manifest.json",
    "status": "passed_cpu_structure_only"
  },
  "ok": true,
  "preference_manifest": {
    "reason": "sft_schema_does_not_use_preference_audit",
    "skipped": true
  },
  "schema_version": "kg1_pre_paid_job_integration_gate_v2",
  "tokenization_manifest": {
    "manifest": "artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\tokenization_gate_real\\v286_generic_tokenization_gate_manifest.json",
    "manifest_max_length": 1024,
    "runtime_expected_max_length": 1024,
    "runtime_length_safe": true,
    "status": "tokenization_gate_passed",
    "train_token_max": 335,
    "validation_token_max": 335
  },
  "train_dataset": {
    "assistant_boxed_only_rows": 0,
    "assistant_final_answer_only_rows": 0,
    "assistant_length_stats": {
      "bit_manipulation": {
        "chars_max": 233,
        "chars_p50": 210,
        "chars_p95": 233,
        "rows": 240
      },
      "equation_transform": {
        "chars_max": 269,
        "chars_p50": 260,
        "chars_p95": 269,
        "rows": 480
      }
    },
    "assistant_multiline_rows": 720,
    "assistant_prefix_counts": {
      "other": 720
    },
    "assistant_rule_prefix_rows": 0,
    "assistant_trace_rows": 0,
    "bad_rows_first30": [],
    "expected_aware_signal_rows_first30": [],
    "family_counts": {
      "bit_manipulation": 240,
      "equation_transform": 480
    },
    "negative_type_counts": {},
    "path": "artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v673_guarded_equation_bit_transfer_train.jsonl",
    "rows": 720,
    "sha256": "69f76195e2a004de5c01c919038210da0987b67476911ca706e7ba9b4160477f",
    "subcategory_counts": {
      "bit_exact_global_binary_replay": 48,
      "bit_exact_global_ternary_replay": 96,
      "bit_fullbyte_ternary_v366_new": 96,
      "equation_numeric_add_direct": 120,
      "equation_numeric_colon_trailing_zero": 120,
      "equation_numeric_minus_signed": 240
    }
  },
  "v438_audit": {
    "skipped": true
  },
  "v666_cpu_gate": {
    "blockers": [],
    "check_count": 8,
    "decision": "gpu_allowed",
    "failed_checks": [],
    "gpu_allowed": true,
    "ok": true,
    "path": "artifacts\\v673_hf_a100_launch\\v673_v666_cpu_gate_stack_after_rowloss_adapter_save_fix.json",
    "schema_version": "kg1_v666_cpu_gate_stack_v1"
  },
  "validation_dataset": {
    "assistant_boxed_only_rows": 0,
    "assistant_final_answer_only_rows": 0,
    "assistant_length_stats": {
      "bit_manipulation": {
        "chars_max": 233,
        "chars_p50": 210,
        "chars_p95": 233,
        "rows": 60
      },
      "equation_transform": {
        "chars_max": 269,
        "chars_p50": 260,
        "chars_p95": 269,
        "rows": 120
      }
    },
    "assistant_multiline_rows": 180,
    "assistant_prefix_counts": {
      "other": 180
    },
    "assistant_rule_prefix_rows": 0,
    "assistant_trace_rows": 0,
    "bad_rows_first30": [],
    "expected_aware_signal_rows_first30": [],
    "family_counts": {
      "bit_manipulation": 60,
      "equation_transform": 120
    },
    "negative_type_counts": {},
    "path": "artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v673_guarded_equation_bit_transfer_val.jsonl",
    "rows": 180,
    "sha256": "df2d44e334de65cb91da935768db93f4727f700edd762dd9fd6d48b3d5d8d14b",
    "subcategory_counts": {
      "bit_exact_global_binary_replay": 12,
      "bit_exact_global_ternary_replay": 24,
      "bit_fullbyte_ternary_v366_new": 24,
      "equation_numeric_add_direct": 30,
      "equation_numeric_colon_trailing_zero": 30,
      "equation_numeric_minus_signed": 60
    }
  }
}
```


## Failure analysis summary
Path: `artifacts\v673_hf_a100_launch\v673_static_safety_gate_active_after_dryrun_tokenization_fix.json`

```text
{
  "file_count": 8,
  "files": [
    "artifacts/v673_hf_a100_launch/launch_v673_hf_a100_guarded_eqbit.py",
    "artifacts/v673_hf_a100_launch/upload_v673_dataset_to_hf.py",
    "scripts/audit_loss_mask_eos_contract.py",
    "scripts/build_v673_guarded_equation_bit_transfer_dataset.py",
    "scripts/hf_job_preflight_gate.py",
    "scripts/hf_job_train_v90.py",
    "scripts/kg1_pre_paid_job_integration_gate.py",
    "scripts/kg1_static_safety_gate.py"
  ],
  "findings": [],
  "ok": true,
  "schema_version": "kg1_static_safety_gate_v1"
}
```


## Previous OpenRouter consensus
Path: `artifacts\openrouter\v672_today_family_gain_from_kaggle_discussions\KG1_V672_OPENROUTER_TODAY_GAIN_CONSENSUS.md`

```text
# KG1 V672 OpenRouter Today-Gain Consensus

Gerado em 2026-05-19 apos auditoria completa das Kaggle discussions V671 e consulta OpenRouter com:

- `openai/gpt-5.5`
- `anthropic/claude-sonnet-4.6`
- `google/gemini-3.1-pro-preview`
- `deepseek/deepseek-v4-pro`
- `qwen/qwen3.6-max-preview`

## Double Check De Cobertura Kaggle

- Topicos listados: `200/200`.
- Detalhes baixados: `200/200`.
- Topicos presentes no corpus achatado: `200/200`.
- Itens analisaveis no corpus: `1224` = `200` topicos + `1024` comentarios.
- Falhas finais: `0`.
- Keyword coverage:
  - metric/eval/parser/boxed: `346` itens em `99` topicos;
  - runtime/submission: `453` itens em `131` topicos;
  - rules/allowed: `225` itens em `55` topicos;
  - RL/GRPO: `122` itens em `36` topicos;
  - training/LoRA/loss: `276` itens em `95` topicos;
  - bit/equation target: `257` itens em `88` topicos;
  - data/labels/synthetic: `267` itens em `90` topicos.

Conclusao honesta: a cobertura esta completa para os topicos/comentarios visiveis retornados pela API do Kaggle. Nao e possivel prometer que nao existam itens privados, deletados ou nao retornados pelo frontend, mas todo o material publico acessivel foi baixado, indexado, ranqueado e absorvido no plano.

## Consenso Dos Modelos

1. Todos os modelos responderam que nao ha tema critico faltando nas discussions visiveis.
2. O caminho de maior probabilidade para hoje nao e treino amplo: e auditoria CPU/local dos misses residuais antes de qualquer GPU.
3. A maioria forte (`GPT-5.5`, `Claude`, `Qwen`) prioriza `equation_numeric` porque precisamos sair de `equation=56` para `>=60`, ha apenas `12` misses numeric, e ja existe sinal CPU/postprocessor chegando a `equation=60`.
4. `Gemini` e `DeepSeek` priorizam `bit_residual` porque V366 mostrou `9` ganhos bit aceitos com `0` perdas, mas isso e secundario para a meta imediata porque `bit=136` ja cumpre o piso promocional.
5. Rota combinada recomendada: auditar hoje os `12` equation numeric e os `24` bit residuals; treinar somente se houver regras deterministicas, label-free, curtas e verificaveis.

## Plano V672 Para Hoje

P0 sem GPU:

1. Criar ledger CSV/JSONL de `36` linhas: `12` equation numeric misses + `24` bit residual misses.
2. Para cada linha registrar: id, prompt hash, familia/subfamilia, baseline V290, candidato CPU, regra candidata, se query operator aparece, ambiguidade, output boxed, `verify_answer`, token estimate, risco de leakage e decisao `trainable/drop/protected-only`.
3. Equation numeric: testar 4 transformacoes, operadores frequentes, 32 operadores, missing-op/absolute-difference, `rbs`, `max_mod_min` e qualquer operador ja presente no nosso DSL.
4. Bit: testar ROT/SHR/SHL, unary/binary, bitsum hash, stride left/right, MAJ3, CHO, fullbyte/bit-serial.
5. Adicionar gate de parser/boxed com casos: zeros a esquerda binarios, resposta contendo `}`, `}}` extra, boxed vazio, texto apos boxed.

P1 ainda sem treino:

- Se `equation_numeric deterministic_unique >= 4`, gerar dataset sintetico nao-weak para esses buckets e seguir para treino curto.
- Se `bit deterministic_unique >= 4` e houver correspondencia com V350/V366 aceitos, adicionar dataset bit curto; caso contrario bit entra so como protected replay.
- Se nenhum dos dois passar, nao gastar A100/H200; atualizar roadmap e consultar OpenRouter com o ledger real.

P2 treino curto somente se P1 passar:

- Parent: V290 `checkpoint-6`, nunca V664.
- Hardware: `a100-large`; H200 bloqueado salvo insuficiencia comprovada.
- Dataset: pequeno, nao-weak, regras verificadas, com protected replay/eval; sem weak val como target.
- Target: curto, exatamente um `\boxed{ANSWER}` final, sem texto pos-boxed. Rationale curto permitido apenas se reduzir erro sem atrasar boxed.
- LoRA: `r=32`, `alpha=32`, `q_proj/v_proj` primeiro; sem DoRA, sem `lm_head`, sem embeddings, sem MoE.
- LR: `5e-5` preferido; `1e-4` apenas se gate de answer-token NLL justificar; `2e-4` bloqueado hoje.
- Steps: `20-40` ou `20-60` max, eval cedo, abortar com qualquer protected backfire, `bit<136`, `equation<60`, `truncated>0`, `boxed_rate<1.0`, `finding_counts.warning>0`, ou loss caindo sem ACC/answer-token NLL melhorar.

## Decisao Cirurgica

O proximo trabalho efetivo nao e chamar GPU. E implementar `scripts/audit_misses.py` ou equivalente para gerar o ledger V672 dos `36` misses residuais. O primeiro candidato de ganho real deve ser `equation_numeric`, com bit residual como segunda frente se o ledger mostrar regras deterministicas. Isso troca tentativa-e-erro por prova por linha antes de gasto.

## Coisas Bloqueadas

- Continuar V664.
- Treino amplo por loss.
- Resolver `equation_symbolic` inteiro hoje.
- Usar gold-conditioned symbolic solver como alvo direto.
- Usar weak labels como seletor/target.
- H200 exploratorio.
- DoRA.
- Tocar `lm_head`, embeddings ou MoE hoje.
- Oversampling acima de `3x` sem prova.
- Submit antes de `total>=196`, `bit>=136`, `equation>=60`, `truncated=0`, `protected_backfire=0`.

```


## Extra artifact 1
Path: `artifacts\openrouter\v674_f2_prelaunch_consult\KG1_V674_CURRENT_STATE.md`

```text
# KG1 V674 Current State For OpenRouter F2 Prelaunch Review

Generated for the 2026-05-19 prelaunch decision. This file is an evidence
block for the OpenRouter prompt; it is not a promotion gate by itself.

## Competition And Objective

- Competition: Kaggle NVIDIA Nemotron Model Reasoning Challenge.
- Submission goal: adapter-only, submit-safe accuracy improvement.
- Metric that matters: final-answer accuracy through label-free extraction and
  `verify_answer`, not eval_loss alone.
- Current weak plateau: `192/315`.
- Active family floors:
  - `bit_manipulation >= 136/160`.
  - `equation_transform >= 60/155`.
  - `total >= 196/315`.
  - `truncated = 0`.
  - `no_box_fallback = 0`.
  - `boxed_rate = 1.0`.
  - protected-row backfire = `0`.
- Current actionable CPU target for V673: simulated `196/315`, with
  `bit=136/160`, `equation=60/155`, lost rows `0`, and no weak-label training.

## Active Operational Rules

- H200 is blocked. Use `a100-large` only.
- No Kaggle submit unless explicitly requested after gates pass.
- Keep `official_like` intact.
- False gains are forbidden. A candidate must pass label-free extraction,
  `verify_answer`, protected-row guard, truncation/box gates, hash checks, and
  LoRA adapter-only checks.
- If a paid job fails or finishes, build a complete OpenRouter prompt with the
  current result before authorizing another paid route.
- Launch code must be committed and pushed before HF job launch because the
  launcher records the current `HEAD` as `EXPECTED_COMMIT` and the remote job
  checks out that commit.

## Recent F2 Bugs Found And Fixed

1. Adapter-only save risk:
   - Canceled A100 job `6a0cada33aba298b21d14304` because PEFT warned that
     `save_embedding_layers=True` would be used automatically due to `lm_head`
     appearing in `target_modules`.
   - This was a real silent packaging bug risk: the adapter could include base
     embeddings/lm_head and stop being clean adapter-only.
   - Fix: `scripts/hf_job_train_v90.py` now saves all checkpoints and final
     output through `save_adapter_only(model, output_dir)`, which calls
     `model.save_pretrained(..., save_embedding_layers=SAVE_EMBEDDING_LAYERS)`.
   - Active default and launcher value: `SAVE_EMBEDDING_LAYERS=0`.

2. Row-loss weighting was silently canceled:
   - With `MICRO_BATCH_SIZE=1`, `LOSS_NORMALIZATION_MODE=example_mean`, and the
     old weighted mean `(per_example_loss * weight).sum() / weight.sum()`, the
     row weight canceled out in every microbatch.
   - This means equation/bit row weights could look configured but have no
     effect on gradient scale.
   - Fix: `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`. Under `example_mean`, the
     denominator is the unweighted active example count, so a single example
     with `loss_weight=2.0` doubles the loss.
   - Self-test in `hf_job_train_v90.py` proves this behavior.

3. Validation loss alignment:
   - Train and validation now both use row-loss weights when row-loss weighting
     is active. This prevents best-loss decisions from measuring a different
     distribution than the optimizer.
   - Loss is still not a promotion metric by itself; it is only useful if ACC
     and protected-row gates agree.

4. Stale threshold:
   - `RESIDUAL_FIRST_MIN_EQUATION` and all active V673 promotion gates are now
     `60`, not `59`.
   - Required floors: total `196`, bit `136`, equation `60`, truncation `0`.

5. Dataset metadata contamination:
   - V673 dataset assistant targets are trace-plus-final-boxed, not boxed-only.
   - Builder fixed `metadata.completion_format` to
     `trace_plus_final_boxed`.
   - New active dataset has `bad_boxed_only_trace=0`.

6. A100 memory guard:
   - Previous A100 reserved memory went above the old 72 GiB abort line without
     proving true OOM.
   - Active launcher uses `ABORT_MAX_RESERVED_GIB=78`.

7. Dry-run report observability:
   - Tokenize-only dry-run report now includes structured
     `tokenization.train` and `tokenization.validation` counters:
     `prompt_truncated`, `fallback_masks`, `completion_tokens_dropped`,
     `offset_masks`, `row_loss_weight_*`.

## Active Dataset V673

- Local root:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T190246Z`.
- Train JSONL:
  `v673_guarded_equation_bit_transfer_train.jsonl`.
- Validation JSONL:
  `v673_guarded_equation_bit_transfer_val.jsonl`.
- Train rows: `720`.
- Validation rows: `180`.
- Train family counts:
  - `equation_transform`: `480`.
  - `bit_manipulation`: `240`.
- Validation family counts:
  - `equation_transform`: `120`.
  - `bit_manipulation`: `60`.
- Train subcategories:
  - `equation_numeric_minus_signed`: `240`.
  - `equation_numeric_add_direct`: `120`.
  - `equation_numeric_colon_trailing_zero`: `120`.
  - `bit_exact_global_ternary_replay`: `96`.
  - `bit_fullbyte_ternary_v366_new`: `96`.
  - `bit_exact_global_binary_replay`: `48`.
- Validation subcategories:
  - `equation_numeric_minus_signed`: `60`.
  - `equation_numeric_add_direct`: `30`.
  - `equation_numeric_colon_trailing_zero`: `30`.
  - `bit_exact_global_ternary_replay`: `24`.
  - `bit_fullbyte_ternary_v366_new`: `24`.
  - `bit_exact_global_binary_replay`: `12`.
- Train SHA256:
  `69f76195e2a004de5c01c919038210da0987b67476911ca706e7ba9b4160477f`.
- Validation SHA256:
  `df2d44e334de65cb91da935768db93f4727f700edd762dd9fd6d48b3d5d8d14b`.
- HF dataset repo:
  `felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts`.
- HF dataset root:
  `v673-guarded-equation-bit-transfer-20260519T190246Z`.
- HF upload commit:
  `f729f85dded2bc0a680b85059b60f2e267ae4c6e`.

## Active Training Recipe V673

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Base revision: `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Parent adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke`,
  subfolder `checkpoint-6`.
- Output repo:
  `felipesp1983/kg1-nemotron-lora-v673-a100-guarded-eqbit-v290ckpt6`.
- HF flavor: `a100-large`.
- Max job time: one hour.
- Train max steps: `20`.
- Save/eval every: `10`.
- `MAX_LENGTH=1024` for training.
- Eval official-like contract remains `max_tokens=7680`,
  `temperature=0`, `top_p=1`.
- Loss normalization: `example_mean`.
- `USE_ROW_LOSS_WEIGHT=1`.
- `REQUIRE_ROW_LOSS_WEIGHT=1`.
- `REQUIRE_VALIDATION_ROW_LOSS_WEIGHT=1`.
- `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`.
- `SAVE_EMBEDDING_LAYERS=0`.
- `LOSS_MASK_STOP_AFTER_EOS=1`.
- `SAMPLING_MODE=weighted_replacement`.
- LoRA:
  - `r=32`.
  - `alpha=32`.
  - dropout `0`.
  - target modules include attention and MLP adapter surfaces as declared by
    the V673 launcher.
  - MoE target parameters:
    `mlp.experts.gate_up_proj,mlp.experts.down_proj`.
  - `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1`.
  - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`.
- Protected rows:
  - `8740ed31=01101000`.
  - `59bee375=10010101`.
  - `55d834d1=00111111`.

## Gates That Passed After Fixes

- V286 tokenizer-real gate:
  - `status=tokenization_gate_passed`.
  - `prompt_truncated=0`.
  - `fallback_masks=0`.
  - `completion_tokens_dropped=0`.
  - max token length `335`.
- V509 dataset integrity:
  - `status=datasets_pass_integrity_audit`.
  - `dataset_count=2`.
  - `blocked_dataset_count=0`.
- V513 learnability:
  - `status=passed_cpu_structure_only`.
  - `blocker=0`.
  - `warning=0`.
- V478 objective:
  - `hf_gpu_allowed=true`.
  - bit effective share `0.148936`.
  - equation effective share `0.851064`.
- EOS/loss mask:
  - final-loss EOS rate `1.0`.
  - no no-loss rows.
- Static safety gate, active files only:
  - `ok=true`.
  - `findings=[]`.
- Pre-paid job integration:
  - `ok=true`.
  - `findings=[]`.
  - confirms `a100-large`, dataset hashes, `ABORT_MAX_RESERVED_GIB=78`,
    `SAVE_EMBEDDING_LAYERS=0`, and
    `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`.
- V666 CPU gate stack:
  - `gpu_allowed=true`.
  - `blockers=[]`.

## Known Failure History To Keep In Mind

- V664 reached only `192/315`, `bit=136/160`, `equation=56/155`, with long
  completions and protected backfire. Loss movement alone did not imply ACC.
- V661 checkpoint-2 regressed to `191/315`, `bit=135/160`,
  `equation=56/155`, `truncated=1`, `no_box_fallback=1`, and protected row
  backfire.
- V653/V660/V661/V662/V663/V664 are frozen as promotional routes.
- Broad H200 exploratory training is removed from the plan.
- Runtime solver/verifier/postprocessor is not acceptable as an adapter-only
  submission route.

## Questions For The External Model

Return only concrete, falsifiable engineering guidance:

1. Is there any remaining mechanism by which the V673 route could produce a
   false gain or hidden backfire despite the gates above?
2. Does `ROW_LOSS_WEIGHT_REDUCTION=scale_mean` correctly fix the microbatch-1
   cancellation issue, or should the weighted objective be implemented
   differently for train and validation?
3. Does `SAVE_EMBEDDING_LAYERS=0` fully close the adapter-only packaging risk
   given that `lm_head` appears in `target_modules`, or should `lm_head` be
   removed from target modules as an additional safety step?
4. Are the active thresholds (`total>=196`, `bit>=136`, `equation>=60`,
   truncation/fallback/protected=0) sufficient, or is any gate missing to
   distinguish bad decoding from adapter drift toward wrong answers?
5. Given the budget constraint and H200 ban, should we launch the bounded
   A100 V673 smoke now, or block and run another CPU-only diagnostic first?

Required response format:

```json
{
  "verdict": "proceed|block|needs_one_more_cpu_gate",
  "top_risks": [
    {
      "risk": "...",
      "evidence": "...",
      "required_fix_or_gate": "...",
      "blocks_a100_launch": true
    }
  ],
  "parameters_to_freeze": {
    "SAVE_EMBEDDING_LAYERS": "0",
    "ROW_LOSS_WEIGHT_REDUCTION": "scale_mean"
  },
  "parameters_to_change_before_launch": {},
  "cheapest_next_action": "...",
  "delete_from_roadmap": ["..."]
}
```

```


## Extra artifact 2
Path: `artifacts\v673_hf_a100_launch\v673_v666_cpu_gate_stack_after_rowloss_adapter_save_fix.json`

```text
{
  "blockers": [],
  "checks": [
    {
      "details": {
        "direct_usable_counts": {
          "bit_residual_miss": 10,
          "equation_numeric_miss": 4
        },
        "gpu_gate": "allow_a100_large_equation_transfer_probe_guarded",
        "gpu_recommendation": "Use a100-large only for a cheap transfer probe, not H200, because the ledger now has direct no-loss candidates.",
        "trainability_decision_counts": {
          "drop": 7,
          "needs_rule_proof": 15,
          "trainable": 12,
          "trainable_guarded": 2
        }
      },
      "name": "v672_residual_miss_ledger",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v672_residual_miss_ledger\\20260519T173138Z\\v672_residual_miss_ledger_manifest.json"
    },
    {
      "details": {
        "status": "tokenization_gate_passed",
        "train": {
          "completion_tokens_dropped": 0,
          "fallback_masks": 0,
          "family_summary": {
            "bit_manipulation": {
              "loss_token_max": 76,
              "loss_token_min": 62,
              "loss_token_p50": 72,
              "rows": 240,
              "token_max": 335,
              "token_p50": 331,
              "token_p90": 333,
              "token_p99": 334
            },
            "equation_transform": {
              "loss_token_max": 115,
              "loss_token_min": 91,
              "loss_token_p50": 110,
              "rows": 480,
              "token_max": 240,
              "token_p50": 232,
              "token_p90": 240,
              "token_p99": 240
            }
          },
          "loss_token_max": 115,
          "loss_token_min": 62,
          "loss_token_p50": 102,
          "offset_masks": 720,
          "prompt_truncated": 0,
          "prompt_truncation_rate": 0.0,
          "rows": 720,
          "token_max": 335,
          "token_mean": 262.189,
          "token_min": 214,
          "token_p50": 238,
          "token_p90": 332,
          "token_p99": 334
        },
        "validation": {
          "completion_tokens_dropped": 0,
          "fallback_masks": 0,
          "family_summary": {
            "bit_manipulation": {
              "loss_token_max": 76,
              "loss_token_min": 62,
              "loss_token_p50": 72,
              "rows": 60,
              "token_max": 335,
              "token_p50": 331,
              "token_p90": 333,
              "token_p99": 334
            },
            "equation_transform": {
              "loss_token_max": 115,
              "loss_token_min": 91,
              "loss_token_p50": 111,
              "rows": 120,
              "token_max": 240,
              "token_p50": 234,
              "token_p90": 240,
              "token_p99": 240
            }
          },
          "loss_token_max": 115,
          "loss_token_min": 62,
          "loss_token_p50": 103,
          "offset_masks": 180,
          "prompt_truncated": 0,
          "prompt_truncation_rate": 0.0,
          "rows": 180,
          "token_max": 335,
          "token_mean": 261.972,
          "token_min": 215,
          "token_p50": 238,
          "token_p90": 332,
          "token_p99": 334
        }
      },
      "name": "v286_tokenization_no_truncation_overlap",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\tokenization_gate_real\\v286_generic_tokenization_gate_manifest.json"
    },
    {
      "details": {
        "blocked_dataset_count": 0,
        "dataset_count": 2,
        "status": "datasets_pass_integrity_audit"
      },
      "name": "v509_dataset_integrity",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v509_integrity\\v673_guarded_equation_bit_transfer_manifest.json"
    },
    {
      "details": {
        "finding_counts": {
          "blocker": 0,
          "info": 2,
          "warning": 0
        },
        "status": "passed_cpu_structure_only"
      },
      "name": "v513_learnability_zero_warning",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v513_learnability\\v513_trace_learnability_gate_manifest.json"
    },
    {
      "details": {
        "findings": [],
        "hf_gpu_allowed": true,
        "thresholds": {
          "max_any_family_effective_share": 0.9,
          "max_equation_effective_share": 0.9,
          "min_bit_effective_share": 0.1,
          "require_row_loss_weight": true,
          "require_validation_row_loss_weight": true,
          "use_row_loss_weight": true
        },
        "train_effective_share_by_family": {
          "bit_manipulation": {
            "share": 0.148936,
            "weight": 84.0
          },
          "equation_transform": {
            "share": 0.851064,
            "weight": 480.0
          }
        },
        "validation_effective_share_by_family": {
          "bit_manipulation": {
            "share": 0.148936,
            "weight": 21.0
          },
          "equation_transform": {
            "share": 0.851064,
            "weight": 120.0
          }
        }
      },
      "name": "v478_objective_alignment_row_weighted",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v478_objective_alignment\\v673_v478_objective_alignment.json"
    },
    {
      "details": {
        "blockers": [],
        "ok": true,
        "train_final_loss_eos_rate": 1.0,
        "train_no_loss_rows": 0,
        "validation_final_loss_eos_rate": 1.0,
        "validation_no_loss_rows": 0
      },
      "name": "loss_mask_eos_contract",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_hf_a100_launch\\v673_loss_mask_eos_contract_after_dataset_metadata_fix.json"
    },
    {
      "details": {
        "mode": "uploaded",
        "train_sha256": "69f76195e2a004de5c01c919038210da0987b67476911ca706e7ba9b4160477f",
        "upload_info": "https://huggingface.co/datasets/felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts/commit/f729f85dded2bc0a680b85059b60f2e267ae4c6e",
        "val_sha256": "df2d44e334de65cb91da935768db93f4727f700edd762dd9fd6d48b3d5d8d14b"
      },
      "name": "hf_dataset_upload",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_hf_a100_launch\\v673_hf_dataset_upload_manifest.json"
    },
    {
      "details": {
        "decision": "surface_gate_passed",
        "environment": null
      },
      "name": "v619_module_surface",
      "ok": true,
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v619_nemotron_module_surface_gate\\v619_module_surface_report.json"
    }
  ],
  "decision": "gpu_allowed",
  "finding_counts": {
    "blocker": 0,
    "failed_checks": 0,
    "passed_checks": 8
  },
  "generated_at_utc": "2026-05-19T19:08:17.432051+00:00",
  "gpu_allowed": true,
  "next_action": "launch bounded V673 A100 train only; first checkpoint weak eval required before any longer run or submit",
  "ok": true,
  "repo_root": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree",
  "route": "v673_guarded_equation_bit_transfer_a100_after_rowloss_adapter_save_fix",
  "schema_version": "kg1_v666_cpu_gate_stack_v1"
}
```


## Final instruction
Give a surgical answer that can change the next roadmap step. Do not repeat generic ML advice.
Focus on concrete implementation, data, masking, decoding, LoRA contract, validation, and gate changes that can improve ACC safely.
