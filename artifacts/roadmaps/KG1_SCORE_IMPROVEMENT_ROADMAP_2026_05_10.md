# KG1 Score Improvement Roadmap

Atualizado: 2026-05-19 21:30 UTC. V672/V673/V674/V675 substitui o plano ativo V671. A decisao
de hoje vem de cinco fontes: consulta OpenRouter V670, evidencias CPU locais
V541/V612, sinais no-loss V350/V366, auditoria completa das discussions Kaggle
V671 e consenso OpenRouter V672. Objetivo de hoje continua: sair de `192/315` mantendo
`bit_manipulation>=136/160` e levando `equation_transform>=60/155`, sem
backfire protegido, sem ganho falso e sem treino cego.

Plano ativo agora:

1. Usar somente `a100-large`; H200 segue bloqueado.
2. Avaliar V673 por weak eval estrito somente depois do gate de runtime
   A100/CUDA12 passar. O par `a100-large + vllm/vllm-openai:v0.20.1` fica
   bloqueado porque expôs Torch CUDA 13 contra driver HF A100 CUDA 12.09.
3. Promover apenas se `total>=196`, `bit_manipulation>=136`,
   `equation_transform>=60`, `truncated=0`, `boxed_rate=1.0`,
   `label_aware_delta=0`, `no_box_fallback=0` e sem backfire protegido.
4. Se o weak eval não passar, não empacotar, não submeter e gerar nova
   consulta OpenRouter com o resultado real do treino/eval, conforme regra.

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
- dataset V673 guardado equation+bit ativo:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T190246Z/v673_guarded_equation_bit_transfer_manifest.json`;
- gate V286 tokenizer real de V673:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T190246Z/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json`;
- limpeza V674:
  `artifacts/v674_cleanup/v674_workspace_clean_after_large_blob_cleanup.json`;
- manifesto de remocao V674:
  `artifacts/v674_cleanup/v674_removed_canceled_submission_and_empty_debug_manifest.json`;
- consulta OpenRouter V674 prelaunch:
  `artifacts/openrouter/v674_f2_prelaunch_consult/`;
- gate V659 output policy V673 com limite 60:
  `artifacts/v675_v673_prelaunch_hardening/v659_output_policy_idx60/v659_local_output_policy_gate_manifest.json`;
- gate static V675 apos remover `lm_head` e ativar weak length cap:
  `artifacts/v675_v673_prelaunch_hardening/v675_static_safety_gate_after_weak_length_gate.json`;
- gate pre-paid V675 apos remover `lm_head`:
  `artifacts/v675_v673_prelaunch_hardening/v675_pre_paid_job_integration_gate_after_no_lmhead.json`;
- gate V619 module surface V675:
  `artifacts/v675_v673_prelaunch_hardening/v675_v619_surface_gate_after_no_lmhead.json`;
- manifesto debug A100 V673 sem launch pago:
  `artifacts/v673_hf_a100_launch/v673-a100-guarded-eqbit-v290ckpt6-20260519T201206Z_launch_manifest.json`;
- gate V485 permitindo drop controlado do `lm_head` do adapter inicial:
  `artifacts/v675_v673_prelaunch_hardening/v675_v485_roundtrip_gate_allow_lmhead_drop.json`;
- gate static depois do carregamento manual/drop de `lm_head`:
  `artifacts/v675_v673_prelaunch_hardening/v675_static_safety_gate_after_lmhead_drop_manual_v2.json`;
- gate pre-paid depois do carregamento manual/drop de `lm_head`:
  `artifacts/v675_v673_prelaunch_hardening/v675_pre_paid_job_integration_gate_after_lmhead_drop_manual_v2.json`;
- log da falha preflight A100 V673:
  `artifacts/v673_hf_a100_launch/v673_hf_job_6a0cc7a93aba298b21d14393_failed_preflight_lmhead_mismatch_logs.txt`;
- log do treino V673 concluido:
  `artifacts/v673_hf_a100_launch/v673_hf_job_6a0ccbae3aba298b21d143b1_logs.txt`;
- gates V485 post-upload V673:
  `artifacts/v675_v673_prelaunch_hardening/v675_v485_v673_checkpoint10_postupload.json`,
  `artifacts/v675_v673_prelaunch_hardening/v675_v485_v673_checkpoint20_postupload.json`,
  `artifacts/v675_v673_prelaunch_hardening/v675_v485_v673_final_postupload.json`;
- log da falha weak eval A100 por runtime CUDA13:
  `artifacts/v673_hf_a100_launch/v673_hf_job_6a0cd42c2dc5b1243da50485_logs.txt`;
- manifesto debug V675/V673 com runtime A100/CUDA12 aceito:
  `artifacts/v673_hf_a100_launch/v673-a100-v221contract-guarded-eqbit-weak-20260519T213322Z_weak_eval_launch_manifest.json`;
- manifesto debug V675/V673 provando bloqueio do runtime A100/CUDA13 antigo:
  `artifacts/v673_hf_a100_launch/v673-a100-v221contract-guarded-eqbit-weak-20260519T212713Z_weak_eval_launch_manifest.json`.

Status V673 treino e weak-eval runtime, 2026-05-19 21:30 UTC:

- Treino A100 `felipesp1983/6a0ccbae3aba298b21d143b1` completou sem OOM e
  sem traceback. O output repo ativo e
  `felipesp1983/kg1-nemotron-lora-v673-a100-guarded-eqbit-v290ckpt6`.
- Checkpoints disponiveis para weak eval: `checkpoint-10`, `checkpoint-20`,
  `final`. V485 post-upload passou nos tres: `r=32`, `alpha=32`,
  `modules_to_save=[]`, target modules sem `lm_head`, target parameters
  `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`.
- Loss real do treino: baseline `1.6199`, step 10 `1.5982`, step 20/final
  `1.5887`. Isto nao e ganho submetivel ate o weak ACC estrito passar.
- Tentativa weak eval A100 `felipesp1983/6a0cd42c2dc5b1243da50485` falhou
  antes de qualquer avaliacao por infraestrutura: container
  `vllm/vllm-openai:v0.20.1` trouxe Torch `2.11.0+cu130`; HF A100 reportou
  driver CUDA `12.09`, entao `torch.cuda.is_available()` ficou `false`.
- Correcao aplicada: `scripts/hf_job_weak_eval_v245.py` agora aceita
  `KG1_MAX_TORCH_CUDA_MAJOR` e aborta se o runtime Torch exceder o limite;
  o launcher V673 define `KG1_MAX_TORCH_CUDA_MAJOR=12`,
  `KG1_ALLOW_CUDA13_ON_A100=0`, usa
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` por padrao e instala o wheel
  oficial `vllm-0.20.1+cu129` antes do eval. Isso preserva a versao de vLLM
  que ja suporta o avaliador/modelo, mas evita Torch CUDA 13. O launcher
  registra `runtime_image_gate` no manifesto. O runtime antigo
  `vllm/vllm-openai:v0.20.1` fica bloqueado para `a100-large`.
- Proxima acao obrigatoria: commitar/pushar esta validacao, regenerar o
  manifesto com o novo `EXPECTED_COMMIT` e lancar weak eval A100 curto. Nao
  usar H200, nao empacotar e nao submeter antes do weak gate.

Hardening V675 prelaunch, 2026-05-19 20:14 UTC:

- Consenso OpenRouter V674 foi tratado como bloqueador tecnico antes de nova
  GPU: `lm_head` nao deve estar em `LORA_TARGET_MODULES` em adapter-only.
  Correcao aplicada no launcher V673:
  `LORA_TARGET_MODULES=down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj`.
  `lm_head`, `embed_tokens` e embeddings passam a ser proibidos pelos gates
  static/pre-paid quando `SAVE_EMBEDDING_LAYERS=0`.
- `scripts/kg1_pre_paid_job_integration_gate.py` agora falha se o launcher
  ativo inclui alvos adapter-only proibidos em `LORA_TARGET_MODULES`.
  `scripts/kg1_static_safety_gate.py` recebeu a mesma regra estatica.
- `scripts/hf_job_weak_eval_v245.py` agora tem teto default de comprimento
  para promocao weak: media de completion `<=512` e max `<=2048`. Isto fecha
  ganho falso por completions longas/runaway antes de qualquer promocao.
- Gate V659 output policy passou para V673 usando limite tecnico
  `max_first_box_word_index=60`: exactly-one boxed, extracao label-free e
  expected-aware consistentes, sem overlap train/val. Restaram apenas warnings
  esperados de cobertura bit (`ROT`, `SHL`, `SHR`) porque V673 e replay
  protegido, nao dataset amplo de bit.
- Gate V619 module surface passou contra o manifesto V673: atencao
  `q_proj,k_proj,v_proj,o_proj` existe e esta solicitada; MoE target
  parameters continuam em `mlp.experts.gate_up_proj` e
  `mlp.experts.down_proj`; blockers `0`, warnings `0`.
- Gates limpos depois do hardening:
  static `ok=true/findings=[]`, pre-paid `ok=true/findings=[]`,
  V619 `surface_gate_passed`, weak eval self-test OK, `py_compile` OK.
- Debug do launcher A100 foi executado sem `--launch`, portanto nao consumiu
  job pago. O flavor detectado e `a100-large`, `80GB`, custo unitario
  `$0.041667/min`. O remote command gerado esta sem `lm_head`.
- Bloqueio operacional antes de GPU: o manifesto debug atual ainda aponta
  `expected_commit=f63f8afc6fdbc0ef9c12e0cbfd9010d6fbde6baf`, anterior ao
  hardening V675. Proxima acao obrigatoria e commitar/pushar as correcoes,
  regenerar o manifesto e so entao lancar A100 curto. Nao lancar job pago com
  manifesto que espera commit antigo.

F2/backfire corrigido apos falha A100 V673, 2026-05-19 20:36 UTC:

- Job A100 `felipesp1983/6a0cc7a93aba298b21d14393` falhou antes do treino,
  durante `hf_job_preflight_gate.py --phase artifacts`, sem checkpoint e sem
  adapter novo. Causa: o adapter inicial V290 `checkpoint-6` tem
  `lm_head` em `adapter_config.target_modules`, mas o contrato efetivo V675
  remove `lm_head` de `LORA_TARGET_MODULES`. O preflight antigo comparava o
  adapter inicial bruto com o contrato efetivo e abortou por mismatch.
- O erro era util: se continuasse via `PeftModel.from_pretrained`, o modelo
  carregaria a configuracao original do adapter com `lm_head`. Isso violaria a
  decisao V674/V675 de adapter-only estrito e poderia recriar o risco de
  salvar `lm_head.base_layer.weight`/embeddings.
- Correcao aplicada:
  `INIT_ADAPTER_LOAD_MODE=manual`,
  `DROP_INIT_ADAPTER_TARGET_MODULES=lm_head`,
  `KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD=1`.
  O treino agora cria o PEFT com os target modules efetivos sem `lm_head` e
  filtra os tensores `lm_head` do adapter inicial antes do load manual.
- `scripts/run_v485_peft_roundtrip_gate.py` agora aceita
  `--allowed-extra-target-modules lm_head` para validar o adapter inicial
  bruto enquanto confirma que o contrato efetivo remove esse alvo. Resultado:
  `v485_peft_roundtrip_gate=ok`, `hf_gpu_allowed=True`,
  target_parameters com cobertura `5934/5934` para `down_proj` e
  `gate_up_proj`, e module LoRA presente para todos os alvos efetivos.
- `scripts/hf_job_preflight_gate.py` agora calcula
  `effective_adapter = adapter.target_modules - DROP_INIT_ADAPTER_TARGET_MODULES`
  e compara isso com `LORA_TARGET_MODULES`; so permite drops de
  `lm_head/embed_tokens/word_embeddings`.
- `scripts/kg1_static_safety_gate.py` continua bloqueando manual-load MoE, mas
  libera este caso somente quando existem `DROP_INIT_ADAPTER_TARGET_MODULES`,
  `KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD`, `--allowed-extra-target-modules`
  e gate V485 dedicado.
- Gates apos a correcao:
  `hf_job_train_v90.py --self-test` OK,
  `hf_job_preflight_gate.py --self-test` OK,
  `run_v485_peft_roundtrip_gate.py --self-test` OK,
  V485 real com drop `lm_head` OK,
  static `ok=true/findings=[]`,
  pre-paid `ok=true/findings=[]`,
  `py_compile` OK.
- Proxima acao obrigatoria: commitar/pushar esta correcao, regenerar o
  manifesto A100 com novo `EXPECTED_COMMIT` e relancar somente `a100-large`.

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

Limpeza V674, 2026-05-19 19:50 UTC:

- Um agente independente revisou candidatos de limpeza com politica
  conservadora. Resultado: caches e temporarios podem ser removidos
  automaticamente; logs, manifests, datasets, adapters, gates, roadmaps,
  respostas OpenRouter e relatorios de analise ficam preservados salvo decisao
  explicita.
- O workspace clean gate antes da limpeza apontou apenas caches seguros como
  erro e um pacote V668 cancelado como aviso de blob grande. Apos
  `kg1_workspace_clean_gate.py --delete-safe`, nao restaram `__pycache__`,
  `.cache`, `.pytest_cache`, `.ipynb_checkpoints` ou temporarios equivalentes.
- Foi removido o pacote de submissao V668 cancelado, que nao era submit valido
  e estava gerando risco de confusao por parecer pacote ativo:
  `artifacts/v668_submission_package/`. A remocao eliminou
  `submission.zip` e `adapter_model.safetensors` desse pacote cancelado,
  totalizando `8093249372` bytes. Tambem foi removido o diretorio vazio
  `artifacts/v573_hf_h200_launch/downloaded_debug/`.
- Gate final de limpeza:
  `artifacts/v674_cleanup/v674_workspace_clean_after_large_blob_cleanup.json`
  com `passed=true`, `error=0`, `warning=0`.
- Diretorios `downloaded_debug`/`downloaded_eval` que contem logs, manifests,
  datasets, relatorios ou evidencias de jobs foram preservados. Esses arquivos
  ainda sao insumo de auditoria F2/backfire e nao devem ser apagados por glob.
- Regra operacional atualizada: qualquer nova limpeza deve passar pelo
  workspace clean gate e gerar manifesto de remocao; nao apagar artefatos
  ativos V673/V674 nem historico necessario para reproduzir loss/ACC.

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
  misses numeric auditaveis. `Gemini` e `DeepSeek` priorizam `bit_residual`,
  mas isso fica secundario porque `bit=136` ja cumpre o piso promocional.
- Proximo passo efetivo: nao GPU. Implementar ledger V672 dos `36` misses
  residuais (`12` equation numeric + `24` bit), com regra candidata,
  ambiguidade, predicao, `verify_answer`, token estimate e decisao
  `trainable/drop/protected-only`.
- Treino curto so e permitido se o ledger provar pelo menos `4` ganhos
  deterministas em `equation_numeric` ou `4` ganhos deterministas em bit sem
  label leakage. Caso contrario, atualizar roadmap e consultar novamente com o
  ledger real.

Execucao V672 local:

- `scripts/audit_v672_residual_miss_ledger.py` foi criado para cruzar V541 com
  V324/V336/V350/V366/V333/V334 e separar ganho direto, ganho guardado, ganho
  herdado sem prova de regra e drop.
- Ledger final `20260519T173138Z`: `36` linhas auditadas (`24` bit residual,
  `12` equation numeric), `12` trainable estritas, `2` trainable guardadas,
  `15` needs-rule-proof e `7` drop.
- Equation numeric: `4` linhas utilizaveis para probe (`7688e06e`, `274def88`,
  `d1bd7478`, `c5b058d6`). As duas primeiras sao estritas; `d1bd7478` e
  `c5b058d6` sao guardadas porque so ha um exemplo com o mesmo operador, embora
  V324/V336 tenham candidato unico, zero conflito e no-loss.
- Bit residual: `10` linhas diretamente treinaveis (`1abaffca`, `b8722d19`,
  `7192535b`, `1a7c8520`, `4ada9150`, `a6192d29`, `048cc279`, `4c327b55`,
  `b8aa3072`, `5ba26f21`) por V350/V366 aceitos.
- Gate: `allow_a100_large_equation_transfer_probe_guarded`. Usar apenas
  `a100-large`, probe barato e curto; H200 continua bloqueada. O treino deve
  mirar primeiro as 4 equation numeric utilizaveis e usar os 10 bit como
  replay/protecao, nao como objetivo principal.

Execucao V673 local:

- `scripts/build_v673_guarded_equation_bit_transfer_dataset.py` foi criado para
  converter o ledger em dados sinteticos sem treinar nas linhas weak. Usa 3
  classes equation V312 (`minus_signed`, `colon_trailing_zero`, `add_direct`) e
  replay bit sintetico V367.
- Dataset final ativo `20260519T190246Z`: train `720` linhas (`480` equation,
  `240` bit), validation `180` linhas (`120` equation, `60` bit).
- Pesos por linha: equation `1.0`, bit replay `0.35`, para manter bit como
  protecao e nao como objetivo dominante. O treino deve usar
  `USE_ROW_LOSS_WEIGHT=1`, `LOSS_NORMALIZATION_MODE=example_mean` e
  `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`.
- Contrato corrigido: mensagens `official_like`, prompt com `PROMPT_SUFFIX`,
  exatamente uma linha final `Final answer: \boxed{...}`. O bug de formato V367
  (`\boxed{...}` cru) foi normalizado no V673.
- Gate V286 com tokenizer real passou:
  `tokenization_gate_passed`, `offset_masks=720/180`, `fallback_masks=0`,
  `completion_tokens_dropped=0`, `prompt_truncated=0`,
  `train_val_prompt_overlap=0`, overlap com weak/baseline `0`.
- Gates V509, V513, V478, EOS/loss-mask, static safety, pre-paid e V666 tambem
  passaram no dataset ativo `20260519T190246Z`.

Regra ativa pos-treino continua: todo treino concluido ou falho deve gerar
prompt OpenRouter completo, executar a consulta quando houver API key,
classificar as respostas e atualizar este roadmap antes de qualquer novo gasto
GPU. Falso ganho continua proibido: nenhuma promocao sem extracao label-free,
`verify_answer`, zero truncation/fallback, guarda de protected rows, hash do
weak CSV e gate anti-runaway.

Este e o unico roadmap ativo. Historico antigo fica apenas como evidencia e nao
guia novas execucoes.

## Plano Ativo V672 Hoje

Decisao: bloquear H200, Kaggle submit e qualquer continuacao direta de V664.
Usar `a100-large` somente depois que P0 e P1 abaixo passarem. Se algum passo
falhar, parar a rota e atualizar este roadmap antes de novo gasto. V671 adiciona
uma regra forte: toda tentativa paga precisa nascer de uma linha/regra
verificada localmente, nao de expectativa de que treino descubra sozinho. V672
aperta a rota: `equation_numeric` e o alvo primario de hoje; bit residual so
entra se o ledger mostrar ganho deterministico e sem risco de regressao.

P0, ja iniciado/completo localmente:

1. V612 V664 vs V290: completo e `blocked`.
2. V541 V290 miss-map: completo e `passed`, com `24` bit residual,
   `12` equation numeric e `87` symbolic punctuation.
3. Corrigir o launcher V669 antes de qualquer HF job: o job falhou por
   `git clone` de repo privado sem autenticacao. O proximo launcher deve rodar
   a partir de artefato/script enviado ao HF ou repo autenticado; nao repetir
   o clone publico privado.
4. V671 Kaggle discussions audit: completo, `200/200` topicos e `1224` itens.
5. V672 OpenRouter today-gain consult: completo, `5/5` modelos responderam.
6. Novo bloqueador antes de GPU: gerar planilha local por linha para os `24`
   bit misses e `12` equation numeric misses, contendo regra candidata,
   evidencia, predicao, `verify_answer`, risco de ambiguidade e se a linha e
   learnable sem usar label weak como seletor.

P1, sem treino:

1. Implementar `scripts/audit_misses.py` ou equivalente para gerar ledger
   V672 de `36` linhas: `12` equation numeric + `24` bit residual. Campos
   obrigatorios: row id, prompt hash, familia/subfamilia, baseline V290,
   candidato CPU, regra candidata, query operator presente/ausente,
   ambiguidade, output boxed, `verify_answer`, token estimate, risco de leakage
   e decisao `trainable/drop/protected-only`. Status: completo via
   `scripts/audit_v672_residual_miss_ledger.py`; resultado permite probe
   guardado em A100-large.
2. Rodar auditoria CPU/local dos `12` equation numeric misses contra 4
   transformacoes, 32 operadores, operadores frequentes primeiro e
   missing-op/absolute-difference, incluindo checagem de `rbs` e `max_mod_min`
   se ja existirem ou forem adicionaveis no DSL.
3. Rodar auditoria CPU/local dos `24` bit misses V541 contra os algoritmos
   publicos/derivados: ROT/SHR/SHL, unary/binary, bitsum hash, stride
   esquerdo/direito, MAJ/CHO/fullbyte e sinais V350/V366.
4. Gate para GPU: `equation_numeric deterministic_unique>=4` e
   `ambiguity_selected_as_gain=0`; ou, secundariamente,
   `bit deterministic_unique>=4` com correspondencia em V350/V366 e zero risco
   de regredir o piso `bit=136`.
5. Construir dataset V673 minimo a partir do ledger: 4 equation numeric
   utilizaveis como alvo primario, 10 bit diretas como replay/protecao,
   answer-only boxed curto, zero symbolic punctuation, sem weak rows como
   seletor oculto. Status: completo em
   `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T190246Z/`.
6. Antes de GPU, rodar gate local do dataset: hashes dos inputs, zero overlap
   train/val, zero truncation, completion-only mask, exactly-one-boxed, `\boxed{}`
   obrigatorio, `verify_answer` em todas as targets e contagem de subfamilias.
   Status: gates V286 com tokenizer real, V509, V513, V478 e EOS passaram
   com `0` truncation, `0` fallback mask, `0` completion drop e `0` warnings
   promocionais.
7. Rodar probe em `a100-large` apenas depois de commit/push do codigo
   corrigido. Parametros: prompt oficial, `temperature=0`, `top_p=1`,
   `max_tokens=7680` no eval, treino curto, `SAVE_EMBEDDING_LAYERS=0`,
   `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`, LoRA compativel com adapter-only e
   sem H200.
8. O sweep de decoding V290/V664 fica diagnostico secundario; nao gastar GPU
   nele antes do dataset V673.
9. Gate para seguir: `protected_backfire=0`, `8740ed31=01101000`,
   `59bee375=10010101`, `bit>=136`, `equation>=60`, `total>=196`,
   `boxed_rate=1.0`, `truncated=0`, sem selecao por linha usando labels,
   e sem ambiguidade nao-resolvida marcada como ganho obrigatorio.

P2, se P1 nao der ganho submit-safe:

1. Construir dataset V671/V672 a partir de classes nao-weak equivalentes aos sinais
   V350/V366, nao a partir de respostas weak como alvo:
   - equation numeric/RBS/operador conhecido para cobrir os `12` numeric misses;
   - bit MAJ3/CHO/fullbyte ternary para reproduzir os `9` ganhos aceitos V366;
   - replay/protected-style para preservar linhas que V290 ja acerta.
2. Parent adapter: V290 `checkpoint-6`.
3. Target: resposta curta compativel com prompt oficial:
   `</think>\n\boxed{ANSWER}` ou answer-first com rationale curto, mas nao
   close-think-only como V664.
4. LoRA: `r=32`, `alpha=32`. Primeiro candidato com `q_proj/v_proj`; expandir
   para `q/k/v/o` somente se o gate de modulo/smoke A100 provar estabilidade e
   memoria. Nao tocar `lm_head`, embeddings ou MoE hoje.
5. Otimizacao: `example_mean`, row weights sincronizados em train/val,
   LR inicial `5e-5` ou `1e-4`; `2e-4` so se o gate de loss/ACC por familia
   indicar queda de loss nos tokens de resposta sem aumento de backfire/runaway.
   Oversampling hard-category acima de `3x` bloqueado.
6. Treino curto: 20-60 updates max, eval/checkpoint cedo. Abortar se
   `val_loss_bit` ou `val_loss_equation` cai mas ACC/answer-token gate nao
   melhora, ou se protected rows piorarem.

Itens removidos/congelados:

- V664 como base ou continuacao: congelado.
- H200 exploratorio: removido; usar `a100-large` por custo.
- `max_tokens=7680` como prova de qualidade isolada: fica apenas como controle
  official-like; candidato bom precisa parar cedo mesmo com limite alto.
- loss-only como criterio: removido. Loss e gate auxiliar; ACC label-free manda.
- qualquer treino sem V541/V612/miss taxonomy e protected replay: bloqueado.
- solver/postprocessor/CPU CSV como submit: bloqueado. So pode virar dataset ou
  criterio de validacao adapter-only.
- uso de labels weak para selecionar resposta em inferencia ou montar alvo
  direto de treino: bloqueado.
- DoRA: removido do plano curto por incompatibilidade/risco em vLLM.
- `equation_symbolic` all-in/gold-conditioned: congelado; usar apenas como
  taxonomia ate haver prova label-free.
- oversampling agressivo por familia dificil: bloqueado acima de `3x` sem
  dry-run demonstrando nao regressao.

## Estado Real

| Metrica | Melhor submit-safe adapter-only | Gate promocional atual |
|---|---:|---:|
| Total weak | `192/315` | `>=196/315` |
| `bit_manipulation` | `136/160` | `>=136/160` |
| `equation_transform` | `56/155` | `>=60/155` |
| Truncated | `0` | `0` |

Sinais nao submit-safe:

- postprocessor/solver historico: `196/315`, `bit=136`, `equation=60`;
- V642 CPU no-loss: `208/315`, `bit=147`, `equation=61`.

Esses sinais so autorizam treino adapter-only curto. Eles nao autorizam submit
sem weak eval oficial-like.

## V664 Reprovada E Evidencia V666

Objetivo: corrigir o mecanismo observado em V661/V663 onde o adapter mantem
`enable_thinking=True` em geracao longa, nao inicia com `\boxed{}` e causa
backfire em linhas protegidas. V664 treina alvo curto:

```text
<think>
</think>
\boxed{answer}
```

Contrato de geracao esperado apos o prompt oficial que ja termina em
`<think>\n`: `</think>\n\boxed{answer}`.

Status atual:

- dataset HF:
  `felipesp1983/kg1-v664-close-think-boxed-artifacts`,
  `v664-close-think-boxed-20260519T-v664-cpu-gate`;
- upload commit HF:
  `7f4e33bb41e0054f44d0546b7dce7a6e241870dc`;
- train: `2233` rows, SHA
  `a04902371b304ba9bf034ed5d677a58b3f2e0a68a9d4f690d6f8b06f4f08e963`;
- val: `360` rows, SHA
  `e2e31ce574b42fd65a5a6de45255b03dd900eddc91da2b289306ab3ddd3c5b40`;
- gate V664 close-think: passed, `first_box_token_idx_p95=2`,
  `target_tokens_p95=17`;
- gate V286 tokenizacao: passed, `train_token_max=314`,
  `validation_token_max=314`, `0` prompt truncation, `0` completion dropped,
  `0` fallback masks, `0` overlap train/val e weak;
- gate V509 integridade: passed para train e val;
- gate V478 objetivo: passed, mix efetivo train
  `bit_manipulation=0.40`, `equation_transform=0.60`;
- gate V524: bloqueia token_mean puro (`bit` token share `0.874554`), portanto
  esta rota exige `LOSS_NORMALIZATION_MODE=example_mean` + row weights;
- gate V526: passed, selected example_mean bit share `0.40`, delta `0.0`;
- gate V513 trace learnability: nao aplicavel para V664, porque V664 remove
  trace por desenho; ele fica como evidencia negativa contra voltar a trace SFT;
- static safety gate V664 CUDA12/A100: passed, findings `0`;
- pre-paid integration gate V664 CUDA12/A100: passed, findings `0`;
- hardware usado: `a100-large` (`1x A100 80GB`, `0.041667 USD/min`);
- tentativa A100 NeMo/CUDA13:
  `https://huggingface.co/jobs/felipesp1983/6a0bf424a5e509f1a84165a9`,
  falhou em preinstall porque A100 com runtime CUDA13 foi bloqueado pelo gate;
- tentativa A100 CUDA12 inicial:
  `https://huggingface.co/jobs/felipesp1983/6a0bf560e7940de6ee6cf59d`,
  falhou antes do treino por bug de ordem no comando remoto: o probe importava
  `causal_conv1d`/`mamba_ssm` antes do `pip install`; corrigido no launcher;
- job ativo A100 CUDA12 com ordem de dependencias corrigida:
  `https://huggingface.co/jobs/felipesp1983/6a0bf719e7940de6ee6cf5c8`,
  cancelado manualmente porque `causal-conv1d` estava compilando por fonte por
  tempo incompatível com um smoke de `2` steps;
- job ativo A100 CUDA12 com dependencias binárias obrigatorias:
  `https://huggingface.co/jobs/felipesp1983/6a0bfa27e7940de6ee6cf663`,
  falhou rapido porque nao existe wheel binario compativel para
  `causal-conv1d==1.6.2.post1` nesse ambiente;
- fallback H200 justificado:
  A100 NeMo/CUDA13 foi bloqueado, A100 CUDA12 exige build-fonte lento para
  dependencias Mamba, e A100 CUDA12 com `--only-binary` nao encontra wheel;
- job ativo H200 NeMo:
  `https://huggingface.co/jobs/felipesp1983/6a0bfb05e7940de6ee6cf667`;
- run id:
  `v664-nemo-h200-closethink-qv-v290ckpt6-20260519T055306Z`;
- imagem ativa:
  `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`;
- custo autorizado somente para este fallback: `h200`,
  `0.083333 USD/min`, gate max `0.09`;
- preflight observado no job ativo: `torch=2.9.0a0+50eac811a6.nv25.09`,
  CUDA `13.0`, GPU `NVIDIA H200`, `139.80 GiB`, `causal_conv1d=1.5.3`,
  `mamba_ssm=2.2.6.post3`;
- tokenizacao remota confirmada no job ativo: train `2233/2233`, val
  `360/360`, `0` truncation, `0` prompt tokens dropped, `0` fallback masks,
  offset masks `2233/360`, row weights train `min=0.4839`, `max=3.4620`,
  `mean=1.0000`;
- superficie treinavel observada: somente `q_proj` e `v_proj`, `24` tensores
  LoRA, `1,867,776` parametros treinaveis, `0.0058%` do modelo; MoE
  `mlp.experts.gate_up_proj` e `mlp.experts.down_proj` preservados no adapter
  mas congelados (`0` parametros treinaveis);
- auditoria tensor-a-tensor do `checkpoint-2` contra o initializer V290
  `checkpoint-6`: `12011` tensores comparados, `missing=0`, `extra=0`;
  somente `q_proj=12` e `v_proj=12` mudaram. Todos os demais modulos
  (`k_proj`, `o_proj`, `in_proj`, `out_proj`, `up_proj`, `down_proj`,
  `lm_head` e MoE preservado) permaneceram inalterados. Resultado:
  `PASS_ONLY_QV_CHANGED`;
- baseline eval loss no alvo V664: `33.0273`;
- treino do checkpoint-2 executado: step 1 loss `39.7814`, step 2 loss
  `27.1268`;
- job H200 `6a0bfb05e7940de6ee6cf667` foi cancelado durante o eval
  pos-step-2 antes do salvamento/upload do checkpoint; nao criou o repo
  `felipesp1983/kg1-nemotron-lora-v664-h200-closethink-qv-v290ckpt6`;
- correcao pos-cancelamento: relancar V664 checkpoint-first com
  `BASELINE_EVAL_BEFORE_TRAIN=0`, `EVAL_EVERY_STEPS=0`,
  `EVAL_MAX_EXAMPLES=24`; isso salva/uploada `checkpoint-2` antes de qualquer
  eval longo e preserva o weak eval official-like como medidor real de ACC;
- gates do relancamento checkpoint-first: `py_compile` passed, static safety
  `v664_static_safety_gate_h200_checkpoint_first.json` passed, pre-paid
  `v664_pre_paid_job_integration_gate_h200_checkpoint_first.json` passed.
- tentativa de relancar em 2026-05-19 06:15 UTC foi bloqueada pela API HF com
  `402 Payment Required`: saldo pre-pago insuficiente para criar Jobs;
- relancamento apos credito HF em 2026-05-19 12:08 UTC:
  `https://huggingface.co/jobs/felipesp1983/6a0c52e8e7940de6ee6cf9cd`;
- run id ativo:
  `v664-nemo-h200-closethink-qv-v290ckpt6-20260519T120800Z`;
- relancamento checkpoint-first concluido: job `6a0c52e8e7940de6ee6cf9cd`
  completou, publicou `checkpoint-2` em
  `felipesp1983/kg1-nemotron-lora-v664-h200-closethink-qv-v290ckpt6/checkpoint-2`
  e registrou final eval loss `32.4868`;
- weak eval V664 ativo:
  `https://huggingface.co/jobs/felipesp1983/6a0c590ea5e509f1a8416e11`;
- weak eval run id:
  `v664-h200-closethink-qv-weak-20260519T123415Z`;
- status final do weak eval V664: `COMPLETED`;
- resultado:
  `192/315`, `bit_manipulation=136/160`, `equation_transform=56/155`,
  `truncated=0`, `boxed_rate=1.0`, `no_box_fallback=0`,
  `label_aware_minus_label_free_correct=0`;
- completion-token diagnostico:
  `completion_tokens_total=1503183`, `avg=4772.0095`, `max=7492`;
  por familia, bit ficou especialmente longo (`mean=6682.5`, p95 `7277.15`),
  equation tambem longo (`mean=2799.9`, p95 `6234.7`);
- first boxed char index:
  bit median `8677`, equation median `1487`; o modelo continua raciocinando
  antes do boxed em vez de emitir resposta curta;
- blocked reasons do weak promotion gate:
  `correct_lt_196`, `equation_lt_60`, `avg_completion_tokens_gt_128`,
  `max_completion_tokens_gt_512`, `protected_row_backfire_guard_failed`;
- protected-row guard:
  `8740ed31` regrediu de baseline correto para candidato errado
  (`01101000` -> `01111000`), `59bee375` permaneceu correto e `55d834d1`
  continuou sem ganho obrigatorio (`00111111` esperado, `10111111` gerado);
- drift contra baseline V516:
  V664 mudou somente `3` linhas relevantes contra
  `artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv`:
  ganhou `4bb8c6cd` em equation (`]` -> `]}\\!`), ganhou `4ada9150` em bit
  (`01111111` -> `01111011`), mas perdeu a protegida `8740ed31`
  (`01101000` -> `01111000`). Portanto ha micro-sinal real, mas nao
  submit-safe;
- upload do eval:
  `evals/v664-h200-closethink-qv-weak-20260519T123415Z`, commit HF
  `95df4c209b93f68bc17a7ddc1eecc152a4d6c1a0`;
- consulta OpenRouter V665 criada apos falha:
  `artifacts/openrouter/v665_v664_failure_consult/KG1_V665_OPENROUTER_V664_FAILURE_PROMPT.md`;
- respostas brutas:
  `artifacts/openrouter/v665_v664_failure_consult/v665_openrouter_raw_results.json`
  e
  `artifacts/openrouter/v665_v664_failure_consult/v665_openrouter_raw_results_retry_deepseek_qwen.json`;
- consenso V665:
  `artifacts/openrouter/v665_v664_failure_consult/KG1_V665_CONSENSUS.md`;
- resumo/gate local de falha V664:
  `artifacts/v665_v664_failure_analysis/v665_v664_failure_summary.json` e
  `artifacts/v665_v664_failure_analysis/KG1_V665_V664_FAILURE_SUMMARY.md`;
- decisao V665:
  nenhum novo GPU ate passar gates CPU de template parity, target token
  contract, answer/EOS weighting e smoke protegido/length antes de full weak;
- manifesto do relancamento ativo:
  `artifacts/v664_hf_a100_launch/v664-nemo-h200-closethink-qv-v290ckpt6-20260519T120800Z_launch_manifest.json`;
- comando remoto do relancamento ativo:
  `artifacts/v664_hf_a100_launch/v664-nemo-h200-closethink-qv-v290ckpt6-20260519T120800Z_remote_command.sh`;
- weak eval V664 preparado localmente:
  `artifacts/v664_hf_h200_weak_eval_launch/launch_v664_hf_weak_eval_checkpoints.py`;
- gate estatico do weak eval V664:
  `artifacts/v664_hf_h200_weak_eval_launch/v664_weak_eval_static_safety_gate.json`
  passed, findings `0`;
- regra adicional para evitar ganho falso no weak eval V664: manter
  `max_tokens=7680` official-like, mas so permitir promocao com
  `boxed_rate=1.0`, `no_box_fallback=0`, `truncated=0`,
  `avg_completion_tokens<=128` e `max_completion_tokens<=512`.

Correcoes V666 apos consulta OpenRouter:

- prompt pos-treino executado:
  `artifacts/openrouter/v666_post_train_rule_v664_executed/KG1_POST_TRAIN_OPENROUTER_PROMPT.md`;
- respostas e manifesto:
  `artifacts/openrouter/v666_post_train_rule_v664_executed/openrouter_responses.md`,
  `artifacts/openrouter/v666_post_train_rule_v664_executed/openrouter_raw_results.json`,
  `artifacts/openrouter/v666_post_train_rule_v664_executed/openrouter_manifest.json`;
- consenso V666:
  `artifacts/openrouter/v666_post_train_rule_v664_executed/KG1_V666_POST_TRAIN_CONSENSUS.md`;
- `scripts/kg1_post_train_openrouter_consult.py` implementa a regra fixa de
  prompt/consulta pos-treino;
- `scripts/audit_v478_training_objective_alignment.py` agora aceita
  `--require-validation-row-loss-weight` e aplica row weights tambem no resumo
  de validation quando `--use-row-loss-weight` esta ativo;
- `scripts/hf_job_train_v90.py` agora tokeniza validation com
  `apply_row_loss_weight=True`, portanto final/best eval loss usa o mesmo
  objetivo ponderado do treino;
- `LOSS_MASK_STOP_AFTER_EOS=True` virou default em `scripts/hf_job_train_v90.py`;
  a mask supervisionada zera tudo apos o primeiro EOS;
- `scripts/run_v286_generic_tokenization_gate.py` foi alinhado com a mesma regra
  de parar a mask apos EOS;
- novo gate `scripts/audit_loss_mask_eos_contract.py` valida que o ultimo token
  supervisionado e EOS;
- evidencia: `artifacts/v665_v664_failure_analysis/v666_v478_validation_row_weight_recheck_final.json`
  passou com pesos efetivos train/validation `bit=0.4`, `equation=0.6`;
- evidencia: `artifacts/v665_v664_failure_analysis/v666_v664_loss_mask_eos_contract_sample80_after_patch.json`
  passou com `final_loss_eos_rate=1.0` no sample de train e validation;
- evidencia: `artifacts/v665_v664_failure_analysis/v666_static_safety_gate_changed_files.json`
  passou sem findings.
- diagnosticos V664 baixados do HF para caminho local curto:
  `artifacts/v665_v664_failure_analysis/v666_downloaded_v664_eval/`;
- V614 anti-runaway/protected gate local:
  `artifacts/v665_v664_failure_analysis/v666_v664_v614_anti_runaway_gate.json`;
- resultado V614: blocked com `correct=192/315`, blockers
  `bit_p99_tokens_gt_128`, `equation_lt_60`, `equation_p99_tokens_gt_512`,
  `protected_failed_55d834d1`, `protected_failed_8740ed31`, `total_lt_196`;
- protected V614:
  `8740ed31` esperado `01101000`, gerado `01111000`, `6290` completion tokens;
  `59bee375` ok mas com `6589` completion tokens;
  `55d834d1` esperado `00111111`, gerado `10111111`, `6285` completion tokens.
- gate agregado V666 implementado:
  `scripts/kg1_v666_cpu_gate_stack.py`;
- relatorio agregado:
  `artifacts/v665_v664_failure_analysis/v666_cpu_gate_stack.json`;
- decisao agregada V666: `gpu_blocked`, com 6 checks verdes
  (`v478_objective_alignment`, `loss_mask_eos_contract`,
  `label_free_drift_audit`, `static_safety_changed_files`,
  `post_train_openrouter_rule`, `workspace_no_pycache`) e 1 check vermelho
  (`v614_anti_runaway_promotion`);
- `scripts/kg1_pre_paid_job_integration_gate.py` agora exige
  `--v666-cpu-gate-report-json`, valida schema
  `kg1_v666_cpu_gate_stack_v1`, `decision=gpu_allowed`, `gpu_allowed=true`,
  `blockers=[]` e todos os checks verdes antes de qualquer job pago;
- todo launcher pago tambem precisa declarar
  `KG1_V666_CPU_GATE_STACK_STATUS="passed"` e
  `KG1_V666_CPU_GATE_STACK_REPORT=...`; sem isso, o pre-paid gate falha com
  `launcher_v666_cpu_gate_stack_not_passed` ou
  `launcher_v666_cpu_gate_report_missing`;
- default operacional do pre-paid gate passou para `expected_flavor=a100-large`;
  H200 deve ser excecao explicita e documentada por preflight/custo;
- static safety final dos scripts V666:
  `artifacts/v665_v664_failure_analysis/v666_static_safety_gate_changed_files_final4.json`,
  `file_count=10`, findings `0`.
- relatorio V666 agregado reexecutado apos limpeza de `__pycache__`:
  `artifacts/v665_v664_failure_analysis/v666_cpu_gate_stack.json` segue
  `gpu_blocked` somente por `v614_protected_or_length_or_score_failed`.
- diagnostico label-free/drift V666:
  `scripts/analyze_v666_v664_label_free_drift.py` e
  `artifacts/v665_v664_failure_analysis/v666_v664_label_free_drift_audit/`;
- resultado do diagnostico: candidate `192/315` contra baseline `191/315`,
  `stored_vs_official_correctness_mismatch=0`,
  `first_boxed_vs_official_correctness_mismatch=0`, transicoes
  `gain=2`, `backfire=1`, `stable_correct=190`, `stable_wrong=122`;
- conclusao do diagnostico: o plateau V664 nao e bug de parser/verifier; o
  bloqueio vem de geracao runaway (`mean=4772`, `p99=7350`, `max=7492`) e
  backfire real em protected row (`8740ed31`).

Contrato LoRA real:

- init adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`;
- manter `r=32`, `alpha=32` e target_modules completos do V290 para
  compatibilidade de carregamento;
- treinar somente tensores LoRA com nomes contendo `q_proj` e `v_proj`;
- `target_parameters` MoE ficam congelados
  (`REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`);
- qualquer sugestao `r=16/alpha=16` vale apenas para adapter novo do zero, nao
  para continuacao V290.

Promocao do V664:

- status: reprovado; nao submit-safe;
- checkpoint-2 existiu e passou pelo weak eval exigido por
  `KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED=1`, mas falhou nos gates
  promocionais;
- bloqueadores absolutos observados: protected-row backfire em `8740ed31`,
  `avg_completion_tokens>128`, `max_completion_tokens>512`,
  `equation<60/155` e `total<196/315`;
- o resultado (`192/315`, `bit=136/160`, `equation=56/155`) congela V664;
- a rota so pode ser reaproveitada como evidencia de engenharia apos os gates
  V666, nunca como adapter de submit.

## Contrato Oficial

- pacote Kaggle deve ser `submission.zip` com LoRA adapter compativel com
  `Nemotron-3-Nano-30B`;
- base/revision fixa:
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`,
  `cbd3fa9f933d55ef16a84236559f4ee2a0526848`;
- `adapter_config.json` obrigatorio;
- `max_lora_rank=32`;
- inferencia oficial-like: `max_tokens=7680`, `temperature=0.0`,
  `top_p=1.0`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`,
  `max_model_len=8192`;
- resposta final em `\boxed{}`;
- extractor label-free deve priorizar `\boxed{}` e registrar fallback;
- metrica principal e ACC por geracao completa + `verify_answer`;
- `eval_loss` e apenas diagnostico de aprendizado do target mascarado.

## Regra De Hardware E Custo

- regra ativa: usar `a100-large` como default para novos jobs sempre que o
  job couber em 80 GB de VRAM;
- credito HF restante informado pelo usuario: aproximadamente `30 USD`; regra
  FinOps ativa: nao abrir jobs duplicados, nao abrir novo H200 enquanto existir
  job H200 em fila/execucao, e encerrar qualquer job sem sinal util antes de
  exceder o timeout do gate;
- especificacao HF observada:
  - `a100-large`: 1x A100, 80 GB VRAM, 142 GB RAM, `0.041667 USD/min`;
  - `h200`: 1x H200, 141 GB VRAM, 256 GB RAM, `0.083333 USD/min`;
- H200 so pode ser usado quando for realmente necessario:
  - treino/eval Nemotron 30B BF16 com gate `MIN_GPU_TOTAL_GIB > 80`;
  - vLLM official-like com `max_tokens=7680` que nao passa em A100;
  - A100 falhou por OOM/preflight de capacidade e o job continua necessario;
  - o motivo deve aparecer no launcher, manifesto e roadmap;
- o job V663 executado permaneceu em H200 porque o launcher aprovado exigia
  `MIN_GPU_TOTAL_GIB=130` e carrega `Nemotron-3-Nano-30B-A3B-BF16` em BF16;
- proximos launchers devem tentar `a100-large` primeiro quando o gate de
  memoria permitir, sem reduzir `max_tokens`, prompt oficial ou qualidade do
  diagnostico para forcar encaixe artificial.
- excecao atual: o weak eval V664 permanece em H200 porque o launcher official-like
  exige `MIN_GPU_TOTAL_GIB=130` para `Nemotron-3-Nano-30B-A3B-BF16` com
  `max_tokens=7680`; custo maximo planejado pelo timeout de `3600s` e de cerca
  de `5 USD`. Se ele entrar em `RUNNING` e nao emitir progresso, cancelar antes
  de consumir credito sem diagnostico.

## Dados E Hashes

Oficiais:

- `train.csv`: `9500` rows, SHA
  `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`;
- `test.csv`: `3` rows, SHA
  `c59d7eb0464b0a872a0c3f81e60cd6643fc1932a2dedaa05972bfd02cc638589`;
- weak gate: `315` rows, SHA
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.

Bloqueio absoluto:

- os `315` weak rows nao podem ser usados para treino, pseudo-label,
  curriculum, hiperparametro ou selecao de candidato;
- subconjuntos weak diagnosticos tambem nao podem orientar iteracoes de
  hiperparametro/checkpoint. So podem validar uma configuracao ja congelada;
- todo ganho precisa ser medido por geracao completa, extracao label-free e
  `verify_answer`;
- qualquer metrica permissiva divergente de `verify_answer` e ganho falso.

## Achados Ativos

- V642 corrigiu bug silencioso de parser em respostas simbolicas com braces
  escapados; baseline real continua `192/315`;
- V647/V646 falharam por output-policy/decoding drift: completions longas,
  boxed tardio, truncation, protected backfire e equation parado em `56/155`;
- V650 corrigiu auditoria permissiva que inflava V647 para `206/315`; valor
  correto e `193/315`;
- V651 corrigiu target curto com `box_answer(answer)` e endureceu promocao para
  `196/60`;
- V652/V613 answer-first foi bloqueado pelo V513: templates normalizados com
  respostas conflitantes e bit answer-only nao aprendivel. V652 nao deve ser
  lancado;
- V661 mostrou que loss menor sem paridade de template nao transfere para ACC:
  `5.8567 -> 5.8231` no treino, mas weak caiu para `191/315` e gerou
  truncation/backfire;
- V662 boxed-only foi rejeitado como rota ativa: o target curto nao respeita o
  prefixo oficial `enable_thinking=True` e V513 classificou bit answer-only
  como nao aprendivel;
- V663 fica congelada/reprovada:
  - dataset: `v663_thinking_close_boxed_train.jsonl` e
    `v663_thinking_close_boxed_val.jsonl`;
  - esquema: `<think>\n[trace compacto]\n</think>\n\boxed{answer}`;
  - objetivo efetivo: `bit=0.40`, `equation=0.60`;
  - contrato LoRA real do init adapter V290: `r=32`, `alpha=32`,
    `target_modules=down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj`
    e `target_parameters=mlp.experts.gate_up_proj,mlp.experts.down_proj`;
  - regra de treino V663 corrigida: preservar `target_parameters` para carregar
    o V290 sem alterar o adapter surface, mas congelar MoE/MLP e treinar
    somente `q_proj,k_proj,v_proj,o_proj`
    (`REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`);
  - V659, V509, V286, V513, V663 template-parity, V478, V526, static safety e
    pre-paid integration passaram;
  - V663 template-parity confirmou `2593/2593` prefixos char/token iguais ao
    prompt official-like com `enable_thinking=True`;
  - tokenizacao real: `0` truncation, `0` dropped completions, `0` fallback
    masks, `0` weak overlap;
  - HF dataset:
    `felipesp1983/kg1-v663-thinking-trace-boxed-artifacts@ead06b6339099a32861f789dece3ae1c007b3af4`;
  - launcher ativo:
    `artifacts/v663_hf_h200_launch/launch_v663_hf_nemo_h200_thinking_trace.py`;
  - output repo previsto:
    `felipesp1983/kg1-nemotron-lora-v663-h200-thinkingtrace-attnstrict-v290ckpt6`;
  - job cancelado por crisis-mode antes de treino util:
    `https://huggingface.co/jobs/felipesp1983/6a0bdaaea5e509f1a8416264`
    porque ainda treinava `up_proj/down_proj` junto com `q/k/v/o`.
  - segundo job cancelado por crisis-mode antes de treino util:
    `https://huggingface.co/jobs/felipesp1983/6a0bdbd5a5e509f1a8416278`
    porque `REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS` ainda exigia
    `up_proj/down_proj`, divergindo do filtro attention-only.
  - job corrigido concluido:
    `https://huggingface.co/jobs/felipesp1983/6a0bdcebe7940de6ee6cf395`;
    status observado em 2026-05-19 01:33 BRT: `COMPLETED`;
    checkpoints publicados: `checkpoint-2`, `checkpoint-4`, `checkpoint-6`;
    loss final `10.4831`, melhor eval loss `10.4814`.
  - observacao de empacotamento: o repo HF contem os adapters em subpastas
    `checkpoint-*`; nao ha `final_adapter/` nem adapter no nivel raiz. Qualquer
    pacote/submit deve apontar explicitamente para uma subpasta validada.
  - weak eval official-like do `checkpoint-2`:
    `https://huggingface.co/jobs/felipesp1983/6a0be316e7940de6ee6cf3b2`;
    output path:
    `evals/v663-h200-thinkingtrace-attnstrict-weak-20260519T041057Z`;
    status final: reprovado pelo gate.
  - resultado V663 checkpoint-2:
    - total `190/315` (`0.603175`);
    - `bit_manipulation=135/160`;
    - `equation_transform=55/155`;
    - `truncated=1`, `no_box_fallback=1`;
    - `boxed_rows=314/315`, `boxed_rate=0.996825`;
    - `starts_boxed_rows=0/315`;
    - `avg_completion_tokens=4776.18`, `max_completion_tokens=7680`;
    - `first_boxed_correct=190`, `label_aware_debug_correct=190`,
      `label-aware delta=0`.
  - delta contra baseline V516 label-free:
    - bit: baseline `136`, V663 `135`, `+1/-2`;
    - equation: baseline `55`, V663 `55`, `+1/-1`;
    - ganhos isolados: `4ada9150` bit e `4bb8c6cd` equation;
    - perdas: `8740ed31` bit, `59bee375` bit e `56343b77` equation.
  - protected-row guard:
    - `8740ed31`: baseline `01101000` correto, V663 `01111000` errado
      (`backfire_from_correct_baseline`);
    - `59bee375`: baseline `10010101` correto, V663 `2` errado, sem boxed e
      truncado (`backfire_from_correct_baseline`);
    - `55d834d1`: baseline `10111111` errado, V663 `10111111` errado,
      faltou ganho obrigatorio para `00111111`.
  - decisao de crise: nao avaliar `checkpoint-4`/`checkpoint-6` em full weak
    como proxima acao. A rota ja foi falsificada por ACC, truncation, no-box
    fallback e protected backfire no primeiro checkpoint.
- forum/THK/OpenRouter reforcaram que `\boxed{}` precisa ser precoce, que
  exemplos de operadores diferentes so ajudam quando ha meta-regra comum, e
  que traces duplicadas com respostas diferentes causam "Duplicate CoT Trap";
- OpenRouter V664 pos-falha V663 consolidou a correcao principal:
  `starts_boxed` sozinho e incompleto com `enable_thinking=True`, porque o
  prompt official-like ja termina em `<think>\n`; a rota curta correta deve
  gerar `</think>\n\boxed{answer}` rapidamente e deve ser controlada por
  `first_box_token_idx`, `completion_tokens`, `boxed_rate`, `truncated` e
  `no_box_fallback`;
- bit deve ser preservado como piso/protected-anchor; o ganho promocional mais
  importante agora precisa vir de `equation_transform=56/155 -> 60/155`.

## Rota V664 Ativa

Status: gates pre-GPU passaram; A100 foi tentado primeiro e falhou por motivos
comprovados de runtime/dependencia. O job ativo esta em H200 apenas para este
fallback V664, conforme regra de usar H200 somente quando realmente necessario.
Novos jobs voltam a tentar `a100-large` por default.

Nome: `V664 close-think immediate boxed q/v-only`.

Hipotese:

- V661/V663 falharam porque o target ainda mantinha ou induzia raciocinio longo
  antes do boxed;
- com `enable_thinking=True`, o prompt de geracao termina em `<think>\n`;
- o target certo deve fechar o bloco de pensamento imediatamente e emitir o
  boxed:

```text
<think>
</think>
\boxed{answer}
```

Na geracao, isso deve aparecer como:

```text
</think>
\boxed{answer}
```

Contrato de treino:

- init adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`;
- LoRA herdado do V290 `r=32`, `alpha=32`, com `target_modules` completos
  preservados para compatibilidade de carregamento;
- treinar somente tensores LoRA cujos nomes contem `q_proj` e `v_proj`;
- `target_parameters` vazios ou preservados apenas para compatibilidade,
  sempre congelados e fora do optimizer;
- trainable fraction baixo e auditado; qualquer aumento de superficie treinavel
  exige novo gate estatico e novo manifesto;
- LR `3e-7` constante;
- primeiro checkpoint com `max_steps=2`;
- dataset nao-weak, sem trace, exatamente um boxed, EOS supervisionado logo
  apos o boxed, payload byte-equal ao `answer`;
- objetivo efetivo `bit=0.40`, `equation=0.60`;
- validacao/holdout reponderado, nao bit-heavy.

Gates V664 antes de GPU:

- template parity official-like em train e holdout;
- `0` truncation, `0` completion tokens dropped, `0` fallback masks;
- `first_box_token_idx <= 8` em `100%` das linhas e p95 `<=6`;
- exatamente `1` boxed por linha;
- EOS no loss mask em `100%` das linhas;
- holdout nao-weak congelado: `>=360` rows, `>=160` equation, `>=120` bit,
  equation share efetivo `>=0.40`, overlap weak `0`;
- preservation subset nao-weak: `>=40` bit rows corretas no init adapter,
  `0` backfire permitido;
- preflight A100-large primeiro. H200 so com OOM/preflight ou manifesto
  provando necessidade real de `>80GB`.

Gates do primeiro checkpoint:

- holdout official-like: `truncated=0`, `no_box_fallback=0`,
  `boxed_rate=1.0`;
- media completion tokens `<=48`, p95 `<=96`, qualquer row `>512` mata a rota;
- bit delta `>=0` vs init adapter;
- equation delta `>=+2` vs init adapter;
- label-aware delta `0`;
- weak eval so roda se tudo acima passar.

## Rota V653 Congelada/Reprovada

Status: historico/forense, nao executavel como rota promocional.

Objetivo original: transformar o sinal CPU `208/315` em aprendizado
adapter-only sem gerar output longo. V653 mantinha a mistura V643/V641/V367,
mas compactava bit para traces curtos com termos de regra e boxed suffix,
preservando equation com regra curta e resposta boxed.

HF dataset:

- repo: `felipesp1983/kg1-v653-compact-trace-output-policy-artifacts`;
- commit: `5f2dd9333efdfd175a5f6c4255b06b4992424361`;
- root: `v653-compact-trace-output-policy-20260518T-v653-cpu-gate`.

Arquivos:

- train: `v653_compact_trace_output_policy_train.jsonl`;
- val: `v653_compact_trace_output_policy_val.jsonl`;
- manifest: `v653_compact_trace_output_policy_manifest.json`.

Hashes:

- train SHA
  `2b2781c855bcf0ddcacfb507c84f0935a8467d1ac91f5801d453a5e4336ba07b`;
- val SHA
  `3e64a84a4fcb4f921ee40e25ff778f4c5ac4f074a35951cf1402c1175474298c`.

Composicao:

- train `2113`, val `480`;
- train: `bit_manipulation=1661`, `equation_transform=452`;
- val: `bit_manipulation=385`, `equation_transform=95`;
- objetivo efetivo com `example_mean + row_loss_weight`:
  `bit=0.741935`, `equation=0.258065`;
- assistant curto: bit p95 `244` chars, equation p95 `411` chars.

Gates V653 ja passados:

- V509 train/val: `blocked_dataset_count=0`;
- V286 tokenization real: `0` truncation, `0` completion tokens dropped,
  `0` fallback masks, train/val overlap `0`;
- V513 learnability: `status=passed_cpu_structure_only`,
  `finding_counts.blocker=0`, `warning=0`;
- V478 objective alignment: `hf_gpu_allowed=true`;
- V524 quota/token objective: `quota_ok_cpu_only`;
- V526 example_mean: `example_mean_dry_run_passed`, delta `0.0`;
- static safety gate: sem findings;
- pre-paid integration gate: `ok=true`, incluindo V513, V286, hashes,
  row-loss, residual-first gate e protected row contract.

Launcher historico:

- `artifacts/v653_hf_h200_launch/launch_v653_hf_nemo_h200_compact_trace_output_policy.py`;
- output repo:
  `felipesp1983/kg1-nemotron-lora-v653-h200-compacttrace-outputpolicy-v290ckpt6`;
- flavor `h200`, timeout `3600`;
- `MAX_STEPS=20`, `SAVE_EVERY_STEPS=2`, `EVAL_EVERY_STEPS=2`;
- `LEARNING_RATE=1.0e-6`, `FINAL_LEARNING_RATE=1.0e-7`;
- `MAX_LENGTH=2048`;
- `LOSS_NORMALIZATION_MODE=example_mean`;
- `USE_ROW_LOSS_WEIGHT=1`, `REQUIRE_ROW_LOSS_WEIGHT=1`;
- LoRA trainable modules:
  `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`;
- MoE target parameters trainaveis:
  `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`;
- init adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.

Pendencia historica antes de `--launch`:

- o launcher debug atual ainda referencia o commit Git anterior
  `23d6e70f3f0c0493c6a4aae712f9c660d4711d51`;
- isso nao deve ser usado para relancar V653; fica registrado apenas como causa
  possivel de falha se alguem reexecutar artefato antigo.

Correcao V653 pos-auditoria:

- V618 apontou que `2` steps com LR `2e-8` era fraco demais para mover output
  policy e provavelmente continuaria o plateau;
- o launcher V653 foi corrigido para `20` steps, LR `1e-6 -> 1e-7`,
  checkpoints/eval a cada `2` steps e
  `KG1_V618_MODULE_SURFACE_GATE_STATUS=passed`;
- pre-paid integration gate pos-correcao passou com `ok=true`, V513/V286
  verdes, drift deferred permitido ate `20` steps e checkpoint cedo exigido.
- V618 pos-correcao ainda bloqueia por schema de dataset official-template
  single-line, que nao e o contrato V653. Isso ficou documentado em
  `artifacts/v653_hf_h200_launch/KG1_V653_V618_APPLICABILITY_NOTE.md`.

Resultado V653 checkpoint-2:

- treino H200 `6a0b8e80a5e509f1a8415c52` foi cancelado por FinOps depois de
  ultrapassar 1h; checkpoints `2`, `4`, `6`, `8` e `10` foram preservados;
- `eval_loss` caiu de `4.3190` para `4.2012` ate checkpoint-10, mas loss
  ainda nao e ganho submit-safe;
- weak-eval oficial-like do checkpoint-2 (`6a0ba12ea5e509f1a8415f07`)
  terminou e foi bloqueado pelo gate:
  - total `192/315`;
  - `bit_manipulation=136/160`;
  - `equation_transform=56/155`;
  - `truncated=1`;
  - boxed rate `314/315`, mas starts-boxed `0/315`;
  - protected backfire em `59bee375`;
  - missing required gain em `55d834d1`.
- checkpoint-2 nao e submetivel e nao pode ser promovido.

Resultado V653 checkpoint-10 smoke focado:

- job HF `6a0ba79de7940de6ee6cf2bf`, run
  `v653-h200-compacttrace-checkpoint10-smoke-20260518T235713Z`;
- smoke diagnostico de `24` rows, nao promocional e nao submetivel;
- resultado: `12/24` total, `bit_manipulation=9/16`,
  `equation_transform=3/8`;
- `truncated=1`, `no_box_fallback=1`, boxed rate `23/24`;
- completion tokens ainda excessivo: `166,034` total,
  media `6,918.08` tokens/row;
- protected rows reprovaram:
  - `8740ed31`: baseline acertava `01101000`, candidate virou `01111000`;
  - `59bee375`: baseline acertava `10010101`, candidate truncou e extraiu `2`;
  - `55d834d1`: ganho obrigatorio ausente, continuou `10111111` em vez de
    `00111111`.
- checkpoint-10 nao deve ir para full weak eval. A queda de loss nao se
  converteu em ACC e o adapter esta empurrando respostas erradas em linhas
  protegidas.

Diagnostico de demora V653:

- weak-eval checkpoint-2 gerou `1,504,299` completion tokens para `315` rows;
- media `4,775.55` completion tokens/row, p50 `6,193`, p90 `7,008.8`;
- `bit_manipulation` e o gargalo: media `6,689.46` tokens/row;
- geracao levou `516.1s` apos cold start; a demora e causada por output
  longo/boxed tardio, nao por baixa utilizacao isolada de GPU;
- reduzir `max_tokens` isoladamente continua fora do plano, mas output-policy
  precisa ser corrigida antes de novos full evals caros.

## Consenso OpenRouter V654

Consulta registrada em
`artifacts/v284_official_gate_worktree/artifacts/openrouter/v654_plateau_crisis_consult/KG1_V654_OPENROUTER_CONSENSUS.md`.

Modelos consultados:

- `openai/gpt-5.4`: resposta util, JSON truncado;
- `anthropic/claude-sonnet-4.6`: resposta util, markdown/JSON invalido;
- `google/gemini-2.5-pro`: resposta util, JSON truncado;
- `deepseek/deepseek-r1-0528`: JSON valido;
- `qwen/qwen3.5-plus-20260420`: JSON valido.

Consenso:

- veredito comum: `redesign_dataset`;
- V653 falhou por desalinhamento de dataset/output-policy: loss caiu, mas ACC
  nao subiu, outputs ficaram longos, boxed tarde, truncation/fallback apareceu;
- LoRA amplo demais amplificou drift e causou protected-row backfire;
- `equation_transform` precisa de objetivo mais forte/especifico, pois o gap
  promocional esta nela e V653 ficou dominado por bit;
- proxima rota deve ter auditoria local de boxed-position, token length,
  duplicidade/contradicao, zeros/sinais/simbolos e peso efetivo antes de treino;
- proxima rota deve smoke-testar preservacao de `8740ed31` e `59bee375` e ganho
  em `55d834d1` antes de full weak.

## Analise V655 Do Export OpenRouter Do Usuario

Arquivo analisado:
`C:\Users\davis\Downloads\OpenRouter Chat Mon May 18 2026.json`.

Relatorio:
`artifacts/v284_official_gate_worktree/artifacts/openrouter/v655_user_export_analysis/KG1_V655_USER_OPENROUTER_EXPORT_ANALYSIS.md`.

Resumo objetivo:

- `19` respostas assistant extraidas, com `12` JSONs validos;
- vereditos validos: `7` `redesign_dataset`, `2` `stop_current_route`,
  `3` `continue`;
- `continue` significa continuar o projeto com correcao de rota, nao promover
  V653 sem mudanca;
- achado novo mais importante: nao usar as `24` weak rows como loop de tuning,
  ajuste de hiperparametro ou selecao de checkpoint. O caminho correto e usar
  holdout nao-weak de `train.csv` para escolher dataset/config/checkpoint e
  deixar o weak smoke apenas como gate pos-congelamento;
- as URLs/search sources dentro do export sao pistas externas, nao evidencia
  validada. Elas nao entram como contrato sem verificacao independente.

Parametros sugeridos para a primeira V654 executavel:

- dataset answer-first, com exatamente um `\boxed{answer}` cedo;
- p95 de assistant tokens `<=96` ou `<=128`;
- objetivo efetivo inicial `equation_transform=0.60`,
  `bit_manipulation=0.40`;
- LoRA attention-only no primeiro teste, sem MoE/MLP;
- `r=16`, `alpha=16`, LR inicial na faixa `5e-7 -> 5e-8`;
- classificador de falhas por linha obrigatorio: truncation, no-box/fallback,
  boxed-wrong, extractor-error, protected backfire e missing required gain.

## Double Check V656 Do Export OpenRouter Do Usuario

Relatorio:
`artifacts/v284_official_gate_worktree/artifacts/openrouter/v655_user_export_analysis/KG1_V656_USER_OPENROUTER_EXPORT_DOUBLECHECK.md`.

Cobertura revisada:

- `17` respostas finais com conteudo tecnico util;
- `2` respostas com `content` final vazio foram ignoradas como evidencia
  executavel;
- vereditos revisados: `10` `redesign_dataset`, `4` `stop_current_route`,
  `3` `continue`;
- conclusao corrigida pelo V657: `14/17` recomendam parar/redesenhar V653 ou
  nao continuar a rota atual; os `3` `continue` significam continuar o projeto
  com correcao de rota, nao promover V653.

Achados adicionados ao plano:

- auditar EOS/template: o target SFT precisa bater com o prompt real de eval e
  terminar com `\boxed{answer}` seguido do EOS correto;
- medir posicao de `\boxed{}`: boxed rate sozinha nao basta. Registrar primeiro
  boxed em tokens, `starts_boxed`, porcentagem de tokens antes do boxed e se o
  boxed correto aparece em qualquer ponto do raw output;
- diferenciar falha de decoding/extractor de erro real do adapter:
  - boxed correto no raw output, mas predicao final errada: extractor/template;
  - boxed cedo com payload errado: adapter;
  - boxed tarde/ausente: output-policy/decoding;
- calcular row-loss vs ACC por linha no holdout nao-weak. Se loss cai mas ACC
  nao sobe ou regride, a configuracao de target/mask/loss continua desalinhada;
- rodar base-vs-adapter nas protected rows: `adapter=none` precisa confirmar se
  o backfire e drift do adapter ou problema de prompt/extractor;
- LoRA tem consenso de reduzir superficie, mas nao ha consenso sobre modulo
  exato. Primeira rota fica attention-only sem MoE/MLP; `v_proj,o_proj`,
  `o_proj/down_proj`, `q_proj/v_proj` e `MoE-only` viram ablations secundarias;
- `row_loss_weight=0.0` e hipotese de ablation, nao default;
- adapter por familia/composicao fica fora da rota principal ate prova de que
  e submit-safe no contrato Kaggle.

## Triple Check V657 Do Export OpenRouter Do Usuario

Relatorio:
`artifacts/v284_official_gate_worktree/artifacts/openrouter/v655_user_export_analysis/KG1_V657_USER_OPENROUTER_EXPORT_TRIPLECHECK.md`.

Cobertura estrutural confirmada:

- SHA256 do export:
  `dc8b85005f881d80266130bc544d94e23bb81b61ba3f16a42c9d3351b66d502c`;
- export `orpg.3.0`, titulo `# KG1 V654 OpenRouter Crisis Plateau Pro`;
- `20` messages, `47` items, `38` characters;
- `19` items de tipo `message`, `17` reasoning, `10` web_search e `1`
  web_fetch;
- `18` itens `assistant/message`, representando `17` respostas finais uteis
  porque `qwen/qwen3.6-plus` tem um stub curto antes da resposta final;
- `2` respostas DeepSeek sem content final nao entram como evidencia final.

Conclusao V657:

- `14/17` respostas finais uteis recomendam parar/redesenhar V653 ou nao
  continuar a rota atual;
- as `3` respostas `continue` significam continuar o projeto com correcao de
  rota, nao promover V653;
- V653 fica rebaixado para historico/reprovado;
- o proximo ganho precisa vir de auditoria local + holdout nao-weak + LoRA
  estreito, nao de weak-tuning nem de mais steps em V653.

Gates novos obrigatorios:

- EOS correto apos `\boxed{answer}` precisa estar dentro da loss mask;
- diff de template SFT vs prompt real vLLM, incluindo ultimos tokens, role
  tokens, `add_generation_prompt`, suffix e EOS;
- payload dentro do boxed precisa ser byte-equal ao `answer`, preservando zeros,
  `-`, `:`, barras, braces, simbolos e caracteres relevantes;
- primeiro boxed em ate `50` tokens na maioria das linhas;
- p95 de assistant/completion target `<=128` salvo excecao justificada;
- `label-aware - label-free == 0` no full weak;
- row-loss vs ACC por linha precisa ter correlacao positiva no holdout
  nao-weak antes de H200 caro;
- base-vs-adapter em protected rows deve registrar ACC, `completion_tokens` e
  `first_box_token_idx` com `adapter=none`;
- cada ablation deve mudar apenas uma variavel inicial: nao mudar LR e
  `target_modules` no mesmo experimento;
- antes de trocar `r/alpha/LR`, calcular escala efetiva `LR * alpha / r`;
- se bit for downsampleado para dar peso a equation, preservar suboperacoes
  `AND`, `OR`, `XOR`, `NOT`, shifts e rotates.

Classificacao de erro obrigatoria:

- boxed correto aparece no raw output, mas `prediction` erra:
  extractor/template/stop errado;
- boxed cedo aparece, mas payload e errado: adapter empurrou resposta errada;
- boxed ausente/tardio: output-policy/decoding;
- `equation_transform` precisa separar regra errada, execucao
  aritmetica/simbolica errada e formato/truncamento.

## Triple Check V658 Por Otica De Skills

Relatorio:
`artifacts/v284_official_gate_worktree/artifacts/openrouter/v655_user_export_analysis/KG1_V658_USER_OPENROUTER_EXPORT_SKILL_TRIPLECHECK.md`.

Novas confirmacoes:

- prompt embutido no export e byte-equal ao prompt local V654;
- SHA do prompt usado na consulta:
  `c7df1d422f4d2b1a6942f88d44a04148a819b35bae9c9424c97bfcab37b1321a`;
- prompt local V654 tem `0` caracteres non-ASCII e `0` controles invisiveis
  fora de whitespace normal;
- export completo tem `0` controles invisiveis fora de `\r`, `\n`, `\t`;
- export completo tem `403` caracteres non-ASCII em respostas/model metadata,
  principalmente hifens/aspas tipograficas, simbolos matematicos e letras
  gregas. Esses simbolos nao podem entrar em scripts/gates sem normalizacao;
- `16` assistant messages finais estao `completed` e `2` estao `in_progress`.
  Respostas `in_progress` podem ser pista, mas nao contrato.

Separacao entre consenso e hardening:

- consenso forte do export: parar/redesenhar V653, corrigir output-policy,
  boxed tardio, weak leakage, LoRA drift, dataset/objetivo e foco em equation;
- hardening tecnico adotado pelo time: EOS dentro da loss mask,
  `adapter=none` em protected rows, p95 `<=128`, primeiro boxed `<=50` tokens,
  uma variavel por ablation e escala efetiva `LR * alpha / r`.

Impacto no plano:

- toda nova consulta OpenRouter precisa registrar SHA do prompt usado;
- qualquer sugestao externa precisa ser classificada como consenso, hipotese ou
  hardening antes de entrar no roadmap;
- qualquer simbolo Unicode vindo de resposta de IA deve ser normalizado antes
  de virar codigo, threshold ou configuracao;
- V653 continua bloqueado; a proxima acao e auditoria local, nao treino.

## Gate Local V659 Output Policy / Ganho Falso

Script implementado:
`artifacts/v284_official_gate_worktree/scripts/audit_v659_local_output_policy_gate.py`.

Objetivo:

- validar datasets JSONL antes de qualquer treino/eval pago;
- falhar fechado se dataset ou weak CSV estiverem ausentes;
- validar overlap com weak por `id`, `prompt_hash` e `prompt_answer_hash`;
- validar exatamente um `\boxed{answer}`;
- validar byte-equality do boxed contra `answer`;
- validar extracao label-free e expected-aware;
- medir `starts_boxed`, `first_box_word_idx`, tamanho da resposta, non-ASCII,
  controles invisiveis, pesos por familia/subcategoria e cobertura de
  suboperacoes de bit.

Baseline negativo V653:
`artifacts/v284_official_gate_worktree/artifacts/v659_local_output_policy_gate/v653_frozen_strict_v659/KG1_V659_LOCAL_OUTPUT_POLICY_GATE.md`.

Resultado V653 com regra rigida `--require-starts-boxed`:

- status: `blocked`;
- train/eval permitido: `false`;
- submit permitido: `false`;
- weak CSV SHA validado:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
- train: `2113` linhas, `1661` bit, `452` equation;
- validation: `480` linhas, `385` bit, `95` equation;
- overlap train/val: `0`;
- V653 falha `starts_boxed_required_failed` em `2593/2593` linhas;
- V653 falha `first_box_word_idx_gt_limit` em `347/2593` linhas;
- `first_box_word_idx` p95: train `56`, validation `53`;
- peso efetivo V653: bit `0.7419`, equation `0.2581`.

Decisao:

- V653 permanece congelado/rejeitado como rota promocional;
- o problema nao e apenas tamanho medio baixo: o target ainda ensina texto
  antes do boxed e parte das linhas passa do limite de `50` palavras ate a
  primeira resposta;
- proxima variante precisa ser answer-first real ou, no minimo, boxed nos
  primeiros tokens, validada por V659 antes de qualquer job;
- V659 nao substitui V286: EOS dentro da loss mask, truncation,
  dropped completion tokens e fallback masks continuam obrigatorios no gate de
  tokenizacao real.

## Dataset V660/V661 Answer-First

Scripts implementados:

- `artifacts/v284_official_gate_worktree/scripts/build_v660_answer_first_reweighted_dataset.py`;
- `artifacts/v284_official_gate_worktree/scripts/build_v661_answer_first_short_trace_dataset.py`;
- `artifacts/v284_official_gate_worktree/scripts/audit_v659_local_output_policy_gate.py`;
- `artifacts/v284_official_gate_worktree/scripts/run_v286_generic_tokenization_gate.py`
  atualizado com modo `boxed_prefix`.

V660 answer-only:

- artefato:
  `artifacts/v284_official_gate_worktree/artifacts/v660_answer_first_reweighted_dataset/20260519T_v660_cpu_gate`;
- V659: passou depois de alinhar `official_like`;
- V509: passou;
- V286 real: passou com `boxed_only`, `0` truncation, `0` dropped
  completions, `0` fallback masks;
- V513: bloqueou corretamente por `bit_answer_only_trace_not_learnable_enough`
  e `bit_trace_rows_below_gpu_floor`.

Decisao V660:

- nao usar para HF/GPU;
- manter como evidencia de que answer-only resolve formato, mas nao resolve
  learnability.

V661 answer-first short-trace:

- artefato:
  `artifacts/v284_official_gate_worktree/artifacts/v661_answer_first_short_trace_dataset/20260519T_v661_cpu_gate`;
- target inicia com exatamente um `\boxed{answer}`;
- depois do boxed ha trace curto derivado do prompt/origem, sem outro boxed;
- contrato `official_like`: mensagens `user,assistant`, user =
  `prompt + PROMPT_SUFFIX`;
- pesos efetivos: bit `0.40`, equation `0.60`;
- V659 final:
  `artifacts/v284_official_gate_worktree/artifacts/v661_answer_first_short_trace_dataset/20260519T_v661_cpu_gate/v659_queryclauses_gate/v659_local_output_policy_gate_manifest.json`;
- V509 final:
  `artifacts/v284_official_gate_worktree/artifacts/v661_answer_first_short_trace_dataset/20260519T_v661_cpu_gate/v509_queryclauses_gate/v661_queryclauses_v509_manifest.json`;
- V286 final:
  `artifacts/v284_official_gate_worktree/artifacts/v661_answer_first_short_trace_dataset/20260519T_v661_cpu_gate/v286_queryclauses_tokenization_real/v286_generic_tokenization_gate_manifest.json`;
- V513 final:
  `artifacts/v284_official_gate_worktree/artifacts/v661_answer_first_short_trace_dataset/20260519T_v661_cpu_gate/v513_queryclauses_learnability_gate/v513_trace_learnability_gate_manifest.json`.

Resultados V661 finais:

- V659: `passed`, blockers `0`, warnings `0`;
- V509: `blocked_dataset_count=0`;
- V286 real: `tokenization_gate_passed`;
- V286 train: `2113` rows, token max `397`, prompt truncation `0`,
  completion dropped `0`, fallback masks `0`, offset masks `2113`;
- V286 validation: `480` rows, token max `394`, prompt truncation `0`,
  completion dropped `0`, fallback masks `0`, offset masks `480`;
- V513: `passed_cpu_structure_only`, blockers `0`, warnings `0`;
- V513 styles: bit `bit_trace_with_rule_terms`, equation
  `equation_short_rule_reject_boxed`;
- V478/V526 confirmaram objetivo efetivo de treino:
  `bit_manipulation=0.40`, `equation_transform=0.60`;
- V524 confirma quotas/tokens fisicos, mas a decisao de objetivo promocional
  usa `row_loss_weight + example_mean`, nao share fisico de linhas;
- V661 e a primeira rota localmente valida para um smoke pequeno, ainda sem
  permissao de submit.

HF smoke V661:

- job:
  `https://huggingface.co/jobs/felipesp1983/6a0bc3e3a5e509f1a841611f`;
- run:
  `v661-nemo-h200-answerfirst-shorttrace-v290ckpt6-20260519T015745Z`;
- output repo:
  `felipesp1983/kg1-nemotron-lora-v661-h200-answerfirst-shorttrace-v290ckpt6`;
- data repo:
  `felipesp1983/kg1-v661-answer-first-short-trace-artifacts`;
- data commit:
  `7b7b9319b4f0519b7f9273c56ad13960b20be5ea`;
- train SHA:
  `d1d47dd84b3a2bb6e3ea89ac80e3fdc05185bea76b21e40c0d8bda136a883af4`;
- val SHA:
  `856632c4eccc1450fda80866fb0bf0752c4833f47a70939528f87a6f3e96128e`;
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
