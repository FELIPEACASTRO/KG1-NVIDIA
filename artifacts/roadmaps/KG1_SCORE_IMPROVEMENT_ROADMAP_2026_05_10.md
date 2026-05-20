# KG1 Score Improvement Roadmap

Atualizado: 2026-05-20 23:10 UTC. Fonte de verdade operacional:
`artifacts/roadmaps/active/KG1_ACTIVE_ROADMAP_2026_05_20.md`.

Atualizacao V713, 2026-05-20 23:17 UTC: consulta OpenRouter/Hugging Face
pos-falha V712 concluida com `5/5` respostas OpenRouter e `6/6` metadados HF.
Consenso local aceito:
`artifacts/openrouter/v713_v712_import_failure_consult/KG1_V713_CONSENSUS_AND_LOCAL_DECISION.md`.
Decisao: primeiro fazer deploy/push do import fix, depois rodar weak-eval-only
A100 do checkpoint ja existente `v712 checkpoint-10`; nao rerodar treino.
Rejeitado para promocao: `KG1_MAX_TOKENS=1024`, remover
`KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=512`, novo treino, H200, package,
full eval, Kaggle submit ou qualquer relaxamento de thresholds antes do weak ACC
label-free/protected-row existir.

Atualizacao V712 falha de weak eval, 2026-05-20 23:10 UTC: o treino A100
terminou, mas o job falhou antes de produzir ACC por
`ModuleNotFoundError: No module named 'scripts.evaluate_lora_adapter'` no
`scripts/evaluate_lora_adapters_batch.py`. Diagnostico aceito: bug silencioso de
packaging/import remoto, pois `scripts/` nao era pacote regular e podia colidir
com pacote terceiro chamado `scripts`. Correcao aplicada: `scripts/__init__.py`
no workspace raiz e no worktree oficial, launcher V712 agora compila
`scripts/evaluate_lora_adapter.py` e roda `scripts_package_gate` +
`weak_eval_import_gate_ok` antes de treino/eval; static gate e pre-paid gate
passaram a bloquear launchers com weak eval sem esse preflight. Gates apos
correcao:
`artifacts/v712_hf_a100_equation_signal_launch/v712_static_safety_gate_after_importfix.json`
e
`artifacts/v712_hf_a100_equation_signal_launch/v712_pre_paid_job_integration_gate_after_importfix.json`,
ambos `ok=true`. V712 loss: baseline `2.6376`, checkpoint-10 `2.6363` melhor,
checkpoint-20/final `2.6373`; tokenizacao limpa, truncation `0`, fallback masks
`0`, trainable efetivo `q_proj,v_proj`. Decisao: nao rerodar treino. Reusar os
checkpoints ja enviados em weak-eval-only A100, primeiro `checkpoint-10`, depois
`checkpoint-20` somente se fizer sentido. Sem package/full eval/Kaggle submit ate
weak ACC/backfire passar os thresholds oficiais internos.

Atualizacao V712, 2026-05-20 22:31 UTC: V712 A100-large foi lancado apos
passar sintaxe, static safety, manifest-only, HF dataset validation e pre-paid
integration gate. Job:
`https://huggingface.co/jobs/felipesp1983/6a0e3565ac8efd7fbbb2aa06`.
Launcher:
`artifacts/v712_hf_a100_equation_signal_launch/launch_v712_hf_a100_equation_signal.py`.
Manifest:
`artifacts/v712_hf_a100_equation_signal_launch/v712-a100-equation-signal-v290ckpt6-20260520T222638Z_launch_manifest.json`.
Gates:
`artifacts/v712_hf_a100_equation_signal_launch/v712_static_safety_gate.json`
e
`artifacts/v712_hf_a100_equation_signal_launch/v712_pre_paid_job_integration_gate.json`,
ambos sem findings. V712 usa A100-only, sem H200, dataset equation-only V708
train `852`/val `195`, `max_steps=20`, `save_every_steps=10`,
`eval_every_steps=10`, LR `2e-6 -> 5e-7`, LoRA `r=32 alpha=32`,
trainavel efetivo `q_proj,v_proj`, MoE target parameters congelados, `lm_head`
ausente, weak eval obrigatoria em `checkpoint-20`. A proposta inicial de `50`
steps foi reduzida para `20` para respeitar o gate de drift/backfire
`deferred_post_checkpoint` sem enfraquecer regras. Nao promover, empacotar,
rodar full eval ou submeter ate a weak eval do proprio job passar thresholds
inalterados: total `>=196`, bit `>=136`, equation `>=60`, truncation `0`,
boxed rate `1.0`, no-box fallback `0`, protected backfire `0`.

Atualizacao V710, 2026-05-20 22:20 UTC: V710 terminou e foi bloqueado
corretamente pelo weak promotion gate. Nao ha jobs HF ativos agora (`hf jobs ps`:
`No jobs found`). O job era weak-eval-only, A100-large, sem treino, sem full
eval, sem package, sem Kaggle submit e sem H200. URL:
`https://huggingface.co/jobs/felipesp1983/6a0e25bc3fef25139d6c9fd0`.
Resultado real: `191/315`, `bit_manipulation=135/160`,
`equation_transform=56/155`, `truncated=1`, `boxed_rows=314/315`,
`no_box_fallback_rows=1`, avg completion `4775.58`, max completion `7680`.
Blockers: `correct_lt_196`, `equation_lt_60`, `bit_lt_136`,
`truncated_gt_0`, `avg_completion_tokens_gt_512`, `no_box_fallback_gt_0`,
`boxed_rate_lt_1.0`, `protected_row_backfire_guard_failed`. Protected rows:
`8740ed31` e `59bee375` backfire real de baseline correto; `55d834d1` ainda
nao aprendeu ganho obrigatorio. Unico ganho vs V703: `4ada9150`.
Decisao: V710 nao e promotable/submittable; consultar OpenRouter/Hugging Face
com esses fatos antes de qualquer novo gasto. Ledger:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/KG1_V710_FAILURE_ANALYSIS.md`.

Atualizacao V711, 2026-05-20 23:00 UTC: consulta OpenRouter/HF executada e
audit local de parametros concluido. Consenso externo inicial apontou possivel
drift de contrato LoRA, mas o manifesto final V708 provou que a superficie
ativa ampla do `adapter_config` e de carregamento/heranca, enquanto o filtro
treinavel efetivo foi `q_proj,v_proj`: `1,867,776` LoRA params treinaveis,
`882,006,016` LoRA params congelados, MoE target parameters com `0` params
treinaveis, `lm_head` ausente. Gate novo:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/v711_lora_trainability_manifest_gate.json`,
`passed=true`. Portanto, a hipotese "treinou todos os modulos" foi resolvida
como falsa suspeita para V708. Causa mais provavel agora: sinal de treino fraco
(`5` steps, batch `2`, LR `5e-7 -> 1e-7`, loss `+0.0018`) combinado com
decoding runaway/protected backfire (`59bee375` truncado/no-box; `8740ed31` e
`55d834d1` boxed mas raciocinaram para bit errado). Artefatos:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/KG1_V711_PARAMETER_AUDIT.md`
e
`artifacts/openrouter/v711_v710_failure_consult/KG1_V711_CONSENSUS_AND_LOCAL_DECISION.md`.

Atualizacao V708, 2026-05-20 19:26 UTC: concluido quadruple check/falso ganho
com as oticas senior ML engineer, senior data engineer, senior data scientist e
seguranca operacional. Artefato:
`artifacts/v708_equation_single_family_dataset/20260520T_v708_cpu_gate/KG1_V708_QUADRUPLE_FALSE_GAIN_AUDIT.md`.
V708 e o dataset CPU ativo de uma familia: train `852`, validation `195`, todos
`equation_transform`; `bit_manipulation` fica apenas como guarda de nao
regressao. Hashes atuais: train
`a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d`, validation
`f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba`, manifest
`1840bb56c251b64661ec671eb84602e068c855d00c5699f00cc2ce6da47d2101`. Correcoes:
sufixo official-like agora falha fechado se aparecer fora do fim, familia/tarefa
original e preservada antes da normalizacao, V478 nao pode mais autorizar GPU no
modo objective-only, e os defaults de protected rows agora incluem
`8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111`. Gates passaram:
V509 train/val, V286 tokenization, loss/EOS, rule-text, label-free/control,
V478 objective-only com `hf_gpu_allowed=false`, e static safety. Gate composto:
`artifacts/v708_equation_single_family_dataset/20260520T_v708_cpu_gate/v708_pre_a100_readiness_gate.json`.
Resultado: `cpu_dataset_ready=true`, `a100_launch_allowed=false`; bloqueador
restante e intencional: `missing_v708_a100_launch_manifest`. Portanto, dataset
pronto nao autoriza GPU. Proximo passo limpo: criar manifesto de launch V708
A100-only com `disable_thinking=0`, `max_tokens=7680`, protected rows, thresholds
`total>=196`, `bit>=136`, `equation>=60`, sem H200, sem submit/package/full eval.

Historico: V703 weak eval de V689 `checkpoint-5`
em `a100-large` terminou e foi bloqueado pelo gate:
`https://huggingface.co/jobs/felipesp1983/6a0de4798229e585f969c787`.
Resultado real: `190/315`, `bit_manipulation=134/160`,
`equation_transform=56/155`, `truncated=1`,
`boxed_rate=0.9968253968253968`, `no_box_fallback_rows=1`,
avg completion `4775.320634920635`, max completion `7680`,
protected-row backfire em `8740ed31` e `59bee375`, missing required gain em
`55d834d1`. Baseline-relative contra V516: net `-1`, bit `-2`,
equation `+1`. Decisao: V689 checkpoint-5 esta bloqueado para promocao,
pacote, submissao e continuacao cega. Ledger:
`artifacts/v703_v689_weak_eval_failure/KG1_V703_V689_CHECKPOINT5_WEAK_EVAL_FAILURE.md`.
Prompt de consulta:
`artifacts/openrouter/v703_v689_checkpoint5_failure_consult/KG1_V703_OPENROUTER_PROMPT.md`.
Consulta externa e trilha Hugging Face concluidas:
`artifacts/openrouter/v703_v689_checkpoint5_failure_consult/KG1_V703_CONSENSUS_AND_LOCAL_DECISION.md`
e `artifacts/v703_v689_weak_eval_failure/KG1_V703_HF_METADATA_TRACK.md`.
Segunda checagem OpenRouter pedida pelo usuario adicionou mais 5 modelos:
Claude Opus 4.7, Gemini 3.1 Pro Preview, GPT-5.5, GPT-OSS-120B free e
GLM-4.5. Ledger das respostas:
`artifacts/openrouter/v703_v689_checkpoint5_failure_consult/KG1_V703_OPENROUTER_RESPONSES.md`.
Decisao aceita: `INVESTIGATE_DECODING_FIRST` com `STOP_BROAD_LORA` aplicado.
Sintese local dos 7 modelos: Qwen, Claude, GPT-OSS, GLM e a parte aceita de
Gemini apontam para diagnostico de decoding/contrato primeiro; DeepSeek bloqueia
LoRA amplo; GPT-5.5 sugere package solver, mas isso fica diferido ate existir
gate proprio de pacote e nao pode contornar weak eval. Caminho ativo: isolar
`raw_output -> extract -> verify_answer`, truncation/no-box e protected-row
backfire antes de qualquer novo treino ou promocao.
Proximo passo obrigatorio: preflight local para V704 bounded decode eval-only
em A100, com `max_tokens=512`, antes de qualquer novo gasto GPU. Esse proximo
job, se lancado, e diagnostico-only; nao promove nem submete.

Atualizacao V704/V705, 2026-05-20 17:45 UTC: implementado e debugado o
launcher V704 bounded-decode eval-only:
`artifacts/v704_hf_a100_bounded_decode_eval/launch_v704_hf_a100_bounded_decode_eval.py`.
Manifest de debug:
`artifacts/v704_hf_a100_bounded_decode_eval/v704-a100-bounded-decode-ckpt5-20260520T173237Z_bounded_decode_launch_manifest.json`.
Decisao de preflight:
`artifacts/v704_hf_a100_bounded_decode_eval/KG1_V704_PREFLIGHT_DECISION.md`.
V704 e A100-only, `diagnostic_only=1`, `max_tokens=512`, mesmo checkpoint-5,
mesmo sufixo official-like `\boxed{}`, sem promocao, package, Kaggle submit,
full eval ou H200 fallback. Replay offline do V703 reproduziu correctness:
`190/315`, `correctness_mismatches=0`, `label_aware_delta=0`; ha `9`
diferencas de string em linhas equation ja erradas, rastreadas como warning de
simbolos/extracao.
V704 A100 diagnostic lancado:
`https://huggingface.co/jobs/felipesp1983/6a0df33eccb6cd133d158c8d`.
Run id: `v704-a100-bounded-decode-ckpt5-20260520T174426Z`. Manifest:
`artifacts/v704_hf_a100_bounded_decode_eval/v704-a100-bounded-decode-ckpt5-20260520T174426Z_bounded_decode_launch_manifest.json`.
Este job continua diagnostico-only e nao pode promover, empacotar ou submeter.

Atualizacao V706, 2026-05-20 18:10 UTC: V704 terminou e foi bloqueado pelo
catastrophic eval guard. Resultado real: `3/315`,
`bit_manipulation=0/160`, `equation_transform=3/155`,
`truncated=288/315`, `boxed_rate=0.08571428571428572`,
`no_box_fallback_rows=288`, avg completion `509.84761904761905`, max
completion `512`. Diagnostico: o limite `max_tokens=512` com
`disable_thinking=false` cortou o raciocinio antes do `\boxed{}`; isso e
falha de decoding/runtime, nao sinal de ganho/perda do adapter. Artefatos
baixados para:
`artifacts/v704_hf_a100_bounded_decode_eval/downloaded_job_artifacts/`.
Consulta OpenRouter/HF:
`artifacts/openrouter/v706_v704_catastrophic_decode_failure/KG1_V706_CONSENSUS_AND_DECISION.md`.
Novo launcher V706:
`artifacts/v706_hf_a100_disable_thinking_decode_eval/launch_v706_hf_a100_disable_thinking_decode_eval.py`.
Gate local de renderizacao:
`artifacts/v706_hf_a100_disable_thinking_decode_eval/KG1_V706_PROMPT_RENDER_GATE.md`.
V706 e o unico proximo diagnostico pago permitido: A100-only, eval-only,
mesmo weak CSV, mesmo checkpoint-5, mesmo sufixo official-like, mas com
`KG1_DISABLE_THINKING=1` e `KG1_REQUIRE_DISABLE_THINKING=1`. Continua sem
promocao, package, full eval, Kaggle submit ou H200. Se V706 nao restaurar
`truncated=0`, `boxed_rate=1.0`, `bit>=136`, `equation>=56` e `total>=191`,
parar GPU e consultar OpenRouter/HF novamente antes de qualquer novo gasto.
V706 lancado:
`https://huggingface.co/jobs/felipesp1983/6a0df97a8229e585f969ca75`.
Run id: `v706-a100-disable-thinking-ckpt5-20260520T181102Z`. Manifest:
`artifacts/v706_hf_a100_disable_thinking_decode_eval/v706-a100-disable-thinking-ckpt5-20260520T181102Z_disable_thinking_launch_manifest.json`.
V706 terminou. Resultado: `17/315`, `bit_manipulation=7/160`,
`equation_transform=10/155`, `truncated=0`, `boxed_rate=1.0`,
`no_box_fallback_rows=0`, `avg_completion_tokens=11.263492063492064`,
`max_completion_tokens=14`, e `175` backfires contra linhas corretas do
baseline V516. Protected-row guard falhou em `8740ed31`, `59bee375` e
`55d834d1`. Conclusao: `disable_thinking=1` corrige formato/termination, mas
destroi raciocinio e esta bloqueado como rota de score, package, full eval ou
submit.

Atualizacao V707, 2026-05-20 18:50 UTC: consulta OpenRouter/HF pos-V706
concluida. Prompt:
`artifacts/openrouter/v707_v706_disable_thinking_failure_consult/KG1_V707_OPENROUTER_PROMPT.md`.
Consenso:
`artifacts/openrouter/v707_v706_disable_thinking_failure_consult/KG1_V707_CONSENSUS_AND_DECISION.md`.
Respostas brutas:
`artifacts/openrouter/v707_v706_disable_thinking_failure_consult/openrouter_results.jsonl`
e retry Gemini:
`artifacts/openrouter/v707_v706_disable_thinking_failure_consult/gemini_retry/openrouter_results.jsonl`.
Decisao aceita: `CPU_FIRST_THEN_ONE_A100_PROBE_IF_GATED`. O proximo passo nao
e GPU: construir residual CPU V516/V703/V706 e microdataset
`equation_transform` minimo. Arquivos:
`artifacts/v707_v706_disable_thinking_failure/KG1_V707_V706_DISABLE_THINKING_FAILURE_ANALYSIS.md`,
`artifacts/v707_v706_disable_thinking_failure/v707_v516_v703_v706_row_residual.csv`
e `artifacts/v707_v706_disable_thinking_failure/v707_residual_summary.json`.
V707 row diff: V703 teve `1` ganho equation (`4bb8c6cd`) e `2` backfires bit
(`8740ed31`, `59bee375`); V706 teve `1` ganho equation (`ea6d926a`), mas
`175` backfires. Proxima rota aceita: partir do baseline limpo V516/V290
checkpoint-6, nao do V689 checkpoint-5; foco positivo somente em
`equation_transform`; `bit_manipulation` vira replay/guard com piso `136/160`;
production weak eval continua `disable_thinking=0`, `max_tokens=7680`.

Consulta V705 para foco em uma familia concluida. Prompt:
`artifacts/openrouter/v705_single_family_equation_focus/KG1_V705_SINGLE_FAMILY_EQUATION_FOCUS_PROMPT.md`.
Consenso:
`artifacts/openrouter/v705_single_family_equation_focus/KG1_V705_CONSENSUS_AND_ROADMAP_PATCH.md`.
Respostas:
`artifacts/openrouter/v705_single_family_equation_focus/KG1_V705_OPENROUTER_RESPONSES.md`.
Trilha HF:
`artifacts/v705_hf_single_family_metadata_track/KG1_V705_HF_METADATA_TRACK.md`.
Votos OpenRouter: `5/5` para `DECODE_FIRST_THEN_EQUATION`. Plano ativo:
primeiro V704 para estabilizar decoding/bit; depois trabalhar somente
`equation_transform`. `bit_manipulation` fica como guarda de nao regressao
com piso `136/160`, nao como alvo de otimizacao. Ganho necessario depois de
um estado limpo `bit=136`, `equation=56` e `+4` equation liquido para chegar
em `equation=60` e total `196`.

Atualizacao anterior: V686 weak eval `official_like` bounded em
`a100-large` terminou e foi bloqueado pelo gate:
`https://huggingface.co/jobs/felipesp1983/6a0da2df8229e585f969c3e5`.
Resultado real: `190/315`, `bit_manipulation=134/160`,
`equation_transform=56/155`, `truncated=1`, `boxed_rate=0.9968253968253968`,
`no_box_fallback_rows=1`, avg completion `4775.396825396825`, max completion
`7680`, protected-row backfire em `8740ed31` e `59bee375`, missing required
gain em `55d834d1`. Baseline-relative contra V516: net `-1`, bit `-2`,
equation `+1`. Decisao: V684 checkpoint-10 esta bloqueado para promocao,
pacote, submissao e continuacao cega.

Fonte de verdade do plano ativo:
`artifacts/roadmaps/active/KG1_ACTIVE_ROADMAP_2026_05_20.md`.

Atualizacao V696, 2026-05-20 16:10 UTC: foi corrigido um bug real no contrato
de weak-eval do checkpoint-5 V689. O launcher combinava
`KG1_DISABLE_THINKING=0` com `KG1_REQUIRE_DISABLE_THINKING=1`, o que tornaria
o runtime policy inconsistente com o modo official-like. Estado atual:
`KG1_DISABLE_THINKING=0`, `KG1_REQUIRE_DISABLE_THINKING=0`,
`KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=512` e
`KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX=7680`. Foi criado o launcher
eval-only `artifacts/v689_hf_a100_launch/launch_v689_hf_a100_checkpoint5_weak_eval.py`
para avaliar apenas `checkpoint-5` depois do treino. Gates locais passaram:
`artifacts/v689_hf_a100_launch/v696_static_safety_gate.json` e
`artifacts/v689_hf_a100_launch/v696_pre_paid_job_integration_gate.json`.
Analise Gemini incorporada em
`artifacts/openrouter/v696_gemini_artifact_review/KG1_V696_GEMINI_RESPONSE_ANALYSIS.md`:
sem GPU por narrativa; exigir artefatos de loss/mask, extracao, protected rows
e runtime A100. Decisao HF Local Apps: `vLLM` segue como backend ativo,
`SGLang` e apenas diagnostico futuro, demais apps locais fora do plano ativo.

Atualizacao V697, 2026-05-20 16:25 UTC: auditados os dumps locais de
discussion/forum previamente baixados para identificar produtos/modelos de IA
mais citados. Resultado em
`artifacts/v697_kaggle_discussion_ai_products_audit/KG1_V697_KAGGLE_DISCUSSION_AI_PRODUCTS_AUDIT.md`.
Top citacoes: Qwen, NVIDIA/Nemotron, Hugging Face, DeepSeek/R1,
GPT/OpenAI/gpt-oss, OpenRouter, Gemini e vLLM. Impacto no plano: usar essa
lista para escolher modelos de consulta externa; nao mudar backend de execucao.
`vLLM` continua runtime ativo, `SGLang` fica diagnostico-only, apps locais nao
sao caminho ativo para score.

Analise do export OpenRouter + V686:
`artifacts/openrouter/v687_v686_failure_user_export_analysis/KG1_V687_OPENROUTER_EXPORT_AND_V686_FAILURE_ANALYSIS.md`.

Nao ha jobs HF ativos no momento da atualizacao. Proximo passo obrigatorio:
analise CPU-first dos artefatos V686 e consulta OpenRouter pos-V686 com os
artefatos reais antes de qualquer novo gasto de GPU.

Atualizacao V687, 2026-05-20 13:05 UTC: consulta OpenRouter pos-V686 concluida
com DeepSeek/Qwen e CPU drift report criado. Decisao combinada:
`no_gpu_now`, V684 lineage bloqueada, diagnostico CPU primeiro. Arquivos:
`artifacts/openrouter/v687_v686_failure_user_export_analysis/KG1_V687_CONSENSUS.md`
e `artifacts/v687_v686_cpu_drift_report/v686_cpu_drift_report.json`.

Atualizacao V688, 2026-05-20 13:35 UTC: catalogo Hugging Face do usuario
analisado. Nova regra operacional: toda consulta externa critica deve usar
OpenRouter com os melhores modelos adequados ao prompt e, quando houver modelo
relevante vivo/custo-seguro, tambem Hugging Face Router/Inference Provider ou
metadados HF. Qwen, DeepSeek, OpenAI `gpt-oss`, Gemma e NVIDIA Nemotron podem
ser candidatos HF; Claude/Gemini fechados nao sao pesos HF. Respostas externas
sao hipoteses e nao substituem weak ACC, `official_like`, extracao label-free,
guards de protected rows e bloqueio de ganho falso. Auditoria:
`artifacts/v688_huggingface_model_catalog_audit/KG1_V688_HUGGINGFACE_MODEL_CATALOG_AUDIT.md`.

Atualizacao V689, 2026-05-20 13:55 UTC: criado e sanitizado o microdataset
CPU-only para corrigir o bug silencioso V684 de texto de regra. Dataset:
train `360`, validation `90`, todos `equation_transform`, prompt
`official_like`, resposta final `\boxed{}`, sem prompt/answer weak no treino.
IDs fracos foram removidos do metadata linha-a-linha dos JSONL; scan textual
dos JSONL encontrou zero ocorrencias de `274def88`, `7688e06e`, `d1bd7478` e
`c5b058d6`. Gate `kg1_v688_rule_text_consistency_gate.py` passou:
`rows_checked=450`, `blocker_count=0`, `warning_count=0`. V689 nao autoriza GPU
ainda; proximos gates obrigatorios sao tokenization/mask, loss/EOS, objective
alignment, static safety, budget, delta weak esperado e kill-switches.
Relatorio:
`artifacts/v689_rule_text_corrected_microdataset/KG1_V689_RULE_TEXT_CORRECTED_MICRODATASET.md`.

Regra operacional adicionada em 2026-05-20 13:30 UTC: toda falha de
treino/eval/job/gate ou valor inseguro exige prompt detalhado com artefatos
reais e consulta OpenRouter antes de novo gasto. Depois de implementar a
correcao, deve haver nova revisao externa com OpenRouter e uma trilha Hugging
Face quando houver modelo/metadado/provider relevante e custo-seguro. Essa
revisao nao substitui gates KG1 e nao autoriza submit.

Revisao externa V689 executada: DeepSeek/Qwen recomendaram
`PROCEED_CPU_GATES`; `gpt-oss-120b:free` pediu mais artefatos por risco de ID
fraco no manifesto. Acao aceita: remover IDs fracos tambem do manifesto
operacional e deixar apenas arquivo audit-only `DO_NOT_UPLOAD`. Leak audit do
diretorio operacional V689: `scanned_file_count=4`,
`weak_ids_present_in_operational_gate_dir=0`,
`ok=true`. Consenso:
`artifacts/openrouter/v689_external_review/KG1_V689_EXTERNAL_REVIEW_CONSENSUS.md`.

Gates CPU V689 adicionais concluidos: V286 tokenizer real passou sem overlap,
sem truncation, sem fallback masks, sem completion drop, offset masks em todas
as `450` linhas e token max `236`; loss/EOS passou com final loss EOS rate
`1.0` em treino e validacao; V478 objective alignment passou com
`findings=[]`; execution dry-run passou com `450/450` final answers extraidos e
queries recalculadas corretamente, alem de `1000` exemplos same-op verificados;
static safety passou no script alterado. Plano protected/kill-switch V689 criado
como audit-only `DO_NOT_UPLOAD`: `4` rows de ganho esperado, `52` rows
protegidas, incluindo `2` backfires V686. GPU continua bloqueada ate existir
launcher/upload manifest que exclua arquivos `DO_NOT_UPLOAD` e passe no
pre-paid gate.

Atualizacao V690/V691, 2026-05-20 14:00 UTC: a regra de consulta externa apos
falha/correcao foi implementada no script
`scripts/kg1_post_train_openrouter_consult.py`, agora com trilha Hugging Face
metadata/provider alem de OpenRouter. Teste V690: HF metadata `6/6`,
OpenRouter pos-correcao `3/3`, static safety `findings=[]`. Teste V691:
criado `artifacts/v689_hf_a100_launch/upload_v689_dataset_to_hf.py`, default
dry-run/no-upload. A primeira execucao falhou por chave errada no leak audit;
foi corrigida para `weak_ids_present_in_operational_gate_dir`. Pos-correcao:
upload preflight seco passou, runtime `DO_NOT_UPLOAD` path count `0`, weak IDs
no diretorio operacional `0`, OpenRouter `1/1`, HF metadata `6/6`. O dataset
operacional V689 foi enviado para o repo privado HF
`felipesp1983/kg1-v689-rule-text-corrected-microdataset` no commit
`fa5ae6e6524f4ef7f66546f7a9b4f8154a62c0f6`; o download-back hash gate passou
com `ok=true`, `blockers=[]`. Nenhuma GPU foi executada. Plano A100 V689 existe apenas como
`DO_NOT_EXECUTE`:
`artifacts/v689_hf_a100_launch/KG1_V689_A100_LAUNCH_PLAN_DO_NOT_EXECUTE.md`.
Antes de A100 ainda faltam launcher real, static safety, pre-paid integration e
gate de generation/decoding drift.

V684 A100 smoke foi lancado em
`a100-large` e gerou checkpoint util no Hub:
`felipesp1983/kg1-nemotron-lora-v684-a100-trace-source-v290ckpt6/checkpoint-10`.
O job foi cancelado manualmente depois do upload completo do checkpoint-10
para preservar credito; isto nao e falha de treino. Resultado de loss:
baseline `3.3707`, checkpoint-5 `3.3682`, checkpoint-10 `3.3674`.
V485 PEFT roundtrip do checkpoint-10 passou localmente. Ainda nao ha ACC desse
adapter; qualquer promocao exige weak eval canonicalizado com
protected/backfire/format gates.
V681 treinou corretamente no A100 com MoE
`target_parameters` frozen-active, mas o weak eval official-like bloqueou o
checkpoint-15: `191/315`, `bit_manipulation=135/160`,
`equation_transform=56/155`, `truncated=1`, `boxed_rate=314/315`,
`no_box_fallback=1`, avg completion `4775.39` e protected backfire em
`8740ed31`/`59bee375` mais missing required gain em `55d834d1`. Portanto V681
esta arquivado como nao-promovivel. V682 no-thinking/max384 tambem terminou e
foi bloqueado: `17/315`, `bit_manipulation=7/160`,
`equation_transform=10/155`, `truncated=0`, `boxed_rate=1.0`,
avg completion `11.29`, mas protected backfire em `8740ed31`/`59bee375` e
missing required gain em `55d834d1`. Conclusao: `disable_thinking=1` corrige
comprimento/formato, mas destroi raciocinio e nao e rota de ACC. Plano ativo
agora e CPU-first: V683 consolidou o ledger residual/protected e V684 criou
um dataset source-only trace-preserving para testar a hipotese oposta ao
V682: preservar raciocinio e finalizar com `Final answer: \boxed{...}`. Novo
treino so e permitido se gates locais provarem chance real de ACC sem falso
ganho.
Objetivo continua:
`total>=196/315`, `bit_manipulation>=136/160`,
`equation_transform>=60/155`, sem truncation, sem fallback, sem backfire,
sem pacote/submissao e sem H200.

Plano ativo agora:

1. Usar somente `a100-large`; H200 e rota H200 estao bloqueados por regra
   estatica `h200_forbidden_by_budget_policy` em
   `scripts/kg1_static_safety_gate.py`.
2. V673 esta morto para treino, weak eval, pacote e submissao. Os launchers
   V673 agora falham antes de launch com `ROUTE_BLOCKED_AFTER_V676_REASON`.
   Os `remote_command.sh` V673 foram removidos para nao simularem rota ativa.
3. V588/V582 interpolation probe tambem esta arquivado/fail-closed. V582 segue
   na blocked adapter list e nao pode alimentar probe, pacote, weak eval ou
   submit.
4. V680 esta arquivado como nao-promovivel. Dataset V680 boxed-only:
   `artifacts/v680_v677_synth_eq_augmented_dataset/20260519T_v680_cpu_gate/`.
   Ele deriva do V677 corrigido e adiciona 180 linhas `equation_transform`
   sinteticas verificadas, sem alterar a validation. A auditoria V681 mostrou
   que os targets eram curtos e corretos, mas o loss efetivo ficou
   `equation=0.887097` contra `bit=0.112903`.
5. V681 esta arquivado como nao-promovivel. O balanceamento 50/50 de loss
   efetivo corrigiu o erro objetivo de V680, mas nao converteu em ACC e ainda
   gerou drift longo/backfire. Nao relancar, continuar ou subir checkpoint-15.
   A correcao de frozen-active MoE permanece como hardening obrigatorio para
   qualquer rota futura.
6. V682/V706 no-thinking estao arquivados como diagnosticos negativos:
   formato ficou perfeito (`boxed_rate=1.0`, `truncated=0`,
   completion tokens curtos), mas ACC caiu para cerca de `17/315`. Bloquear
   qualquer plano que tente resolver plateau apenas com `disable_thinking=1`
   ou cap curto de tokens. Se usar cap/decoding no futuro, deve preservar
   thinking/raciocinio e passar protected rows antes de GPU.
7. Promover qualquer rota futura apenas se `total>=196`,
   `bit_manipulation>=136`, `equation_transform>=60`, `truncated=0`,
   `boxed_rate=1.0`, `label_aware_delta=0`, `no_box_fallback=0`, sem
   protected-row backfire e sem queda baseline-relative contra `196/136/60`.
8. Todo final de treino ou falha de weak eval exige consulta externa post-train
   com prompt completo e artefatos reais antes de novo gasto. Usar OpenRouter
   com os melhores modelos adequados ao prompt e tambem Hugging Face quando
   houver modelo relevante disponivel/custo-seguro. A
   consulta V682 pos-falha V681 foi executada com
   `deepseek/deepseek-v4-pro` e `qwen/qwen3.6-max-preview`; consenso:
   `artifacts/openrouter/v682_v681_weak_eval_failure_consult/KG1_V682_CONSENSUS.md`.
   Consulta V685 pos-V684 executada com GPT-5.5, Gemini, Qwen, DeepSeek e
   Claude; consenso:
   `artifacts/openrouter/v685_post_v684_loop_consult/KG1_V685_CONSENSUS.md`.
   Regra operacional consolidada: `execute -> measure -> diagnose ->
   OpenRouter/HuggingFace -> roadmap -> next bounded step`. Toda falha de
   treino, weak eval, loss/ACC, formato, truncation, protected row ou gate
   precisa gerar novo prompt com artefatos reais antes de novo gasto.
9. Consulta V683 pos-falha V682 executada com `deepseek/deepseek-v4-pro` e
   `qwen/qwen3.6-max-preview`; consenso consolidado:
   `artifacts/openrouter/v683_v682_nothink_failure_consult/KG1_V683_CONSENSUS.md`.
   Decisao: bloquear no-thinking/short-cap, bloquear V681/V682, e exigir
   ledger CPU de misses + microdataset source-only verificado + extractor
   parity + protected safety antes de qualquer A100.
   Ledger V683 concluido em
   `artifacts/v683_current_residual_ledger/`: `38` linhas, decisao
   `cpu_only_no_gpu`, `12` source-only trainable, `2` source-only guarded,
   `14` needs-rule-proof, `7` drop e `3` protected guard. Regra: esses IDs
   weak sao apenas seletores de regra/proxy, nunca linhas de treino diretas.
   V684 implementou a variante CPU trace-preserving:
   `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/`.
   Dataset: train `640` (`320` bit, `320` equation), validation `160`
   (`80`/`80`), `loss_weight=1.0`, effective share `50/50`, trace lines
   `min=5`, `p50=8`, `max=10`. Gates CPU V684 passaram: V679
   `assistant_contract=boxed_suffix`, V286 real `prompt_truncated=0`,
   `completion_tokens_dropped=0`, `fallback_masks=0`, token max `332`,
   loss/EOS `ok=true`, V478 strict `findings=[]`, V513
   `blocker=0/warning=0`, V564 contract/mask alignment com
   `protected_mode=absent_required` passou (`mask_ok=640/160`,
   protected IDs ausentes por desenho source-only), static safety e workspace
   clean sem findings.
   V540 extractor parity inicialmente detectou artefato historico stale
   (`prediction/correct` antigo no baseline), entao foi criado
   `scripts/canonicalize_prediction_csv_label_free.py` para recomputar
   `prediction` e `correct` somente de `raw_output -> extract_final_answer ->
   verify_answer`, preservando `original_prediction/original_correct`.
   Baseline canonico passou V540 sem blockers: `192/315`, `bit=136/160`,
   `equation=56/155`, protected rows ok, prompt template parity ok. V682
   canonico continua arquivado como falha real por protected backfire
   (`8740ed31`, `59bee375`) e nao autoriza treino.
10. Achados do arquivo externo `ANALISE_DESAFIO_IAS_16.txt` entram como
   hardening, nao como novo treino cego:
   - aplicar padrao NeMo Curator antes de novo dataset: limpeza, qualidade,
     deduplicacao exata/fuzzy/semantica, source ledger por linha e manifestos;
   - aplicar padrao Data Designer apenas para gerar microdados condicionais
     validados por Python/verificador deterministico e, se usado, por juiz LLM;
   - se usar logprobs de vLLM para loss/row-loss, registrar versao vLLM e
     bloquear versoes/rotas com divergencia de logprob nao auditada; a doc
     NeMo RL alerta bug antes de vLLM `0.17.0`;
   - usar NeMo Evaluator/Gym apenas como referencia de estrutura de avaliacao:
     rollouts, metricas agregadas e perfis por tarefa. O gate KG1 continua
     label-free e baseado em `raw_output -> extract -> verify_answer`.
11. Nenhuma URL do arquivo trouxe gabarito novo, score direto ou dataset pronto
   para `bit_manipulation`/`equation_transform`. Datasets Nemotron gerais
   ficam fora do plano curto, exceto como referencia de curadoria/validacao.
12. Consenso V679 OpenRouter forte (`deepseek/deepseek-v4-pro` e
    `qwen/qwen3.6-max-preview`): inserir achados externos apenas como gates e
    experimento condicionado:
    - `G-SYNTH`: todo novo synthetic deve passar verificador Python
      deterministico, formato `boxed_only`, dedup contra train/val/protected e
      rerun dos gates de tokenizacao/loss/EOS antes de qualquer GPU;
    - `G-EVAL-FAMILY`: todo eval deve reportar `bit_manipulation` e
      `equation_transform` separadamente via `raw_output -> extract ->
      verify_answer`; nenhum agregado pode mascarar regressao de familia;
    - `G-VLLM-LOGPROB`: se uma rota usar logprob vLLM para diagnostico/loss,
      registrar versao vLLM, bloquear `<0.17.0` ou divergencia nao auditada, e
      nao usar logprob para promocao sem referencia fixa;
    - `E-SYNTH-EQ`: proximo experimento barato e CPU-first: gerar 100-200
      linhas `equation_transform` por templates deterministicas, validar 100%,
      deduplicar, e so entao permitir um unico SFT A100-large attention-only se
      os gates CPU nao encontrarem contaminacao.
13. Excluir explicitamente do plano curto: merge de datasets Nemotron gerais,
    RAG/agent cookbooks, GRPO/DAPO/RLVR, Brev/Discord/YouTube e qualquer rota
    que nao tenha verificador KG1. Estes itens ficam como referencia, nao como
    acao.
14. Gate V679 implementado e executado no V677:
    `scripts/kg1_v679_dataset_family_gate.py`.
    Resultado final em V677: `decision=pass`, `blockers=[]`, `warnings=[]`.
    O gate cobre `G-SYNTH`, `G-EVAL-FAMILY` e `G-VLLM-LOGPROB`; valida
    `boxed_only`, determinismo por regra/expr, train/val overlap, protected
    source, ledger bit, contagens por familia e politica vLLM logprob. O fuzzy
    overlap contra protected compara apenas exemplos/query, nao boilerplate
    oficial-like, para evitar falso blocker.

Status V680, 2026-05-20 01:12 UTC:

- Dataset V680: train `900` rows (`bit=240`, `equation=660`), validation
  `180` rows (`bit=60`, `equation=120`), `assistant_format=boxed_only`,
  `boxed_rate=1.0`.
- Hashes V680: train
  `9210b6caa6298034a288026f9425df9a16332cd848343ce2392c3d0c24234900`;
  validation `7a3f4ed1c9a1c66ce3484fc5c5e669b8b88de4621964a15f134a709190f0bbed`.
- V679 dataset/family gate V680: `decision=pass`, `blockers=[]`,
  `warnings=[]`.
- V286 tokenization real V680: `prompt_truncated=0`,
  `completion_tokens_dropped=0`, `fallback_masks=0`,
  `offset_masks=900/180`, `train_token_max=273`, `val_token_max=273`.
- Loss/EOS contract V680: `ok=true`, `final_loss_eos_rate=1.0`,
  `no_loss_rows=0`, `no_offset_rows=0`.
- V478 objective alignment V680: train effective shares
  `bit=0.112903`, `equation=0.887097`; validation effective shares
  `bit=0.148936`, `equation=0.851064`; `findings=[]`.
- Static safety e pre-paid integration no launcher de treino V680: `ok=true`,
  `findings=[]`; custo `a100-large=0.041667 USD/min`, dentro do teto
  `0.05 USD/min`; commit remoto confirmado em `origin/v230-v226-complementarity`.
- Dataset V680 enviado ao HF:
  `felipesp1983/kg1-v680-v677-synth-eq-augmented-artifacts`,
  commit `ab13db2603e91dc3f466d0f02fa7d130c0b57a2f`.
- Job V680 A100 de treino concluido:
  `https://huggingface.co/jobs/felipesp1983/6a0cfc492dc5b1243da5077f`,
  run id `v680-a100-v677-synth-eq-v290ckpt6-20260520T001046Z`.
  Preflight remoto passou: A100/CUDA/mamba/causal-conv ok, V485 adapter
  roundtrip ok e objective alignment remoto ok. Resultado de loss:
  baseline eval loss `0.1509`, checkpoint-10 `0.1489`, checkpoint-20/final
  `0.1479`. Isto e ganho de loss, nao ganho de ACC.
- V485 PEFT roundtrip local no output repo V680:
  checkpoint-20 e `final` passaram com `hf_gpu_allowed=true`, `r=32`,
  `alpha=32`, base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`,
  target_modules `down/in/k/o/out/q/up/v` e target_parameters
  `mlp.experts.gate_up_proj,mlp.experts.down_proj`.
- Weak eval V680 A100 official-like concluido e bloqueado:
  `https://huggingface.co/jobs/felipesp1983/6a0d06c43aba298b21d14567`,
  run id `v680-a100-v221contract-synth-eq-weak-20260520T005531Z`.
  Controles: `disable_thinking=0`, `max_tokens=7680`, `max_num_seqs=64`,
  prompt suffix oficial `\boxed{}`, GPU memory `0.85`, A100-only,
  no H200 fallback. Promotion gate exige `total>=196`, `bit>=136`,
  `equation>=60`, `truncated=0`, `boxed_rate=1.0`,
  `label_aware_delta=0`, `no_box_fallback=0`, sem protected-row backfire e sem
  queda baseline-relative contra `196/136/60`.
- Resultado real V680 weak eval: `191/315` (`accuracy=0.606349`),
  `bit_manipulation=134/160`, `equation_transform=57/155`, `truncated=1`,
  `boxed_rate=314/315`, `no_box_fallback_rows=1`, avg completion `4775.26`,
  max completion `7680`. Protections bloquearam:
  `8740ed31` backfire `01101000 -> 01111000`, `59bee375` backfire
  `10010101 -> 2` com truncation, e `55d834d1` missing required gain
  `00111111` ainda predito como `10111111`.
- Decisao: V680 nao e promovivel, nao pode ir para pacote/submissao e nao deve
  receber continuacao cega. OpenRouter V681 foi consultado e a auditoria local
  confirmou que o alvo supervisionado nao era longo: o erro raiz acionavel e a
  combinacao de desbalanceamento efetivo de loss (`equation=88.7%`) com drift
  de decodificacao official-like e MoE target_parameters aumentando o blast
  radius. A proxima rota deve atacar isso, nao apenas baixar loss.

Artefatos V680 arquivados/diagnostico:

- Builder:
  `scripts/build_v680_v677_synth_eq_augmented_dataset.py`;
- dataset manifest:
  `artifacts/v680_v677_synth_eq_augmented_dataset/20260519T_v680_cpu_gate/v680_v677_synth_eq_augmented_manifest.json`;
- V679 gate:
  `artifacts/v680_v677_synth_eq_augmented_dataset/20260519T_v680_cpu_gate/v679_dataset_family_gate/v680_v679_dataset_family_gate.json`;
- tokenization gate:
  `artifacts/v680_v677_synth_eq_augmented_dataset/20260519T_v680_cpu_gate/v286_tokenization_real/v286_generic_tokenization_gate_manifest.json`;
- loss/EOS gate:
  `artifacts/v680_v677_synth_eq_augmented_dataset/20260519T_v680_cpu_gate/v680_loss_mask_eos_contract.json`;
- CPU gate stack:
  `artifacts/v680_v677_synth_eq_augmented_dataset/20260519T_v680_cpu_gate/v680_cpu_gate_stack.json`;
- HF upload manifest:
  `artifacts/v680_hf_a100_launch/v680_hf_dataset_upload_manifest.json`;
- launcher:
  `artifacts/v680_hf_a100_launch/launch_v680_hf_a100_synth_eq_augmented.py`;
- launch manifest:
  `artifacts/v680_hf_a100_launch/v680-a100-v677-synth-eq-v290ckpt6-20260520T001046Z_launch_manifest.json`.
- weak-eval launcher:
  `artifacts/v680_hf_a100_launch/launch_v680_hf_a100_weak_eval.py`;
- weak-eval launch manifest:
  `artifacts/v680_hf_a100_launch/v680-a100-v221contract-synth-eq-weak-20260520T005531Z_weak_eval_launch_manifest.json`;
- PEFT roundtrip gates:
  `artifacts/v680_hf_a100_launch/v680_v485_peft_roundtrip_checkpoint20.json`,
  `artifacts/v680_hf_a100_launch/v680_v485_peft_roundtrip_final.json`;
- weak-eval static gate:
  `artifacts/v680_hf_a100_launch/v680_weak_eval_static_safety_gate.json`.
- weak-eval downloaded diagnostics:
  `artifacts/v680_hf_a100_launch/downloaded_weak_eval_v680_final/`;
  mirror curto para CSVs longos em `C:/kg1_v680_eval/`.
- failure analysis V681:
  `artifacts/v680_hf_a100_launch/v680_weak_eval_failure_analysis.json`;
- wrong-row sample:
  `artifacts/v680_hf_a100_launch/v680_weak_eval_wrong_rows_top_completion_tokens.csv`;
- OpenRouter V681 prompt/responses/consensus:
  `artifacts/openrouter/v681_v680_weak_eval_failure_consult/KG1_V681_OPENROUTER_PROMPT.md`,
  `artifacts/openrouter/v681_v680_weak_eval_failure_consult/KG1_V681_OPENROUTER_RESPONSES.json`,
  `artifacts/openrouter/v681_v680_weak_eval_failure_consult/KG1_V681_CONSENSUS.md`.

Status V681, 2026-05-20 02:05 UTC:

- Objetivo: recuperar pelo menos `196/315` sem falso ganho, buscando
  `bit>=136/160` e `equation>=60/155`.
- Primeiro passo obrigatorio: gerar/auditar dataset balanceado por familia e
  por loss efetivo. Regra: `equation_loss_effective_share <= 0.60` salvo
  justificativa e gate explicito; preferencia inicial `50/50`.
- `loss_weight` nao pode repetir `bit=0.35` com equation massivo. Usar pesos
  uniformes ou bit-protective e registrar o share efetivo no manifest.
- Proximo treino permitido somente se gates CPU passarem:
  V679 family gate, V286 tokenization real, loss/EOS, workspace clean, static
  safety, PEFT contract preflight, no protected source, no overlap train/val,
  no truncation/token dropped/fallback masks.
- Config inicial para GPU:
  `a100-large`, max 15 steps, eval/probe a cada 5, `max_length=1024`,
  `save_embedding_layers=0`, `lm_head` fora, target_modules dense/attention
  com MoE target_parameters do V290 preservados como frozen-active
  (`REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`).
- Abort gates antes de weak full:
  protected IDs `8740ed31`, `59bee375`, `55d834d1` sem backfire; boxed
  `315/315` no probe aplicavel; `truncated=0`; `no_box_fallback=0`;
  `label_aware_delta=0`; bit nao pode cair contra baseline; se falhar, nao
  gastar weak full.
- Itens explicitamente fora do plano curto: continuar V680 por loss menor,
  adicionar `lm_head` sem gate proprio, usar H200, promover por loss ou por
  metrica agregada.
- Dataset V681 CPU-first implementado:
  `artifacts/v681_balanced_eqbit_dataset/20260520T0205Z_cpu_gate/`.
  Origem: V680 bloqueado, sem labels novos. Selecao: todos os `240` bit de
  treino + `240` equation balanceados (`80` por subcategoria); validation
  `60` bit + `60` equation (`20` por subcategoria).
- V681 corrige o erro objetivo V680:
  `loss_weight=1.0` em todas as linhas, train effective share
  `bit=0.5`, `equation=0.5`; validation effective share `bit=0.5`,
  `equation=0.5`.
- Gates CPU V681 executados:
  - builder static safety: `ok=true`, `findings=[]`;
  - V679 dataset/family gate: `decision=pass`, `blockers=[]`, `warnings=[]`;
  - V286 tokenization real: train `480`, val `120`, `prompt_truncated=0`,
    `completion_tokens_dropped=0`, `fallback_masks=0`, `offset_masks=480/120`,
    `train_token_max=273`, `val_token_max=273`;
  - loss/EOS: `ok=true`, `final_loss_eos_rate=1.0`, `no_loss_rows=0`,
    `no_offset_rows=0`;
  - V478 objective alignment: `hf_gpu_allowed=true`, `findings=[]`,
    train/validation effective share `50/50`;
  - workspace clean V681: `error=0`, `warning=0`;
  - CPU gate stack V681: `decision=pass`, `blockers=[]`.
- Dataset V681 enviado ao HF:
  `felipesp1983/kg1-v681-balanced-eqbit-artifacts`, commit
  `e0c052a8baa0f2f6835d54bc2de474a314358754`.
- Launcher V681 A100:
  `artifacts/v681_hf_a100_launch/launch_v681_hf_a100_balanced_eqbit.py`.
  Static safety apos correcao: `ok=true`, `findings=[]`; pre-paid gate apos
  correcao: `ok=true`, `findings=[]`. O pre-paid gate tambem foi corrigido
  para aceitar `hf_gpu_allowed=true` nos CPU stack schemas novos, sem
  enfraquecer blockers/failed checks, e agora bloqueia launcher com
  `LORA_TARGET_PARAMETERS` quando
  `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` sem
  `FREEZE_LORA_TARGET_PARAMETERS=1`.
- Job V681 inicial falhou antes de treino:
  `https://huggingface.co/jobs/felipesp1983/6a0d15503aba298b21d14601`.
  Causa: `decoding_vs_adapter_drift_gate` exigia `passed`, enquanto o launcher
  carregava `first_checkpoint_required`; nao houve checkpoint nem promocao.
  Correcao aplicada: `KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS` =
  `deferred_post_checkpoint` e
  `KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT=1`, mantendo
  `KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED=1`, `MAX_STEPS=15`,
  `SAVE/EVAL_EVERY_STEPS=5`.
- Segundo job V681 A100-large foi cancelado antes de checkpoint:
  `https://huggingface.co/jobs/felipesp1983/6a0d16502dc5b1243da50981`,
  run id `v681-a100-balanced-eqbit-v290ckpt6-20260520T020150Z`. O preflight
  remoto, tokenizacao e load de adapter passaram, mas o log mostrou
  `target_parameters_trainability_mode=trainable` e
  `target_parameter_trainable_lora_params` de aproximadamente `432,791,552`
  para cada MoE target_parameter. Isto contradiz o plano V681
  frozen-active e aumenta o blast radius. Decisao: cancelado, sem checkpoint
  promovivel e sem weak eval.
- Correcao V681 aplicada apos cancelamento:
  `scripts/hf_job_train_v90.py` ganhou `FREEZE_LORA_TARGET_PARAMETERS=1`,
  congela qualquer LoRA tensor que casa com
  `mlp.experts.gate_up_proj`/`mlp.experts.down_proj`, reporta
  `freeze_lora_target_parameters`, e falha se algum target_parameter continuar
  treinavel. O launcher V681 exporta `FREEZE_LORA_TARGET_PARAMETERS=1` e exige
  apenas `q/k/v/o` como substrings treinaveis obrigatorias. Teste local com
  modelo dummy confirmou `target_parameters_trainability_mode=frozen_active`,
  `target_parameter_trainable_lora_tensors=0`, e `q/k/v/o` treinaveis.
- Gate adicional aplicado:
  `scripts/kg1_pre_paid_job_integration_gate.py` agora resolve referencias de
  env/constantes sem casar sufixos de nomes (`LORA_TARGET_PARAMETERS` nao pode
  ser lido de `FREEZE_LORA_TARGET_PARAMETERS`) e bloqueia
  `launcher_lora_target_parameters_not_frozen`. Self-test do gate passou.
- Gates pos-correcao:
  `artifacts/v681_hf_a100_launch/v681_static_safety_gate_after_reference_resolution_fix.json`
  (`ok=true`, `findings=[]`) e
  `artifacts/v681_hf_a100_launch/v681_pre_paid_job_integration_gate_after_target_literal_fix.json`
  (`ok=true`, `findings=[]`, contrato LoRA com
  `freeze_lora_target_parameters=true`).
- Commit/push da correcao executado em `67a27bcb2ed6a4e9856adb61ece63516a0b29637`.
- Terceiro job V681 A100-large concluiu treino com contrato correto:
  `https://huggingface.co/jobs/felipesp1983/6a0d1c132dc5b1243da509e9`,
  output repo
  `felipesp1983/kg1-nemotron-lora-v681-a100-balanced-eqbit-v290ckpt6`.
  Checkpoints publicados: `checkpoint-5`, `checkpoint-10`, `checkpoint-15`.
  Loss: baseline `0.2907`, checkpoint-5 `0.2904`, checkpoint-10 `0.2897`,
  checkpoint-15 `0.2893`. Contrato remoto: `target_parameters_trainability_mode=frozen_active`,
  `target_parameter_trainable_lora_tensors=0`, q/k/v/o treinaveis, MoE
  target_parameters preservados mas congelados.
- Weak eval V681 A100-large concluido e bloqueado:
  `https://huggingface.co/jobs/felipesp1983/6a0d254d2dc5b1243da50a76`,
  run id `v681-a100-v221contract-balanced-eqbit-weak-20260520T030548Z`,
  upload commit `9844327eb28744c05e9eb2294460fe30e4249195`.
  Resultado real: `191/315`, `bit_manipulation=135/160`,
  `equation_transform=56/155`, `truncated=1`, `boxed_rate=314/315`,
  `no_box_fallback_rows=1`, avg completion `4775.39`,
  completion tokens `1,504,248`. Label-free, first-boxed e label-aware todos
  iguais a `191`, logo parser/expected-aware nao e a causa principal.
  Protected guard bloqueou `8740ed31` (`01101000 -> 01111000`),
  `59bee375` (`10010101 -> 2`) e missing required gain em `55d834d1`
  (`10111111`, expected `00111111`).
- Decisao V681 final: nao-promovivel, nao-submetivel, nao continuar por loss
  menor, nao relancar por LR/steps. V681 provou que balancear loss efetivo
  sem resolver drift de geracao/formato nao sai do plato.
- OpenRouter V682 pos-falha V681 executado:
  `artifacts/openrouter/v682_v681_weak_eval_failure_consult/KG1_V682_OPENROUTER_PROMPT.md`,
  `artifacts/openrouter/v682_v681_weak_eval_failure_consult/KG1_V682_OPENROUTER_RESPONSES.md`,
  `artifacts/openrouter/v682_v681_weak_eval_failure_consult/KG1_V682_CONSENSUS.md`.
  Consenso: bloquear V681, rejeitar mais steps/LR sweep, atacar geracao longa
  e protected backfire por gates CPU antes de qualquer novo A100.
- Auditoria V682 CPU dos outputs V681 concluida:
  `artifacts/v681_hf_a100_launch/v682_cpu_output_audit/v681_checkpoint15_summary.json`
  e
  `artifacts/v681_hf_a100_launch/v682_cpu_output_audit/v681_checkpoint15_format_length_audit.json`.
  Achados: `starts_boxed=0/315`, `starts_final_answer_boxed=0/315`,
  `boxed_zero_rows=1`, completion tokens p50 `6193`, p95 `7155`, max `7680`.
  Em bit, completion p50 `6703` e first boxed char p50 `8677`; em equation,
  completion p50 `707` mas p75 `5743`. Early extraction mostrou
  `early_512_correct=0`, `early_1024_correct=2`, `early_2048_correct=9`.
  Conclusao: hard cap simples em `max_tokens` nao resolve; a rota precisa
  mudar o comportamento para responder mais cedo ou separar explicitamente
  ganho de decoding/config de ganho de adapter.
- Auditoria V682/V610 de drift linha-a-linha concluida:
  `artifacts/v681_hf_a100_launch/v682_cpu_row_drift_audit/v610_adapter_drift_summary.json`
  e `artifacts/v681_hf_a100_launch/v682_cpu_row_drift_audit/v610_row_drift.csv`.
  Resultado: baseline CSV `191/315` (`bit=136`, `equation=55`) contra V681
  `191/315` (`bit=135`, `equation=56`), net `0`, ganhos `2`, perdas `2`.
  Perdas: `8740ed31` e `59bee375`; ganhos: `4ef88f92` e `4bb8c6cd`.
  Bloqueadores: abaixo do piso submit-safe `196/136/60`, protected-row guard,
  `stored_prediction_mismatch_nonzero` e `truncated_rows_nonzero`. Conclusao:
  V681 e troca de familia com risco, nao ganho real.
- Gate formal V614 anti-runaway rodado contra V681:
  `artifacts/v681_hf_a100_launch/v682_v681_v614_anti_runaway_gate.json`.
  Resultado: `decision=blocked`, `correct=191/315`, `boxed_rate=0.9968`,
  `truncated=1`, bit p99 `7390`, equation p99 `6397`; blockers incluem
  `bit_lt_136`, `equation_lt_60`, `total_lt_196`,
  `finish_reason_length_nonzero`, `no_box_fallback_gt_0`,
  `protected_failed_8740ed31`, `protected_failed_59bee375` e
  `protected_failed_55d834d1`.
- Launcher V682 no-thinking diagnostic implementado e lançado em A100-large:
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/launch_v682_hf_a100_nothink_diagnostic.py`.
  Job: `https://huggingface.co/jobs/felipesp1983/6a0d324c2dc5b1243da50b05`,
  run id `v682-a100-nothink-v681ckpt15-diag-20260520T040116Z`.
  Contrato: `diagnostic_only=true`, `disable_thinking=1`, `max_tokens=384`,
  `KG1_REQUIRE_DISABLE_THINKING=1`, `KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1`,
  `hf_job_timeout_s=3600`, A100-large only, H200 bloqueado. Este job pode
  diagnosticar se o gargalo e decoding/config, mas nao e ganho de adapter,
  nao e promotable e nao e submit-safe.
- V682 final baixado, auditado e bloqueado:
  upload commit HF
  `2e2beb0eebedf6ab09e12933a153f91928c406c7`. Resultado:
  `17/315`, `bit_manipulation=7/160`, `equation_transform=10/155`,
  `truncated=0`, `boxed_rate=1.0`, `no_box_fallback=0`,
  `avg_completion_tokens=11.2889`, `max_completion_tokens=14`.
  V614 local: `decision=blocked` por `total_lt_196`, `bit_lt_136`,
  `equation_lt_60` e protected failures em `8740ed31`, `59bee375`,
  `55d834d1`.
  V610 drift local: baseline `191/315` contra V682 `17/315`, net `-174`,
  ganhos `1`, perdas `175`, bit losses `129`, equation losses `46`.
  Conclusao operacional: `disable_thinking=1` e cap curto nao atacam o
  problema de ACC; eles removem reasoning e causam resposta curta errada.
- Hardening novo de filesystem/Windows: V614, V610 e
  `analyze_eval_predictions.py` agora usam long-path-safe open/read/write.
  Motivo: o CSV V682 baixado existia, mas o caminho longo gerava
  `FileNotFoundError` em Python no Windows. O gate foi reexecutado pelo caminho
  longo original e bloqueou corretamente, sem depender do mirror curto.
- Hardening novo de extractor/simbolos: V682 revelou 4 rows de
  `equation_transform` com divergencia entre stored prediction remoto e
  re-extraction local por payloads com `\\{`/`\\}` dentro de `\\boxed{}`. Essas
  4 rows estavam erradas nos dois caminhos, entao nao houve falso ganho, mas
  qualquer novo job precisa sincronizar o commit remoto com o extractor local
  atual antes de gerar novo weak eval. Sem isso, existe risco de drift de
  simbolos/backslash entre treino, eval local e HF.
- OpenRouter V683 pos-falha V682 executado:
  `artifacts/openrouter/v683_v682_nothink_failure_consult/KG1_V683_OPENROUTER_PROMPT.md`,
  `artifacts/openrouter/v683_v682_nothink_failure_consult/KG1_V683_OPENROUTER_RESPONSES.md`,
  `artifacts/openrouter/v683_v682_nothink_failure_consult/KG1_V683_DEEPSEEK_RETRY_RESPONSE.md`,
  `artifacts/openrouter/v683_v682_nothink_failure_consult/KG1_V683_CONSENSUS.md`.
  Consenso aceito: proximo passo e CPU residual-miss ledger + microdataset
  source-only verificado + extractor parity + protected safety; sem novo A100
  ate essas evidencias existirem. Sugestoes agressivas de `r=32`, MoE/MLP e
  LR `2e-4` foram marcadas como condicionais/rejeitadas por enquanto porque
  V673/V681 ja mostraram regressao com blast radius alto.
- Ledger V683 atual:
  `scripts/build_v683_current_residual_ledger.py`,
  `artifacts/v683_current_residual_ledger/v683_current_residual_ledger.csv`,
  `artifacts/v683_current_residual_ledger/v683_current_residual_ledger_manifest.json`,
  `artifacts/v683_current_residual_ledger/KG1_V683_CURRENT_RESIDUAL_LEDGER.md`.
  Decisao `cpu_only_no_gpu`; nao usar linhas weak como treino direto.
- Dataset V684 trace-preserving source-only:
  `scripts/build_v684_v683_trace_preserving_source_dataset.py`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_v683_trace_preserving_source_manifest.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_v683_trace_preserving_source_train.jsonl`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_v683_trace_preserving_source_val.jsonl`.
  Gates: V679
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v679_dataset_family_gate/v684.json`,
  V286
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v286_tokenization_real/v286_generic_tokenization_gate_manifest.json`,
  loss/EOS
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_loss_mask_eos_contract.json`,
  V478
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_v478_objective_alignment.json`,
  V513
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v513_trace_learnability/v513_trace_learnability_gate_manifest.json`,
  V564 contract/mask alignment
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_v564_contract_mask_alignment_absent_required.json`,
  static/workspace clean
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_static_safety_gate.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_workspace_clean_gate.json`.
  Extractor/canonical CSV:
  `scripts/canonicalize_prediction_csv_label_free.py`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/canonical_prediction_csvs/v516_baseline_current_extractor_manifest.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/canonical_prediction_csvs/v516_baseline_current_extractor.csv`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_extractor_parity_canonical_baseline/v684_extractor_parity_canonical_baseline_summary.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_extractor_static_safety_gate.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_v564_extractor_static_safety_gate.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_workspace_clean_gate_after_extractor.json`,
  `artifacts/v684_v683_trace_preserving_source_dataset/20260520T_v684_cpu_gate/v684_workspace_clean_gate_after_v564.json`.
- Artefatos V681:
  `scripts/build_v681_balanced_eqbit_dataset.py`,
  `artifacts/v681_balanced_eqbit_dataset/20260520T0205Z_cpu_gate/v681_balanced_eqbit_manifest.json`,
  `artifacts/v681_balanced_eqbit_dataset/20260520T0205Z_cpu_gate/v681_cpu_gate_stack.json`,
  `artifacts/v681_hf_a100_launch/v681_pre_paid_job_integration_gate_after_drift_defer_fix.json`,
  `artifacts/v681_hf_a100_launch/v681_pre_paid_job_integration_gate_after_target_literal_fix.json`,
  `artifacts/v681_hf_a100_launch/v681_static_safety_gate_after_reference_resolution_fix.json`,
  `artifacts/v681_hf_a100_launch/v681-a100-balanced-eqbit-v290ckpt6-20260520T020150Z_launch_manifest.json`,
  `artifacts/v681_hf_a100_launch/v681_hf_job_6a0d15503aba298b21d14601_logs.txt`,
  `artifacts/v681_hf_a100_launch/v681_hf_job_6a0d16502dc5b1243da50981_logs.txt`,
  `artifacts/v681_hf_a100_launch/v681_hf_job_6a0d1c132dc5b1243da509e9_logs.txt`,
  `artifacts/v681_hf_a100_launch/v681_weak_eval_job_6a0d254d2dc5b1243da50a76_logs.txt`,
  `artifacts/v681_hf_a100_launch/v681_weak_eval_failure_analysis.json`,
  `artifacts/v681_hf_a100_launch/v681_weak_eval_wrong_rows_sample.csv`,
  `artifacts/v681_hf_a100_launch/v682_cpu_output_audit/v681_checkpoint15_summary.json`,
  `artifacts/v681_hf_a100_launch/v682_cpu_output_audit/v681_checkpoint15_format_length_audit.json`,
  `artifacts/v681_hf_a100_launch/v682_cpu_row_drift_audit/v610_adapter_drift_summary.json`,
  `artifacts/v681_hf_a100_launch/v682_cpu_row_drift_audit/v610_row_drift.csv`,
  `artifacts/v681_hf_a100_launch/v682_v681_v614_anti_runaway_gate.json`,
  `artifacts/v681_hf_a100_launch/downloaded_weak_eval_v681_final/`;
  mirror curto dos CSVs em `C:/kg1_v681_eval/`.
- Artefatos V682:
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/launch_v682_hf_a100_nothink_diagnostic.py`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682-a100-nothink-v681ckpt15-diag-20260520T040116Z_launch_manifest.json`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682-a100-nothink-v681ckpt15-diag-20260520T040116Z_remote_command.sh`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/downloaded_eval/`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/short_eval/v682_predictions.csv`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682_final_v614_anti_runaway_gate.json`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682_final_v614_longpath_verified.json`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682_final_output_audit/`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682_final_output_audit_longpath_verified/`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682_final_row_drift_audit/`,
  `artifacts/v682_hf_a100_nothink_diagnostic_launch/v682_final_row_drift_audit_longpath_verified/`.

Status V677, 2026-05-19 23:05 UTC:

- Causa raiz candidata corrigida no builder: V673 usava `v673_source_ledger_ids`
  agregado, mas linhas bit selecionadas podiam vir de `metadata.source_id`
  protegido ou nao autorizado. Agora cada row bit deve ter source real permitido,
  source protegido bloqueado, e ledger por row igual a `[metadata.source_id]`.
- Fontes protegidas bloqueadas: `8740ed31`, `59bee375`, `55d834d1`.
- Dataset V677: train `720` rows (`bit=240`, `equation=480`), validation
  `180` rows (`bit=60`, `equation=120`), `assistant_format=boxed_only`,
  `boxed_rate=1.0`.
- Hashes V677: train
  `8456966d2e5131179596183d85eb38a130acf969de5e09350bc60973b915528c`;
  validation `7a3f4ed1c9a1c66ce3484fc5c5e669b8b88de4621964a15f134a709190f0bbed`.
- Auditoria root-cause V676 rerun em V677: `decision=pass`, `blockers=[]`,
  `warnings=[]`, `weak_overlap=0`, `protected_nonzero_loss=0`,
  `protected_source_count=0`, `unauthorized_bit_source_count=0`,
  `bit_source_ledger_mismatch_count=0`, `assistant_not_boxed_first_count=0`.
- Tokenization V286 real rerun em V677: `prompt_truncated=0`,
  `completion_tokens_dropped=0`, `fallback_masks=0`, `offset_masks=720/180`,
  `train_token_max=273`, `val_token_max=273`, tokenizer real Nemotron
  `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Loss/EOS contract V677: `ok=true`, `loss_contains_eos_rate=1.0`,
  `final_loss_eos_rate=1.0`, `no_loss_rows=0`, `no_offset_rows=0`.
- Static safety gate em `scripts`, `src`, V673 launch e V677 dataset:
  `ok=true`, `findings=[]`. Workspace clean gate nos escopos ativos:
  `finding_counts.error=0`, `warning=0`.
- V679 dataset/family gate em V677:
  `decision=pass`, `blockers=[]`, `warnings=[]`; train/val sem overlap exato,
  targets `boxed_only`, equacoes verificadas deterministicamente, bit rows
  verificadas por `metadata.expr`, protected sources ausentes e ledger bit
  consistente.
- Nao ha ganho ACC novo ainda. O ganho real so existe quando o novo adapter
  V677 passar weak eval label-free com os pisos acima. V677 apenas remove os
  bugs silenciosos que invalidavam V673.

Artefatos ativos:

- Dataset V677:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v673_guarded_equation_bit_transfer_manifest.json`;
- root-cause audit V677:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v676_root_cause_audit_rerun/v676_v673_dataset_root_cause_audit.json`;
- tokenization gate V677:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v286_tokenization_real_rerun/v286_generic_tokenization_gate_manifest.json`;
- loss/EOS gate V677:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v677_loss_mask_eos_contract_rerun.json`;
- static gate ativo pos-cleanup:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v677_static_safety_gate_active_dirs_after_cleanup.json`;
- workspace clean scripts:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v677_workspace_clean_gate_scripts_deleted_pycache.json`;
- workspace clean V673 launch:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v677_workspace_clean_gate_v673_after_remote_cleanup.json`;
- workspace clean V677 tree:
  `artifacts/v677_guarded_equation_bit_transfer_dataset/20260519T224818Z/v677_workspace_clean_gate_dataset_tree_final.json`;
- V679 dataset/family gate final:
  `artifacts/v679_dataset_family_gate/v677_20260519T224818Z_final/v679_v677_dataset_family_gate_final.json`;
- consulta OpenRouter V676:
  `artifacts/openrouter/v676_v673_weak_eval_failure_consult/KG1_V676_OPENROUTER_CONSENSUS.md`.

Historico V673/V676, mantido apenas como evidencia e nao como plano ativo:

- Treino A100 `felipesp1983/6a0ccbae3aba298b21d143b1` completou sem OOM e
  sem traceback no repo
  `felipesp1983/kg1-nemotron-lora-v673-a100-guarded-eqbit-v290ckpt6`, mas
  isto nao e ganho. O weak eval estrito real bloqueou todos os candidatos.
- Weak eval A100 `felipesp1983/6a0cddf23aba298b21d1442e` terminou em erro
  proposital de gate depois de subir os diagnosticos. Resultados:
  `checkpoint-10=18/315` (`bit=8/160`, `equation=10/155`,
  `truncated=0`, `boxed_rate=1.0`, `max_completion_tokens=14`) e
  `checkpoint-20=17/315` (`bit=7/160`, `equation=10/155`,
  `truncated=0`, `boxed_rate=1.0`, `max_completion_tokens=14`). O candidato
  `final` nao foi avaliado porque o stop antecipado economizou credito.
- V673 esta bloqueado para promocao, pacote e submissao. A falha nao foi
  comprimento, falta de `\boxed{}` ou truncation; foi drift de conteudo:
  o adapter empurrou respostas erradas. Protected rows `8740ed31` e
  `59bee375` backfired; `55d834d1` continuou sem aprender ganho obrigatorio.
- Consenso OpenRouter V676 com 5/5 modelos: bloquear qualquer novo GPU ate
  auditar dado/loss/mask/LoRA; remover MoE/MLP da proxima micro-rota;
  usar no maximo atencao `q_proj,k_proj,v_proj,o_proj`; nao usar
  `lm_head`, embeddings, `mlp.experts.gate_up_proj` nem
  `mlp.experts.down_proj` em novo micro-rescue.
- Auditoria CPU V676 encontrou causa raiz candidata no dataset V673:
  `weak_overlap=0`, mas `protected_source_nonzero_loss` existe em train
  (`10` linhas) e validation (`3` linhas), todas derivadas de `55d834d1`
  com `loss_weight=0.35`. Isto pode misturar replay protegido com gradiente
  positivo e gerar backfire silencioso. Tambem ha
  `assistant_target_not_boxed_first` em todas as linhas porque o treino usa
  trace + resposta final, enquanto o weak eval V673 forcou resposta boxed
  curta sem raciocinio.
- Validacao implementada apos V673: o weak eval agora tem guarda
  catastrophic baseline-relative (`192/136/56` como baseline de total/bit/eq)
  e `KG1_STOP_ON_PROTECTED_BACKFIRE=1`. Assim um colapso como `18/315`
  falha mesmo com `truncated=0` e `boxed_rate=1.0`.
- Proxima acao obrigatoria e CPU-only: auditar mascara de loss/EOS/pesos por
  linha, confirmar se os protected IDs entram no gradiente, e montar somente
  depois um V677 attention-only. Nao relancar V673, nao avaliar `final`, nao
  empacotar e nao submeter.

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
- excecao atual: V706 pode usar `max_tokens=512` somente como diagnostico
  eval-only com `disable_thinking=true` para validar o fechamento do bloco de
  raciocinio apos a falha V704. Essa excecao nao altera o contrato oficial, nao
  autoriza promocao, nao autoriza pacote e nao autoriza submit;
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
- H200 esta bloqueado pela regra operacional atual do usuario. Somente A100
  pode ser usado para novos jobs; qualquer excecao futura exige pedido humano
  explicito e um motivo tecnico registrado no launcher, manifesto e roadmap;
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
- continuar V681 checkpoint-15 por loss menor;
- relancar V681 50/50 uniforme com mais steps, LR sweep ou mais dados sem nova
  hipotese testavel;
- aumentar LoRA para `r=64/alpha=128`, destravar router, destravar MoE
  `target_parameters` ou adicionar `lm_head` como resposta ao backfire V681;
- usar `enable_thinking=False` ou `max_new_tokens` menor como ganho de adapter
  sem classificar separadamente como efeito de decoding/config;

## Proxima Acao Executavel

1. Manter congeladas V653/V660/V661/V662/V663/V664/V680/V681 como rotas
   ativas:
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
   - V680 caiu para `191/315`, `bit=134`, `equation=57`, com geracao longa e
     protected backfire.
   - V681 caiu para `191/315`, `bit=135`, `equation=56`, com `truncated=1`,
     `no_box_fallback=1`, `boxed_rate=314/315`, avg completion `4775.39` e
     protected backfire.
2. Executar V682 CPU-only e fechar diagnostico no-thinking antes de qualquer novo GPU:
   - feito: parsear `C:/kg1_v681_eval/predictions.csv` e gerar auditoria de
     comprimento/formato. Resultado: `starts_boxed=0/315`, bit first boxed
     muito tardio, `early_512_correct=0`; hard cap simples nao e rota de
     ganho;
   - feito: rodar drift baseline-relative V610. Resultado: V681 tem net `0`
     contra baseline, com troca `bit -1` por `equation +1` e protected losses
     em `8740ed31`/`59bee375`;
   - feito: rodar V614 anti-runaway promotion gate. Resultado: bloqueado por
     score, p99 de tokens, truncation, boxed rate, no-box fallback e protected
     rows;
   - feito: probe/baseline de decoding sem thinking somente como diagnostico:
     `disable_thinking=1`, `temperature=0`, `max_tokens=384`, job
     `6a0d324c2dc5b1243da50b05`. Resultado final `17/315`
     (`bit=7/160`, `equation=10/155`), `truncated=0`, `boxed_rate=1.0`,
     avg tokens `11.29`, protected failures em `8740ed31`, `59bee375`,
     `55d834d1`. Decisao: rota rejeitada; nao usar no-thinking/cap curto como
     plano de ACC;
   - rodar overlap/protected-source gate em qualquer dataset V682:
     weak id/prompt/prompt+answer overlap `0`, protected source `0`,
     train/val overlap `0`;
   - construir/auditar dataset curto somente se preservar thinking/raciocinio;
     V682 provou que targets/decoding curtos sem raciocinio podem ficar
     format-perfect e mesmo assim destruir ACC. Targets devem ser byte-equal,
     ter EOS dentro da loss mask, `0` truncation, `0` dropped completion
     tokens, `0` fallback masks, e passar protected rows em CPU/smoke;
   - rodar smoke protegido para `8740ed31`, `59bee375`, `55d834d1`; qualquer
     regressao bloqueia GPU.
3. Nova frente CPU V683/V684, antes de qualquer treino:
   - feito: ledger residual V683 dos misses/protected atuais, separando bit,
     equation, protected rows, ganhos/perdas V681 e colapso V682. Resultado:
     `38` linhas, `decision=cpu_only_no_gpu`, `12` source-only trainable,
     `2` source-only guarded, `14` needs-rule-proof, `7` drop, `3`
     protected guard;
   - regra obrigatoria: os IDs weak do V683 autorizam somente regra/proxy; nao
     podem virar exemplo supervisionado direto;
   - feito: V684 source-only weak-excluded com verificador por row e target
     trace-preserving (`Final answer: \boxed{...}`), train `640`
     (`320/320`), validation `160` (`80/80`), effective share `50/50`;
   - feito: V679 `boxed_suffix`, V286, loss/EOS, V478, V513, V564
     `protected_mode=absent_required`, static safety e workspace clean
     passaram sem blocker/warning estrutural;
   - feito: extractor parity V540 com fixtures `\\{`, `\\}`, braces
     literais e payloads com backslash. O primeiro audit contra CSVs
     historicos falhou porque o baseline antigo tinha `prediction/correct`
     stale em simbolos de equation; a correcao foi canonica, nao label-aware:
     `scripts/canonicalize_prediction_csv_label_free.py` gerou
     `canonical_prediction_csvs/v516_baseline_current_extractor.csv` a partir
     de `raw_output` e preservou as colunas antigas em `original_*`.
     Resultado canonico: `192/315`, `bit=136/160`, `equation=56/155`,
     `prediction_changed_from_original=10`, `correct_changed_from_original=1`.
     O gate V540 canonico passou sem blockers/warnings em
     `v684_extractor_parity_canonical_baseline/`. Regra nova: todo CSV de eval
     remoto deve ser canonicalizado ou provar `prediction == extract(raw_output)`
     antes de comparacao, pacote ou decisao de novo treino;
   - feito/arquivado: V682 canonico passou paridade de extrator, mas segue
     bloqueado por protected backfire em `8740ed31` e `59bee375`; nao usar V682
     como candidato, somente como diagnostico negativo;
   - feito: dry-run V684 A100 validou upload HF, hashes remoto/local,
     `a100-large` (`A100 80GB`, `0.041667 USD/min`), gates locais
     V679/V286/loss-EOS/V478/V513/V540/V564, static safety e objective
     alignment 50/50. Launcher:
     `artifacts/v684_hf_a100_launch/launch_v684_hf_a100_trace_source.py`;
   - primeira tentativa A100
     `https://huggingface.co/jobs/felipesp1983/6a0d8c378229e585f969c29d`
     falhou no preflight por bloqueio conservador de CUDA 13 em A100. O
     driver gate imediatamente anterior ja havia provado `cuda_available=true`,
     A100 detectada e alocacao CUDA OK;
   - segunda tentativa A100
     `https://huggingface.co/jobs/felipesp1983/6a0d8cb6ccb6cd133d158898`
     falhou cedo porque o launcher ainda nao exportava todo o bloco
     residual-first V541/V516/stale/protected exigido pelo preflight remoto;
   - corrigido: launcher V684 agora passa `kg1_static_safety_gate` e
     `kg1_pre_paid_job_integration_gate` com `findings=[]`, incluindo
     `v684_cpu_gate_stack.json`, `KG1_V541_*`, `KG1_V516_*`,
     protected IDs, `KG1_CPU_SIMULATED_TOTAL/BIT/EQUATION=196/136/60`,
     `KG1_MAX_TOKEN_HEADROOM_RATIO=0.325` e drift gate
     `deferred_post_checkpoint` limitado a 10 steps;
   - concluido/cancelado por custo apos checkpoint valido: V684 A100 smoke
     `https://huggingface.co/jobs/felipesp1983/6a0d8f4bccb6cd133d1588b2`,
     run id `v684-a100-tracesource-v290ckpt6-20260520T103805Z`.
     O job passou preinstall/postinstall, CUDA13/A100 driver gate,
     dependency probe, residual-first V541 e drift-deferred gate, treinou
     `10/10` steps e publicou `checkpoint-5` e `checkpoint-10`.
     Loss remoto: baseline `3.3707`, checkpoint-5 `3.3682`, checkpoint-10
     `3.3674`. O checkpoint-10 contem `adapter_config.json`,
     `adapter_model.safetensors`, tokenizer e chat template completos. O job
     foi cancelado manualmente durante eval final repetido, depois do upload,
     para nao queimar credito. Proibido promover sem weak eval canonicalizado
     e sem passar protected/backfire/format gates.
   - feito: V485 PEFT roundtrip local do checkpoint-10 passou:
     `artifacts/v684_hf_a100_launch/v684_checkpoint10_v485_peft_roundtrip_gate.json`.
     Contrato observado: `r=32`, `alpha=32`, target_modules
     `down/in/k/o/out/q/up/v`, target_parameters
     `mlp.experts.gate_up_proj,mlp.experts.down_proj`,
     target_parameter_lora_tensors `5934/5934`, e q/k/v/o com tensores LoRA
     presentes. Isto valida estrutura do adapter, nao ACC.
   - feito: V685 OpenRouter pos-V684 consult:
     `artifacts/openrouter/v685_post_v684_loop_consult/KG1_V685_OPENROUTER_PROMPT.md`,
     `artifacts/openrouter/v685_post_v684_loop_consult/KG1_V685_OPENROUTER_RESPONSES.md`,
     `artifacts/openrouter/v685_post_v684_loop_consult/KG1_V685_OPENROUTER_GPT_RETRY.md`,
     `artifacts/openrouter/v685_post_v684_loop_consult/KG1_V685_CONSENSUS.md`.
     Voto: GPT-5.5/Gemini/Qwen/Claude pedem `run_v684_weak_eval_now`;
     DeepSeek pediu micro-eval antes, mas tambem em A100. Decisao consolidada:
     preparar um unico weak eval `official_like` A100 bounded do checkpoint-10,
     com protected rows/format/canonicalizacao/cost gates e sem promocao por
     loss.
4. Parametros permitidos para uma futura tentativa A100, somente apos gates:
   - hardware: `a100-large`;
   - LoRA: `q_proj,k_proj,v_proj,o_proj`, MoE frozen-active, `lm_head` fora;
   - preferir menor blast radius (`r=8/alpha=8` ou `r=16/alpha=32`), rejeitar
     `r=64`, router treinavel e MoE destravado ate prova local especifica;
   - checkpoint/probe curto, abortando cedo se `avg_completion_tokens>256`,
     `max_completion_tokens>512`, `truncated>0`, `boxed_rate<1.0`,
     `no_box_fallback>0`, ou protected backfire.
5. Depois de cada treino concluido ou falho:
   - gerar prompt OpenRouter atualizado com artefatos reais do run;
   - executar consulta quando `OPENROUTER_API_KEY` estiver disponivel;
   - salvar prompt, raw results, responses, manifest e consenso em
     `artifacts/openrouter`;
   - classificar cada sugestao como consenso, hipotese, hardening ou rejeitada
     antes de alterar o roadmap.
6. Promover somente se:
   - total `>=196/315`;
   - bit `>=136/160`;
   - equation `>=60/155`;
   - truncation `0`;
   - no-box fallback `0`;
   - boxed rate `1.0`;
   - protected backfire `0`;
   - `label-aware - label-free == 0`.

## V697 Post Runtime Assert Consensus

- New local decision artifact:
  `artifacts/openrouter/v697_post_runtime_assert_review/KG1_V697_CONSENSUS_AND_LOCAL_DECISION.md`
- Qwen: proceed with one V689 5-step `a100-large` probe after launcher debug.
- DeepSeek: block due to bit-regression and loss-ACC risks.
- Local decision:
  - accept the bit-regression warning;
  - keep the checkpoint-5 weak-eval launcher as mandatory safety
    infrastructure;
  - reject CPU-only 30B gradient surgery and MoE target-parameter unfreeze for
    this run;
  - do not change LR or answer-span weights without a new CPU gate/review;
  - proceed only to launcher debug, then one 5-step A100 probe if clean.
- Promotion remains impossible without checkpoint-5 weak eval passing
  `total>=196`, `bit_manipulation>=136`, `equation_transform>=60`,
  `truncated=0`, `boxed_rate=1.0`, `no_box_fallback=0`,
  `avg_completion_tokens<=512`, `label_aware_delta=0`, and zero protected-row
  backfire.

## V689 A100 Probe Launch

- Job:
  `https://huggingface.co/jobs/felipesp1983/6a0ddc5accb6cd133d158a63`
- Run id:
  `v689-a100-ruletext-micro-v290ckpt6-20260520T160645Z`
- Manifest:
  `artifacts/v689_hf_a100_launch/v689-a100-ruletext-micro-v290ckpt6-20260520T160645Z_launch_manifest.json`
- Launch was allowed only after:
  - static safety gate passed with `findings=[]`;
  - pre-paid integration gate passed with `findings=[]`;
  - A100-large live cost was `0.041667 USD/min`;
  - active paid KG1 jobs were `[]`;
  - debug fixed two launcher issues before spend: remote f-string braces and
    missing explicit `KG1_DATA_REPO`/`KG1_DATA_ROOT` env propagation.
- This is a 5-step probe only. No package, promotion, weak eval promotion, or
  Kaggle submit is authorized until checkpoint-5 weak eval passes all gates.

## V700 Remote Preflight Correction And Relaunch

- Consensus artifact:
  `artifacts/openrouter/v700_v689_remote_preflight_failure_review/KG1_V700_CONSENSUS_AND_LOCAL_DECISION.md`
- The first V689 A100 train launch failed before training in remote preinstall
  because the remote gate did not accept the local alias
  `checkpoint5_eval_required`.
- Correction kept the reviewed experiment frozen and restored the pinned remote
  contract:
  `KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS=deferred_post_checkpoint` plus
  `KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT=1`.
- Remote preflight-only proof:
  `https://huggingface.co/jobs/felipesp1983/6a0ddf858229e585f969c728`,
  run id `v689-a100-ruletext-micro-v290ckpt6-20260520T162015Z`,
  status `COMPLETED`, marker `remote_preflight_only_ok`.
- Corrected full V689 train probe:
  `https://huggingface.co/jobs/felipesp1983/6a0de01b8229e585f969c732`,
  run id `v689-a100-ruletext-micro-v290ckpt6-20260520T162245Z`,
  manifest
  `artifacts/v689_hf_a100_launch/v689-a100-ruletext-micro-v290ckpt6-20260520T162245Z_launch_manifest.json`.
- No H200, no submit, no package, no promotion by loss. After completion,
  checkpoint-5 weak eval is mandatory and must pass all ACC/format/protected
  gates before any promotion decision.

## Criterio De Submit

Submit ao Kaggle somente se:

- adapter-only gerar ganho weak real;
- pacote contem adapter/config correto;
- eval oficial-like confirma ganho sem fallback;
- gates de hash, prompt, max_tokens, LoRA contract, parser e protected rows
  passam;
- o resultado nao depende de solver runtime nem de weak labels.
