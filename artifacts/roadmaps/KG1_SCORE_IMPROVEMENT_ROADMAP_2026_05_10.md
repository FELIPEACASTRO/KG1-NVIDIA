# KG1 Score Improvement Roadmap

Atualizado: 2026-05-15

Este e o roadmap ativo a partir do V435. O historico pre-V435 foi arquivado em:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V435_CLEANUP_2026_05_15.md`

O objetivo desta limpeza e remover rotas antigas, hipoteses sem transferencia e itens que nao serao mais usados. A execucao agora deve seguir apenas o plano abaixo.

## Objetivo

Melhorar o score submit-safe adapter-only nas familias fracas, sem perder o que ja esta bom.

Gate minimo para considerar ganho:

| Metrica | Baseline submit-safe | Minimo para promover |
|---|---:|---:|
| Total weak | 192/315 | > 192/315 |
| equation_transform | 56/155 | > 56/155, ideal >= 60/155 |
| bit_manipulation | 136/160 | >= 136/160 |
| truncated | 0 | 0 |
| Full official-like | 823/947 | > 823/947 |

Regra central: ganho so conta se aparecer no adapter/package. Teacher CPU, solver, verifier e postprocessor sao diagnostico ate virarem comportamento do adapter valido.

## Snapshot Atual

| Item | Resultado | Status |
|---|---:|---|
| Melhor adapter-only submit-safe, V291/V290 checkpoint-6 | weak 192/315, equation 56/155, bit 136/160, trunc 0 | referencia ativa |
| Full V291/V290 checkpoint-6 | 823/947, equation 56, bit 135, trunc 1 | melhor pacote conhecido |
| Melhor teacher CPU V414/V366 | weak 222/315, equation 63, bit 159 | nao submit-safe |
| Melhor projecao solver integrada V405 | weak 201/315, equation 63, bit 138 | nao submit-safe |
| OpenRouter V434/V434B/V434C | sem ganho pronto; trouxe filtros e rota V435 | evidencia para plano |
| V435 CPU pair gate inicial | 0/3558 pares aprovados | bloqueou GPU ate existir raw output real |
| V435C/V435D raw-output audit | 280 probes, 133 erros reais do adapter | base permitida para hard negatives |
| V435E antigo misto | 200 rows, 67 format-only negatives | arquivado; contaminava mean-NLL preference |
| V435E corrigido hard-negative-only | 133 rows, equation 120, bit 13 | unico dataset preference permitido agora |
| V435F corrigido | hard-only passa; dataset antigo e bloqueado | GPU so com novo launcher e novo kill-switch |
| V436B H200 hard-negative-only | checkpoint-3 piorou preference 6/24 -> 4/24 | cancelado; sem weak/full |
| V440 H200 V439 final-answer-only | checkpoint-3 empatou baseline 8/24; equation 7/22; bit 1/2 | cancelado; sem weak/full |
| V441 OpenRouter consult | DeepSeek/Qwen/Gemini consultados; 2 respostas completas dizem SIM com gates; Gemini truncou mas iniciou SIM | V441 boxed-payload e justificado, com kill-switch |
| V441 H200 boxed-payload | baseline 7/24; checkpoint-3 7/24; equation 6/22; bit 1/2 | cancelado; sem weak/full |
| V442 post-V441 route audit | 133 pares V439 auditados; 133 source-ok; 0 rule-certified | bloqueia novo GPU preference; volta para CPU certified builder |
| V445 prediction/parse audit | current, official reextract, first boxed e last boxed todos `192/56/136/0` | parser/extractor nao e gargalo |
| OpenRouter V446 uploaded consult | 16 slots, 11 respostas finais; consenso CPU-first target-alignment | evidencia para V446/V447 |
| V446 Tong-source target-alignment gate | `1310` traces aceitos: bit `848`, equation `462`; `hf_gpu_allowed=true` | sinal material novo; exige dataset/token gate antes de GPU |
| OpenRouter V447 public mining consensus | 6 modelos uteis; consenso condicional | minerar notebooks publicos em paralelo, mas priorizar V446 builder |
| V449 ACC metric integrity audit | weak scorer usa `verify_answer`; `answers_equivalent` superconta bit e foi bloqueado para `official_correct` | metrica de promocao confirmada; script diagnostico corrigido |
| V448 H200 V447 clean trace weak eval | checkpoint-3 `190/315`, equation `56/155`, bit `134/160`, trunc `1` | cancelado por FinOps; rota clean-trace SFT bloqueada |
| V450 transfer-debug audit | ACC path, weak scorer, family mapping e exact binary auditados | sem erro ativo de scoring; gargalo e transferencia adapter-only |
| V451 equation DSL v2 gap audit | V324 tem `+6` CPU solver-only; V443 tem `0` pares certificados e `120` no_unique_certified_rule | V452 precisa ampliar DSL/certificador antes de GPU |
| V452 equation DSL v2 certified builder | 133 rows auditadas; 7 candidatos; 2 pares certificados; 5 candidatos numericos reprovados | `hf_gpu_allowed=false`; nao abrir H200 por esta rota |
| V453 public Kaggle kernel mining | 30 kernels listados/analisados; 29 pull ok; raw notebooks apagados apos triagem | sem ganho submit-ready; reforca `lm_head`/target modules e mineracao publica CPU-only |
| V454 bit guardrail decision | teacher CPU chega a bit `159/160`, mas adapter-transfer V359/V368 fica em `134-135/160` | bit-only GPU bloqueado; bit vira replay/guardrail |
| V455 equation target audit | V324 tinha 6 candidatos no-loss; V452 cobriu 2 pares; faltam 4 rows verificadas em 3 classes numericas; simbolico verificado 0 | `hf_gpu_allowed=false`; V456 precisa fechar classes faltantes antes de GPU |
| V456 missing numeric class decision | 3 classes faltantes auditadas; 2 ja tinham sintético massivo sem transferir; 1 precisa raw probe; 0 elegiveis para treino | `hf_gpu_allowed=false`; nao repetir SFT sintetico |
| V457 public-train numeric probe pack | 22 prompts public-train sem answer para `minus_signed_opposite_sign_guarded`; pack sem chaves answer-like | `hf_raw_probe_allowed=true`; treino ainda bloqueado ate raw outputs reais |
| V458 HF raw-output probe | H200 inference-only nos 22 prompts V457; 22 outputs, stop-only, 0 labels no input | concluido; habilitou V459 CPU audit |
| V459 numeric hard-negative audit | 22 rows auditadas; adapter acerta 15, erra 7 exatamente no padrao opposite-sign; postprocessor corrige 22/22 | sinal real, mas 1 classe so; `hf_gpu_allowed=false` |
| V460 one-rule micro dataset | train 146 rows: 18 equation, 128 bit replay; val 36 rows; token gate passou, trunc 0 | GPU bloqueada ate aceitar risco explicito de micro-smoke de uma classe |
| V461 synthetic numeric probe pack | 56 prompts sem labels para 4 classes numericas | permitido apenas inference-only |
| V462 HF raw-output probe | H200 inference-only; 56/56 outputs, stop-only, 0 labels no input | concluido; habilitou V463 CPU audit |
| V463 synthetic numeric hard-negative audit | 26 hard negatives reais em 3 classes; prompt hashes 56/56; postprocessor 56/56 | autoriza V464 dataset CPU; treino ainda bloqueado |
| V464 numeric multirule dataset | train 558 rows: 46 equation, 512 bit replay; val 138 rows; token gate passou, trunc 0 | dataset pronto para smoke V465 |
| V465 H200 numeric multirule smoke | dataset/gate enviados ao HF; launcher com H200 <=1h, 16 steps, checkpoints 4/8/12/16 | proxima execucao paga, com kill-switch weak |

## Regras Permanentes

1. Weak/full sao somente avaliacao e gate. Nunca usar weak/full para construir pares, escolher misses, selecionar candidatos, gerar `chosen/rejected`, treinar, desempatar regra ou fazer cherry-pick.
2. `answer` nao pode participar da decisao de regra, filtro, tiebreak, `chosen`, selecao de trace ou selecao de candidato. A resposta correta so pode aparecer em auditoria posterior, depois da regra estar congelada.
3. `id`, prompt hash de weak/full, oracle, solver runtime, verifier runtime e postprocessor nao podem entrar no submit.
4. Submit valido e apenas adapter-only: `adapter_config.json` e `adapter_model.safetensors` no root do pacote. Sem script, tokenizer, prompt prefix, soft-prompt, `embed_tokens`, `lm_head`, decoder patch, logit mask, constrained decoding, runtime abstention ou confidence threshold.
5. Decisao e por ACC/truncation. `eval_loss`, `train_loss`, preference accuracy interna e probabilidades de LLM nao liberam submit nem GPU.
6. ACC de promocao deve ser calculada com `src.competition_utils.verify_answer`. `answers_equivalent` e diagnostico-only; ela superconta strings binarias por tolerancia numerica e nao pode alimentar `official_correct`, weak/full gate, promocao ou submit.
7. HF GPU so pode rodar depois de CPU gate com sinal material, manifest auditavel e `hf_gpu_allowed=true`.
8. FinOps: cancelar ou nao iniciar qualquer job que nao possa mais superar `total>192`, `equation>56`, `bit>=136`, `truncated=0`.
9. Toda nova versao precisa quadro comparativo contra V291/V290 e decisao explicita: promover, repetir CPU, cancelar ou arquivar.
10. Se notebook for criado ou alterado, precisa passar `python scripts/notebook_release_gate.py <notebook>` antes de entrega/push.
11. Se job estiver rodando, analisar logs periodicamente e aplicar kill-switch sem esperar gasto inutil.
12. Todo script, job launcher, workflow ou notebook criado/alterado precisa passar `python scripts/kg1_static_safety_gate.py <paths>` antes de entrega/push/execucao. O gate bloqueia V435E misto arquivado, `format_negative_*` em treino ativo, `ALLOW_FORMAT_NEGATIVES` em job/notebook e uso permissivo de `answers_equivalent` para `official_correct`.
13. H200 esta autorizada ate 1 hora por execucao. Se uma execucao precisar passar de 1 hora, parar e pedir autorizacao humana antes de continuar.
14. Todo erro novo deve entrar no ledger `artifacts/roadmaps/KG1_ERROR_LEDGER_2026_05_15.md` com evidencia, impacto, regra preventiva e status antes de abrir novo job pago.
15. Antes de qualquer job pago ou notebook operacional novo/alterado, rodar auditoria de integracao: launcher, dataset correto, conteudo do dataset, hashes, schema, targets, paths HF, adapter inicial, gates, kill-switch e comparacao contra baseline. Para HF jobs, usar `scripts/kg1_pre_paid_job_integration_gate.py` alem do static gate.
16. Acesso a modelos/datasets Hugging Face deve reutilizar o `HF_TOKEN` ja usado para criar/executar jobs. Nunca imprimir, commitar ou gravar a chave em artefatos. `.env*` fica ignorado; commitar apenas `.env.example` sem segredos.
17. Todo novo dataset de equation precisa passar por gate de alinhamento de target antes de GPU: alvo nativo verificado por rejection sampling ou alvo canonico com logprob/score pre-registrado contra base/V291. Target plausivel e condicao necessaria, nao opcional.
18. Todo treino que toque `equation_transform` precisa de bit replay/anchor pre-registrado. Piso exploratorio: `>=200` rows limpas; preferencia para H200 final: `>=800` rows, se disponiveis sem leakage.
19. Antes de qualquer GPU, rodar leakage forte: `id`, `prompt_sha256`, prompt normalizado e overlap `13-gram` contra weak/full. Qualquer hit bloqueia o dataset.
20. Antes de qualquer GPU, carregar um scaffold LoRA pelo caminho oficial vLLM/LoRA em prompt dummy. Se nao carrega como adapter-only, nao treinar.
21. One-shot policy: nao repetir a mesma receita paga se ela ja falhou weak/family gate. Nova GPU exige evidencia CPU nova e material.
22. Mineracao de notebooks publicos e permitida somente como extracao de tecnica/dataset/trace, nunca como uso direto de adapter, peso ou submissao de terceiros. Downloads grandes precisam ser apagados depois da triagem; o roadmap guarda apenas o achado validado.
23. V444/V448 provaram que target-aligned trace SFT, mesmo limpo e com token gate, nao transfere ganho para adapter-only. Nova GPU nao pode repetir essa classe de treino sem prova CPU nova de que o adapter parseia e emite a resposta curta correta, nao apenas que o teacher/trace esta correto.

## Achados Consolidados V434C

O triple check do arquivo `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026.json` e dos agentes convergiu para estes pontos:

| Achado | Decisao |
|---|---|
| ORPO/DPO pode ser util | Ativo somente se houver hard negatives reais do V291/V290 em dados permitidos |
| MDL, Leave-One-Out e renaming stability | Ativo como certificado label-free, nao como texto narrativo |
| Operator-conditioned slot alignment | Ativo; deve virar nucleo de `equation_pair_builder` |
| Bit-pair/bitsum/stride | Ativo como guardrail e trace curto para preservar bit |
| ANF-SAT/MaxSAT | P2 barato para bit, nao libera GPU sozinho |
| Targeted modules/freeze | P2; so apos V435 provar sinal |
| Prefix tokens, hidden-state distillation, runtime abstention | Removidos do plano ativo |
| Mais SFT amplo, mais epochs, LR sweep, H200 relaunch | Removidos ate existir dado novo |

## V435 - CPU Adapter-Level Pair Builder

Status: ativo.

Objetivo: construir um artefato CPU que prove, antes de gastar GPU, que ha pares submit-safe capazes de atacar erro real do V291/V290 e preservar bit.

V435 nao e treino. V435 e gate, contrato e gerador de pares. Se V435 nao passar, nao abrir HF GPU.

### V435.1 equation_pair_builder

Entrada permitida:

- Public train e sintetico auditado.
- Sem weak/full como fonte.
- Sem `answer` antes de congelar a regra.
- Sem linhas com overlap por `id`, `prompt_sha256` ou prompt normalizado contra weak/full.

Tecnicas permitidas:

- Anti-unification / E-generalization.
- CEGIS / SyGuS para programas pequenos.
- Slot alignment e substring alignment condicionado por operador.
- MDL para preferir regra curta.
- Leave-One-Out para rejeitar regra fragil.
- Renaming stability com renomeacoes bijetivas de simbolos.
- Rejeicao obrigatoria em caso de empate, near-tie ou candidato ambiguo.

Saida esperada:

- Pares `chosen/rejected` apenas quando a regra for unica e label-free.
- `rejected` deve vir de inferencia real do V291/V290 congelado sobre dados permitidos, com raw output, prompt hash e decode config registrados.
- Nao aceitar near-miss inventado como `rejected`.

### V435.2 bit_guardrail_builder

Objetivo: preservar `bit_manipulation>=136/160` enquanto equation tenta subir.

Tecnicas:

- Bit-pair / bitsum / stride curto inspirado na discussao de Tong Hui Kang.
- Traces deterministas curtos, preferencialmente <= 80-120 tokens.
- Hard negatives verificados por programa.
- ANF-SAT/MaxSAT somente como teacher P2 barato, se nao aumentar custo.

Regras:

- Nao perseguir ganho de bit via postprocess.
- Nao aceitar dataset de bit com contradicao de boxed answer.
- Manter proporcao minima de replay/guardrail contra pares de equation.

### V435.3 training_feasibility_gate

Artefato obrigatorio: `v435_pair_manifest.json`.

Schema minimo por par:

| Campo | Obrigatorio |
|---|---|
| source e split | sim |
| prompt_sha256 e prompt_normalized_sha256 | sim |
| family e rule_class | sim |
| program ou regra congelada | sim |
| MDL score e numero de candidatos | sim |
| Leave-One-Out result | sim |
| renaming stability stats | sim |
| slot/substring alignment stats | sim |
| chosen e rejected final answer | sim |
| V291/V290 raw wrong output | sim |
| adapter commit/path e decode config | sim |
| token lengths e truncation check | sim |
| leakage checks por id/hash/prompt | sim |
| `locked_before_answer_audit=true` | sim |

Condicoes para `hf_gpu_allowed=true`:

1. Pelo menos 4 modos de erro/regra independentes em equation, nao apenas 4 linhas.
2. Cada modo precisa hard negative real do V291/V290 em dados permitidos.
3. Volume de pares limpos suficiente por regra para justificar ORPO/DPO curto.
4. Zero overlap weak/full por `id`, `prompt_sha256` e prompt normalizado.
5. Tokenization com truncation 0 e offset/mask valido.
6. Bit guardrail pronto e sem contradicao.
7. Comparativo contra V291/V290 gerado automaticamente.
8. Manifest declara explicitamente `hf_gpu_allowed=true`.

Se qualquer condicao falhar, V435 termina com `hf_gpu_allowed=false` e nao abre GPU.

### Resultado V435 2026-05-15

Artefatos:

- `artifacts/v435_adapter_level_pair_gate/20260515T_v435_cpu_gate/v435_adapter_level_pair_gate_manifest.json`
- `artifacts/v435_adapter_level_pair_gate/20260515T_v435_cpu_gate/v435_adapter_level_pair_gate_pair_audit.csv`
- `artifacts/v435_adapter_level_pair_gate/20260515T_v435_cpu_gate/v435_adapter_level_pair_gate_decision.md`

Resultado:

| Item | Valor |
|---|---:|
| candidate pairs auditados | 3558 |
| approved pairs | 0 |
| approved equation rule modes | 0 |
| bit replay train/validation | 720 / 160 |
| programmatic bit hard negatives | 0 |
| `hf_gpu_allowed` | false |

Bloqueios principais:

- todos os pares existentes faltam raw output real do V291/V290;
- todos faltam identidade/config de decode do adapter;
- todos faltam `locked_before_answer_audit=true`;
- todos faltam certificados MDL, Leave-One-Out e renaming stability;
- bit tem replay, mas nao tem hard negatives programaticos prontos.

Decisao: nao abrir GPU de treino. O proximo passo seguro e coletar raw outputs reais do V291/V290 em prompts permitidos, sem usar answers.

### Resultado V435B 2026-05-15

Artefatos:

- `artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_manifest.json`
- `artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_prompts.jsonl`
- `artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_prompts.csv`

Resultado:

| Item | Valor |
|---|---:|
| public train rows vistos | 9500 |
| bit_manipulation vistos | 1602 |
| equation_transform vistos | 1555 |
| prompts exportados equation | 600 |
| prompts exportados bit | 240 |
| total prompts exportados | 840 |
| answers exportadas | 0 |
| weak/full rows removidas por overlap | 315 |

Decisao: o pack V435B e permitido para coleta de raw outputs reais do adapter. Ele nao e dataset de treino e nao libera V436 sozinho.

### V435C - Adapter Probe Raw Outputs

Status: concluido HF inference-only.

Artefatos preparados:

- `scripts/run_v435c_adapter_probe_raw_outputs.py`
- `artifacts/v435c_adapter_probe_raw_output_launch/launch_v435c_hf_adapter_probe_raw_outputs.py`

Contrato:

| Item | Valor |
|---|---|
| adapter | V291/V290 checkpoint-6 |
| input | V435B prompt-only pack |
| labels no input | sim |
| scoring | nao |
| treino | nao |
| submit/package | nao |
| default hardware | `h200` |
| default caps | equation 200, bit 80 |
| cost gate | `unit_cost_usd <= 0.09/min` |

Objetivo: obter `raw_output`, `prediction`, prompt hash, rendered prompt hash, decode config e identidade do adapter para reexecutar V435 com hard negatives reais. Se V435C falhar por OOM/infra, tentar H200 somente se a coleta ainda for o menor gasto para desbloquear o gate; caso contrario cancelar por FinOps.

Execucoes:

| Run | Hardware | Status | Decisao |
|---|---|---|---|
| `v435c-adapter-probe-raw-20260515T141708Z` | `a100-large` | erro rapido | driver A100 incompativel com imagem CUDA 13; launcher agora bloqueia essa combinacao por default |
| `v435c-adapter-probe-raw-20260515T141924Z` | `h200` | concluido | `280` raw outputs coletados sem labels; `80` bit, `200` equation, `0` truncation |

Artefatos:

- HF job: `https://huggingface.co/jobs/felipesp1983/6a072bade48bea4538b9e4e2`
- HF dataset output: `https://huggingface.co/datasets/felipesp1983/kg1-v435c-adapter-probe-raw-outputs/tree/main/runs/v435c-adapter-probe-raw-20260515T141924Z`
- local: `artifacts/v435c_adapter_probe_raw_outputs/20260515T141924Z_hf_download/`

### V435D - Adapter Probe Output Analysis

Status: concluido.

Artefato:

- `scripts/analyze_v435d_adapter_probe_outputs.py`

Objetivo: depois que V435C publicar `raw_outputs.csv`, juntar os outputs com `competition_train.csv` por `id`, somente entao usar a resposta publica de treino para medir quais prompts permitidos o adapter errou. Essa etapa gera inventario de misses por familia e deve alimentar o construtor de pares certificados. Ela nao libera treino sozinha.

Resultado H200 2026-05-15:

| Familia | Rows | Correct | ACC | Misses | Trunc |
|---|---:|---:|---:|---:|---:|
| `bit_manipulation` | 80 | 67 | 83.75% | 13 | 0 |
| `equation_transform` | 200 | 80 | 40.00% | 120 | 0 |
| OVERALL | 280 | 147 | 52.50% | 133 | 0 |

Artefatos:

- `artifacts/v435d_adapter_probe_output_analysis/20260515T141924Z_hf/v435d_adapter_probe_output_analysis_manifest.json`
- `artifacts/v435d_adapter_probe_output_analysis/20260515T141924Z_hf/v435d_adapter_probe_output_analysis_summary.csv`
- `artifacts/v435d_adapter_probe_output_analysis/20260515T141924Z_hf/v435d_adapter_probe_output_analysis_misses.csv`

Decisao: V435D produziu o primeiro inventario permitido de erros reais do adapter congelado fora de weak/full. Isto libera construir pares exact-wrong, mas ainda nao e ganho de ACC.

### V435E - Adapter Exact-Wrong Preference Dataset

Status: corrigido em 2026-05-15. A versao antiga foi arquivada.

Artefato:

- `scripts/build_v435e_adapter_probe_preference_dataset.py`

Politica corrigida:

- Usar somente public-train probes de V435D.
- `chosen`: completion curta com resposta publica correta em `\boxed{}` com escape de braces/backslash.
- `rejected`: completion curta com a predicao errada real do adapter V291/V290.
- Guardrail de bit: rows bit corretas nao entram como preferencia por padrao.
- Negativos de formato (`format_negative_*`) sao permitidos apenas para diagnostico explicito com `--include-format-negatives`; nao podem alimentar GPU preference training.
- `weak/full`: usados somente para filtro de overlap, nunca como treino.

Erro encontrado:

- A versao anterior de V435E misturou `133` hard negatives semanticos com `67` negativos so de formato (`format_negative_format_no_box`).
- Isso contaminava o objetivo mean-NLL: o rejected podia ser semanticamente correto e apenas menor/sem box, entao o treino podia aprender preferencia de formato/comprimento, nao resolver puzzle.
- Esse erro explica por que o V436 teve metrica interna ruim e nao deve ser repetido.

Resultado antigo arquivado:

| Split | Rows | bit | equation | hard negatives | bit replay |
|---|---:|---:|---:|---:|---:|
| train | 160 | 62 | 98 | 109 | 51 |
| validation | 40 | 18 | 22 | 24 | 16 |
| all | 200 | 80 | 120 | 133 | 67 |

Resultado corrigido hard-negative-only:

| Split | Rows | bit | equation | hard negatives | format negatives |
|---|---:|---:|---:|---:|---:|
| train | 109 | 11 | 98 | 109 | 0 |
| validation | 24 | 2 | 22 | 24 | 0 |
| all | 133 | 13 | 120 | 133 | 0 |

Rule classes de equation cobertas:

- `equation_numeric_operator_to_number`: 20
- `equation_numeric_operator_to_symbolic`: 4
- `equation_symbolic_sequence`: 64
- `equation_symbolic_short`: 32

Hashes corrigidos:

- train: `9dfe89d6da803e593da566ec72865815ccbadb4b42385f82bb7ae2e9d0ad240b`
- validation: `92cd619bc7e315e930742cdf14978452f93d25ae07c28e7ded2ba91816c2d503`
- manifest: `artifacts/v435e_adapter_probe_preference_dataset/20260515T_v435e_hardneg_only/v435e_adapter_probe_preference_dataset_manifest.json`

HF dataset status:

- O dataset antigo em HF com path `20260515T_v435e_from_h200_probe` esta arquivado para diagnostico e nao deve ser usado para treino.
- O hard-negative-only foi publicado em `felipesp1983/kg1-nemotron-training` no path `data/v435e_adapter_probe_preference/20260515T_v435e_hardneg_only`.
- Commit HF dataset: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/ccc53c6412cb6f94a03f1b7ec5482e4c7f0bf7cb`.
- Verificacao HF: `4` arquivos presentes no path novo.

### V435F - Adapter Probe Preference Gate

Status: corrigido em 2026-05-15.

Artefato:

- `scripts/run_v435f_adapter_probe_preference_gate.py`
- `artifacts/v435f_adapter_probe_preference_gate/20260515T_v435f_hardneg_only/v435f_adapter_probe_preference_gate_manifest.json`
- `artifacts/v435f_adapter_probe_preference_gate/20260515T_v435f_old_mixed_recheck_after_fix/v435f_old_mixed_recheck_after_fix_manifest.json`

Gate corrigido sobre hard-negative-only:

| Condicao | Resultado |
|---|---|
| all rows approved | true |
| approved rows >= 120 | true (`133`) |
| equation hard negatives >= 100 | true (`120`) |
| bit hard negatives >= 10 | true (`13`) |
| format negatives absent | true (`0`) |
| equation rule classes >= 4 | true (`4`) |

Recheck do dataset antigo misto:

| Condicao | Resultado |
|---|---|
| rows | `200` |
| hard negatives | `133` |
| format negatives | `67` |
| `format_negatives_absent_for_preference` | false |
| decisao | `v435f_blocks_gpu` |

Decisao: o gate corrigido permite apenas o hard-negative-only. O launcher V436 antigo foi travado fail-closed; qualquer novo GPU smoke deve usar um launcher novo, hashes hard-only e kill-switch no primeiro checkpoint.

## V436 - Short Adapter-Only Smoke

Status: executado e cancelado por FinOps no primeiro checkpoint.

Objetivo: testar transferencia real para adapter, com gasto minimo.

Configuracao inicial:

- Preservar config e target modules do V291/V290.
- Preference objective curto, usando somente pares exact-wrong aprovados no V435F.
- Evitar loss medio diluido: ponderar span final da resposta/boxed answer quando a implementacao permitir.
- Sem mudanca de tokenizer, prompt oficial, runtime ou pacote.
- Config V436 inicial: H200, `24` steps, checkpoint/eval a cada `4`, LR `2e-8 -> 4e-9`, `PREF_BETA=0.20`, `PREF_LOSS_WEIGHT=1.5`, `CHOSEN_CE_WEIGHT=0.30`.

Resultado 2026-05-15:

- Job: `https://huggingface.co/jobs/felipesp1983/6a0732913308d79117b90436`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v436-v435e-preference-v290ckpt6`.
- Preflight remoto OK: H200 CUDA 12.8, hashes V435E OK, tokenizacao sem truncation, adapter V290 checkpoint-6 carregado com `12011/12011` tensores mapeados.
- Baseline preference eval V435E: `6/40 = 15.0%`; bit `2/18`, equation `4/22`.
- Step 4 preference eval: `5/40 = 12.5%`; bit `2/18`, equation `3/22`.
- Decisao: cancelado. O primeiro checkpoint piorou a metrica de preferencia e nao justificou continuar gastando H200.
- Artefato: `artifacts/v436_hf_h200_v435e_preference_launch/v436-v435e-pref-v290ckpt6-20260515T144849Z_finops_decision.json`.

Kill-switch no primeiro checkpoint:

| Condicao | Acao |
|---|---|
| total <= 192 | cancelar |
| equation <= 56 | cancelar |
| bit < 136 | cancelar |
| truncated > 0 | cancelar |
| package incompatibilidade | cancelar |
| ganho weak e bit preservado | continuar para full gate |

Conclusao: V436 nao trouxe ganho submit-safe e nao autoriza V437. O motivo provavel foi dado contaminado por negativos so de formato. Nao repetir o launcher V436 antigo.

## V436B - Hard-Negative-Only Preference Smoke

Status: executado em H200 e cancelado por FinOps no primeiro checkpoint.

Objetivo: repetir o smoke de preference usando somente V435E hard-negative-only corrigido, sem os `67` negativos so de formato.

Configuracao:

- Job: `https://huggingface.co/jobs/felipesp1983/6a073c74e48bea4538b9e652`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v436b-v435e-hardneg-v290ckpt6`.
- Dataset HF: `felipesp1983/kg1-nemotron-training/data/v435e_adapter_probe_preference/20260515T_v435e_hardneg_only`.
- Train/val hashes confirmados: `9dfe89d6da803e593da566ec72865815ccbadb4b42385f82bb7ae2e9d0ad240b` e `92cd619bc7e315e930742cdf14978452f93d25ae07c28e7ded2ba91816c2d503`.
- Static safety gate remoto OK, preinstall/artifacts/postinstall gates OK, tokenizacao sem truncation, adapter V290 checkpoint-6 carregado com `12011/12011` tensores mapeados.
- Trainable LoRA: `8,015,872` parametros, `0.0247%` do modelo, modulos `q_proj,k_proj,v_proj,o_proj,lm_head`.

Resultado:

| Metrica interna | Baseline V290 ckpt-6 | V436B checkpoint-3 | Decisao |
|---|---:|---:|---|
| preference hard-negative total | 6/24 | 4/24 | piorou |
| equation_transform preference | 4/22 | 2/22 | piorou |
| bit_manipulation preference | 2/2 | 2/2 | preservou |

Decisao: cancelado. O primeiro checkpoint piorou a preferencia hard-negative e nao autoriza weak/full eval, package ou submit.

Implicacao tecnica: o problema nao era apenas o dataset misto. O objetivo `mean_nll` chosen/rejected continua sem transferencia confiavel para ACC. A linha preference direta fica bloqueada ate existir novo gate CPU que prove orientacao correta do objetivo ou uma forma de treinar resposta final sem mover a distribuicao contra os pares.

## V438 - Error Ledger e Preference Objective Audit

Status: executado localmente em CPU.

Artefatos:

- `scripts/audit_v438_preference_objective.py`.
- `artifacts/v438_preference_objective_audit/20260515T_v438_v435e_hardneg_only/v438_v435e_hardneg_only_preference_objective_audit_manifest.json`.
- `artifacts/roadmaps/KG1_ERROR_LEDGER_2026_05_15.md`.
- `artifacts/openrouter/KG1_V438_ERROR_LEDGER_EXTERNAL_API_PROMPT_2026_05_15.md`.

Resultado:

| Check | Resultado |
|---|---:|
| answer boxes semanticamente corretos | 133/133 |
| rejected boxes semanticamente iguais ao adapter wrong | 133/133 |
| chosen menciona adapter prediction errado | 123/133 |
| chosen menciona public-train label audit | 133/133 |
| chosen tokens medio | 34.08 |
| rejected tokens medio | 26.80 |

Decisao: o erro dominante agora e estrutural no target, nao no label. O chosen ensina texto de auditoria e repete a resposta errada, enquanto o submit precisa aprender uma resposta final curta. Novo GPU job fica bloqueado ate V439 gerar e auditar targets `final-answer-only` sem esses contaminantes.

## V439 - Final-Answer-Only Pair Cleanup

Status: executado localmente em CPU; pronto para publicacao HF se decidirmos rodar smoke.

Artefatos:

- `scripts/build_v439_final_answer_only_pairs.py`.
- `artifacts/v439_final_answer_only_pairs/20260515T_v439_final_answer_only/v439_final_answer_only_pairs_manifest.json`.
- `artifacts/v438_preference_objective_audit/20260515T_v438_v439_final_answer_only/v438_v439_final_answer_only_audit_manifest.json`.

Dataset:

| Split | Rows | equation | bit | SHA256 |
|---|---:|---:|---:|---|
| train | 109 | 98 | 11 | `bc032da2f7cada19aef295aa91aef6098e03c7b85215e7729f1ddd71b3e5079a` |
| val | 24 | 22 | 2 | `57321347f9293e9c0f2f17e6c9de1d88f1246fee4154125574b2e60251aee3a6` |

V438 audit sobre V439:

| Check | Resultado |
|---|---:|
| answer boxes corretos | 133/133 |
| rejected boxes iguais ao adapter wrong | 133/133 |
| chosen menciona adapter prediction fora do boxed | 0/133 |
| chosen menciona public-train label audit | 0/133 |
| chosen tokens medio | 4.83 |
| rejected tokens medio | 4.80 |

Decisao: V439 corrige E003. Ele nao prova ganho de ACC, mas e o primeiro dataset limpo para um smoke curto. Se rodar, usar H200 por menos de 1 hora, checkpoint/eval no step 3 e cancelar se nao melhorar a metrica interna contra o baseline V439.

## V440 - H200 Smoke Final-Answer-Only

Status: executado em H200 e cancelado por FinOps no checkpoint-3.

Objetivo: testar se V439 final-answer-only, sem contaminacao textual do chosen,
melhora o objetivo interno de preference antes de gastar weak/full.

Configuracao:

- Job: `https://huggingface.co/jobs/felipesp1983/6a07467be48bea4538b9e722`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v440-v439-final-answer-v290ckpt6`.
- Dataset HF: `felipesp1983/kg1-nemotron-training/data/v439_final_answer_only_pairs/20260515T_v439_final_answer_only`.
- Adapter inicial: `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.
- H200, max `12` steps, eval/save no step `3`, timeout `3600s`.
- Integration gate local e remoto: OK, zero findings.
- Tokenizacao: `0` truncation, offset masks OK.
- Adapter load: `12011/12011` tensores mapeados.
- Trainable LoRA: `8,015,872` parametros, `0.0247%`.

Resultado:

| Metrica interna V439 val | Baseline V290 ckpt-6 | V440 checkpoint-3 | Decisao |
|---|---:|---:|---|
| preference hard-negative total | 8/24 | 8/24 | empatou |
| equation_transform preference | 7/22 | 7/22 | empatou |
| bit_manipulation preference | 1/2 | 1/2 | empatou |

Decisao: cancelado. V440 corrigiu o target contaminado de V436B, mas nao gerou
sinal material no primeiro checkpoint. Nao autoriza weak/full, package ou
submit.

Implicacao tecnica: a linha `mean_nll` preference, mesmo com final-answer-only,
nao e suficiente. O proximo passo nao deve ser mais epoch nem LR sweep nessa
mesma formulacao. Precisamos mudar o objetivo ou voltar para CPU gate de solver
/ DSL que produza novos pares com cobertura/sinal diferente.

## V441 - API Consult e Boxed-Payload Preference

Status: executado e cancelado por FinOps; nao promove weak/full/submit.

Consulta externa:

- Artefato: `artifacts/openrouter/v441_boxed_payload_decision_api_consult/`.
- Modelos consultados: `deepseek/deepseek-v3.2`,
  `qwen/qwen3.6-max-preview`, `google/gemini-3.1-pro-preview`.
- DeepSeek: SIM, com ressalvas; V441 e justificavel porque foca o gradiente no
  payload, mas pode nao resolver raciocinio interno.
- Qwen: SIM; considera V441 a correcao mecanica direta para a falha de V440,
  desde que haja mascara nao vazia, drift check e kill-switch no checkpoint-3.
- Gemini: resposta truncada, mas iniciou com SIM e a mesma justificativa de
  diluicao de sinal.

Decisao: V441 e permitido como smoke curto porque muda tecnicamente o objetivo.
Nao e repeticao de V440.

Mudanca tecnica:

- Novo `PAIR_SCORE_MODE=boxed_payload_mean_nll`.
- O trainer calcula `score_loss_mask` somente nos tokens dentro do payload do
  ultimo `\boxed{...}`.
- Boilerplate `Final answer:` e wrappers `\boxed{}` deixam de entrar no score
  preference e no chosen CE.
- O trainer aborta se qualquer payload mask tiver zero tokens ou se houver drift
  entre tokenizacao do chat template e a mascara.

Validacoes obrigatorias:

- Dataset V439 hash/rows/family/subcategory preservados.
- `chosen` e `rejected` continuam final-answer-only, sem format negatives.
- `score_loss_mask` nao pode ter zero tokens em nenhum par.
- Integration gate exige explicitamente
  `--expected-pair-score-mode boxed_payload_mean_nll`.
- Primeiro checkpoint precisa melhorar preference accuracy interna contra o
  baseline V439; caso contrario cancelar por FinOps.

Preflight local executado:

- `py_compile`: OK.
- `kg1_static_safety_gate.py`: OK, sem findings.
- `kg1_pre_paid_job_integration_gate.py`: OK, sem findings.
- Tokenize-only dry-run com tokenizer Nemotron real: OK.
- Pares tokenizados: treino `109/109`, validacao `24/24`.
- Truncation: `0`.
- Offset mask fallback: `0`.
- Score mask payload: treino `chosen=339`, `rejected=376`; validacao
  `chosen=75`, `rejected=78`.

Risco honesto: mesmo que V441 melhore preference interna, ainda pode nao
transferir para weak ACC. Preference interna e so kill-switch barato; promocao
continua exigindo weak/full por familia.

Resultado HF:

- Job: `https://huggingface.co/jobs/felipesp1983/6a075046e48bea4538b9e7d3`.
- Output repo parcial: `felipesp1983/kg1-nemotron-lora-v441-v439-boxed-payload-v290ckpt6/checkpoint-3`.
- Baseline V441 val: `7/24`, equation `6/22`, bit `1/2`.
- Checkpoint-3: `7/24`, equation `6/22`, bit `1/2`.
- Decisao: cancelado. A mascara de payload funcionou tecnicamente, mas nao
  produziu sinal interno no primeiro checkpoint. Nao ha base para weak/full ou
  submit.

## V442 - Post-V441 Route Audit

Status: executado em CPU; bloqueia novo GPU preference sobre V435E/V439.

Artefatos:

- `scripts/analyze_v442_post_v441_route_audit.py`
- `artifacts/v442_post_v441_route_audit/20260515T_v442_post_v441_route_audit/v442_post_v441_route_audit_manifest.json`
- `artifacts/v442_post_v441_route_audit/20260515T_v442_post_v441_route_audit/v442_post_v441_route_audit_pair_certification_audit.csv`
- `artifacts/v442_post_v441_route_audit/20260515T_v442_post_v441_route_audit/v442_post_v441_route_audit_report.md`

Resultado:

| Item | Valor |
|---|---:|
| pares V439 auditados | 133 |
| source-ok rows | 133 |
| weak/full training rows | 0 |
| rule-certified rows | 0 |
| V441 checkpoint-3 delta preference total | 0 |
| V441 checkpoint-3 delta equation | 0 |
| V441 checkpoint-3 delta bit | 0 |

Interpretacao:

- O dataset V439/V435E esta limpo contra vazamento weak/full e serve como
  diagnostico.
- Ele nao tem `rule_unique_label_free`, `program_or_rule`, `mdl_score`,
  `leave_one_out_pass`, `renaming_stability_pass`, `slot_alignment_stats` nem
  `rule_frozen_before_answer`.
- Portanto, o dataset nao justifica outro job pago por si so. A ausencia do
  certificado explica por que V436B/V440/V441 mudaram o objetivo mas nao
  transferiram ACC.

Decisao:

- Nao relancar V435E/V439 preference com mais steps, LR, H200 ou payload loss.
- Proxima implementacao obrigatoria: CPU certified equation pair builder.
- Novo HF GPU so volta se o builder gerar pares com regra unica e pelo menos
  quatro modos independentes de `equation_transform`, preservando `bit>=136`.

## Regra De Integracao Pre-Job

Status: implementada em `scripts/kg1_pre_paid_job_integration_gate.py`.

O gate deve rodar antes de qualquer execucao paga ou longa. Ele verifica:

- launcher aponta para o dataset esperado, hashes esperados, adapter inicial e output repo corretos;
- timeout H200 fica em `3600` segundos e custo unitario respeita o teto;
- primeiro checkpoint/eval existe no step `3` para kill-switch cedo;
- dataset local tem row count, SHA, families, subcategories e `negative_type` esperados;
- `chosen` e `rejected` usam template final-answer-only, exatamente um `\boxed{}`;
- prompt de sistema do launcher deve estar alinhado ao target final-answer-only;
- `chosen` nao contem auditoria, resposta errada, texto de adapter ou contaminacao de target;
- flags `gate_rows_used_for_training`, `weak_gate_rows_used_for_training` e `full_gate_rows_used_for_training` sao `false`;
- manifest V438 declara `hf_gpu_allowed_for_same_objective=true` e zero mismatches.

Decisao: se esse gate falhar, nao abrir HF/Kaggle GPU. Primeiro corrigir o dado ou o launcher.

## V437 - Full Gate, Package e Submit

Status: bloqueado. V436 e V436B falharam o primeiro kill-switch.

Regras:

1. Rodar full official-like somente se weak gate superar V291/V290.
2. Package somente adapter-only root.
3. Submit somente com ganho medido contra o submit historico.
4. A descricao do submit deve declarar versao, checkpoint, weak, full, equation, bit e truncation.
5. Se full nao superar 823/947 ou trouxer regressao de bit/truncation, arquivar e nao submeter.

## P2 Somente Apos V435

Estes itens nao estao ativos agora. So entram se houver novo gate CPU mais forte que V435E:

- Targeted module/freeze controlado para reduzir regressao de bit.
- TIES/DARE/SLERP apenas se houver adapters novos com sinais complementares medidos.
- Distilacao de teacher externa apenas como dados permitidos e auditados, sem weak/full.
- ANF-SAT/MaxSAT para bit se V435 mostrar que ha ganho barato e sem risco.
- Dataset sintetico adicional apenas se passar anti-leakage, MDL/LOO e gate de transferencia.

## Removidos Do Plano Ativo

| Item removido | Motivo |
|---|---|
| Broad SFT, mais epochs, LR sweep | Repetiu teto `equation=56` e frequentemente perdeu bit |
| V391/V398/V413/V416 rawstyle/teacher transfer | CPU bom nao transferiu para adapter |
| Prompt sweep amplo, thinking variants | Sem ganho submit-safe |
| Adapter soups V388/V389 | Sem complemento real e risco de regressao |
| DSLs V421-V433 ja fechadas | Zero ganho novo ou dependencia de label/tiebreak |
| Solver/verifier/postprocessor direto | Nao e package adapter-only |
| Runtime abstention, constrained decoding, logit masks | Fora do contrato submit-safe |
| Prefix/soft-prompt/`embed_tokens` | Risco de nao carregar no vLLM LoRA oficial |
| Hidden-state distillation | Incompativel e exigiria runtime externo |
| Public adapters drop-in | Nao superaram baseline ou quebram compatibilidade |
| Google Drive/CSV archaeology sem novo sinal | Arquivado como evidencia, nao acao |
| `competition_test.csv` como eval | Contem poucas linhas e pode sobrepor train |
| Weak/full misses como treino | Leakage/cherry-pick |
| Decisao por `eval_loss` | Nao correlacionou com ACC das familias |
| H200/A100 relaunch sem V435 | Gasto sem novo sinal |
| V436 preference variants sobre V435E antigo | Dataset misto tinha `67` format negatives; launcher antigo agora falha fechado |
| Preference mean-NLL direto sobre V435E hard-only | V436B piorou checkpoint-3 de `6/24` para `4/24` |
| Chosen com texto `public-train label audit` e candidato errado | V438 mostrou contaminacao de target em `133/133` e `123/133` linhas |

## Atualizacao V443/V444 - 2026-05-15

Estado medido antes do proximo job:

| Item | Valor |
|---|---:|
| melhor adapter-only weak | `192/315` |
| `equation_transform` weak | `56/155` |
| `bit_manipulation` weak | `136/160` |
| truncation weak | `0` |
| melhor full official-like conhecido | `823/947` |

Resultado V443:

| Item | Valor |
|---|---:|
| linhas auditadas | `133` |
| linhas equation analisadas | `120` |
| candidatos certificados | `0` |
| pares certificados | `0` |

Interpretacao: o builder de regras simples/slot-map nao encontrou uma regra
unica, LOO e renaming-stable. Isso fecha a rota de gerar pares certificados por
substituicao textual simples. Nao ha autorizacao tecnica para relancar V439 ou
V441 com mais steps.

Resultado V444 CPU:

| Item | Valor |
|---|---:|
| fonte | `sft_reconstructed.jsonl` |
| status mantidos | `rule_found`, `hypothesis_formed` |
| status removido | `rule_unknown` |
| train rows | `1848` |
| val rows | `172` |
| train bit/equation | `1219 / 629` |
| val bit/equation | `112 / 60` |
| train SHA | `4b064ed04401c6632798c470f76225688e0af3b0771dc65225d32cc283f439cc` |
| val SHA | `7a6ba5a60575f34f04f721b3c2312147a33fbbea6d3e27fbf9063ab8f4ef361e` |
| tokenization gate | passed |
| prompt truncation | `0` |
| completion truncation | `0` |

Interpretacao: V444 e diferente de V397/V398 porque remove `822` linhas
`rule_unknown`, que provavelmente diluiram o sinal. Ainda nao e ganho. E apenas
o menor teste pago justificavel depois que V443 fechou a rota de regras simples.

## Atualizacao V445/V446 - 2026-05-15

V444 H200 foi executado e o weak eval foi cancelado por FinOps no primeiro
checkpoint avaliado.

Resultado V444 checkpoint-2:

| Item | Baseline submit-safe | V444 ckpt-2 | Decisao |
|---|---:|---:|---|
| Total weak | `192/315` | `190/315` | reprova |
| `equation_transform` | `56/155` | `56/155` | sem ganho |
| `bit_manipulation` | `136/160` | `134/160` | regressao |
| truncated | `0` | `1` | regressao |

Job: `felipesp1983/6a075f1a3308d79117b907ff`.

Decisao: cancelar antes de avaliar checkpoint-4. O primeiro checkpoint nao
manteve `bit>=136`, nao aumentou equation e introduziu truncation. Isso fecha a
rota V444 high-confidence SFT como fonte de submit.

Resultado V445 prediction/parse audit:

| Estrategia | Total weak | Equation | Bit | Trunc | Decisao |
|---|---:|---:|---:|---:|---|
| current prediction | `192/315` | `56/155` | `136/160` | `0` | baseline |
| official reextract raw | `192/315` | `56/155` | `136/160` | `0` | sem ganho |
| first boxed | `192/315` | `56/155` | `136/160` | `0` | sem ganho |
| last boxed | `192/315` | `56/155` | `136/160` | `0` | sem ganho |

Interpretacao: parser, boxed extraction e re-extracao nao explicam o teto. O
problema e geracao/raciocinio transferido para o adapter, nao pos-processamento.

Resultado do arquivo OpenRouter V446:

- Fonte: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026 (1).json`.
- Artefato local: `artifacts/openrouter/v446_uploaded_chat_analysis/KG1_V446_OPENROUTER_FILE_ANALYSIS_SUMMARY.md`.
- 16 slots/modelos no export; 11 respostas finais usaveis; 5 sem resposta final
  acionavel.

Consenso util:

| Achado | Decisao no roadmap |
|---|---|
| FinOps e promotion gates estao corretos | manter e endurecer com novos pre-gates |
| SFT amplo, mais epochs, LR sweep e preference sem novo dado repetem falha | nao rodar |
| Falta target treinavel com estado intermediario | implementar target-alignment gate |
| Equation precisa de trace/trajectory ou DSL composicional verificavel | V446/V447 CPU-first |
| Bit deve ser preservado, nao reaprendido | bit anchor/replay obrigatorio |
| Weak/full nao podem virar fonte de treino | rejeitar sugestoes baseadas em weak misses |

Rejeicoes explicitas do V446:

| Sugestao vista nas respostas | Motivo de rejeicao |
|---|---|
| Treinar com weak misses/120 misses | leakage/cherry-pick |
| Parser/verifier no adapter package | viola adapter-only |
| Modulo/layer sweep sem target certificado | custo sem evidencia |
| Synthetic generico sem certificacao | ruido e regressao provaveis |
| Relaxar `bit` para `134/135` | perde baseline submit-safe |

Double check V446D de URLs/fontes do arquivo OpenRouter:

- Fonte auditada: `C:\Users\davis\Downloads\OpenRouter Chat Fri May 15 2026 (1).json`.
- Artefato: `artifacts/openrouter/v446_uploaded_chat_analysis/KG1_V446_URL_DOUBLE_CHECK.md`.
- URLs extraidas: `101`; URLs de conteudo/desafio: `61`; metadados de
  provedor/status/TOS/favicon: `40`.
- Kaggle CLI confirmou paginas oficiais de `rules`, `evaluation` e
  `data-description`; a avaliacao continua sendo adapter-only LoRA rank `<=32`
  via vLLM, `temperature=0.0`, `max_tokens=7680`, `max_model_len=8192`.
- Tiebreak oficial: em empate, vence a submissao enviada primeiro. Isso reforca
  que uma submissao nova so faz sentido com ganho medido, nao com empate.

Achados acionaveis do V446D:

| Fonte | Achado | Decisao |
|---|---|---|
| `tonghuikang/nemotron` | repo publico da submissao Progress Prize; contem `reasoners`, `corpus`, metricas e pipeline | usar como fonte de algoritmos para gate CPU, nao como submit pronto |
| `reasoners/equation_numeric.py` | inventario concreto: concat/reverse concat, add/sub/mul, abs diff, `+1/-1`, div/mod, digitos, determinante, reversao de operandos/resultado, prefix/suffix | base obrigatoria da DSL v2 |
| `reasoners/bit_manipulation.py` | bit-pair/bitsum/stride, matching por colunas e preservacao de regra | bit deve ser protegido com replay/anchor; nao reaprendido por SFT amplo |
| `andy279/nemotron-reasoning-challenge*` | README documenta SFT/teacher traces, solver-guided transformation/bit traces; payload esta gated/401 neste ambiente | usar somente copias locais ja fornecidas e apenas apos provenance + anti-leakage gate |
| `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` | espelho publico Apache-2.0 de `train.csv`/`test.csv`, 9.5k train | serve para hash/provenance, nao e sinal novo |
| NVIDIA/vLLM/NeMo docs | confirmam scaffold e compatibilidade de loader | manter scaffold/LoRA package gate |

Rejeicoes adicionais do V446D:

| Fonte | Motivo |
|---|---|
| Kaggle discussion URLs no export | HTTP retorna shell generico; Kaggle CLI 2.0.2 nao expoe corpo de discussion; sem novo conteudo extraido |
| `passagereptile455/*` | HF API mostra somente `.gitattributes`; nao ha adapter/pesos |
| `GaryNENE/nemotron-nano-8b-reasoning-lora` | base 8B e rank 128; nao e submit-compativel com Nemotron 3 Nano 30B A3B rank<=32 |
| datasets genericos NVIDIA Math/ReasoningGym/logical-puzzles | P2 para fixtures/estilo de trace; nao autorizam GPU sem target-alignment |

## Atualizacao V446/V447 - Target Alignment e Public Mining

V446 CPU gate foi implementado e executado.

Artefatos:

- Script: `scripts/run_v446_tong_source_target_alignment_gate.py`.
- Manifesto: `artifacts/v446_tong_source_target_alignment_gate/20260515T_v446_cpu_gate/v446_tong_source_target_alignment_gate_manifest.json`.
- Source inventory: `artifacts/v446_tong_source_target_alignment_gate/20260515T_v446_cpu_gate/v446_tong_source_target_alignment_gate_source_inventory.json`.
- Audit CSV: `artifacts/v446_tong_source_target_alignment_gate/20260515T_v446_cpu_gate/v446_tong_source_target_alignment_gate_candidate_audit.csv`.

Resultado V446:

| Item | Resultado |
|---|---:|
| Rows auditadas em `sft_reconstructed.jsonl` | `9500` |
| Rows aceitas preliminarmente | `1310` |
| `bit_manipulation` aceitas | `848` |
| `equation_transform` aceitas | `462` |
| Rows bloqueadas | `8190` |
| Weak row contract | passou |
| Tong source commit | `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85` |
| `hf_gpu_allowed` | `true`, condicionado a builder/token gate |

O source inventory confirmou que a fonte Tong contem os operadores que o plano
precisa:

| Familia | Inventario confirmado |
|---|---|
| `bit_manipulation` | bit-pair, bitsum, stride, operadores booleanos |
| `equation_transform` | concat, reverse, add/sub/mul, div/mod, digitos/determinante |

Interpretacao: pela primeira vez apos dias de tentativas, existe material novo
suficiente para justificar uma rota curta de transferencia para adapter, mas
isso ainda nao e ganho de ACC. A decisao correta e construir dataset final
V447, rodar tokenizacao/pair gate e somente entao abrir um H200 smoke curto.

V447 OpenRouter/API consensus:

- Artefato: `artifacts/openrouter/v447_public_submission_mining_consensus/KG1_V447_PUBLIC_SUBMISSION_MINING_CONSENSUS.md`.
- Prompt: `artifacts/openrouter/v447_public_submission_mining_consensus/v447_prompt.txt`.
- Modelos uteis: `openai/gpt-5.2`, `anthropic/claude-sonnet-4.6`,
  `deepseek/deepseek-r1-0528`, `qwen/qwen3-max-thinking`,
  `google/gemini-3.1-pro-preview`, `perplexity/sonar-reasoning-pro`.
- Consenso: minerar notebooks/submissoes publicas e valido, mas nao substitui
  o V446. A mineracao deve rodar em CPU e extrair tecnica/trace/dataset
  permitido, nao usar artefato de terceiros.

Decisao V447:

| Opcao | Decisao |
|---|---|
| Mais SFT generico, mais epochs, LR sweep | rejeitado |
| Minerar notebooks publicos antes do V446 builder | rejeitado como caminho principal; permitido em paralelo CPU |
| V446 -> builder -> token gate -> H200 smoke <= 1h | caminho principal |
| Usar solver/verifier runtime no submit | proibido |
| Usar adapter/peso/submissao publica de terceiro | proibido |

V447 dataset clean:

| Item | V447 bruto | V447 clean | Decisao |
|---|---:|---:|---|
| Total rows | `1310` | `1293` | usar clean |
| Rows descartadas por `boxed` contraditorio | `0` | `17` | bloqueio obrigatorio |
| Train rows | `1178` | `1164` | passa minimo |
| Val rows | `132` | `129` | passa minimo |
| Train `bit_manipulation` | `763` | `763` | preservado |
| Train `equation_transform` | `415` | `401` | remove ruido contraditorio |
| Val `bit_manipulation` | `85` | `85` | preservado |
| Val `equation_transform` | `47` | `44` | remove ruido contraditorio |
| `last_boxed_mismatch` | `17` potenciais | `0` | obrigatorio antes de GPU |

Gate de tokenizacao V447 clean:

| Check | Resultado |
|---|---|
| Manifesto | `artifacts/v447_v446_trace_dataset/20260515T_v447_tokenization_gate_clean/v286_generic_tokenization_gate_manifest.json` |
| `prompt_truncation_rate` | `0.0` |
| `completion_tokens_dropped` | `0` |
| `fallback_masks` | `0` |
| `offset_masks` | `1164/1164` train, `129/129` val |
| `train_val_prompt_overlap` | `0` |
| `train_val_prompt_answer_overlap` | `0` |
| Token max train/val | `8048` / `7995` |
| Decisao | `tokenization_gate_passed` |

Dataset HF V447 clean:

| Split | HF path | SHA256 |
|---|---|---|
| train | `felipesp1983/kg1-nemotron-training::data/v447_v446_trace_dataset/20260515T_cpu_gate_clean/v447_v446_trace_train.jsonl` | `08a4d36adf61fd20dcd5f2536eff6e5f39825b3a6b3a3a64d21e5ea4399c1ca7` |
| val | `felipesp1983/kg1-nemotron-training::data/v447_v446_trace_dataset/20260515T_cpu_gate_clean/v447_v446_trace_val.jsonl` | `19a8a360444d0fae61181fc77ce1f53180e5bed05661dc0c2f6cd7c4d3f00f31` |

Nova regra permanente: qualquer dataset/trace com ultimo `\boxed{}` diferente
da resposta oficial de treino deve ser bloqueado por padrao. Override so pode
existir via flag explicita, nunca silenciosamente. Essa regra entra em todos os
builders, notebooks e jobs novos/alterados.

## Atualizacao V448 - 2026-05-15

V448 foi executado e avaliado no contrato weak V221. O treino H200 completou,
mas o primeiro weak eval util falhou o gate e o job foi cancelado antes de
avaliar checkpoint-6.

Artefatos:

- Launcher de treino: `artifacts/v448_hf_h200_v447_trace_launch/launch_v448_hf_h200_v447_trace.py`.
- Launcher de weak eval: `artifacts/v448_hf_h200_v447_trace_launch/launch_v448_hf_weak_eval.py`.
- Job de weak eval: `https://huggingface.co/jobs/felipesp1983/6a07832e3308d79117b90a27`.
- Resultado registrado: `artifacts/v448_hf_h200_v447_trace_launch/V448_WEAK_EVAL_RESULT.md`.

Resultado V448 checkpoint-3:

| Item | Baseline submit-safe | V444 ckpt-2 | V448 ckpt-3 | Decisao |
|---|---:|---:|---:|---|
| Total weak | `192/315` | `190/315` | `190/315` | reprovado |
| `equation_transform` | `56/155` | `56/155` | `56/155` | sem ganho |
| `bit_manipulation` | `136/160` | `134/160` | `134/160` | regressao |
| `truncated` | `0` | `1` | `1` | reprovado |

Interpretacao: V446/V447 continham material novo e limpo, mas o adapter nao
aprendeu a converter esses traces em melhoria submit-safe. O padrao repetiu
V444: equation fica travado em `56`, bit perde `2`, e aparece truncation.
Portanto, repetir V448 com mais steps, mais epochs, H200 maior, LR sweep ou
novo checkpoint do mesmo dataset e gasto sem base tecnica.

## Atualizacao V453 - Public Kernel Mining

V453 executou mineracao CPU-only de notebooks publicos do Kaggle para procurar
tecnicas de terceiros que tratem `bit_manipulation` e `equation_transform`
melhor que a nossa rota atual.

Artefatos:

- Script: `scripts/mine_v453_public_kernels.py`.
- Manifesto: `artifacts/v453_public_kernel_mining/20260515T_cpu_mining/v453_public_kernel_mining_manifest.json`.
- Sumario CSV: `artifacts/v453_public_kernel_mining/20260515T_cpu_mining/v453_public_kernel_mining_summary.csv`.
- Relatorio: `artifacts/v453_public_kernel_mining/20260515T_cpu_mining/KG1_V453_PUBLIC_KERNEL_MINING.md`.

Resultado:

| Item | Valor |
|---|---:|
| Kernels listados | `30` |
| Kernels analisados | `30` |
| Pull failures | `1` |
| Raw notebooks retidos | `0` |
| `hf_gpu_allowed` | `false` |

Achados acionaveis:

| Achado | Interpretacao |
|---|---|
| `huikang/end-to-end-finetuning-for-lb-0-85` e forks reforcam `lm_head` manual e target modules amplos | confirma tecnica ja conhecida; nao e ganho novo porque nossas rotas `lm_head`/rank32 ja foram testadas e nao passaram gate |
| `matthewblakeward/steinifrank` tem classificadores por regex e hard negatives por familia | util como inspiracao para triagem/local routing CPU, mas nao prova regra `equation_transform` ou bit que melhore weak |
| notebooks `nvidia-nemotron-trained-models-submission` listam fontes Kienngx/Tinker/LoRA diversas | material para inventario, nao para uso direto de pesos/submits de terceiros |
| guias `0.86` reforcam que compatibilidade estrutural de adapter nao basta | valida nosso bloqueio contra mais treino generico sem sinal CPU |

Decisao: V453 nao libera GPU e nao altera baseline. A mineracao publica continua
permitida somente em CPU e somente para extrair regra implementavel localmente.
Qualquer achado futuro precisa virar builder/probe e provar ganho por
`verify_answer` antes de entrar em treino.

## Atualizacao V454 - Bit Guardrail

V454 consolidou as rotas de bit ja testadas para decidir se ainda vale abrir
GPU bit-only.

Artefatos:

- Script: `scripts/build_v454_bit_guardrail_decision.py`.
- Manifesto: `artifacts/v454_bit_guardrail_decision/20260515T_cpu_gate/v454_bit_guardrail_decision_manifest.json`.
- Relatorio: `artifacts/v454_bit_guardrail_decision/20260515T_cpu_gate/V454_BIT_GUARDRAIL_DECISION.md`.

Resultado:

| Fonte | Resultado | Decisao |
|---|---:|---|
| V296 stride train | `1201/1602`, gains `154`, losses `218` | diagnostico lossy |
| V333 Tong train | `1364/1602` | teacher forte |
| V333 Tong weak replace | `192/315`, bit `136/160`, gains `1`, losses `1` | nao deployable |
| V366 CPU teacher | `222/315`, bit `159/160`, losses `0` | teacher only |
| V359 adapter transfer | `190/315`, bit `134/160`, trunc `1` | rejeitado |
| V368 adapter transfer | `191/315`, bit `135/160`, trunc `0` | rejeitado |

Decisao: `hf_gpu_allowed=false` para treino bit-only. O problema nao e falta
de solver/teacher de bit; o problema e transferencia para adapter-only. A partir
daqui bit entra apenas como replay/guardrail quando a rota de equation provar
ganho CPU. Novo job bit-only so pode existir se houver evidencia nova que
ataque diretamente a falha de transferencia, nao apenas teacher melhor.

## Atualizacao V455 - Equation Target Audit

V455 comparou o ganho CPU solver-only do V324 contra o que o builder legal V452
conseguiu transformar em pares treinaveis. O objetivo era separar falta de regra
de falta de material treinavel permitido.

Artefatos:

- Script: `scripts/build_v455_equation_target_audit.py`.
- Manifesto: `artifacts/v455_equation_target_audit/20260515T_cpu_gate/v455_equation_target_audit_manifest.json`.
- Relatorio: `artifacts/v455_equation_target_audit/20260515T_cpu_gate/V455_EQUATION_TARGET_AUDIT.md`.

Resultado:

| Item | Valor |
|---|---:|
| Equation misses auditados pelo V324 | `99` |
| Misses numericos | `16` |
| Misses simbolicos/pontuacao | `83` |
| Candidatos V324 aceitos no-loss | `6` |
| Pares treinaveis certificados pelo V452 | `2` |
| Rows verificadas ainda ausentes do builder | `4` |
| Classes verificadas ainda ausentes | `3` |
| Candidatos simbolicos verificados | `0` |
| `hf_gpu_allowed` | `false` |

Lacunas concretas:

| Classe | V324 verified | V452 promoted | Gap |
|---|---:|---:|---:|
| `v274_guarded_numeric_add_direct_over_model_add_variant` | `1` | `0` | `1` |
| `v274_guarded_numeric_colon_absdiff_restore_trailing_zero` | `1` | `0` | `1` |
| `v274_guarded_numeric_minus_direct_negative_restore_sign` | `2` | `2` | `0` |
| `v274_guarded_numeric_minus_signed_opposite_sign_guarded` | `2` | `0` | `2` |

Decisao: V455 nao autoriza GPU. O proximo passo correto e V456 CPU: tentar
construir pares legais para as 3 classes numericas ausentes ou provar que elas
nao possuem material treinavel permitido. Weak/full labels continuam proibidos
para treino, filtro e tiebreak.

## Atualizacao V456/V457 - Missing Numeric Classes

V456 auditou as 3 classes numericas que V455 marcou como ausentes:

| Classe | Gap V455 | Evidencia | Decisao |
|---|---:|---|---|
| `add_direct_over_model_add_variant` | `1` | `6160` ocorrencias sinteticas historicas em V290/V293/V294 | bloqueada: sintetico ja falhou transferencia |
| `colon_absdiff_restore_trailing_zero` | `1` | builder existe, mas sem cobertura sintetica historica e sem hard-negative publico | precisa raw probe publico antes de qualquer treino |
| `minus_signed_opposite_sign_guarded` | `2` | `6160` ocorrencias sinteticas historicas em V290/V293/V294 | bloqueada para novo sintetico; precisa hard-negative real |

Decisao V456: `hf_gpu_allowed=false`. O achado importante e negativo: repetir
dataset sintetico para `minus_signed` e `add_direct` e gasto ruim, porque ja foi
tentado em escala e o adapter ficou preso em `equation=56`.

V457 entao construiu um pack de raw-output probe, usando apenas public-train:

- Script: `scripts/build_v457_public_train_numeric_probe_pack.py`.
- Manifesto: `artifacts/v457_public_train_numeric_probe_pack/20260515T_cpu_gate/v457_public_train_numeric_probe_pack_manifest.json`.
- Prompt pack: `artifacts/v457_public_train_numeric_probe_pack/20260515T_cpu_gate/v457_public_train_numeric_probe_pack_prompts.jsonl`.
- Rows: `22` prompts public-train de `minus_signed_opposite_sign_guarded`.
- O prompt pack omite `answer` e nao contem chaves `answer`, `label`, `target`,
  `correct`, `is_correct` ou `solution`.
- `hf_raw_probe_allowed=true` apenas para inferencia e coleta de raw output.
- `hf_gpu_train_allowed=false` ate os raw outputs provarem hard negatives reais
  do adapter, sem overlap weak/full.

Proxima acao tecnica: executar raw-output probe curto no HF para estes 22
prompts, depois analisar se o adapter V291/V290 realmente erra com o padrao que
o V274 corrige. Se nao houver hard-negative real, arquivar a rota sem treino.

## Atualizacao V458/V459/V460 - Raw Probe e Micro Dataset

V458 executou o raw-output probe em HF H200, sem labels no input:

- Job: `https://huggingface.co/jobs/felipesp1983/6a07989d3308d79117b90d62`.
- Output HF: `https://huggingface.co/datasets/felipesp1983/kg1-v458-v457-numeric-raw-probe/tree/main/runs/v458-v457-numeric-raw-probe-20260515T220411Z`.
- Commit HF output: `https://huggingface.co/datasets/felipesp1983/kg1-v458-v457-numeric-raw-probe/commit/12b10166887cebb2d3490503edcc7b7368484abd`.
- Manifesto local: `artifacts/v458_hf_v457_numeric_raw_probe_outputs/runs/v458-v457-numeric-raw-probe-20260515T220411Z/v458_v457_numeric_raw_probe_manifest.json`.

Resultado V458:

| Item | Valor |
|---|---:|
| rows geradas | `22` |
| family | `equation_transform` |
| completion tokens | `128278` |
| finish reason | `stop=22` |
| generation elapsed | `115.49s` |
| H200 total job | cerca de `6min` |

V459 juntou as labels public-train somente apos a coleta de raw outputs:

- Script: `scripts/build_v459_v458_numeric_hard_negative_audit.py`.
- Manifesto: `artifacts/v459_v458_numeric_hard_negative_audit/20260515T_v459_cpu_audit/v459_v458_numeric_hard_negative_audit_manifest.json`.
- Relatorio: `artifacts/v459_v458_numeric_hard_negative_audit/20260515T_v459_cpu_audit/v459_v458_numeric_hard_negative_audit.md`.

Resultado V459:

| Classe | Rows | Adapter correto | Adapter igual ao erro simulado | Postprocessor correto | Hard negatives reais |
|---|---:|---:|---:|---:|---:|
| `v274_guarded_numeric_minus_signed_opposite_sign_guarded` | `22` | `15` | `7` | `22` | `7` |

Interpretacao: existe sinal real do adapter, porque 7 linhas public-train
produzem exatamente a resposta opposite-sign errada que o V274 corrige. Isso e
mais forte que sintetico, mas ainda e estreito: uma unica classe. Por isso
`hf_gpu_allowed=false` no V459.

V460 materializou uma proposta de dataset micro, ainda CPU-only:

- Script: `scripts/build_v460_numeric_one_rule_micro_dataset.py`.
- Manifesto: `artifacts/v460_numeric_one_rule_micro_dataset/20260515T_v460_cpu_dataset/v460_numeric_one_rule_micro_dataset_manifest.json`.
- Token gate: `artifacts/v460_numeric_one_rule_micro_dataset/20260515T_v460_tokenization_gate/v286_generic_tokenization_gate_manifest.json`.

Resultado V460:

| Split | Rows | equation | bit replay | Hard negatives reais | Trunc/token gate |
|---|---:|---:|---:|---:|---|
| train | `146` | `18` | `128` | `7` | passou |
| validation | `36` | `4` | `32` | `0` | passou |

Token gate V460:

- `prompt_truncation_rate=0.0`;
- `completion_tokens_dropped=0`;
- `fallback_masks=0`;
- offset masks completas;
- train/val prompt overlap `0`.

Decisao: V460 ainda nao libera GPU por padrao porque e uma rota de uma classe
so. O caminho agressivo possivel e um micro-smoke de um checkpoint em H200, mas
isso deve ser tratado como risco explicito: ele so tenta transferir a classe
`minus_signed_opposite_sign_guarded`. Se rodar, o kill-switch e imediato:
promover somente se weak `>192`, equation `>56`, bit `>=136`, trunc `0`;
caso contrario cancelar e arquivar.

## Proxima Acao Ativa

Rota ativa: V465 numeric multirule smoke. Esta e a primeira rota desde V448 que
tem hard negatives reais multi-classe do adapter atual antes de treino.

1. V462 foi concluido:
   - job: `https://huggingface.co/jobs/felipesp1983/6a07a0aa3308d79117b90da2`;
   - output: `felipesp1983/kg1-v462-v461-synthetic-raw-probe`;
   - resultado: `56/56` outputs, todos `stop`, prompt pack sem labels.
2. V463 foi fechado:
   - script: `artifacts/v463_v462_synthetic_numeric_hard_negative_audit/build_v463_v462_synthetic_numeric_hard_negative_audit.py`;
   - artefato: `artifacts/v463_v462_synthetic_numeric_hard_negative_audit/20260515T_cpu_gate/`;
   - resultado: `26` hard negatives reais em `3` classes;
   - classes: `add_direct_over_model_add_variant`, `minus_direct_negative_restore_sign`, `minus_signed_opposite_sign_guarded`;
   - sanity: ids 56/56 unidos, prompt hashes 56/56, finish `stop` 56/56, postprocessor 56/56.
3. V464 foi fechado:
   - script: `scripts/build_v464_v463_numeric_multirule_dataset.py`;
   - artefato: `artifacts/v464_v463_numeric_multirule_dataset/20260515T_cpu_gate/`;
   - train: `558` rows = `46` equation + `512` bit replay;
   - validation: `138` rows = `10` equation + `128` bit replay;
   - hard negatives no treino: `22` em `3` classes;
   - token gate: `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `fallback_masks=0`, train/val overlap `0`.
4. V465 deve rodar apenas como smoke:
   - launcher: `artifacts/v465_hf_h200_v464_numeric_multirule_launch/launch_v465_hf_h200_v464_numeric_multirule.py`;
   - dataset HF: `felipesp1983/kg1-nemotron-training/data/v464_v463_numeric_multirule_dataset/20260515T_cpu_gate`;
   - gate HF: `runtime_artifacts/v464_v463_numeric_multirule_dataset/20260515T_tokenization_gate`;
   - H200, timeout `3600s`, `MAX_STEPS=16`, checkpoints `4/8/12/16`;
   - pesos: equation hard-negative classes sobem, bit replay preserva piso.
5. Depois do treino V465:
   - rodar weak eval por checkpoint, nao full eval direto;
   - promover somente se `total > 192/315`, `equation > 56/155`, `bit >= 136/160`, `truncated = 0`;
   - se checkpoint-4 ja mostrar bit <136 ou equation sem ganho, cancelar/arquivar por FinOps;
   - se nenhum checkpoint passar, nao repetir receita: voltar para mining CPU de novas classes reais.

Regra FinOps continua: se o primeiro checkpoint ou gate parcial nao indicar
caminho para `total>192`, `equation>56`, `bit>=136`, `truncated=0`, cancelar.
`eval_loss`, `train_loss` e accuracy interna nao promovem submit; elas apenas
ajudam a matar job cedo.

## Atualizacao V466/V468 - Crisis Mode Silent Bug

V466 avaliou o adapter V465 treinado sobre V464 e confirmou que a rota nao
gerou ganho submit-safe:

| Checkpoint | Total weak | equation | bit | truncated | Decisao |
|---|---:|---:|---:|---:|---|
| V465 checkpoint-4 | `189/315` | `56/155` | `133/160` | `1` | rejeitar |
| V465 checkpoint-8 | `192/315` | `56/155` | `136/160` | `1` | rejeitar |

FinOps: o job V466 foi cancelado antes de checkpoint-12/16/final, porque
checkpoint-8 manteve `equation=56`, nao passou `truncated=0` e nao superou
`192/315`.

Auditoria posterior achou um bug silencioso no dataset V464:

| Split V464 antigo | equation rows | traces com `candidate == answer` |
|---|---:|---:|
| train | `46` | `24` |
| validation | `10` | `6` |

Exemplo do bug: a trace dizia que o candidato `'30'` era rejeitado e, na mesma
linha, finalizava com `\boxed{30}`. Isso torna a supervisao contraditoria: o
loss pode cair, mas a regra ensinada fica semanticamente errada.

Correcoes implementadas:

- `scripts/build_v464_v463_numeric_multirule_dataset.py` agora seleciona um
  `rejected_candidate` que obrigatoriamente difere da resposta pelo
  `verify_answer`.
- O builder falha em CPU se nao existir candidato errado valido.
- O manifesto registra `rejected_candidate` e
  `rejected_candidate_source`.
- `scripts/run_v286_generic_tokenization_gate.py` agora bloqueia qualquer
  dataset em que `metadata.rejected_candidate` ou o texto
  `candidate 'X' is rejected` verifique igual ao gabarito.

V468 e o rebuild corrigido:

| Split V468 | Rows | equation | bit replay | contradiction gate |
|---|---:|---:|---:|---|
| train | `558` | `46` | `512` | `0` |
| validation | `138` | `10` | `128` | `0` |

Tokenization gate V468:

- tokenizer real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`;
- `prompt_truncation_rate=0.0`;
- `completion_tokens_dropped=0`;
- `fallback_masks=0`;
- offset masks completas;
- train/val prompt overlap `0`.

Decisao atual: V464/V465 ficam bloqueados para novos treinos. A unica proxima
execucao permitida nesta rota e um smoke HF novo usando V468 corrigido. O gate
de promocao permanece:

- weak total `>192/315`;
- `equation_transform >56/155`;
- `bit_manipulation >=136/160`;
- `truncated = 0`.

Se checkpoint-4 do novo treino V468 nao passar nesses criterios, cancelar
imediatamente por FinOps e voltar para mining CPU de novas classes reais, sem
treinar mais epochs sobre o mesmo sinal.

## Atualizacao V471 - Crisis Triple-Check

Status real apos V470:

| Estado | Total weak | equation_transform | bit_manipulation | truncated | Decisao |
|---|---:|---:|---:|---:|---|
| Melhor submit-safe travado | `192/315` | `56/155` | `136/160` | `0` | manter |
| V470, checkpoint-4 de V469 | `190/315` | `56/155` | `134/160` | `1` | rejeitar |

Conclusao: a rota V468 -> V469 -> V470 nao transfere ganho para ACC. Mesmo com
dataset corrigido e loss aparentemente saudavel, o weak gate mostrou regressao
de bit e truncation. Esta rota esta encerrada para GPU.

Correcoes de gate implementadas em V471:

- `scripts/evaluate_lora_adapter.py` e `scripts/evaluate_lora_adapters_batch.py`
  agora falham se o CSV de solucao nao tiver `answer` ou se a quantidade de
  outputs vLLM divergir da quantidade de prompts.
- `scripts/hf_job_train_v90.py` agora falha se truncation remover qualquer token
  supervisionado da completion.
- `scripts/run_v286_generic_tokenization_gate.py` agora exige que o texto do
  assistant seja extraivel por `extract_final_answer` e verifique contra
  `answer`; isso bloqueia simbolos/braces/backslashes que parecam validos, mas
  que o metric path nao entenderia.
- `scripts/build_v447_v446_trace_dataset.py` agora so preserva/apende final
  answer se o resultado final for metricamente extraivel e correto.
- `scripts/hf_job_weak_eval_v245.py` agora tem gate de promocao executavel:
  `total>=193`, `equation>=57`, `bit>=136`, `truncated=0`.
- `scripts/notebook_release_gate.py` foi apertado para bloquear notebooks com
  `WEAK_BIT_MIN_FOR_FULL = 133` ou `WEAK_MAX_TRUNC_FOR_FULL = 3`; o piso atual e
  bit `136` e truncation `0`.

Evidencias:

- `artifacts/v471_crisis_solution_audit/V471_CRISIS_AUDIT_RESULT.md`;
- `artifacts/v471_crisis_solution_audit/v470_metric_integrity/v470_metric_integrity_manifest.json`;
- `artifacts/v471_crisis_solution_audit/v470_parse_audit/v470_parse_audit_manifest.json`;
- `artifacts/v470_hf_h200_v469_checkpoint4_weak_eval_launch/V470_TERMINAL_RESULT.md`.

Proximo passo obrigatorio:

1. parar treino amplo e qualquer repeat da rota V468/V469;
2. voltar para CPU mining dos `99` misses de `equation_transform`;
3. so liberar HF GPU quando um gate CPU encontrar pelo menos `+4` equation,
   `0` perdas, `bit>=136`, e todos os simbolos finais passarem pelo mesmo
   `extract_final_answer`/`verify_answer` usado no weak eval;
4. em qualquer smoke futuro, o primeiro checkpoint precisa passar o gate de
   promocao. Caso contrario, cancelar por FinOps.
