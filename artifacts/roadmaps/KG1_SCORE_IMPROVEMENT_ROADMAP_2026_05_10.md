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

## Regras Permanentes

1. Weak/full sao somente avaliacao e gate. Nunca usar weak/full para construir pares, escolher misses, selecionar candidatos, gerar `chosen/rejected`, treinar, desempatar regra ou fazer cherry-pick.
2. `answer` nao pode participar da decisao de regra, filtro, tiebreak, `chosen`, selecao de trace ou selecao de candidato. A resposta correta so pode aparecer em auditoria posterior, depois da regra estar congelada.
3. `id`, prompt hash de weak/full, oracle, solver runtime, verifier runtime e postprocessor nao podem entrar no submit.
4. Submit valido e apenas adapter-only: `adapter_config.json` e `adapter_model.safetensors` no root do pacote. Sem script, tokenizer, prompt prefix, soft-prompt, `embed_tokens`, `lm_head`, decoder patch, logit mask, constrained decoding, runtime abstention ou confidence threshold.
5. Decisao e por ACC/truncation. `eval_loss`, `train_loss`, preference accuracy interna e probabilidades de LLM nao liberam submit nem GPU.
6. HF GPU so pode rodar depois de CPU gate com sinal material, manifest auditavel e `hf_gpu_allowed=true`.
7. FinOps: cancelar ou nao iniciar qualquer job que nao possa mais superar `total>192`, `equation>56`, `bit>=136`, `truncated=0`.
8. Toda nova versao precisa quadro comparativo contra V291/V290 e decisao explicita: promover, repetir CPU, cancelar ou arquivar.
9. Se notebook for criado ou alterado, precisa passar `python scripts/notebook_release_gate.py <notebook>` antes de entrega/push.
10. Se job estiver rodando, analisar logs periodicamente e aplicar kill-switch sem esperar gasto inutil.
11. Todo script, job launcher, workflow ou notebook criado/alterado precisa passar `python scripts/kg1_static_safety_gate.py <paths>` antes de entrega/push/execucao. O gate bloqueia V435E misto arquivado, `format_negative_*` em treino ativo e `ALLOW_FORMAT_NEGATIVES` em job/notebook.
12. H200 esta autorizada ate 1 hora por execucao. Se uma execucao precisar passar de 1 hora, parar e pedir autorizacao humana antes de continuar.
13. Todo erro novo deve entrar no ledger `artifacts/roadmaps/KG1_ERROR_LEDGER_2026_05_15.md` com evidencia, impacto, regra preventiva e status antes de abrir novo job pago.
14. Antes de qualquer job pago ou notebook operacional novo/alterado, rodar auditoria de integracao: launcher, dataset correto, conteudo do dataset, hashes, schema, targets, paths HF, adapter inicial, gates, kill-switch e comparacao contra baseline. Para HF jobs, usar `scripts/kg1_pre_paid_job_integration_gate.py` alem do static gate.
15. Acesso a modelos/datasets Hugging Face deve reutilizar o `HF_TOKEN` ja usado para criar/executar jobs. Nunca imprimir, commitar ou gravar a chave em artefatos.

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

## Proxima Acao Unica

Parar a linha V436/V436B/V440/V441 e nao abrir novo GPU job de preference
simples sem uma mudanca tecnica nova no dado. V441 ja testou a hipotese de
score somente no boxed payload e nao gerou sinal; V442 confirmou que o V439
tem `0` linhas com certificado de regra label-free.

Objetivo:

1. Implementar V443 CPU certified equation pair builder.
   - Entrada: public train permitido e probes V435D/V439 apenas como diagnostico.
   - Prioridade: `equation_symbolic_sequence` e `equation_symbolic_short`.
   - Exigir regra congelada antes do answer, MDL, Leave-One-Out, renaming
     stability, candidate count unico e slot/substring alignment stats.
2. Gerar pares somente para regras certificadas.
   - chosen: resposta final curta derivada da regra congelada.
   - rejected: erro real do adapter V291/V290 sobre o mesmo prompt.
   - Sem weak/full como fonte, sem oracle de gate, sem tiebreak por answer.
3. Rodar gate CPU contra baseline V291/V290.
   - So avanca se houver novo sinal medido para `equation_transform` e zero
     regressao esperada de bit.
4. Se e somente se V443 gerar pelo menos quatro modos independentes de equation,
   publicar dataset e rodar integration gate pre-pago.
5. Qualquer novo GPU job precisa mostrar, antes de rodar, quadro comparativo
   contra V291/V290 e condicao objetiva de parada no primeiro checkpoint.
6. Promover para weak/full/package/submit somente se o gate superar o melhor
   adapter-only atual: weak `192/315`, equation `56/155`, bit `136/160`,
   trunc `0`.

Regra FinOps: V436 revelou bug de dataset; V436B provou que hard-negative-only ainda nao basta. O objetivo segue sendo ganho medido de ranking; loss interno e preference accuracy so servem para matar job cedo, nao para promover.
