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
| V435 CPU pair gate | 0/3558 pares aprovados | bloqueia GPU; faltam raw outputs reais V291/V290 e certificados |
| V435B prompt pack | 840 prompts permitidos, 0 answers exportadas | pronto para coleta de raw outputs, nao libera treino |
| HF GPU treino | bloqueado | liberar somente com V435 `hf_gpu_allowed=true` |
| HF V435C inferencia | preparado | permitido apenas para coletar raw outputs V291/V290, com caps e kill-switch |

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

Status: concluido e enviado ao HF dataset de treino.

Artefato:

- `scripts/build_v435e_adapter_probe_preference_dataset.py`

Politica:

- Usar somente public-train probes de V435D.
- `chosen`: completion curta com resposta publica correta em `\boxed{}` com escape de braces/backslash.
- `rejected`: completion curta com a predicao errada real do adapter V291/V290.
- Guardrail de bit: rows bit corretas viram format replay `format_negative_no_box`.
- `weak/full`: usados somente para filtro de overlap, nunca como treino.

Resultado:

| Split | Rows | bit | equation | hard negatives | bit replay |
|---|---:|---:|---:|---:|---:|
| train | 160 | 62 | 98 | 109 | 51 |
| validation | 40 | 18 | 22 | 24 | 16 |
| all | 200 | 80 | 120 | 133 | 67 |

Rule classes de equation cobertas:

- `equation_numeric_operator_to_number`: 20
- `equation_numeric_operator_to_symbolic`: 4
- `equation_symbolic_sequence`: 64
- `equation_symbolic_short`: 32

HF dataset:

- `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/tree/main/data/v435e_adapter_probe_preference/20260515T_v435e_from_h200_probe`

### V435F - Adapter Probe Preference Gate

Status: passou.

Artefato:

- `scripts/run_v435f_adapter_probe_preference_gate.py`
- `artifacts/v435f_adapter_probe_preference_gate/20260515T_v435f_from_h200_probe/v435f_adapter_probe_preference_gate_manifest.json`

Gate:

| Condicao | Resultado |
|---|---|
| all rows approved | true |
| approved rows >= 180 | true (`200`) |
| equation hard negatives >= 100 | true (`120`) |
| bit hard negatives >= 10 | true (`13`) |
| bit replay >= 50 | true (`67`) |
| equation rule classes >= 4 | true (`4`) |

Decisao: `hf_gpu_allowed=true` para um unico V436 short smoke. Este e um desbloqueio de treino, nao autorizacao de submit.

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

Conclusao: V436 nao trouxe ganho submit-safe e nao autoriza V437. Nao repetir variantes V436 sem um novo gate CPU que demonstre sinal direto de transferencia, nao apenas dataset novo ou loss interno.

## V437 - Full Gate, Package e Submit

Status: bloqueado. V436 falhou o primeiro kill-switch.

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
| V436 preference variants sobre V435E | Primeiro checkpoint regrediu de `6/40` para `5/40` |

## Proxima Acao Unica

Parar a linha V436 e nao abrir novo GPU job ate existir um gate CPU com hipotese de transferencia mais direta.

Objetivo:

1. Converter os achados V435D/V435E em diagnostico de por que o adapter prefere negativos: comparar `chosen_mean_nll`, `rejected_mean_nll`, tamanho de respostas, tipo de negativo e familia.
2. Se houver bug de formato/dataset, corrigir em CPU e reconstruir V435E; se nao houver bug, arquivar a linha preference-LoRA.
3. Qualquer novo HF GPU job precisa passar um gate CPU novo com uma metrica que nao tenha regredido no checkpoint inicial anterior.
4. Promover para weak/full/package/submit somente se o novo gate superar o melhor adapter-only atual: weak `192/315`, equation `56/155`, bit `136/160`, trunc `0`.

Regra FinOps: V436 ja acionou o kill-switch. O objetivo segue sendo ganho medido de ranking, nao mais perda de tempo com loss interno.
