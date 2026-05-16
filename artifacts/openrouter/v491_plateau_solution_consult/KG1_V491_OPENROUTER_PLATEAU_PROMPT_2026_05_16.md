You are a senior ML systems auditor, Kaggle competition engineer, and PEFT/LoRA/Nemotron specialist. We need a rigorous, non-generic decision for a stalled adapter-only Kaggle solution. Do not hallucinate. If a fact is not supported by the evidence below, label it UNKNOWN. We need actions that can be implemented and gated, not broad advice.

# Problem
We are working on the NVIDIA Nemotron Model Reasoning Challenge. The submission must be an adapter-only LoRA/PEFT package: adapter_config.json + adapter_model.safetensors. Runtime solvers, postprocessors, verifiers, row-label oracles, logit masks, prompt hacks, and cherry-picking by weak/full labels are NOT submit-safe.

We are stuck after many days of experiments. Loss/eval_loss changes, but weak ACC does not improve submit-safely. The two important weak families are:
- bit_manipulation: current submit-safe baseline 136/160
- equation_transform: current submit-safe baseline 56/155
- total weak baseline: 192/315, truncated=0
Promotion minimum: total > 192, equation > 56, bit >= 136, truncated = 0. Full official-like known floor: 823/947; only package/submit if full-like improves.

Recent best failed run V488: total 191/315, equation 57/155, bit 134/160, truncated=1. It gained one equation row but lost two bit rows and introduced truncation. We need a plan that produces real ACC gain, not lower loss.

# Critical Constraints
1. Adapter-only submission. No runtime solver/verifier/postprocessor.
2. Weak/full rows are eval/gate only; they cannot be used to create train labels, chosen/rejected pairs, or cherry-pick.
3. ACC uses strict verify_answer. Binary-like answers matching [01]+ must be exact strings, not numeric tolerance.
4. Expected-aware extraction may only disambiguate the LAST boxed payload for symbolic braces/backslashes; it must never select an earlier boxed answer because it matches the validation answer.
5. FinOps: cancel any job that cannot beat total>192, equation>56, bit>=136, truncated=0.
6. Next useful experiment must include a first-checkpoint weak micro-ACC kill switch.

# Current Roadmap Summary
# KG1 Score Improvement Roadmap

Atualizado: 2026-05-16

Este e o roadmap ativo e limpo apos a revisao V484 dos arquivos OpenRouter de
16/05/2026. O historico detalhado foi arquivado em:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V435_CLEANUP_2026_05_15.md`
- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_PRE_V484_OPENROUTER_CLEANUP_2026_05_16.md`

A partir deste arquivo, itens antigos so valem como evidencia historica. O plano
executavel e somente o que esta abaixo.

## Estado Real

| Metrica | Melhor submit-safe atual | Promocao minima |
|---|---:|---:|
| Total weak | 192/315 | > 192/315 |
| equation_transform | 56/155 | > 56/155, alvo inicial 60/155 |
| bit_manipulation | 136/160 | >= 136/160 |
| truncated | 0 | 0 |
| Full official-like conhecido | 823/947 | > 823/947 |

Valores melhores que dependem de solver, verifier, postprocessor, teacher CPU,
oracle ou selecao por weak/full nao sao submit-safe ate virarem comportamento do
adapter LoRA valido.

Ultima evidencia operacional relevante:

| Versao | Resultado | Decisao |
|---|---:|---|
| V291/V290 checkpoint-6 | weak 192, equation 56, bit 136, trunc 0 | baseline ativo |
| V477 ckpt-2 | weak 192, equation 57, bit 135, trunc 0 | nao promove; ganhou equation mas perdeu bit |
| V475 CPU solver projection | weak 196, equation 60, bit 136 | sinal CPU; ainda nao submit-safe |
| V480/V483 linha recente | loss mexe, ACC nao sai do plateau | suspeita forte de PEFT continuity bug |
| V485 seed PEFT metadata gate | `hf_gpu_allowed=true`; 12011 tensors; target params 5934/5934; `modules_to_save=[]` | seed V290/V291 estruturalmente liberado |
| V487 treino H200 | treino completo, checkpoint-10 melhor `eval_loss=1.3519`; target params 5934/5934 ativos | continuidade PEFT corrigida, mas nao prova ACC |
| V488 ckpt-10 weak eval | weak 191, equation 57, bit 134, trunc 1 | nao promove; target params nao eram o unico gargalo |
| V489 audit integridade | metrica ACC estrita correta; V488 teve +1 equation, -2 bit, +1 trunc; F2 frozen-active nao era visivel no manifesto; expected-aware antigo podia vazar boxed anterior | corrigir observabilidade/guard/extracao antes de novo GPU |
| V490 debug double check | compilacao, self-tests, static gate, dataset V390/V326, tokenization e metric path OK; HF jobs ativos 0 | proximo passo deve mudar mecanismo treinavel, nao repetir V487 |

## Achado Principal V484

Os dois arquivos OpenRouter de 16/05/2026 reforcam um ponto tecnico mais forte
que qualquer nova busca de dados: a linha de treino precisa provar continuidade
PEFT/LoRA antes de gastar GPU.

Consenso acionavel:

| Achado | Status | Acao |
|---|---|---|
| `target_parameters` perdido ou carregado por caminho manual pode quebrar continuidade do adapter MoE | risco alto, agora bloqueado por gate | usar `PeftModel.from_pretrained(..., is_trainable=True)` como padrao |
| `adapter_config.json` precisa bater com env de treino | obrigatorio | gate compara `r`, `alpha`, `target_modules`, `target_parameters` |
| key/shape/dtype de `adapter_model.safetensors` precisa ser auditado | obrigatorio antes de GPU | criar/rodar CPU round-trip gate |
| `modules_to_save` nao pode carregar pesos cheios | obrigatorio | permitir `lm_head` apenas como LoRA em `target_modules`, nunca como modulo salvo inteiro |
| `answer_span_loss_weight=12.0` pode mascarar ACC | risco nao comprovado, mas recorrente no double check | logar componentes e usar micro-ACC como kill-switch; nao aumentar peso sem evidencia |
| `eval_loss` baixo nao comprova ACC | regra permanente | promover so por weak/full ACC e truncation |
| Mais epochs, LR sweep ou H200 longo sem novo gate e desperdicio | removido | FinOps cancela antes de custo |

Evidencia externa verificada:

- PEFT documenta `target_parameters` para parametros MoE que nao sao `nn.Linear`.
- PEFT documenta que, quando o adapter original usa `target_parameters`, a
  injecao a partir de `state_dict` exige a config PEFT correta.
- A suite PEFT possui testes especificos para `target_parameters` em modelos
  Llama4/GPT-OSS, incluindo `mlp.experts.gate_up_proj` e
  `mlp.experts.down_proj`.
- PR PEFT `#2710` corrigiu problemas de `target_parameters` e menciona riscos
  de erro silencioso com multiplos adapters; portanto nossa regra deve ser
  fail-closed.

Fontes: `artifacts/v484_openrouter_uploaded_audit/V484_OPENROUTER_UPLOAD_AUDIT.md`.

## Regras Ativas

1. Submit valido e somente adapter-only: `adapter_config.json` e
   `adapter_model.safetensors` no pacote. Sem runtime solver, verifier,
   postprocessor, logit mask, prompt hack ou threshold.
2. Weak/full sao apenas avaliacao e gate. Nao podem construir dataset, pares,
   chosen/rejected, desempate de regra ou cherry-pick.
3. ACC de promocao usa `src.competition_utils.verify_answer`. `answers_equivalent`
   e diagnostico-only.
4. Extracao expected-aware so pode desambiguar o ultimo `\boxed{}` usando
   `verify_answer`; nunca pode escolher um `\boxed{}` anterior por bater com o
   gabarito.
5. Nenhum job pago roda se `target_parameters` estiver ausente, divergente ou
   carregado por modo manual sem round-trip CPU aprovado.
6. `modules_to_save` deve ficar vazio no seed e no pacote final. `lm_head` pode
   aparecer em `target_modules` como LoRA, mas nao como peso cheio salvo.
7. Se um launcher usa `LORA_TARGET_PARAMETERS` MoE junto com allowlist
   `TRAINABLE_LORA_MODULES`, ele deve declarar explicitamente se esses
   `target_parameters` precisam ser treinaveis:
   `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` ou `1`.
8. Todo launcher/job/notebook novo ou alterado passa por
   `scripts/kg1_static_safety_gate.py`.
9. Antes de job pago, rodar `scripts/kg1_pre_paid_job_integration_gate.py` e
   `scripts/hf_job_preflight_gate.py`; o preflight deve falhar se qualquer linha
   de treino vier marcada como gate/weak/full usada para treino.
10. FinOps: cancelar job que nao possa mais superar `total>192`,
   `equation>56`, `bit>=136`, `truncated=0`.
11. H200 pode ser usada ate 1 hora por execucao. Acima disso exige autorizacao
   humana.
12. Todo erro novo entra em `KG1_ERROR_LEDGER_2026_05_15.md` antes de novo job
   pago.
13. Toda versao nova precisa quadro comparativo contra V291/V290.

## Plano Cronologico Ativo

### P0 - Congelar Gasto Pendente

Objetivo: parar custo enquanto a continuidade PEFT nao estiver provada.

Executar:

- Nao abrir novo H200/A100 para repetir V476/V477/V480.
- Cancelar jobs que ja nascem abaixo do gate ou dependem de `eval_loss`.
- Manter apenas CPU/HF CPU barato ate o round-trip passar.

Promove para P1 quando: nenhum launcher ativo usa `INIT_ADAPTER_LOAD_MODE=manual`
com `LORA_TARGET_PARAMETERS`.

### P1 - Corrigir Continuidade PEFT

Objetivo: garantir que o adapter inicial V290/V291 e carregado no mesmo espaco
estrutural que gerou 192/315 e que o manifesto declare claramente quais LoRA
ficaram treinaveis.

Executar:

- Padrao de treino: `PeftModel.from_pretrained(base, init_adapter, is_trainable=True)`.
- Bloquear `INIT_ADAPTER_LOAD_MODE=manual` para adapters com
  `target_parameters`.
- Verificar no gate:
  - `adapter_config.json` preserva `target_parameters`.
  - `adapter_config.json` tem `modules_to_save` vazio.
  - LoRA tensors de `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`
    existem.
  - `target_parameter_lora_tensors` nao e vazio.
  - `target_parameter_trainable_lora_tensors` e
    `target_parameters_trainability_mode` ficam registrados no manifesto.
  - nomes treinaveis contem os modulos obrigatorios.
  - nao ha warnings de missing adapter keys.
  - SHA256 de `adapter_config.json` e fingerprints de keys/shapes/dtypes sao
    registrados antes e depois do treino.

Promove para P2 quando: CPU preflight e static gate passam sem excecao.

### P2 - CPU Round-Trip Gate V484/V485

Status: implementado e aprovado no adapter seed V290 checkpoint-6.

Objetivo: provar equivalencia estrutural antes de qualquer GPU.

Implementar/rodar um gate CPU que:

- baixa/carrega o adapter seed V290/V291;
- carrega com `PeftModel.from_pretrained(..., is_trainable=True)`;
- salva em diretorio temporario;
- recarrega;
- compara `adapter_config.json`, lista de keys, shapes, dtypes e contagem de
  tensores LoRA;
- confirma `modules_to_save=[]` ou `null`;
- roda um micro forward/backward em batch dummy quando aplicavel e confirma se
  os parametros LoRA esperados estao treinaveis ou explicitamente
  `frozen_active`;
- emite manifesto com `hf_gpu_allowed=true` somente se tudo bater.

Implementacao atual:

- usa metadados Hub/safetensors para evitar download multi-GB;
- valida `target_modules`, `target_parameters`, `modules_to_save`, keys,
  shapes, dtypes, contagens LoRA e fingerprints;
- aceita o alias estrutural real do Nemotron em que
  `mlp.experts.gate_up_proj` aparece como LoRA em
  `mixer.experts.<id>.up_proj`;
- aceita apenas o `lm_head.base_layer.weight` conhecido do seed como tensor
  nao-LoRA fingerprintado; qualquer `modules_to_save` segue bloqueado;
- foi conectado ao launcher V391 antes do download de dataset e antes de
  qualquer treino pago.

Resultado V485 seed:

| Campo | Valor |
|---|---:|
| resolved revision | `75909c9b40d8b7fa846d379d9d764fa33daeb9e2` |
| adapter_model bytes | 4259063856 |
| tensor_count | 12011 |
| target_parameter_lora_tensors gate_up | 5934 |
| target_parameter_lora_tensors down | 5934 |
| modules_to_save | `[]` |
| allowed non-LoRA key | `base_model.model.lm_head.base_layer.weight` |
| hf_gpu_allowed | `true` |

Manifest: `artifacts/v485_peft_roundtrip_gate/v485_seed_adapter_manifest.json`.

Promove para P3 quando: round-trip manifesto aprovado.

### V391/V486 Objective Balance Update

V391 foi lancado em H200 mas parou antes do treino, no gate V478. Isso foi
correto: os pesos `equation_numeric_* = 10.00` faziam equation dominar o
objetivo efetivo.

| Versao | Status | Bit share efetivo treino | Equation share efetivo treino | Decisao |
|---|---|---:|---:|---|
| V391 | rejeitado antes do treino | 0.135975 | 0.864025 | nao treinar; objetivo desequilibrado |
| V486 | probe CPU aprovado | 0.207788 | 0.792212 | candidato a smoke curto |

Artefatos:

- `artifacts/version_diffs/V486_VS_V391.md`
- `artifacts/v486_objective_weight_probe/V486_OBJECTIVE_WEIGHT_PROBE.md`
- `artifacts/v486_objective_weight_probe/eq_6.json`

Regra atualizada: qualquer novo HF job deve passar o gate V478 e manter
`bit_manipulation` com pressao efetiva minima. `eval_loss` continua sendo
diagnostico secundario; promocao depende de weak micro-ACC.

Atualizacao V487: V486 passou V485/V478 e falhou antes do treino porque o
script de treino nao aplicava o alias estrutural de `target_parameters`
(`mlp.experts.gate_up_proj` salvo/carregado como `mixer.experts.<id>.up_proj`).
O matcher foi alinhado com V485 e V487 e o relancamento correto.

Artefato: `artifacts/version_diffs/V487_VS_V486.md`.

Atualizacao V488: o treino V487 completou em H200 e confirmou LoRA ativa para
os `target_parameters`, mas a weak eval focada do checkpoint-10 produziu
`191/315`, `equation_transform=57/155`, `bit_manipulation=134/160` e
`truncated=1`. Portanto a continuidade PEFT era um bug real, mas nao era
suficiente para romper o plateau. A rota de repetir o mesmo SFT/mesmo objetivo
esta bloqueada por FinOps ate existir novo sinal CPU que preserve bit e
truncation.

Atualizacao V489: o diff linha a linha confirmou que a metrica estrita esta
correta e que V488 teve exatamente um ganho real de equation (`518deb39`) e duas
regressoes reais de bit (`8740ed31`, `59bee375`), sendo uma com truncation. A
auditoria tambem mostrou um gap de F2/observabilidade: V487 carregava
`target_parameters`, mas a allowlist treinavel era `q/k/v/o/lm_head`; logo
`up_proj/down_proj` ficavam frozen-active, nao comprovadamente treinados. O
script de treino agora grava `target_parameters_trainability_mode` e os contadores
de tensores trainaveis por `target_parameter`; launchers futuros precisam
declarar explicitamente se esperam `target_parameters` treinaveis. A auditoria
tambem corrigiu dois bugs silenciosos de validação: uma chave duplicada no
static gate que anulava checks de `hf_job_train_v90.py`, e a extracao
expected-aware que agora so pode desambiguar o ultimo boxed.

Artefato: `artifacts/v489_solution_integrity_audit/V489_SOLUTION_INTEGRITY_AUDIT.md`.

Atualizacao V490: o double check em modo debug confirmou que o gap atual nao e
um erro simples de ACC, split, tokenizacao ou threshold. V488 realmente ganhou
1 linha de equation (`518deb39`) e perdeu 2 linhas de bit (`8740ed31`,
`59bee375`), com truncation em `59bee375`. O dataset V390/V326 tem 5031/532
linhas, IDs/prompts unicos, zero overlap train/val e flags `gate/weak/full`
como `False`. A proxima tentativa so faz sentido se for estruturalmente
diferente: testar `target_parameters` MoE como treinaveis, nao apenas
frozen-active.

Artefato: `artifacts/v490_debug_double_check/V490_DEBUG_DOUBLE_CHECK_2026_05_16.md`.

### P3 - Smoke HF Minimo

Objetivo: verificar se o bug de continuidade era o gargalo sem gastar longo.

Executar:

- max 2 a 4 steps na primeira leitura de ACC;
- mesmo dataset limpo V390/V475;
- bit replay obrigatorio;
- log de componentes de loss, incluindo answer-span;
- weak micro-ACC no primeiro checkpoint, nao apenas `eval_loss`;
- kill-switch no primeiro checkpoint.
- proxima variante deve declarar `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`
  e incluir os LoRA `up_proj/down_proj` ligados aos `target_parameters`, porque
  V487 provou que `q/k/v/o/lm_head` com MoE frozen-active nao basta.

Gate:

| Metrica | Requisito |
|---|---:|
| Total weak | > 192 |
| equation_transform | > 56 |
| bit_manipulation | >= 136 |
| truncated | 0 |

Se o smoke der `equation=57` com perda de bit, nao promove. Esse caso ja
ocorreu duas vezes: V477 (`equation=57`, `bit=135`, `trunc=0`) e V488
(`equation=57`, `bit=134`, `trunc=1`). O ganho isolado de uma linha em
equation nao vale se derruba o guardrail de bit ou truncation.

### P4 - Equation Somente Se P3 Sinalizar

Objetivo: transformar os 4 ganhos CPU de equation em comportamento do adapter.

Executar somente se P3 mostrar ganho real sem perder bit. Se P3 repetir
`equation=57` com `bit<136` ou truncation, nao continuar GPU; voltar para CPU.

- construir dataset hard-negative curto com apenas regras CPU verificadas;
- target final-answer-only, sem auditoria textual;
- rejected = resposta exata errada do baseline;
- chosen = resposta correta curta;
- leakage gate por `id`, `prompt_sha256`, prompt normalizado e n-gram.

Nao executar:

- SFT amplo;
- mais epochs sem novo sinal;
- dataset sintetico que ja falhou transfer;
- treino com labels selecionados por weak/full.

### P5 - Bit Como Guardrail

Objetivo: preservar os 136/160 enquanto equation sobe.

Executar:

- manter replay de bit em todo treino de equation;
- validar que qualquer ganho de equation nao derruba bit;
- usar bit-pair/bitsum/stride somente para gerar traces curtos e verificados se
  houver cobertura nova maior que a linha V304.

Nao abrir job bit-only enquanto teacher CPU nao transfere para adapter.

### P6 - Package/Submit

Objetivo: submeter apenas quando existir ganho real.

Condicao minima:

- weak > 192;
- equation > 56;
- bit >= 136;
- trunc 0;
- full official-like > 823/947 ou evidencia equivalente via gate oficial-like.

Sem isso, nao packagear e nao submeter.

## Itens Removidos Do Plano Ativo

| Item | Motivo |
|---|---|
| Repetir H200 longo por eval_loss | loss nao correlacionou com ACC |
| Mais epochs/steps sem novo dado | quatro dias de plateau; custo sem sinal |
| V435E misto e format negatives | contaminado e bloqueado |
| V447/V448 trace SFT limpo | nao transferiu para adapter |
| V464/V468/V469 derivados | rota contaminada/quarentenada |
| Public adapters/submissions de terceiros | somente tecnica, nunca peso/submissao direta |
| Solver/verifier no runtime submit | contra regra adapter-only |
| Prompt hack, logit mask, constrained decoding | nao submit-safe |
| OpenRouter/provider/legal URLs | ruido; nao afeta ACC |

## Proxima Acao Imediata

1. Usar o V489/V490 audit como baseline de diagnostico: qualquer novo treino deve
   mostrar no manifesto se `target_parameters` estao `frozen_active`,
   `partially_trainable` ou `trainable`.
2. Criar gate CPU de regressao que bloqueie qualquer dataset/objetivo que
   reproduza o padrao `equation +1` com `bit -1/-2` ou truncation.
3. Rodar auditoria de extracao raw em todo weak eval novo:
   `simple_extracted` vs `expected_aware_extracted`, com delta documentado por
   familia.
4. Criar o proximo smoke com `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1` e
   `up_proj/down_proj` treinaveis junto ao conjunto minimo necessario. O job so
   pode seguir alem do primeiro checkpoint se mantiver `bit>=136`, `trunc=0` e
   superar `equation=56`.
5. Minerar `kishanvavdara/nemotron-reasoning-traj` apenas como fonte de
   padroes/fixtures apos anti-leakage; nao treinar direto sem gate.
6. So voltar para H200 se o gate CPU mostrar uma mudanca verificavel com
   `bit>=136`, `trunc=0` e pelo menos `equation>=57` sem regressao total.
7. Se o CPU diff mostrar que o erro vem de truncation/formato, corrigir formato
   e parser de treino antes de novo SFT. Se mostrar erro semantico, voltar ao
   DSL/trace curto e nao ao broad SFT.


# V490 Debug Double Check Summary
# V490 Debug Double Check - 2026-05-16

## Objetivo

Revalidar, em modo debug, as pecas que influenciam diretamente `loss`,
`eval_loss`, `ACC`, gate weak e decisao de FinOps depois do resultado V488.

## Estado Verificado

| Item | Resultado | Decisao |
|---|---:|---|
| V291/V290 baseline submit-safe | weak `192/315`, equation `56/155`, bit `136/160`, trunc `0` | baseline ativo |
| V488 checkpoint-10 | weak `191/315`, equation `57/155`, bit `134/160`, trunc `1` | bloqueado |
| Delta V488 vs V291/V290 | `+1` equation, `-2` bit, `+1` truncation | regressao real |
| HF jobs ativos | nenhum | sem custo pendente |

## Debug De Metricas

Comandos reexecutados:

- `python -m py_compile ...`
- `python scripts/audit_v449_acc_metric_integrity.py --self-test`
- `python scripts/hf_job_train_v90.py --self-test`
- `python scripts/kg1_static_safety_gate.py --self-test`
- `python scripts/kg1_static_safety_gate.py <critical files>`

Resultado:

- compilacao OK;
- self-tests OK;
- static safety gate OK;
- nenhum uso ativo de `answers_equivalent` como metrica oficial;
- thresholds ativos continuam `total>192`, `equation>56`, `bit>=136`,
  `truncated=0`.

Teste manual do extrator:

| Caso | Esperado |
|---|---|
| ultimo `\boxed{00000100}` | extrai `00000100`, correto |
| `\boxed{1010}` anterior e `\boxed{1011}` final | extrai `1011`, rejeita contra `1010` |
| simbolico escapado `\boxed{]\}\\!}` | extrai `]}\!`, correto |

Conclusao: o caminho de ACC nao esta inflando acertos por selecionar um boxed
anterior. A extracao expected-aware fica restrita ao ultimo boxed e serve apenas
para preservar payload simbolico com `}`, `{` ou `\`.

## Debug Do Resultado V488

Manifestos usados:

- `artifacts/v489_solution_integrity_audit/v489_v488_metric_integrity_v2_manifest.json`
- `artifacts/v489_solution_integrity_audit/v489_v488_vs_v290_diff_manifest.json`
- `artifacts/v489_solution_integrity_audit/v489_v488_vs_v290_row_diff.csv`

Diff confirmado:

| id | familia | Delta |
|---|---|---|
| `518deb39` | equation_transform | ganho real, V488 correto e baseline errado |
| `8740ed31` | bit_manipulation | regressao real |
| `59bee375` | bit_manipulation | regressao real e truncation |

Conclusao: o plateau nao esta vindo de erro simples de score. V488 realmente
transferiu um pequeno ganho de equation, mas comprou esse ganho com perda de
bit e truncation.

## Debug De Dataset

Dataset V390/V326 usado pela linha V487:

| Split | Rows | IDs unicos | Prompt dup | Familias |
|---|---:|---:|---:|---|
| train | 5031 | 5031 | 0 | bit `4231`, equation `800` |
| val | 532 | 532 | 0 | bit `332`, equation `200` |

Flags de contaminacao no metadata:

- `gate_rows_used_for_training=False` em todas as linhas;
- `weak_gate_rows_used_for_training=False` em todas as linhas;
- `full_gate_rows_used_for_training=False` em todas as linhas.

Tokenization gate V390/V326:

- offset masks: OK;
- fallback masks: `0`;
- prompt truncation: `0`;
- completion truncation: `0`;
- `boxed_suffix` validado.

Conclusao: nao apareceu sujeira simples no dataset V390/V326. O problema nao e
duplicidade, split leak obvio ou truncation no treino. A regressao aparece no
comportamento gerado pelo adapter.

## Debug De LoRA/Trainability

V487 corrigiu o alias estrutural de `target_parameters`, mas treinou somente:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `lm_head`

Os parametros MoE `mlp.experts.gate_up_proj` e `mlp.experts.down_proj` ficaram
ativos no forward, mas nao eram obrigados a ficar treinaveis naquela receita.
O script agora registra:

- `target_parameter_trainable_lora_tensors`;
- `target_parameter_trainable_lora_params`;
- `target_parameters_trainability_mode`;
- `require_lora_target_parameters_trainable`;
- `trainable_parameter_report_after_filter`.

Conclusao: a proxima tentativa ousada nao deve repetir V487. Ela deve testar
explicitamente uma variante `target_parameters_trainability_mode=trainable` com
kill-switch de ACC no primeiro checkpoint.

## Diagnostico Honesto

O que esta correto:

- metrica weak estrita;
- gate de thresholds;
- dataset V390/V326;
- tokenizacao;
- hash/row counts;
- preflight anti-gate-row;
- observabilidade de LoRA apos o patch.

O que ainda impede ganho:

1. O treino otimiza `loss/eval_loss`, mas o ganho que precisamos e row-level
   ACC em duas familias pequenas. O loss pode cair sem alterar as respostas
   finais corretas.
2. O ganho de equation e fragil: V477 e V488 chegaram a `equation=57`, mas
   ambos perderam bit ou truncation.
3. O adapter parece sensivel a mudancas pequenas de formato/decoding; portanto
   qualquer recipe sem micro-ACC no primeiro checkpoint e desperdicio.
4. O proximo caminho util e mexer no mecanismo de transferencia, nao procurar
   mais dados genericos.

## Proximo Caminho Ousado E Responsavel

1. Criar uma variante de smoke que treine tambem `up_proj/down_proj` LoRA
   associados a `target_parameters`, com
   `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`.
2. Limitar a execucao a checkpoint curto e abortar se:
   - total `<=192`;
   - `bit<136`;
   - `truncated>0`;
   - equation nao passar de `56`.
3. Se a variante der `equation=57` e bit `136`, continuar somente ate o proximo
   checkpoint. Se repetir `bit<136`, cancelar.
4. Se a variante falhar, abandonar broad SFT e voltar ao caminho CPU: minerar
   rows `518deb39`, `8740ed31`, `59bee375` e construir hard negatives
   especificos, nao dataset amplo.

## Decisao

Nao foi encontrado novo bug silencioso que explique sozinho o plateau. O bug
real anterior era observabilidade/trainability de `target_parameters`; ele foi
corrigido no codigo, mas V487/V488 provaram que isso sozinho nao basta.

O proximo experimento valido precisa ser mais ousado no conjunto treinavel
(`up_proj/down_proj` trainable), mas mais rigido em FinOps: micro-ACC no
primeiro checkpoint e cancelamento imediato se houver regressao.


# V489 Metric Audit Manifest
```json
{
  "builtin_metric_tests": [
    {
      "answer": "10101010",
      "expected": true,
      "name": "bit_exact_match",
      "observed": true,
      "passed": true,
      "prediction": "10101010",
      "reason": "8-bit binary answers must match exactly."
    },
    {
      "answer": "11100011",
      "expected": false,
      "name": "bit_near_numeric_must_not_pass",
      "observed": false,
      "passed": true,
      "prediction": "11100010",
      "reason": "Numeric tolerance would overcount binary strings; strict verifier must reject this."
    },
    {
      "answer": "42",
      "expected": true,
      "name": "numeric_tolerance_non_binary",
      "observed": true,
      "passed": true,
      "prediction": "42.1",
      "reason": "Non-binary numeric answers use 1% relative tolerance."
    },
    {
      "answer": "101",
      "expected": false,
      "name": "binary_like_answer_exact",
      "observed": false,
      "passed": true,
      "prediction": "101.0",
      "reason": "The project verifier mirrors the historical Kaggle verify_kaggle helper: all [01]+ answers are exact."
    },
    {
      "answer": "Alice",
      "expected": true,
      "name": "case_insensitive_symbolic",
      "observed": true,
      "passed": true,
      "prediction": "alice",
      "reason": "Non-numeric symbolic answers compare case-insensitively."
    }
  ],
  "decision": "metric_path_ok",
  "generated_at_utc": "2026-05-16T15:05:59.526683+00:00",
  "prediction_metric_audits": [
    {
      "column_audits": [
        {
          "disagreement_by_family": {
            "bit_manipulation": 14
          },
          "disagreement_examples": [
            {
              "answer": "11100011",
              "family_norm": "bit_manipulation",
              "id": "e6d2a064",
              "permissive_correct": true,
              "prediction": "11100010",
              "strict_correct": false
            },
            {
              "answer": "01000000",
              "family_norm": "bit_manipulation",
              "id": "0e70c867",
              "permissive_correct": true,
              "prediction": "01001100",
              "strict_correct": false
            },
            {
              "answer": "11001100",
              "family_norm": "bit_manipulation",
              "id": "f507b7f1",
              "permissive_correct": true,
              "prediction": "11011100",
              "strict_correct": false
            },
            {
              "answer": "11001001",
              "family_norm": "bit_manipulation",
              "id": "78d02fc5",
              "permissive_correct": true,
              "prediction": "11001011",
              "strict_correct": false
            },
            {
              "answer": "01010111",
              "family_norm": "bit_manipulation",
              "id": "4ef88f92",
              "permissive_correct": true,
              "prediction": "01010011",
              "strict_correct": false
            },
            {
              "answer": "01100000",
              "family_norm": "bit_manipulation",
              "id": "1a7c8520",
              "permissive_correct": true,
              "prediction": "01100110",
              "strict_correct": false
            },
            {
              "answer": "11111101",
              "family_norm": "bit_manipulation",
              "id": "202af98d",
              "permissive_correct": true,
              "prediction": "11111111",
              "strict_correct": false
            },
            {
              "answer": "11111011",
              "family_norm": "bit_manipulation",
              "id": "3ace787f",
              "permissive_correct": true,
              "prediction": "11111111",
              "strict_correct": false
            },
            {
              "answer": "01001010",
              "family_norm": "bit_manipulation",
              "id": "3a7dd604",
              "permissive_correct": true,
              "prediction": "01001011",
              "strict_correct": false
            },
            {
              "answer": "01111011",
              "family_norm": "bit_manipulation",
              "id": "4ada9150",
              "permissive_correct": true,
              "prediction": "01111111",
              "strict_correct": false
            },
            {
              "answer": "11110010",
              "family_norm": "bit_manipulation",
              "id": "06120e47",
              "permissive_correct": true,
              "prediction": "11111110",
              "strict_correct": false
            },
            {
              "answer": "11111010",
              "family_norm": "bit_manipulation",
              "id": "e1f3ffbb",
              "permissive_correct": true,
              "prediction": "11111011",
              "strict_correct": false
            },
            {
              "answer": "11011100",
              "family_norm": "bit_manipulation",
              "id": "4c327b55",
              "permissive_correct": true,
              "prediction": "11011110",
              "strict_correct": false
            },
            {
              "answer": "11001101",
              "family_norm": "bit_manipulation",
              "id": "82ae858c",
              "permissive_correct": true,
              "prediction": "11111101",
              "strict_correct": false
            }
          ],
          "permissive_accuracy": 0.653968253968254,
          "permissive_correct": 206,
          "prediction_column": "prediction",
          "strict_accuracy": 0.6095238095238096,
          "strict_correct": 192,
          "strict_vs_permissive_disagreement_rows": 14
        }
      ],
      "path": "artifacts\\v489_solution_integrity_audit\\inputs\\v290_baseline_predictions.csv",
      "prediction_columns": [
        "prediction"
      ],
      "rows": 315
    },
    {
      "column_audits": [
        {
          "disagreement_by_family": {
            "bit_manipulation": 15
          },
          "disagreement_examples": [
            {
              "answer": "11100011",
              "family_norm": "bit_manipulation",
              "id": "e6d2a064",
              "permissive_correct": true,
              "prediction": "11100010",
              "strict_correct": false
            },
            {
              "answer": "01000000",
              "family_norm": "bit_manipulation",
              "id": "0e70c867",
              "permissive_correct": true,
              "prediction": "01001100",
              "strict_correct": false
            },
            {
              "answer": "01101000",
              "family_norm": "bit_manipulation",
              "id": "8740ed31",
              "permissive_correct": true,
              "prediction": "01111000",
              "strict_correct": false
            },
            {
              "answer": "11001100",
              "family_norm": "bit_manipulation",
              "id": "f507b7f1",
              "permissive_correct": true,
              "prediction": "11011100",
              "strict_correct": false
            },
            {
              "answer": "11001001",
              "family_norm": "bit_manipulation",
              "id": "78d02fc5",
              "permissive_correct": true,
              "prediction": "11001011",
              "strict_correct": false
            },
            {
              "answer": "01010111",
              "family_norm": "bit_manipulation",
              "id": "4ef88f92",
              "permissive_correct": true,
              "prediction": "01010011",
              "strict_correct": false
            },
            {
              "answer": "01100000",
              "family_norm": "bit_manipulation",
              "id": "1a7c8520",
              "permissive_correct": true,
              "prediction": "01100110",
              "strict_correct": false
            },
            {
              "answer": "11111101",
              "family_norm": "bit_manipulation",
              "id": "202af98d",
              "permissive_correct": true,
              "prediction": "11111111",
              "strict_correct": false
            },
            {
              "answer": "11111011",
              "family_norm": "bit_manipulation",
              "id": "3ace787f",
              "permissive_correct": true,
              "prediction": "11111111",
              "strict_correct": false
            },
            {
              "answer": "01001010",
              "family_norm": "bit_manipulation",
              "id": "3a7dd604",
              "permissive_correct": true,
              "prediction": "01001011",
              "strict_correct": false
            },
            {
              "answer": "01111011",
              "family_norm": "bit_manipulation",
              "id": "4ada9150",
              "permissive_correct": true,
              "prediction": "01111111",
              "strict_correct": false
            },
            {
              "answer": "11110010",
              "family_norm": "bit_manipulation",
              "id": "06120e47",
              "permissive_correct": true,
              "prediction": "11111110",
              "strict_correct": false
            },
            {
              "answer": "11111010",
              "family_norm": "bit_manipulation",
              "id": "e1f3ffbb",
              "permissive_correct": true,
              "prediction": "11111011",
              "strict_correct": false
            },
            {
              "answer": "11011100",
              "family_norm": "bit_manipulation",
              "id": "4c327b55",
              "permissive_correct": true,
              "prediction": "11011110",
              "strict_correct": false
            },
            {
              "answer": "11001101",
              "family_norm": "bit_manipulation",
              "id": "82ae858c",
              "permissive_correct": true,
              "prediction": "11111101",
              "strict_correct": false
            }
          ],
          "permissive_accuracy": 0.653968253968254,
          "permissive_correct": 206,
          "prediction_column": "prediction",
          "strict_accuracy": 0.6063492063492063,
          "strict_correct": 191,
          "strict_vs_permissive_disagreement_rows": 15
        }
      ],
      "path": "artifacts\\v489_solution_integrity_audit\\inputs\\v488_predictions.csv",
      "prediction_columns": [
        "prediction"
      ],
      "rows": 315
    }
  ],
  "raw_extraction_audits": [
    {
      "answer_csv": "artifacts\\v489_solution_integrity_audit\\inputs\\v488_predictions.csv",
      "correctness_delta_by_family": {
        "equation_transform": 1
      },
      "correctness_delta_examples": [
        {
          "answer": "]}\\!",
          "expected_aware_correct": true,
          "expected_aware_extracted": "]}\\!",
          "family_norm": "equation_transform",
          "id": "4bb8c6cd",
          "simple_correct": false,
          "simple_extracted": "]"
        }
      ],
      "correctness_delta_rows": 1,
      "expected_aware_correct": 191,
      "expected_aware_minus_simple_correct": 1,
      "extraction_disagreement_rows": 1,
      "path": "artifacts\\v489_solution_integrity_audit\\inputs\\v488_raw_predictions_pre_score.csv",
      "raw_output_column": "raw_output",
      "rows": 315,
      "simple_correct": 190
    }
  ],
  "rule": "Promotion ACC must be computed with src.competition_utils.verify_answer. answers_equivalent is diagnostic-only because it numerically overcounts binary bit strings. Expected-aware extraction may only disambiguate the last boxed payload; it must not select an earlier boxed answer because that would leak the validation answer into extraction.",
  "schema_version": "kg1_v449_acc_metric_integrity_v1",
  "weak_answer_audit": {
    "binary_like_answer_by_family": {
      "bit_manipulation": 160,
      "equation_transform": 2
    },
    "binary_like_equation_rows": [
      {
        "answer_s": "101",
        "id": "209c43f0"
      },
      {
        "answer_s": "100",
        "id": "f333b67f"
      }
    ],
    "path": "artifacts\\v489_solution_integrity_audit\\inputs\\v488_predictions.csv",
    "rows": 315
  }
}

```

# V489 V488 vs V290 Row Diff Manifest
```json
{
  "baseline_path": "artifacts\\v489_solution_integrity_audit\\inputs\\v290_baseline_predictions.csv",
  "baseline_sha256": "910a051d8b8e652e37c0b0814ac59fe4a400b95cb432945b6a0244f97f5b31bf",
  "candidate_path": "artifacts\\v489_solution_integrity_audit\\inputs\\v488_predictions.csv",
  "candidate_sha256": "f42c632252d82f80406d7890914b1d563ef2b2c253af193698dcf1ace61e2da6",
  "column_correct_mismatches": [],
  "family_summary": {
    "bit_manipulation": {
      "both_correct": 134,
      "both_wrong": 24,
      "regression": 2,
      "rows": 160,
      "v488_truncated": 1
    },
    "equation_transform": {
      "both_correct": 56,
      "both_wrong": 98,
      "gain": 1,
      "rows": 155
    }
  },
  "gate": {
    "baseline_by_family": {
      "bit_manipulation": 136,
      "equation_transform": 56
    },
    "baseline_total": 192,
    "candidate_by_family": {
      "bit_manipulation": 134,
      "equation_transform": 57
    },
    "candidate_total": 191,
    "promote": false,
    "reason": "V488 has total 191, bit 134, equation 57, truncated 1; fails total/bit/truncation gate."
  },
  "overall": {
    "both_correct": 190,
    "both_wrong": 122,
    "gain": 1,
    "regression": 2
  },
  "rows": 315,
  "schema_version": "kg1_v489_v488_vs_v290_diff_v1"
}

```

# V390/V326 Dataset Manifest Used by V487
```json
{
  "generated_at_utc": "2026-05-14T19:42:07.385049+00:00",
  "inputs": {
    "v304_manifest_json": "artifacts\\v304_solver_trace_distill_dataset\\20260512T1430Z\\v304_solver_trace_distill_manifest.json",
    "v304_manifest_sha256": "8b226d63304328fc7baae296e3ecb2ab0c65d6a0b197d82b7c43769a8963109e",
    "v304_train": {
      "path": "artifacts\\v304_solver_trace_distill_dataset\\20260512T1430Z\\v304_solver_trace_distill_train.jsonl",
      "sha256": "7935ff999cdd8318de67538922de3651170c59baa2664a10beac3334dfcf9082"
    },
    "v304_val": {
      "path": "artifacts\\v304_solver_trace_distill_dataset\\20260512T1430Z\\v304_solver_trace_distill_val.jsonl",
      "sha256": "2b06224afe035c5085798f4a4be27e764ffaebde3ff7eee11c558c0cd5bdd29d"
    },
    "v325_manifest_json": "artifacts\\v390_v325_equation_no_loss_distill_dataset\\20260514T193847Z\\v390_v325_equation_no_loss_distill_manifest.json",
    "v325_manifest_sha256": "6b4d7c2ee8c27e907f2342dd00e572b8c652839565f5545f6657cf10af72431b",
    "v325_train": {
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v390_v325_equation_no_loss_distill_dataset\\20260514T193847Z\\v390_v325_equation_no_loss_distill_sft_train.jsonl",
      "sha256": "d78704157cd6ea55f6340dd5218f45c372dbe72ba227e9fd1e5851430d670793"
    },
    "v325_val": {
      "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v390_v325_equation_no_loss_distill_dataset\\20260514T193847Z\\v390_v325_equation_no_loss_distill_sft_val.jsonl",
      "sha256": "bdca7d2b7787e87eb1645ed6cb5b1e8687f1445fa76d58510b768de8eae1b5fa"
    }
  },
  "outputs": {
    "manifest_json": "artifacts\\v390_v326_equation_bit_replay_mix_dataset\\20260514T193847Z\\v390_v326_equation_bit_replay_mix_manifest.json",
    "train_jsonl": "artifacts\\v390_v326_equation_bit_replay_mix_dataset\\20260514T193847Z\\v390_v326_equation_bit_replay_mix_train.jsonl",
    "train_sha256": "92db92f4ce5a14ae9a27d1bd21cf64966fb94b63ccf35c6ebbbafa57f9b8a959",
    "val_jsonl": "artifacts\\v390_v326_equation_bit_replay_mix_dataset\\20260514T193847Z\\v390_v326_equation_bit_replay_mix_val.jsonl",
    "val_sha256": "90ec8a24d74d0fcc921665f20493469fae6e1e0d5a96730e430750d38fda0d82"
  },
  "overlap_summary": {
    "train_val_id_overlap": 0,
    "train_val_id_overlap_sample": [],
    "train_val_prompt_overlap": 0,
    "train_val_prompt_overlap_sample": []
  },
  "recommended_hf_controls": {
    "first_checkpoint_kill_switch": "bit>=136 and equation>56 on weak gate",
    "no_full_eval_or_submit_until": "adapter-only weak gate shows measured gain with no bit regression",
    "suggested_source_weights": {
      "v304_bit_replay_only": 1.0,
      "v325_equation_no_loss_distill": 4.0
    }
  },
  "schema_version": "kg1_v326_equation_bit_replay_mix_dataset_v1",
  "source_policy": {
    "bit_rows": "all V304 bit_manipulation rows retained for non-regression replay",
    "equation_rows": "V325 only; V304 broad equation replay intentionally excluded",
    "final_answer_format": "boxed_suffix",
    "physical_duplicates": false,
    "weak_or_full_gate_rows_used_for_training": false
  },
  "train_summary": {
    "bad_rows_first10": [],
    "duplicate_ids": 0,
    "duplicate_prompts": 0,
    "family_counts": {
      "bit_manipulation": 4231,
      "equation_transform": 800
    },
    "label": "train",
    "origin_counts": {
      "v304_bit_replay_only": 4231,
      "v325_equation_no_loss_distill": 800
    },
    "rows": 5031,
    "source_counts": {
      "v215_replay_anchor": 253,
      "v216_base_clean_safe_strict_bit": 1796,
      "v216_synthetic_kg1_bit_rules": 646,
      "v304_solver_trace_bit_fullbyte_distill_exact": 1056,
      "v304_solver_trace_bit_fullbyte_distill_random": 480,
      "v325_equation_no_loss_distill": 800
    },
    "subcategory_counts": {
      "bit_fullbyte_binary": 120,
      "bit_fullbyte_safe_ternary": 360,
      "bit_fullbyte_v300_gain_pattern": 1056,
      "bit_manipulation": 646,
      "equation_numeric_add_direct": 160,
      "equation_numeric_colon_absdiff": 160,
      "equation_numeric_colon_trailing_zero": 160,
      "equation_numeric_minus_direct_negative": 160,
      "equation_numeric_minus_signed": 160,
      "unknown": 2049
    },
    "unique_ids": 5031,
    "unique_prompt_hashes": 5031
  },
  "training_authorization": "blocked_until_v286_tokenization_gate_and_hf_smoke_kill_switch",
  "validation_summary": {
    "bad_rows_first10": [],
    "duplicate_ids": 0,
    "duplicate_prompts": 0,
    "family_counts": {
      "bit_manipulation": 332,
      "equation_transform": 200
    },
    "label": "validation",
    "origin_counts": {
      "v304_bit_replay_only": 332,
      "v325_equation_no_loss_distill": 200
    },
    "rows": 532,
    "source_counts": {
      "v215_replay_anchor": 30,
      "v216_base_clean_safe_strict_bit": 80,
      "v216_synthetic_kg1_bit_rules": 54,
      "v304_solver_trace_bit_fullbyte_distill_exact": 88,
      "v304_solver_trace_bit_fullbyte_distill_random": 80,
      "v325_equation_no_loss_distill": 200
    },
    "subcategory_counts": {
      "bit_fullbyte_binary": 20,
      "bit_fullbyte_safe_ternary": 60,
      "bit_fullbyte_v300_gain_pattern": 88,
      "bit_manipulation": 54,
      "equation_numeric_add_direct": 40,
      "equation_numeric_colon_absdiff": 40,
      "equation_numeric_colon_trailing_zero": 40,
      "equation_numeric_minus_direct_negative": 40,
      "equation_numeric_minus_signed": 40,
      "unknown": 110
    },
    "unique_ids": 532,
    "unique_prompt_hashes": 532
  }
}

```

# V390/V326 Tokenization Gate Manifest
```json
{
  "blocked_actions": [
    "full_eval",
    "package",
    "kaggle_submit"
  ],
  "config": {
    "assistant_final_answer_mode": "boxed_suffix",
    "max_length": 1024,
    "max_prompt_truncation_rate": 0.0,
    "min_train_rows": 5031,
    "min_val_rows": 532,
    "require_offset_mask": true
  },
  "dataset_manifest_json": "artifacts\\v390_v326_equation_bit_replay_mix_dataset\\20260514T193847Z\\v390_v326_equation_bit_replay_mix_manifest.json",
  "dataset_manifest_sha256": "6b6951aca6a794591a3cff20246d293a8853bbc756095432afb9461f2cf617e0",
  "decision": {
    "next_action": "Only consider a tiny HF smoke train if roadmap risk/budget gates approve it.",
    "reason": "train_rows=5031; val_rows=532; train_token_max=749; val_token_max=748; completion_truncation=0",
    "status": "tokenization_gate_passed"
  },
  "generated_at_utc": "2026-05-14T19:42:36.110891+00:00",
  "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
  "model_revision": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
  "outputs": {
    "manifest_json": "artifacts\\v390_v326_tokenization_gate\\20260514T193847Z\\v286_generic_tokenization_gate_manifest.json"
  },
  "schema_version": "kg1_v286_generic_tokenization_gate_v1",
  "tokenization": {
    "train": {
      "completion_tokens_dropped": 0,
      "fallback_masks": 0,
      "family_summary": {
        "bit_manipulation": {
          "loss_token_max": 479,
          "loss_token_min": 18,
          "loss_token_p50": 18,
          "rows": 4231,
          "token_max": 749,
          "token_p50": 317,
          "token_p90": 748,
          "token_p99": 748
        },
        "equation_transform": {
          "loss_token_max": 116,
          "loss_token_min": 92,
          "loss_token_p50": 106,
          "rows": 800,
          "token_max": 262,
          "token_p50": 243,
          "token_p90": 260,
          "token_p99": 262
        }
      },
      "loss_token_max": 479,
      "loss_token_min": 18,
      "loss_token_p50": 18,
      "offset_masks": 5031,
      "prompt_truncated": 0,
      "prompt_truncation_rate": 0.0,
      "rows": 5031,
      "token_max": 749,
      "token_mean": 422.724,
      "token_min": 235,
      "token_p50": 307,
      "token_p90": 748,
      "token_p99": 748
    },
    "validation": {
      "completion_tokens_dropped": 0,
      "fallback_masks": 0,
      "family_summary": {
        "bit_manipulation": {
          "loss_token_max": 478,
          "loss_token_min": 18,
          "loss_token_p50": 421,
          "rows": 332,
          "token_max": 748,
          "token_p50": 691,
          "token_p90": 748,
          "token_p99": 748
        },
        "equation_transform": {
          "loss_token_max": 116,
          "loss_token_min": 92,
          "loss_token_p50": 106,
          "rows": 200,
          "token_max": 262,
          "token_p50": 243,
          "token_p90": 260,
          "token_p99": 262
        }
      },
      "loss_token_max": 478,
      "loss_token_min": 18,
      "loss_token_p50": 106,
      "offset_masks": 532,
      "prompt_truncated": 0,
      "prompt_truncation_rate": 0.0,
      "rows": 532,
      "token_max": 748,
      "token_mean": 415.038,
      "token_min": 236,
      "token_p50": 288,
      "token_p90": 747,
      "token_p99": 748
    }
  },
  "tokenizer_info": {
    "class": "TokenizersBackend",
    "eos_token": "<|im_end|>",
    "is_fast": true,
    "pad_token": "<|im_end|>",
    "toy": false
  },
  "validation": {
    "train": {
      "family_counts": {
        "bit_manipulation": 4231,
        "equation_transform": 800
      },
      "prompt_answer_hash_count": 5031,
      "prompt_hash_count": 5031,
      "rows": 5031,
      "sha256_expected": "92db92f4ce5a14ae9a27d1bd21cf64966fb94b63ccf35c6ebbbafa57f9b8a959",
      "source_counts": {
        "v215_replay_anchor": 253,
        "v216_base_clean_safe_strict_bit": 1796,
        "v216_synthetic_kg1_bit_rules": 646,
        "v304_solver_trace_bit_fullbyte_distill_exact": 1056,
        "v304_solver_trace_bit_fullbyte_distill_random": 480,
        "v325_equation_no_loss_distill": 800
      },
      "subcategory_counts": {
        "bit_fullbyte_binary": 120,
        "bit_fullbyte_safe_ternary": 360,
        "bit_fullbyte_v300_gain_pattern": 1056,
        "bit_manipulation": 646,
        "equation_numeric_add_direct": 160,
        "equation_numeric_colon_absdiff": 160,
        "equation_numeric_colon_trailing_zero": 160,
        "equation_numeric_minus_direct_negative": 160,
        "equation_numeric_minus_signed": 160,
        "unknown": 2049
      },
      "unique_ids": 5031
    },
    "train_val_prompt_answer_overlap": 0,
    "train_val_prompt_overlap": 0,
    "validation": {
      "family_counts": {
        "bit_manipulation": 332,
        "equation_transform": 200
      },
      "prompt_answer_hash_count": 532,
      "prompt_hash_count": 532,
      "rows": 532,
      "sha256_expected": "90ec8a24d74d0fcc921665f20493469fae6e1e0d5a96730e430750d38fda0d82",
      "source_counts": {
        "v215_replay_anchor": 30,
        "v216_base_clean_safe_strict_bit": 80,
        "v216_synthetic_kg1_bit_rules": 54,
        "v304_solver_trace_bit_fullbyte_distill_exact": 88,
        "v304_solver_trace_bit_fullbyte_distill_random": 80,
        "v325_equation_no_loss_distill": 200
      },
      "subcategory_counts": {
        "bit_fullbyte_binary": 20,
        "bit_fullbyte_safe_ternary": 60,
        "bit_fullbyte_v300_gain_pattern": 88,
        "bit_manipulation": 54,
        "equation_numeric_add_direct": 40,
        "equation_numeric_colon_absdiff": 40,
        "equation_numeric_colon_trailing_zero": 40,
        "equation_numeric_minus_direct_negative": 40,
        "equation_numeric_minus_signed": 40,
        "unknown": 110
      },
      "unique_ids": 532
    }
  }
}
```

# V485 PEFT Roundtrip Gate for Seed Adapter
```json
{
  "file_info": {
    "adapter_model_size": 4259063856,
    "config_filename": "checkpoint-6/adapter_config.json",
    "repo_id": "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
    "resolved_revision": "75909c9b40d8b7fa846d379d9d764fa33daeb9e2",
    "revision": "75909c9b40d8b7fa846d379d9d764fa33daeb9e2",
    "weights_filename": "checkpoint-6/adapter_model.safetensors"
  },
  "generated_at_utc": "2026-05-16T13:18:27.924672+00:00",
  "summary": {
    "config": {
      "adapter_config_sha256": "4dfe5ec8ca33dade558c27af9a2f2f542f1f0d05ff0ee9d85bb5e5a57f12aae4",
      "lora_alpha": 32,
      "modules_to_save": [],
      "r": 32,
      "target_modules": [
        "down_proj",
        "in_proj",
        "k_proj",
        "lm_head",
        "o_proj",
        "out_proj",
        "q_proj",
        "up_proj",
        "v_proj"
      ],
      "target_parameters": [
        "mlp.experts.down_proj",
        "mlp.experts.gate_up_proj"
      ]
    },
    "coverage": {
      "allowed_non_lora_tensor_keys": [
        "base_model.model.lm_head.base_layer.weight"
      ],
      "module_lora_params": {
        "down_proj": 432791552,
        "in_proj": 9562112,
        "k_proj": 565248,
        "lm_head": 4280320,
        "o_proj": 1302528,
        "out_proj": 4993024,
        "q_proj": 1302528,
        "up_proj": 432791552,
        "v_proj": 565248
      },
      "module_lora_tensors": {
        "down_proj": 5934,
        "in_proj": 46,
        "k_proj": 12,
        "lm_head": 2,
        "o_proj": 12,
        "out_proj": 46,
        "q_proj": 12,
        "up_proj": 5934,
        "v_proj": 12
      },
      "modules_to_save_key_count": 0,
      "non_lora_tensor_key_count": 0,
      "target_parameter_lora_params": {
        "mlp.experts.down_proj": 432791552,
        "mlp.experts.gate_up_proj": 432791552
      },
      "target_parameter_lora_tensors": {
        "mlp.experts.down_proj": 5934,
        "mlp.experts.gate_up_proj": 5934
      }
    },
    "errors": [],
    "hf_gpu_allowed": true,
    "safetensors": {
      "dtype_counts": {
        "BF16": 1,
        "F32": 12010
      },
      "key_shape_dtype_sha256": "fc540b014aaba5eeef5098abbb638b70e639f5e7ba04386535012f54b8d98a95",
      "tensor_count": 12011,
      "total_parameters": 1240475648
    }
  },
  "version": "V485_PEFT_ROUNDTRIP_GATE"
}

```

# Key Code: verify/extract metric path
```python
0210: def extract_final_answer_for_expected(text: str | None, expected: object) -> str:
0211:     """Extract a final answer, using the expected answer only to disambiguate.
0212: 
0213:     Symbolic tasks may have literal ``{`` or ``}`` inside the answer.  A raw
0214:     model output such as ``\boxed{?}}`` is ambiguous without knowing whether the
0215:     second ``}`` is payload or delimiter.  Evaluation already has the expected
0216:     answer, so this helper prefers an exact boxed payload that verifies against
0217:     that expected answer, then falls back to the public extraction order.
0218:     """
0219: 
0220:     if text is None:
0221:         return "NOT_FOUND"
0222:     value = str(text)
0223:     expected_text = str(expected).strip()
0224:     if expected_text:
0225:         marker = r"\boxed{"
0226:         cursor = 0
0227:         marker_positions: list[int] = []
0228:         raw_expected_variants = [expected_text]
0229:         escaped_expected = escape_boxed_answer(expected_text)
0230:         if escaped_expected != expected_text:
0231:             raw_expected_variants.insert(0, escaped_expected)
0232:         while True:
0233:             marker_pos = value.find(marker, cursor)
0234:             if marker_pos == -1:
0235:                 break
0236:             marker_positions.append(marker_pos)
0237:             cursor = marker_pos + len(marker)
0238:         if marker_positions:
0239:             start = marker_positions[-1] + len(marker)
0240:             tail = value[start:]
0241:             for variant in raw_expected_variants:
0242:                 if not variant or not tail.startswith(variant):
0243:                     continue
0244:                 after = start + len(variant)
0245:                 if after >= len(value) or value[after] in "}\r\n\t `.,;)]":
0246:                     observed_text = canonical_boxed_payload(variant)
0247:                     if verify_answer(expected_text, observed_text):
0248:                         return observed_text
0249:     return extract_final_answer(value)
0250: 
0251: 
0252: def verify_answer(stored_answer: object, predicted: object) -> bool:
0253:     """Verify a prediction with the public Kaggle metric behavior."""
0254: 
0255:     expected = str(stored_answer).strip()
0256:     observed = str(predicted).strip()
0257:     if re.fullmatch(r"[01]+", expected):
0258:         return observed.lower() == expected.lower()
0259:     try:
0260:         return math.isclose(float(expected), float(observed), rel_tol=1e-2, abs_tol=1e-5)
0261:     except Exception:
0262:         return observed.lower() == expected.lower()
0263: 
0264: 
0265: def canonical_answer(value: object) -> str:
0266:     if value is None:
0267:         return ""
0268:     text = unicodedata.normalize("NFKC", str(value))
0269:     return re.sub(r"\s+", " ", text).strip()
0270: 
0271: 
0272: def escape_boxed_answer(value: object) -> str:
0273:     return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
0274: 
0275: 
0276: def unescape_latex_braces(value: object) -> str:
0277:     return str(value).replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
0278: 
0279: 
0280: def canonical_boxed_payload(value: object) -> str:
0281:     return canonical_answer(unescape_latex_braces(value))
0282: 
0283: 
0284: def parse_finite_number(value: object) -> float | None:
0285:     text = canonical_answer(value).replace(",", "")
0286:     if not text:
0287:         return None
0288:     try:
0289:         number = float(text)
0290:     except ValueError:
0291:         return None
0292:     return number if math.isfinite(number) else None
0293: 
0294: 
0295: def answers_equivalent(
0296:     expected: object,
0297:     observed: object,
0298:     *,
0299:     rel_tol: float = 1e-2,
0300:     abs_tol: float = 1e-5,
0301:     observed_is_boxed_payload: bool = False,
0302: ) -> bool:
0303:     expected_text = canonical_answer(expected)
0304:     observed_text = canonical_boxed_payload(observed) if observed_is_boxed_payload else canonical_answer(observed)
0305:     expected_number = parse_finite_number(expected_text)
0306:     observed_number = parse_finite_number(observed_text)
0307:     if expected_number is not None and observed_number is not None:
0308:         return math.isclose(expected_number, observed_number, rel_tol=rel_tol, abs_tol=abs_tol)
0309:     return expected_text.lower() == observed_text.lower()
0310: 
0311: 
0312: def box_answer(value: object) -> str:
0313:     return f"\\boxed{{{escape_boxed_answer(value)}}}"
```

# Key Code: LoRA trainable filter and target_parameters observability
```python
0602: def apply_trainable_lora_module_filter(model: torch.nn.Module) -> dict[str, Any]:
0603:     """Freeze loaded LoRA params except a deliberate module allowlist.
0604: 
0605:     The 0.86 adapter contains a full 9-module Huikang-style adapter.  Loading it
0606:     as fully trainable uses roughly 888M trainable params and can OOM on long
0607:     examples.  For a conservative delta run, keep the full adapter active in the
0608:     forward pass but update only the lighter routing/attention/mamba projection
0609:     modules named by TRAINABLE_LORA_MODULES.
0610:     """
0611: 
0612:     modules = parse_csv_items(TRAINABLE_LORA_MODULES)
0613:     name_substrings = parse_csv_items(TRAINABLE_LORA_NAME_SUBSTRINGS)
0614:     required_trainable_substrings = parse_csv_items(REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS)
0615:     target_parameters = parse_target_parameters(LORA_TARGET_PARAMETERS)
0616:     target_parameter_lora_tensors: dict[str, int] = {item: 0 for item in target_parameters}
0617:     target_parameter_lora_params: dict[str, int] = {item: 0 for item in target_parameters}
0618:     target_parameter_trainable_lora_tensors: dict[str, int] = {item: 0 for item in target_parameters}
0619:     target_parameter_trainable_lora_params: dict[str, int] = {item: 0 for item in target_parameters}
0620:     required_trainable_lora_tensors: dict[str, int] = {item: 0 for item in required_trainable_substrings}
0621:     required_trainable_lora_params: dict[str, int] = {item: 0 for item in required_trainable_substrings}
0622: 
0623:     if not modules and not name_substrings:
0624:         trainable_lora_params = 0
0625:         frozen_lora_params = 0
0626:         trainable_lora_tensors = 0
0627:         frozen_lora_tensors = 0
0628:         for name, param in model.named_parameters():
0629:             if ".lora_" not in name:
0630:                 continue
0631:             count = int(param.numel())
0632:             matched_target_parameters = [
0633:                 item for item in target_parameters if target_parameter_name_matches(item, name)
0634:             ]
0635:             for target_parameter in matched_target_parameters:
0636:                 target_parameter_lora_tensors[target_parameter] += 1
0637:                 target_parameter_lora_params[target_parameter] += count
0638:             if param.requires_grad:
0639:                 trainable_lora_params += count
0640:                 trainable_lora_tensors += 1
0641:                 for target_parameter in matched_target_parameters:
0642:                     target_parameter_trainable_lora_tensors[target_parameter] += 1
0643:                     target_parameter_trainable_lora_params[target_parameter] += count
0644:             else:
0645:                 frozen_lora_params += count
0646:                 frozen_lora_tensors += 1
0647:         missing_target_parameters = [
0648:             item for item, tensors in target_parameter_lora_tensors.items() if tensors <= 0
0649:         ]
0650:         if REQUIRE_LORA_TARGET_PARAMETER_MATCH and missing_target_parameters:
0651:             raise RuntimeError(
0652:                 "LORA_TARGET_PARAMETERS were configured but no matching LoRA tensors were found: "
0653:                 + ", ".join(missing_target_parameters)
0654:             )
0655:         missing_trainable_target_parameters = [
0656:             item for item, tensors in target_parameter_trainable_lora_tensors.items() if tensors <= 0
0657:         ]
0658:         if target_parameters and len(missing_trainable_target_parameters) == len(target_parameters):
0659:             target_parameters_trainability_mode = "frozen_active"
0660:         elif missing_trainable_target_parameters:
0661:             target_parameters_trainability_mode = "partially_trainable"
0662:         elif target_parameters:
0663:             target_parameters_trainability_mode = "trainable"
0664:         else:
0665:             target_parameters_trainability_mode = "not_configured"
0666:         if REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE and missing_trainable_target_parameters:
0667:             raise RuntimeError(
0668:                 "LORA_TARGET_PARAMETERS were configured but are not all trainable under the current "
0669:                 "LoRA selector. Missing trainable target_parameters: "
0670:                 + ", ".join(missing_trainable_target_parameters)
0671:             )
0672:         return {
0673:             "enabled": False,
0674:             "modules": [],
0675:             "name_substrings": [],
0676:             "trainable_lora_params": trainable_lora_params,
0677:             "frozen_lora_params": frozen_lora_params,
0678:             "trainable_lora_tensors": trainable_lora_tensors,
0679:             "frozen_lora_tensors": frozen_lora_tensors,
0680:             "trainable_by_module": {},
0681:             "trainable_by_name_substring": {},
0682:             "frozen_by_module": {},
0683:             "target_parameter_lora_tensors": target_parameter_lora_tensors,
0684:             "target_parameter_lora_params": target_parameter_lora_params,
0685:             "target_parameter_trainable_lora_tensors": target_parameter_trainable_lora_tensors,
0686:             "target_parameter_trainable_lora_params": target_parameter_trainable_lora_params,
0687:             "target_parameters_trainability_mode": target_parameters_trainability_mode,
0688:             "require_lora_target_parameters_trainable": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
0689:             "required_trainable_lora_tensors": required_trainable_lora_tensors,
0690:             "required_trainable_lora_params": required_trainable_lora_params,
0691:         }
0692: 
0693:     trainable_by_module: dict[str, int] = {module: 0 for module in modules}
0694:     trainable_by_name_substring: dict[str, int] = {substring: 0 for substring in name_substrings}
0695:     frozen_by_module: dict[str, int] = {}
0696:     trainable_lora_params = 0
0697:     frozen_lora_params = 0
0698:     trainable_lora_tensors = 0
0699:     frozen_lora_tensors = 0
0700: 
0701:     for name, param in model.named_parameters():
0702:         if ".lora_" not in name:
0703:             continue
0704:         matched_module = next((module for module in modules if f".{module}." in name), None)
0705:         matched_name_substrings = [substring for substring in name_substrings if substring in name]
0706:         matched_target_parameters: list[str] = []
0707:         for target_parameter in target_parameters:
0708:             if target_parameter_name_matches(target_parameter, name):
0709:                 matched_target_parameters.append(target_parameter)
0710:                 count = int(param.numel())
0711:                 target_parameter_lora_tensors[target_parameter] += 1
0712:                 target_parameter_lora_params[target_parameter] += count
0713:         is_trainable_by_name = bool(matched_name_substrings)
0714:         if matched_module or is_trainable_by_name:
0715:             param.requires_grad_(True)
0716:             count = int(param.numel())
0717:             trainable_lora_params += count
0718:             trainable_lora_tensors += 1
0719:             if matched_module:
0720:                 trainable_by_module[matched_module] = trainable_by_module.get(matched_module, 0) + count
0721:             for substring in matched_name_substrings:
0722:                 trainable_by_name_substring[substring] = (
0723:                     trainable_by_name_substring.get(substring, 0) + count
0724:                 )
0725:             for target_parameter in matched_target_parameters:
0726:                 target_parameter_trainable_lora_tensors[target_parameter] += 1
0727:                 target_parameter_trainable_lora_params[target_parameter] += count
0728:             for required_substring in required_trainable_substrings:
0729:                 if required_substring in name:
0730:                     required_trainable_lora_tensors[required_substring] += 1
0731:                     required_trainable_lora_params[required_substring] += count
0732:         else:
0733:             param.requires_grad_(False)
0734:             count = int(param.numel())
0735:             frozen_lora_params += count
0736:             frozen_lora_tensors += 1
0737:             module_name = "unknown"
0738:             for candidate in [
0739:                 "q_proj",
0740:                 "k_proj",
0741:                 "v_proj",
0742:                 "o_proj",
0743:                 "in_proj",
0744:                 "out_proj",
0745:                 "up_proj",
0746:                 "down_proj",
0747:                 "lm_head",
0748:             ]:
0749:                 if f".{candidate}." in name:
0750:                     module_name = candidate
0751:                     break
0752:             frozen_by_module[module_name] = frozen_by_module.get(module_name, 0) + count
0753: 
0754:     if trainable_lora_params <= 0:
0755:         raise RuntimeError(
0756:             "Trainable LoRA selector matched no LoRA parameters: "
0757:             f"TRAINABLE_LORA_MODULES={TRAINABLE_LORA_MODULES!r}, "
0758:             f"TRAINABLE_LORA_NAME_SUBSTRINGS={TRAINABLE_LORA_NAME_SUBSTRINGS!r}"
0759:         )
0760:     if REQUIRE_LORA_TARGET_PARAMETER_MATCH:
0761:         missing_target_parameters = [
0762:             item for item, tensors in target_parameter_lora_tensors.items() if tensors <= 0
0763:         ]
0764:         if missing_target_parameters:
0765:             raise RuntimeError(
0766:                 "LORA_TARGET_PARAMETERS were configured but no matching LoRA tensors were found: "
0767:                 + ", ".join(missing_target_parameters)
0768:             )
0769:     missing_trainable_target_parameters = [
0770:         item for item, tensors in target_parameter_trainable_lora_tensors.items() if tensors <= 0
0771:     ]
0772:     if target_parameters and len(missing_trainable_target_parameters) == len(target_parameters):
0773:         target_parameters_trainability_mode = "frozen_active"
0774:     elif missing_trainable_target_parameters:
0775:         target_parameters_trainability_mode = "partially_trainable"
0776:     elif target_parameters:
0777:         target_parameters_trainability_mode = "trainable"
0778:     else:
0779:         target_parameters_trainability_mode = "not_configured"
0780:     if REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE and missing_trainable_target_parameters:
0781:         raise RuntimeError(
0782:             "LORA_TARGET_PARAMETERS were configured but are not all trainable under the current "
0783:             "LoRA selector. Missing trainable target_parameters: "
0784:             + ", ".join(missing_trainable_target_parameters)
0785:         )
0786:     missing_required_trainable = [
0787:         item for item, tensors in required_trainable_lora_tensors.items() if tensors <= 0
0788:     ]
0789:     if missing_required_trainable:
0790:         raise RuntimeError(
0791:             "Required trainable LoRA name substrings were not matched: "
0792:             + ", ".join(missing_required_trainable)
0793:         )
0794: 
0795:     return {
0796:         "enabled": True,
0797:         "modules": modules,
0798:         "name_substrings": name_substrings,
0799:         "trainable_lora_params": trainable_lora_params,
0800:         "frozen_lora_params": frozen_lora_params,
0801:         "trainable_lora_tensors": trainable_lora_tensors,
0802:         "frozen_lora_tensors": frozen_lora_tensors,
0803:         "trainable_by_module": trainable_by_module,
0804:         "trainable_by_name_substring": trainable_by_name_substring,
0805:         "frozen_by_module": frozen_by_module,
0806:         "target_parameter_lora_tensors": target_parameter_lora_tensors,
0807:         "target_parameter_lora_params": target_parameter_lora_params,
0808:         "target_parameter_trainable_lora_tensors": target_parameter_trainable_lora_tensors,
0809:         "target_parameter_trainable_lora_params": target_parameter_trainable_lora_params,
0810:         "target_parameters_trainability_mode": target_parameters_trainability_mode,
0811:         "require_lora_target_parameters_trainable": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
0812:         "required_trainable_lora_tensors": required_trainable_lora_tensors,
0813:         "required_trainable_lora_params": required_trainable_lora_params,
0814:     }
```

# Key Code: answer span masking/tokenization risk
```python
1236:         marker_positions: list[int] = []
1237:         for marker in ("Final answer:", "ANSWER:"):
1238:             pos = assistant_text.rfind(marker)
1239:             if pos >= 0:
1240:                 marker_positions.append(pos)
1241:         if marker_positions:
1242:             return max(marker_positions), len(assistant_text)
1243:         return None
1244: 
1245:     assistant_start = full_text.rfind(assistant_text)
1246:     weighted_tokens = 0
1247:     if assistant_start >= 0:
1248:         try:
1249:             encoded = tokenizer(
1250:                 full_text,
1251:                 add_special_tokens=False,
1252:                 return_offsets_mapping=True,
1253:             )
1254:             input_ids = list(encoded["input_ids"])
1255:             offsets = encoded.get("offset_mapping")
1256:             if offsets and len(offsets) == len(input_ids):
1257:                 loss_mask: list[float] = []
1258:                 answer_span = answer_char_span()
1259:                 answer_start = answer_end = -1
1260:                 if answer_span is not None and ANSWER_SPAN_LOSS_WEIGHT > 1.0:
1261:                     answer_start = assistant_start + answer_span[0]
1262:                     answer_end = assistant_start + answer_span[1]
1263:                 for start, end in offsets:
1264:                     token_start = int(start)
1265:                     token_end = int(end)
1266:                     if token_end <= assistant_start:
1267:                         loss_mask.append(0.0)
1268:                     elif (
1269:                         answer_end > answer_start
1270:                         and token_end > answer_start
1271:                         and token_start < answer_end
1272:                     ):
1273:                         loss_mask.append(float(ANSWER_SPAN_LOSS_WEIGHT))
1274:                         weighted_tokens += 1
1275:                     else:
1276:                         loss_mask.append(1.0)
1277:                 return input_ids, loss_mask, True, False, weighted_tokens
1278:         except (NotImplementedError, TypeError, ValueError):
1279:             pass
1280: 
1281:     full_ids = tokenizer.encode(full_text, add_special_tokens=False)
1282:     prompt_messages = [m for m in messages if m.get("role") != "assistant"]
1283:     # enable_thinking=True aligns with Tong recipe (Progress Prize winner)
1284:     # so the `<think>` scaffold is preserved in the prompt and the completion
1285:     # can emit `</think>\n\boxed{answer}` naturally.
1286:     try:
1287:         prompt_text = tokenizer.apply_chat_template(
1288:             prompt_messages,
1289:             tokenize=False,
1290:             add_generation_prompt=True,
1291:             enable_thinking=True,
1292:         )
1293:     except TypeError:
1294:         prompt_text = tokenizer.apply_chat_template(
1295:             prompt_messages, tokenize=False, add_generation_prompt=True
1296:         )
1297:     prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
1298:     prefix_mismatch = full_ids[: len(prompt_ids)] != prompt_ids
1299:     prompt_len = min(len(prompt_ids), len(full_ids))
1300:     loss_mask = [0.0] * prompt_len + [1.0] * (len(full_ids) - prompt_len)
1301:     return full_ids, loss_mask, False, prefix_mismatch, 0
1302: 
1303: 
1304: def tokenize_examples(
1305:     examples: list[dict[str, Any]],
1306:     tokenizer: Any,
1307:     label: str,
1308: ) -> list[dict[str, Any]]:
1309:     tokenized: list[dict[str, Any]] = []
1310:     skipped_missing_messages = 0
1311:     skipped_no_loss = 0
1312:     truncated_count = 0
1313:     prompt_truncated_count = 0
1314:     prompt_tokens_dropped_est = 0
1315:     offset_mask_count = 0
1316:     fallback_mask_count = 0
1317:     fallback_prefix_mismatch_count = 0
1318:     answer_span_weighted_examples = 0
1319:     answer_span_weighted_tokens = 0
1320:     for ex in examples:
1321:         msgs = ex.get("messages", [])
1322:         if not msgs:
1323:             skipped_missing_messages += 1
1324:             continue
1325: 
1326:         try:
1327:             full_text = tokenizer.apply_chat_template(
1328:                 msgs,
1329:                 tokenize=False,
1330:                 add_generation_prompt=False,
1331:                 enable_thinking=True,
1332:             )
1333:         except TypeError:
1334:             full_text = tokenizer.apply_chat_template(
1335:                 msgs, tokenize=False, add_generation_prompt=False
1336:             )
1337:         full_ids, loss_mask, used_offsets, prefix_mismatch, weighted_tokens = build_completion_mask(
1338:             full_text, msgs, tokenizer
1339:         )
1340:         if used_offsets:
1341:             offset_mask_count += 1
1342:         else:
1343:             fallback_mask_count += 1
1344:         if prefix_mismatch:
1345:             fallback_prefix_mismatch_count += 1
1346:         if weighted_tokens:
1347:             answer_span_weighted_examples += 1
1348:             answer_span_weighted_tokens += weighted_tokens
1349: 
1350:         if len(full_ids) > MAX_LENGTH:
1351:             first_loss_idx = next((idx for idx, value in enumerate(loss_mask) if value), len(loss_mask))
1352:             overflow = len(full_ids) - MAX_LENGTH
1353:             dropped_loss = float(sum(loss_mask[:overflow]))
1354:             if dropped_loss > 0:
1355:                 raise RuntimeError(
1356:                     f"{label} truncation would drop supervised completion tokens for {ex.get('id', '')}: "
1357:                     f"overflow={overflow} dropped_loss_weight={dropped_loss:.4f} max_length={MAX_LENGTH}"
1358:                 )
1359:             dropped_prompt_tokens = min(overflow, first_loss_idx)
1360:             if dropped_prompt_tokens > 0:
1361:                 prompt_truncated_count += 1
1362:                 prompt_tokens_dropped_est += dropped_prompt_tokens
1363:             full_ids = full_ids[overflow:]
1364:             loss_mask = loss_mask[overflow:]
1365:             truncated_count += 1
1366: 
1367:         if sum(loss_mask) == 0:
1368:             skipped_no_loss += 1
1369:             continue
1370: 
1371:         tokenized.append(
1372:             {
1373:                 "id": ex.get("id", ""),
1374:                 "input_ids": full_ids,
1375:                 "loss_mask": loss_mask,
1376:                 "category": ex.get("family", ex.get("category", "unknown")),
1377:                 "subcategory": (
1378:                     (ex.get("metadata") or {}).get("subcategory")
1379:                     or (ex.get("metadata") or {}).get("subtype")
1380:                     or ex.get("subcategory")
1381:                     or ex.get("subtype")
1382:                     or "unknown"
1383:                 ),
1384:                 "source": ex.get("source") or (ex.get("metadata") or {}).get("source") or "unknown",
1385:             }
1386:         )
1387: 
1388:     print(
1389:         f"{label} tokenization summary: raw={len(examples)} tokenized={len(tokenized)} "
1390:         f"truncated={truncated_count} prompt_truncated={prompt_truncated_count} "
1391:         f"prompt_tokens_dropped_est={prompt_tokens_dropped_est} "
1392:         f"skipped_missing_messages={skipped_missing_messages} "
1393:         f"skipped_no_loss={skipped_no_loss} offset_masks={offset_mask_count} "
1394:         f"fallback_masks={fallback_mask_count} "
1395:         f"fallback_prefix_mismatches={fallback_prefix_mismatch_count} "
1396:         f"answer_span_loss_weight={ANSWER_SPAN_LOSS_WEIGHT} "
1397:         f"answer_span_weighted_examples={answer_span_weighted_examples} "
1398:         f"answer_span_weighted_tokens={answer_span_weighted_tokens}"
1399:     )
1400:     if ANSWER_SPAN_LOSS_WEIGHT > 1.0 and answer_span_weighted_examples <= 0:
1401:         raise RuntimeError(
1402:             f"{label} ANSWER_SPAN_LOSS_WEIGHT={ANSWER_SPAN_LOSS_WEIGHT} "
1403:             "but no explicit Final answer/ANSWER/boxed spans were weighted."
1404:         )
1405:     if (
1406:         ANSWER_SPAN_LOSS_WEIGHT > 1.0
1407:         and ANSWER_SPAN_MIN_WEIGHTED_TOKENS > 0
1408:         and answer_span_weighted_tokens < ANSWER_SPAN_MIN_WEIGHTED_TOKENS
1409:     ):
1410:         raise RuntimeError(
1411:             f"{label} weighted answer-span tokens below floor: "
1412:             f"{answer_span_weighted_tokens} < {ANSWER_SPAN_MIN_WEIGHTED_TOKENS}"
1413:         )
1414:     if REQUIRE_OFFSET_MASK and fallback_mask_count:
1415:         raise RuntimeError(
1416:             f"{label} tokenization used {fallback_mask_count} fallback completion masks. "
1417:             "Set REQUIRE_OFFSET_MASK=0 only for a deliberate diagnostic run."
1418:         )
```

# Key Code: weak promotion gate
```python
0283: def weak_promotion_gate(summary: dict[str, Any]) -> dict[str, Any]:
0284:     total_min = env_int("KG1_WEAK_PROMOTE_TOTAL_MIN", 193)
0285:     equation_min = env_int("KG1_WEAK_PROMOTE_EQUATION_MIN", 57)
0286:     bit_min = env_int("KG1_WEAK_PROMOTE_BIT_MIN", 136)
0287:     trunc_max = env_int("KG1_WEAK_PROMOTE_TRUNC_MAX", 0)
0288:     rows = summary.get("rows", [])
0289:     if not isinstance(rows, list):
0290:         rows = []
0291:     candidates: list[dict[str, Any]] = []
0292:     for row in rows:
0293:         if not isinstance(row, dict):
0294:             continue
0295:         status_ok = str(row.get("status", "")).lower() == "ok"
0296:         correct = int(row.get("correct", 0) or 0)
0297:         equation = int(row.get("equation_transform_correct", 0) or 0)
0298:         bit = int(row.get("bit_manipulation_correct", 0) or 0)
0299:         truncated = int(row.get("truncated", 0) or 0)
0300:         passed = (
0301:             status_ok
0302:             and correct >= total_min
0303:             and equation >= equation_min
0304:             and bit >= bit_min
0305:             and truncated <= trunc_max
0306:         )
0307:         candidates.append(
0308:             {
0309:                 "name": str(row.get("name", "")),
0310:                 "status_ok": status_ok,
0311:                 "correct": correct,
0312:                 "equation_transform_correct": equation,
0313:                 "bit_manipulation_correct": bit,
0314:                 "truncated": truncated,
0315:                 "passed": passed,
```

# Key Code: V487 launcher training env
```bash
0105: export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
0106: export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
0107: export DATA_REPO='felipesp1983/kg1-nemotron-training'
0108: export DATA_FILE="$KG1_TRAIN_FILE"
0109: export VAL_FILE="$KG1_VAL_FILE"
0110: export EXPECTED_TRAIN_SHA256="$KG1_TRAIN_SHA"
0111: export EXPECTED_VAL_SHA256="$KG1_VAL_SHA"
0112: export MIN_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
0113: export MIN_VAL_EXAMPLES="$KG1_VAL_ROWS"
0114: export MIN_TOKENIZED_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
0115: export MIN_TOKENIZED_VAL_EXAMPLES="$KG1_VAL_ROWS"
0116: export OUTPUT_DIR='/tmp/kg1_v487_output'
0117: export OUTPUT_REPO="$KG1_OUTPUT_REPO"
0118: export RUN_ID="$KG1_RUN_ID"
0119: export UPLOAD_TO_HF=1
0120: export UPLOAD_CHECKPOINTS_DURING_TRAINING=1
0121: export INIT_ADAPTER_REPO="$KG1_INIT_ADAPTER_REPO"
0122: export INIT_ADAPTER_SUBFOLDER="$KG1_INIT_ADAPTER_SUBFOLDER"
0123: export INIT_ADAPTER_LOAD_MODE='peft'
0124: export PEFT_MANUAL_LOAD_METHOD='auto'
0125: export FAIL_ON_MISSING_ADAPTER_KEYS=1
0126: export LORA_R=32
0127: export LORA_ALPHA=32
0128: export LORA_DROPOUT=0.0
0129: export LORA_TARGET_MODULES='down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj'
0130: export LORA_TARGET_PARAMETERS="$KG1_LORA_TARGET_PARAMETERS"
0131: export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'
0132: export TRAINABLE_LORA_NAME_SUBSTRINGS=''
0133: export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='q_proj,k_proj,v_proj,o_proj,lm_head'
0134: export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1
0135: export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0
0136: export MAX_TRAINABLE_PARAM_RATIO=0.035
0137: export MAX_LENGTH=1024
0138: export BATCH_SIZE=4
0139: export MICRO_BATCH_SIZE=1
0140: export LEARNING_RATE=4.0e-8
0141: export FINAL_LEARNING_RATE=1.0e-8
0142: export NUM_EPOCHS=1
0143: export MAX_STEPS=12
0144: export SAVE_EVERY_STEPS=2
0145: export EVAL_EVERY_STEPS=2
0146: export EVAL_MAX_EXAMPLES=96
0147: export LOG_EVERY_STEPS=1
0148: export MICRO_LOG_EVERY=0
0149: export ANSWER_SPAN_LOSS_WEIGHT="$KG1_ANSWER_SPAN_LOSS_WEIGHT"
0150: export ANSWER_SPAN_MIN_WEIGHTED_TOKENS="$KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS"
0151: export BASELINE_EVAL_BEFORE_TRAIN=1
0152: export REQUIRE_FINAL_EVAL_LTE_BASELINE=0
0153: export ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA=0.16
0154: export MAX_FINAL_EVAL_REGRESSION=0
0155: export ABORT_TRAIN_RISE_POINTS=0
0156: export ABORT_MAX_RESERVED_GIB=78
0157: export SAMPLING_MODE='weighted_replacement'
0158: export SUBCATEGORY_WEIGHTS="$KG1_SUBCATEGORY_WEIGHTS"
0159: export SOURCE_WEIGHTS="$KG1_SOURCE_WEIGHTS"
0160: export MAX_PROMPT_TRUNCATION_RATE=0.0
0161: export REQUIRE_OFFSET_MASK=1
0162: export TOKENIZE_ONLY_DRY_RUN=0
0163: export DRY_RUN_VALIDATE_ONLY=0
0164: export USE_BITSANDBYTES=0
0165: export MODEL_DEVICE_MAP='auto'
0166: export ATTN_IMPLEMENTATION='eager'
0167: export TORCH_ALLOW_TF32=1
0168: export TORCH_FLOAT32_MATMUL_PRECISION='high'
0169: export GRADIENT_CHECKPOINTING=1
0170: $PYBIN scripts/hf_job_preflight_gate.py --phase preinstall
0171: $PYBIN scripts/hf_job_preflight_gate.py --phase artifacts
0172: $PYBIN scripts/run_v485_peft_roundtrip_gate.py \
0173:   --adapter-repo "$KG1_INIT_ADAPTER_REPO" \
0174:   --adapter-subfolder "$KG1_INIT_ADAPTER_SUBFOLDER" \
0175:   --expected-r 32 \
0176:   --expected-alpha 32 \
0177:   --expected-target-modules 'down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj' \
0178:   --expected-target-parameters "$KG1_LORA_TARGET_PARAMETERS" \
0179:   --output-json /tmp/kg1_v485_peft_roundtrip_gate_manifest.json
0180: $PYBIN - <<'PY'
```

# What We Already Tried and What Failed
- Broad SFT / replay training lowered loss but did not improve weak ACC; bit often regressed.
- Solver/verifier/postprocessor CPU projections can reach around total 196, equation 60, bit 136, but are not submit-safe until transferred into adapter behavior.
- Objective imbalance V391-like route was blocked because equation weight dominated and bit pressure was too low.
- V487 corrected PEFT target_parameter aliasing enough to train, but it used TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head while target_parameters up/down were active but not required trainable. V488 eval showed equation 57 but bit 134 and truncation 1.
- Dataset V390/V326 appears clean: no duplicate IDs/prompts, train/val overlap 0, gate flags false, tokenization offset masks OK, no truncation in training tokenization.
- Metric audit fixed a silent risk: answers_equivalent cannot be used for official ACC; expected-aware extractor cannot select earlier boxed answers.

# Current Hypothesis
The next valid route may be equation-first with bit as hard guardrail, and specifically testing whether making target_parameters MoE LoRA (up_proj/down_proj aliases) trainable can transfer equation gains without destroying bit. But this may increase trainable params and risk bit regression/OOM, so we need a carefully gated micro experiment or an alternative.

# Required Output Format
Respond in Markdown with exactly these sections:

## 1. Verdict
A concise decision: what is the most likely blocker, and what should we do next? Say whether equation_transform-first is correct.

## 2. Root Cause Ranking
Rank the top 5 likely blockers. For each: confidence %, evidence from the prompt, and how to falsify it quickly.

## 3. Implementation Bugs or Gaps To Check
List concrete bugs/gaps in code, dataset, metrics, weights, PEFT loading, target_parameters, loss masking, decoding, or gates. Only include items that could plausibly explain loss moving without ACC improving or equation +1 causing bit -2.

## 4. Exact Next Experiment
Give one minimal next experiment. Include:
- trainable modules / target_parameters setting
- LR/max_steps/save_every/eval_every
- answer_span_loss_weight recommendation
- dataset mix/weights recommendation
- first-checkpoint kill-switch thresholds
- expected cost/risk level
- expected best-case and worst-case weak outcome

## 5. Alternative If That Experiment Fails
A fallback path that does not repeat broad SFT. Include CPU-only or cheap probes if possible.

## 6. Stop Doing
What must stop immediately because it is wasting time/money or creating false signal.

## 7. Roadmap Patch
Write 5-10 exact bullets that should be inserted into our roadmap. No noise.

# Important
Be strict. If you think the current plan is wrong, say so. If you think the implementation likely has a specific bug, name the file/logic and explain the exact test to confirm. Do not recommend illegal/non-submit-safe runtime solvers. Do not give generic ML advice.
