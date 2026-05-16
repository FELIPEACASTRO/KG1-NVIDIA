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
| Total weak adapter raw label-free | 191/315 | > 192/315 |
| equation_transform adapter raw label-free | 55/155 | alvo inicial 60/155 |
| bit_manipulation | 136/160 | >= 136/160 |
| truncated | 0 | 0 |
| Full official-like conhecido | 823/947 | > 823/947 |

Valores melhores que dependem de solver, verifier, postprocessor, teacher CPU,
oracle ou selecao por weak/full nao sao submit-safe ate virarem comportamento do
adapter LoRA valido.

Ultima evidencia operacional relevante:

| Versao | Resultado | Decisao |
|---|---:|---|
| V291/V290 checkpoint-6 | stored weak 192, equation 56, bit 136, trunc 0; V505 label-free recalcula 191, equation 55, bit 136 | baseline forense; promocao precisa metrica label-free |
| V477 ckpt-2 | weak 192, equation 57, bit 135, trunc 0 | nao promove; ganhou equation mas perdeu bit |
| V475 CPU solver projection | weak 196, equation 60, bit 136 | sinal CPU; ainda nao submit-safe |
| V480/V483 linha recente | loss mexe, ACC nao sai do plateau | suspeita forte de PEFT continuity bug |
| V485 seed PEFT metadata gate | `hf_gpu_allowed=true`; 12011 tensors; target params 5934/5934; `modules_to_save=[]` | seed V290/V291 estruturalmente liberado |
| V487 treino H200 | treino completo, checkpoint-10 melhor `eval_loss=1.3519`; target params 5934/5934 ativos | continuidade PEFT corrigida, mas nao prova ACC |
| V488 ckpt-10 weak eval | weak 191, equation 57, bit 134, trunc 1 | nao promove; target params nao eram o unico gargalo |
| V489 audit integridade | metrica ACC estrita correta; V488 teve +1 equation, -2 bit, +1 trunc; F2 frozen-active nao era visivel no manifesto; expected-aware antigo podia vazar boxed anterior | corrigir observabilidade/guard/extracao antes de novo GPU |
| V490 debug double check | compilacao, self-tests, static gate, dataset V390/V326, tokenization e metric path OK; HF jobs ativos 0 | proximo passo deve mudar mecanismo treinavel, nao repetir V487 |
| V491 OpenRouter consult | GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, Qwen 3.6 Max convergem em MoE trainability, freeze `lm_head`, loss weight 1.0 e kill-switch cedo | roteiro de smoke alterado; nao repetir treino attention+`lm_head` |
| V492 uploaded OpenRouter double check | 12 modelos adicionais reforcam MoE `up_proj/down_proj` frozen-active como principal suspeita; tambem alertam que o +1 equation pode ser extracao, nao aprendizado bruto | roadmap limpo para um unico experimento fail-fast, depois pivot/stop |
| V493 H200 smoke | treino completo; `target_parameters_trainability_mode=trainable`; `up_proj/down_proj` treinaveis; `lm_head` congelado; eval loss `1.9233 -> 1.9152`; checkpoint-2 uploaded | loss saudavel mas ganho nao comprovado; seguir para V494 weak eval |
| V494 loss/ACC sync audit | loss path correto como CE mascarada; ACC path correto como geracao+extracao+`verify_answer`; loss nao e proxy matematico de ACC; V245 precisa controles long-context explicitos | static gate atualizado; rodar weak eval promocional com `KG1_MAX_TOKENS=7680`, thinking on e gate bloqueante |
| V494 V493 checkpoint-2 weak eval | weak 190, equation 57, bit 133, trunc 1; simple extraction 189; strict-vs-permissive bit overcount 15 | nao promove; invalida o mix antigo V390/V326 nesse mecanismo, mas nao testa o dataset V475 CPU-gated |
| V494 dataset mismatch audit | V493/V494 usou `data/v390_equation_bit_replay_mix`, nao `data/v475_equation_bit_replay_mix`; V475 tem 1312/328 linhas, token max 331, trunc 0 e projecao CPU `equation 56 -> 60` | justificou V495; apos V496, V475 SFT tambem esta bloqueado sem novo sinal CPU |
| V495 H200 V475 smoke | treino tecnico OK; MoE `up_proj/down_proj` treinaveis; `lm_head` congelado; eval loss `1.695015 -> 1.694518` | loss saudavel, mas muito pequeno; decisao dependeu do V496 weak eval |
| V496 V495 checkpoint-2 weak eval | weak 191, equation 57, bit 134, trunc 1; diff real vs V290: +1 equation (`518deb39`), -2 bit (`8740ed31`, `59bee375`) | nao promove; V475 SFT transfer tambem bloqueada; proximo passo e CPU teacher/guardrail, nao mais H200 SFT amplo |
| V497 CPU residual transfer audit | baseline 192; V324 CPU projeta 196; V496 projeta 191; 4 ganhos CPU nao transferiram; 1 ganho simbolico V496; 2 perdas bit | confirma que o gargalo e transferencia teacher->LoRA, nao metrica/loss; proximo passo e trace numeric curto + bit guardrail |
| V498 numeric teacher trace pack | 1712/428 linhas; 3 regras numeric com hard negatives; bit replay guardrail; zero overlap id/prompt/prompt+answer com weak/full; token max 331; offset masks completos | dataset elegivel para um unico smoke H200 curto, nao para treino amplo |
| V499 H200 smoke | dataset V498 uploaded no HF commit `c7e27fd39c598dd23cb25481f567787bdff50820`; V478 objetivo passou; H200 rodou, mas final eval `2.8125 -> 2.8162` e answer-span inativo | bloqueado; nao weak-eval/package/submit |
| V501 H200 answer-span | answer-span ativo: train `1712` exemplos e `15197` tokens; MoE `gate_up/down` treinaveis; `lm_head` congelado; final eval `1.9919 -> 1.9923` | bloqueado por kill-switch; nao rodar weak/full/package/submit |
| V504 crisis root-cause audit | achou bugs reais em metrica/gates: ACC label-aware, thresholds weak antigos, weak defaults diagnosticos, full best selection, anti-leak flags ausentes, pre-paid gate sem schema SFT | corrigir gates e revalidar candidatos com extracao label-free antes de qualquer novo GPU |
| V505 label-free revalidation | 30 CSVs varridos; 22 weak315; raw-output adapter top `191/315`, `equation=55`, `bit=136`; reference-only solver/postprocessor chega a `222/315`, mas nao e adapter-only | bloquear promocao de CSV sem `raw_output`; proximo ganho precisa converter sinal solver em comportamento do adapter ou pacote valido |
| V506 reference signal gap | compara melhor adapter raw vs melhor reference-only: `31` targets (`23` bit, `8` equation), `0` reference-loss risk, `93` ambos errados | inventario alvo para transferencia; ainda nao e submit-safe |

## Achados Principais V484-V492

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

Atualizacao V491/V492: o double check com o prompt completo e o arquivo
`C:\Users\davis\Downloads\OpenRouter Chat Sat May 16 2026 (2).json` reforcou
quatro conclusoes praticas:

| Achado | Decisao no plano |
|---|---|
| V487/V488 carregavam MoE `target_parameters`, mas a allowlist treinavel era `q/k/v/o/lm_head`; isso deixa `up_proj/down_proj` em modo `frozen_active` | proximo smoke deve exigir `target_parameters_trainability_mode="trainable"` e tensores treinaveis nao nulos para `up_proj/down_proj` |
| `lm_head` treinavel e suspeito direto de flip de bit e truncation, porque altera distribuicao de `0/1` e EOS | remover `lm_head` de `TRAINABLE_LORA_MODULES` no proximo smoke; so reativar como ablation documentada |
| `ANSWER_SPAN_LOSS_WEIGHT=12.0` pode baixar loss sem melhorar ACC estrito | pin em `1.0` para smokes oficiais; qualquer valor maior vira experimento separado e nao promocional |
| O `equation=57` de V488 precisa ser revalidado em raw output, porque pode vir de extracao expected-aware e nao de melhoria real do adapter | todo ganho novo precisa ter diff `raw_output`, `simple_extracted`, `expected_aware_extracted` e familia antes de promocao |

Consenso util dos modelos: a proxima tentativa nao e "mais treino"; e um teste
de mecanismo. Se o MoE treinavel com `lm_head` congelado nao preservar
`bit>=136` no primeiro checkpoint, broad SFT segue bloqueado por FinOps.

Atualizacao V494: o teste de mecanismo foi executado. O adapter treinou com
`target_parameters_trainability_mode="trainable"`, `up_proj/down_proj`
treinaveis, `lm_head` congelado e `ANSWER_SPAN_LOSS_WEIGHT=1.0`, mas a weak
eval longa retornou `190/315`, `equation_transform=57/155`,
`bit_manipulation=133/160` e `truncated=1`. A auditoria local confirmou que a
metrica estrita esta sincronizada e que o valor de loss nao deve mais ser usado
como preditor de ACC. O unico ganho liquido vs V290 checkpoint-6 foi
`518deb39` em equation, contra tres perdas em bit (`5b9964c7`, `8740ed31`,
`59bee375`). Alem disso, a extracao expected-aware adicionou 1 acerto em
equation (`4bb8c6cd`) que nao existe na extracao simples.

Correcao critica do double check: V493/V494 nao treinou o dataset V475. O
launcher usou `data/v390_equation_bit_replay_mix/20260514T193847Z`, com 5031
linhas de treino e pressao efetiva `bit=0.926178`, `equation=0.073822`. O
V475 CPU-gated que projetava `equation 56 -> 60` e `weak 196` e outro dataset:
`data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix`,
com 1312/328 linhas, `equation=800/200`, `bit=512/128`, token max `331`,
truncation `0`, offset masks completos e zero overlap com weak/full. Isso
justificou o smoke V495 no V475, mas V496 mostrou que esse sinal CPU nao
transferiu para LoRA submit-safe.

Atualizacao V495/V496: o smoke curto V475 foi executado. O treino confirmou que
as pecas tecnicas estavam no lugar (`target_parameters` MoE treinaveis,
`lm_head` congelado, `ANSWER_SPAN_LOSS_WEIGHT=1.0`, hashes V475 corretos), mas
a weak eval bloqueou promocao:

| Metrica | V290 checkpoint-6 | V496 V495 checkpoint-2 | Delta |
|---|---:|---:|---:|
| Total weak | 192/315 | 191/315 | -1 |
| equation_transform | 56/155 | 57/155 | +1 |
| bit_manipulation | 136/160 | 134/160 | -2 |
| truncated | 0 | 1 | +1 pior |

Auditoria V496:

- metric path OK (`verify_answer` estrito);
- V496 mudou 17 linhas, mas so 3 mudaram corretude;
- ganho real: `518deb39` em equation;
- perdas reais: `8740ed31` e `59bee375` em bit;
- o extra `4bb8c6cd` de expected-aware extraction ja existia no baseline V290,
  portanto nao e aprendizado novo;
- V496 gerou `1,504,306` completion tokens em `516.9s`; o gargalo e geracao
  longa com thinking/max_tokens oficiais, nao falta de H200.

Decisao: V475 SFT transfer tambem esta bloqueada. A rota mais rapida agora e
CPU-first: descobrir teacher/verifier de equation que projete pelo menos
`equation>=60`, `bit>=136`, `trunc=0` antes de qualquer novo H200.

Atualizacao V497: a auditoria CPU residual separou o problema em tres grupos:

| Grupo | Evidencia | Acao |
|---|---|---|
| 4 ganhos V324 numeric nao transferidos | `7688e06e`, `274def88`, `d1bd7478`, `c5b058d6` sao corretos no CPU solver e continuam errados no adapter | transformar em traces numericos deterministas curtos, com hard negatives e sem weak/full labels como alvo de treino |
| 1 ganho V496 simbolico isolado | `518deb39` virou correto no adapter, mas nao veio do V324 numeric gate | usar apenas como padrao de diagnostico de formato/pontuacao; nao treinar diretamente pelo label weak |
| 2 perdas bit bloqueantes | `8740ed31` virou `01111000`; `59bee375` virou `2` | qualquer novo dataset/job deve falhar se nao preservar essas duas linhas e saida binaria exata |

Top clusters residuais continuam majoritariamente `equation_symbolic_punct`
opacos. Portanto, para ganhar hoje, o alvo mais barato e mais controlavel nao e
"resolver todos os simbolicos"; e transferir corretamente os 4 numeric gains ja
verificados pelo CPU gate sem perder bit.

Atualizacao V498/V499: o pacote V498 foi criado para atacar somente as tres
classes numericas que explicam os quatro ganhos CPU de V324. Ele nao usa linhas
weak/full como treino; essas linhas entram apenas como fingerprints proibidos.
O gate real V286 passou com `prompt_truncation_rate=0`, `completion_tokens_dropped=0`,
`offset_masks=train 1712/1712` e `validation 428/428`. O V478 local e o debug
HF do V499 confirmaram que os pesos do launcher nao repetem a sub-representacao
de bit: o objetivo efetivo fica `bit=0.390244` e `equation=0.609756`. Isso
autoriza um unico smoke H200 curto. Se o primeiro checkpoint repetir o padrao
V496 (`equation=57` com `bit<136` ou truncation), cancelar por FinOps e voltar
ao CPU teacher, sem broad SFT.

## Regras Ativas

1. Submit valido e somente adapter-only: `adapter_config.json` e
   `adapter_model.safetensors` no pacote. Sem runtime solver, verifier,
   postprocessor, logit mask, prompt hack ou threshold.
2. Weak/full sao apenas avaliacao e gate. Nao podem construir dataset, pares,
   chosen/rejected, desempate de regra ou cherry-pick.
3. ACC de promocao usa predicao label-free:
   `extract_final_answer(raw_output)` seguido de
   `src.competition_utils.verify_answer`. `answers_equivalent` e
   diagnostico-only.
4. Extracao expected-aware e diagnostico-only para datasets com label; ela nao
   pode produzir a coluna `prediction` usada em gate, package ou submit.
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
9. Antes de job pago, rodar `scripts/kg1_pre_paid_job_integration_gate.py` no
   schema correto (`preference` ou `sft`) e `scripts/hf_job_preflight_gate.py`;
   o preflight deve falhar se qualquer linha de treino vier marcada como
   gate/weak/full usada para treino ou se as flags anti-leakage estiverem
   ausentes.
10. FinOps: cancelar job que nao possa mais superar `total>192`,
   `equation>56`, `bit>=136`, `truncated=0`.
11. H200 pode ser usada ate 1 hora por execucao. Acima disso exige autorizacao
   humana.
12. Todo erro novo entra em `KG1_ERROR_LEDGER_2026_05_15.md` antes de novo job
   pago.
13. Toda versao nova precisa quadro comparativo contra V291/V290.
14. Job promocional com `LORA_TARGET_PARAMETERS` deve declarar
    `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`, incluir `up_proj/down_proj`
    na allowlist treinavel e falhar se o manifesto nao registrar modo
    `trainable`.
15. `lm_head` fica fora de `TRAINABLE_LORA_MODULES` nos smokes de melhoria.
    Reativar `lm_head` exige ablation isolada, porque V477/V488 sugerem risco
    de flip de bit e truncation.
16. `ANSWER_SPAN_LOSS_WEIGHT` fica em `1.0` nos smokes promocionais. Valores
    maiores sao diagnostico-only ate provarem ganho de ACC estrito.
17. Nenhum ganho baseado apenas em `expected_aware_extracted` promove pacote.
    O diff precisa mostrar que a saida bruta ou a extracao simples tambem nao
    introduz regressao ilegal.
18. `PRETOKENIZED_VAL_COPY_ONLY=1` e diagnostico-only. Ele invalida
    independencia de `eval_loss` porque pode copiar treino para validacao; o
    static gate bloqueia isso em jobs/notebooks promocionais.
19. Weak eval promocional com `scripts/hf_job_weak_eval_v245.py` deve declarar
    controles long-context: `KG1_DISABLE_THINKING=0`, `KG1_NO_PROMPT_SUFFIX=0`,
    `KG1_MAX_TOKENS=7680`, `KG1_MAX_MODEL_LEN=8192`,
    `KG1_MAX_NUM_SEQS=64`; os defaults agora tambem seguem esse modo.
20. Weak eval promocional falha por padrao se nao passar
    `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`. Para sweep barato,
    marcar explicitamente `KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1`.
21. V499 e V501 sao artefatos de forense, nao candidatos. Qualquer launcher que
    referencie os repos desses adapters deve falhar no static/pre-paid gate.

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

### P3 - V493 Checkpoint-2 Weak Eval V494

Objetivo: testar se a correcao de mecanismo V493 transferiu para ACC real. O
loss melhorou pouco (`1.9233 -> 1.9152`), entao a unica decisao valida vem do
weak eval.

Status: executado em H200.

- Rodou V494 H200 weak eval no `checkpoint-2` do repo
  `felipesp1983/kg1-nemotron-lora-v493-nemo-h200-moe-trainable-no-lmhead-v290ckpt6`.
- Usou controles long-context:
  - `KG1_DISABLE_THINKING=0`
  - `KG1_NO_PROMPT_SUFFIX=0`
  - `KG1_MAX_TOKENS=7680`
  - `KG1_MAX_MODEL_LEN=8192`
  - `KG1_MAX_NUM_SEQS=64`
- Gate enforcado:
  - `total > 192`
  - `equation_transform > 56`
  - `bit_manipulation >= 136`
  - `truncated = 0`

Resultado:

| Metrica | V290 checkpoint-6 baseline | V494 checkpoint-2 | Delta |
|---|---:|---:|---:|
| Total weak | 192/315 | 190/315 | -2 |
| equation_transform | 56/155 | 57/155 | +1 |
| bit_manipulation | 136/160 | 133/160 | -3 |
| truncated | 0 | 1 | +1 pior |

Auditoria de metrica:

- `scripts/audit_v449_acc_metric_integrity.py` passou com
  `decision=metric_path_ok`.
- `simple_correct=189` e `expected_aware_correct=190`; logo 1 acerto de
  equation (`4bb8c6cd`) depende de extracao expected-aware e nao pode ser
  tratado como ganho bruto do adapter.
- Strict vs permissive divergiu em 15 linhas de bit; `verify_answer` estrito
  esta correto e impede overcount numerico em strings binarias.
- Diff vs V290 checkpoint-6: ganho `518deb39` em equation; perdas
  `5b9964c7`, `8740ed31` e `59bee375` em bit.

Decisao atualizada: falhou para o mix antigo V390/V326 e tambem para o V475
CPU-gated apos V495/V496. Nao repetir V390/V326 nem V475 em H200. SFT pago fica
bloqueado ate existir novo sinal CPU independente que projete `equation>=60`,
`bit>=136`, `trunc=0` e `total>192`.

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

Atualizacao V489/V492: o diff linha a linha confirmou que a metrica estrita esta
correta e que V488 teve exatamente um ganho observado de equation (`518deb39`) e
duas regressoes reais de bit (`8740ed31`, `59bee375`), sendo uma com truncation.
O ganho de equation ainda precisa ser classificado como aprendizado bruto do
adapter ou efeito de extracao expected-aware antes de qualquer promocao. A
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

Atualizacao V490/V492: o double check em modo debug confirmou que o gap atual
nao e um erro simples de ACC, split, tokenizacao ou threshold. V488 observou
1 linha a mais de equation (`518deb39`) e perdeu 2 linhas de bit (`8740ed31`,
`59bee375`), com truncation em `59bee375`. O dataset V390/V326 tem 5031/532
linhas, IDs/prompts unicos, zero overlap train/val e flags `gate/weak/full`
como `False`. A proxima tentativa so faz sentido se for estruturalmente
diferente: testar `target_parameters` MoE como treinaveis, nao apenas
frozen-active, e auditar se qualquer ganho de equation aparece na saida bruta.

Artefato: `artifacts/v490_debug_double_check/V490_DEBUG_DOUBLE_CHECK_2026_05_16.md`.

### P3 - Smoke HF Minimo V491/V492

Objetivo: testar o mecanismo apontado por V491/V492 sem gastar longo: MoE LoRA
treinavel, `lm_head` congelado, answer-span sem peso artificial e bit como
guardrail duro.

Executar:

- max 2 steps no primeiro smoke, com leitura obrigatoria de ACC no checkpoint 2;
- mesmo dataset limpo V390/V326;
- bit replay obrigatorio;
- log de componentes de loss, incluindo answer-span;
- weak micro-ACC no primeiro checkpoint, nao apenas `eval_loss`;
- kill-switch no primeiro checkpoint.
- configuracao minima do launcher:
  - `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`
  - `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`
  - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`
  - `ANSWER_SPAN_LOSS_WEIGHT=1.0`
  - `lm_head` ausente da allowlist treinavel
  - `LEARNING_RATE=2.0e-8`
  - `FINAL_LEARNING_RATE=5.0e-9`
- pressao de dados bit-protective para smoke:
  - usar fontes concretas V390 `v304_solver_trace_bit_fullbyte_distill_exact=2.0`
    e `v304_solver_trace_bit_fullbyte_distill_random=2.0`
  - `v325_equation_no_loss_distill=1.0`
  - qualquer peso equation maior exige justificativa e aborta se bit cair.
  - como V390 ja e fisicamente dominado por bit, o V478 gate promocional usa
    `bit_effective_share>=0.35`, `equation_effective_share<=0.65` e
    `any_family_effective_share<=0.95`.

Gate:

| Metrica | Requisito |
|---|---:|
| Total weak | > 192 |
| equation_transform | > 56 |
| bit_manipulation | >= 136 |
| truncated | 0 |

Manifesto obrigatorio antes de aceitar o resultado:

- `target_parameters_trainability_mode == "trainable"`;
- `target_parameter_trainable_lora_tensors` nao vazio para
  `mlp.experts.gate_up_proj` e `mlp.experts.down_proj`;
- `trainable_by_module["up_proj"] > 0` e `trainable_by_module["down_proj"] > 0`;
- `trainable_by_module["lm_head"] == 0`;
- `ANSWER_SPAN_LOSS_WEIGHT` deve estar explicito e auditado; para treino
  promocional de ACC, o valor `1.0` agora e considerado smoke-only, nao
  suficiente para alinhar loss com resposta final;
- `simple_extracted` vs `expected_aware_extracted` auditado por familia.

Se o smoke der `equation=57` com perda de bit, nao promove. Esse caso ja
ocorreu duas vezes: V477 (`equation=57`, `bit=135`, `trunc=0`) e V488
(`equation=57`, `bit=134`, `trunc=1`). O ganho isolado de uma linha em
equation nao vale se derruba o guardrail de bit ou truncation.

### P4 - Equation Somente Se P3 Sinalizar

Objetivo: transformar os 4 ganhos CPU de equation em comportamento do adapter.

Executar somente em CPU enquanto nao houver novo sinal. P3 repetiu
`equation=57` com `bit<136` e truncation, entao nao continuar GPU nesta rota.
O objetivo agora e descobrir um teacher/verifier que produza ganhos de equation
sem usar weak/full como label de treino e sem perder bit.

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
- hard negatives diretamente extraidos de IDs weak/full como labels de treino.
  `518deb39`, `8740ed31` e `59bee375` so podem ser usados como diagnostico de
  diff/gate, nao como exemplos de treino promocional.

Como P3 falhou no V390/V326 e depois V495/V496 falhou no V475:

- nao continuar o mesmo job;
- nao abrir nova H200 para repetir V493/V494/V495/V496, V390/V326 ou V475
  sem novo sinal CPU;
- rodar CPU-only para auditar os residuos de equation e bit:
  - mapear os 99 misses de equation por padrao simbolico;
  - separar ganho bruto vs ganho por extracao;
  - identificar qualquer regra que resolva pelo menos +4 equation com zero
    perda de bit quando convertida para trace/teacher;
  - produzir dataset curto somente se o solver CPU independente passar
    leakage/contract gate.
- broad SFT fica encerrado ate existir novo sinal CPU submit-safe.

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
| `lm_head` treinavel no smoke principal | risco de bit flip/truncation; somente ablation |
| Treino promocional com `ANSWER_SPAN_LOSS_WEIGHT=1.0` | V499 mostrou `0` exemplos com answer-span weighting e loss final sem melhora |
| Treinar diretamente nos IDs weak/full que regrediram ou ganharam | viola regra de usar weak/full apenas como gate |
| Promover ganho visto apenas por expected-aware extractor | pode ser melhoria de parser, nao de adapter |

## Proxima Acao Imediata

1. Manter V290 checkpoint-6 como unico adapter submit-safe ate haver weak/full
   gate melhor com metrica label-free.
2. V499 e V501 estao bloqueados:
   - V499: final eval loss `2.8125 -> 2.8162`, answer-span inativo;
   - V501: answer-span ativo, MoE trainavel e `lm_head` congelado, mas final
     eval loss `1.9919 -> 1.9923`.
3. Nao rodar weak/full/package/submit em V499/V501. A decisao FinOps correta e
   nao gastar com ACC caro quando o kill-switch local ja bloqueou o candidato.
4. Revalidar inventario de candidatos anteriores usando somente
   `submit_safe_label_free_prediction`. Qualquer ganho que exista apenas em
   `label_aware_debug_prediction` e descartado.
5. Atualizar/usar os gates corrigidos antes de qualquer novo job:
   - `kg1_static_safety_gate.py`;
   - `kg1_pre_paid_job_integration_gate.py --dataset-schema sft|preference`;
   - `hf_job_preflight_gate.py` com flags anti-leakage obrigatorias;
   - weak eval promocional official-like e bloqueante por default.
6. Se a revalidacao label-free nao produzir candidato `>192/315`, voltar para
   CPU teacher/verifier discovery. Nao abrir H200 ate existir novo sinal CPU
   que projete `equation>=60`, preserve `bit>=136` e tenha `trunc=0`.
7. Se surgir novo candidato, weak eval promocional deve usar thinking ligado,
   `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64` e falhar se nao
   passar `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`.

## Atualizacao V500 - Auditoria De Parametros V499

Artefatos:

- `artifacts/v500_v499_parameterization_audit/KG1_V500_V499_PARAMETERIZATION_AUDIT.md`;
- `artifacts/v500_v499_parameterization_audit/v500_v499_parameterization_audit_manifest.json`.

Conclusao: as pecas de execucao do V499 estao sincronizadas, mas os valores nao
estao corretos para esperar ganho de ACC. O run comprovou que o caminho tecnico
funciona, nao que o adapter aprendeu os quatro ganhos de equation. O novo gate
obrigatorio e: treino para ACC precisa ativar answer-span weighting e provar que
esse weighting foi aplicado no tokenizador antes de gastar H200.

## Atualizacao V502 - Double Check De Parametrizacao

Artefatos:

- `artifacts/v502_solution_parameterization_double_check/KG1_V502_SOLUTION_PARAMETERIZATION_DOUBLE_CHECK.md`;
- `artifacts/v502_solution_parameterization_double_check/v502_solution_parameterization_double_check_manifest.json`;
- `artifacts/v501_hf_nemo_h200_v498_answer_span_weighted_launch/launch_v501_hf_nemo_h200_v498_answer_span_weighted.py`.

Resultado do double check:

- dataset V498: `1712` train, `428` val, zero duplicados, zero falha de
  extracao/verificacao do `Final answer`, zero caracteres suspeitos nos
  answers;
- tokenizacao: `token_max=331`, `MAX_LENGTH=1024` seguro, zero truncation,
  offset masks completos;
- ACC metric: raw CoT nao e aceito diretamente; avaliacao precisa extrair
  resposta final antes de `verify_answer`, como esperado;
- V499: tecnicamente correto, mas bloqueado por objetivo local
  (`2.8125 -> 2.8162`, delta `+0.0037`, answer-span inativo);
- V501: corrigido para `ANSWER_SPAN_LOSS_WEIGHT=4.0`,
  `ANSWER_SPAN_MIN_WEIGHTED_TOKENS=1000`, `MAX_STEPS=4`, V290 checkpoint-6,
  MoE `gate_up/down` treinaveis, `lm_head` congelado, bit replay efetivo
  `39.02%`;
- `kg1_static_safety_gate.py` atualizado para permitir peso de answer-span
  apenas em rota explicitamente `answer_span_weighted` e bloquear qualquer caso
  sem min-token gate.

Decisao:

- V499 nao deve receber weak eval pago.
- V501 foi lancado como smoke H200 curto depois do debug de parametrizacao e
  foi bloqueado pelo kill-switch: final eval `1.9919 -> 1.9923`.
- Nao rodar weak/full/package/submit em V499/V501; ambos sao artefatos
  forenses, nao candidatos.

## Atualizacao V503/V504 - Causa Raiz Do Plateau

Artefatos:

- `artifacts/v503_live_hf_job_parameterization/KG1_V503_LIVE_HF_JOB_PARAMETERIZATION.md`;
- `artifacts/v504_crisis_root_cause_audit/KG1_V504_CRISIS_ROOT_CAUSE_AUDIT.md`;
- `artifacts/v504_crisis_root_cause_audit/v504_crisis_root_cause_audit_manifest.json`.

Conclusao: as pecas de treino V501 estavam tecnicamente presentes, mas a rota
de promocao tinha bugs silenciosos suficientes para explicar falsas esperancas
de ganho:

| Problema | Correcao |
|---|---|
| `prediction` podia ser extraida com `extract_final_answer_for_expected`, usando label esperado | `prediction` agora e label-free; expected-aware fica somente em `label_aware_debug_prediction` |
| expected-aware aceitava prefixo inseguro como `\boxed{30 wrong}` para expected `30` | helper agora aceita apenas delimitador real `}` |
| weak eval V245 tinha defaults diagnosticos (`max_tokens=96`, thinking off) e threshold `equation>=57` | defaults agora sao official-like e promocao exige `total>=196`, `equation>=60`, `bit>=136`, `trunc=0` |
| weak eval podia terminar com exit 0 sem candidato promovido se env nao estivesse setado | promocao e bloqueante por padrao; sweeps baratos exigem `KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1` |
| full eval escolhia maior `correct` antes de aplicar truncation gate | agora escolhe primeiro entre candidatos que passam correct/truncation |
| pre-paid gate era preference-only e nao cobria V498/V501 SFT | adicionado `--dataset-schema sft` e save/eval steps configuraveis |
| flags anti-leakage ausentes eram apenas contadas | agora bloqueiam preflight promocional |
| launchers pagos podiam ser criados sem declarar que os guards F2/backfire/FinOps estavam ativos | `kg1_static_safety_gate.py` e `kg1_pre_paid_job_integration_gate.py` agora exigem `KG1_CRISIS_MODE_BACKFIRE_GUARD=1` |
| CSV de solver/postprocessor podia parecer candidato promocional | V505 separa `raw_output` adapter-only de reference-only; CSV sem `raw_output` nao promove |

O que ficou provado como nao raiz: hashes V498, row counts, offset masks,
truncation do dataset, MoE `gate_up/down` treinaveis, `lm_head` congelado,
memoria H200 e custo.

Decisao operacional: V499 e V501 entram na lista de adapters bloqueados. A
revalidacao V505 ja foi executada e nao encontrou candidato adapter-only acima
do gate. O melhor CSV `raw_output` de adapter ficou em `191/315`, enquanto
solvers/postprocessors reference-only chegam a `222/315` mas nao sao
submittable. Sem novo candidato adapter-only `>192/315`, nao ha novo GPU; volta
para CPU teacher/verifier discovery com obrigacao de converter sinal em
comportamento LoRA ou pacote permitido.

## Atualizacao V506 - Alvo Real De Transferencia

Artefatos:

- `artifacts/v506_reference_signal_gap/v506_reference_signal_gap_manifest.json`;
- `artifacts/v506_reference_signal_gap/v506_reference_signal_gap_rows.csv`;
- `artifacts/v506_reference_signal_gap/v506_reference_gain_targets.csv`.

V506 comparou o melhor adapter `raw_output` label-free contra o melhor sinal
reference-only. Resultado:

| Status | Total | bit | equation |
|---|---:|---:|---:|
| both_correct | 191 | 136 | 55 |
| reference_gain_target | 31 | 23 | 8 |
| both_wrong | 93 | 1 | 92 |
| reference_loss_risk | 0 | 0 | 0 |

Quebra dos `31` targets:

- `13` bit por `bit_exact_global_ternary_unique_prediction`;
- `4` bit por `bit_fullbyte_ternary_op_CHO`;
- `4` bit por `bit_fullbyte_ternary_op_MAJ3`;
- `1` bit por `bit_exact_global_binary_XOR`;
- `1` bit sem regra preenchida;
- `8` equation sem `source_rule` preenchido no CSV reference-only, exigindo
  inspeção/rotulagem antes de qualquer dataset sintetico.

Interpretacao: existe sinal tecnico forte para subir `191 -> 222` no weak, mas
ele ainda esta fora do comportamento adapter-only. A tarefa agora nao e "mais
treino" generico; e converter esses `31` targets em saida do adapter sem
quebrar os `191` atuais. Qualquer job pago precisa declarar
`KG1_CRISIS_MODE_BACKFIRE_GUARD=1`, passar pre-paid gate, e so pode ser
promovido apos weak eval `raw_output` label-free.
