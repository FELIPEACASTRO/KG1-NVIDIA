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
4. Nenhum job pago roda se `target_parameters` estiver ausente, divergente ou
   carregado por modo manual sem round-trip CPU aprovado.
5. `modules_to_save` deve ficar vazio no seed e no pacote final. `lm_head` pode
   aparecer em `target_modules` como LoRA, mas nao como peso cheio salvo.
6. Todo launcher/job/notebook novo ou alterado passa por
   `scripts/kg1_static_safety_gate.py`.
7. Antes de job pago, rodar `scripts/kg1_pre_paid_job_integration_gate.py` e
   `scripts/hf_job_preflight_gate.py`.
8. FinOps: cancelar job que nao possa mais superar `total>192`,
   `equation>56`, `bit>=136`, `truncated=0`.
9. H200 pode ser usada ate 1 hora por execucao. Acima disso exige autorizacao
   humana.
10. Todo erro novo entra em `KG1_ERROR_LEDGER_2026_05_15.md` antes de novo job
   pago.
11. Toda versao nova precisa quadro comparativo contra V291/V290.

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

Objetivo: garantir que o adapter inicial V290/V291 e carregado e treinado no
mesmo espaco estrutural que gerou 192/315.

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
- roda um micro forward/backward em batch dummy e confirma gradiente nos
  parametros LoRA esperados;
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

### P3 - Smoke HF Minimo

Objetivo: verificar se o bug de continuidade era o gargalo sem gastar longo.

Executar:

- max 2 a 4 steps;
- mesmo dataset limpo V390/V475;
- bit replay obrigatorio;
- log de componentes de loss, incluindo answer-span;
- weak micro-ACC no primeiro checkpoint, nao apenas `eval_loss`;
- kill-switch no primeiro checkpoint.

Gate:

| Metrica | Requisito |
|---|---:|
| Total weak | > 192 |
| equation_transform | > 56 |
| bit_manipulation | >= 136 |
| truncated | 0 |

Se o smoke der `equation=57` e `bit=135`, nao promove. Esse caso ja ocorreu e
nao e submit-safe.

### P4 - Equation Somente Se P3 Sinalizar

Objetivo: transformar os 4 ganhos CPU de equation em comportamento do adapter.

Executar somente se P3 mostrar ganho real sem perder bit:

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

1. Validar as alteracoes de gate que bloqueiam `INIT_ADAPTER_LOAD_MODE=manual`
   com `target_parameters`.
2. Implementar/rodar o CPU round-trip PEFT gate V485.
3. Se V485 passar, abrir smoke curto HF. Se nao passar, corrigir carga PEFT
   antes de qualquer treino.
