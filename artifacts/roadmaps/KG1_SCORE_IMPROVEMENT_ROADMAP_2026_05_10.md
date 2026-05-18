# KG1 Score Improvement Roadmap

Atualizado: 2026-05-18, V653 checkpoint-2 weak-eval bloqueado; rota ativa
segue em diagnostico de output longo/backfire antes de qualquer novo full eval.

Este e o unico roadmap ativo. Historico antigo fica apenas como evidencia e
nao guia novas execucoes.

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
- forum/THK/OpenRouter reforcaram que `\boxed{}` precisa ser precoce, que
  exemplos de operadores diferentes so ajudam quando ha meta-regra comum, e
  que traces duplicadas com respostas diferentes causam "Duplicate CoT Trap";
- bit ainda e a familia mais promissora para ganho curto, mas nao pode quebrar
  protected rows nem reduzir equation.

## Rota Ativa: V653

Objetivo: transformar o sinal CPU `208/315` em aprendizado adapter-only sem
gerar output longo. V653 mantem a mistura V643/V641/V367, mas compacta bit
para traces curtos com termos de regra e boxed suffix, preservando equation com
regra curta e resposta boxed.

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

Launcher ativo:

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

Pendencia critica antes de `--launch`:

- o launcher debug atual ainda referencia o commit Git anterior
  `23d6e70f3f0c0493c6a4aae712f9c660d4711d51`;
- e obrigatorio commitar/pushar os arquivos V653 e o gate atualizado, depois
  rerodar debug e pre-paid com o novo HEAD. Caso contrario o job remoto clona
  codigo antigo e falha.

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

Diagnostico de demora V653:

- weak-eval checkpoint-2 gerou `1,504,299` completion tokens para `315` rows;
- media `4,775.55` completion tokens/row, p50 `6,193`, p90 `7,008.8`;
- `bit_manipulation` e o gargalo: media `6,689.46` tokens/row;
- geracao levou `516.1s` apos cold start; a demora e causada por output
  longo/boxed tardio, nao por baixa utilizacao isolada de GPU;
- reduzir `max_tokens` isoladamente continua fora do plano, mas output-policy
  precisa ser corrigida antes de novos full evals caros.

## Bloqueadores Permanentes

Cancelar ou reprovar se qualquer item ocorrer:

- CUDA/GPU fora do contrato;
- dataset hash diferente;
- weak overlap, weak label aware selection ou train/val overlap invalido;
- prompt duplicado contraditorio;
- `finding_counts.warning > 0` em gate promocional;
- `blocked_dataset_count > 0`;
- truncation > `0`;
- completion tokens dropped > `0`;
- fallback masks > `0`;
- resposta sem `\boxed{}` em smoke de output policy;
- extractor usa fallback numerico para promover;
- protected rows com backfire;
- raw output correto mas extractor/config erra;
- raw output errado por adapter drift;
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
- lancar V652.

## Proxima Acao Executavel

1. Usar os artefatos baixados do checkpoint-2 para diagnosticar:
   - a row truncada/backfire `59bee375`;
   - a row de ganho obrigatorio ausente `55d834d1`;
   - se o erro vem de decoding tardio, ausencia de `\boxed{}`, extractor ou
     adapter empurrando resposta errada.
2. Criar um smoke focado antes de qualquer novo full eval:
   - protected ids `8740ed31`, `59bee375`, `55d834d1`;
   - top rows longas de `bit_manipulation`;
   - amostra pequena de `equation_transform`;
   - criterios: truncation `0`, fallback `0`, protected backfire `0`,
     boxed presente, bit protegido preservado.
3. Avaliar checkpoint-10 primeiro no smoke focado, porque checkpoint-10 tem
   loss menor que checkpoint-2 (`4.2012` vs `4.2914`), mas ainda precisa
   provar ACC.
4. Rodar full weak eval do checkpoint-10 somente se o smoke focado passar.
5. Promover somente se:
   - total `>=196/315`;
   - bit `>=136/160`;
   - equation `>=60/155`;
   - truncation `0`;
   - protected backfire `0`;
   - fallback audit ok.

## Criterio De Submit

Submit ao Kaggle somente se:

- adapter-only gerar ganho weak real;
- pacote contem adapter/config correto;
- eval oficial-like confirma ganho sem fallback;
- gates de hash, prompt, max_tokens, LoRA contract, parser e protected rows
  passam;
- o resultado nao depende de solver runtime nem de weak labels.
