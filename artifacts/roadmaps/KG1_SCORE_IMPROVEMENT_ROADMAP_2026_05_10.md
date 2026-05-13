# KG1 NVIDIA - Roadmap ativo de melhoria por familia

Atualizado em: 2026-05-13

Este arquivo e o roadmap ativo. Ele contem apenas o que sera usado daqui para frente para tentar melhorar `bit_manipulation` e `equation_transform`.

O historico completo de tentativas, buscas, logs e rotas rejeitadas foi movido para:

- `artifacts/roadmaps/archive/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10_FULL_HISTORY_2026_05_13.md`

## Verdade operacional

As evidencias fortes reunidas hoje mostram que ha ganho possivel, mas o ganho comprovado ainda esta em solver/verifier, nao em LoRA puro.

| Estado | Weak / Full | Equation | Bit | Decisao |
|---|---:|---:|---:|---|
| Melhor adapter-only weak atual | `192/315` | `56/155` | `136/160` | baseline operacional LoRA |
| Melhor full adapter-only conhecido/packageado | `823/947` | `56/155` | `135/160` | V291 package rank<=32; referencia associada ao melhor score conhecido `0.86` |
| V274/V275 solver/verifier | `196/315` | `60/155` | `136/160` | ganho real CPU, ainda nao LoRA puro |
| V324+V329 solver/verifier integrado no V336A | `197/315` confirmado | `61/155` | `136/160` | ganho real CPU, ainda nao LoRA puro |
| V306/V302 full verifier local | `838/947` potencial | `60/155` | `146/160` | depende de verifier/postprocessor |
| V335 LoRA mixed trace replay | `190/315` | `56/155` | `134/160` | falhou; cancelado por FinOps |
| V338B LoRA minimal transfer weak eval | `190/315` | `56/155` | `134/160` | checkpoints 2 e 4 falharam; cancelado por FinOps |

Conclusao: a busca/documentacao gerou conhecimento util e ganho tecnico real. O erro foi assumir que SFT curto/misto transferiria automaticamente essas regras para LoRA. V303, V326, V331 e V335 falsificaram essa hipotese.

Atualizacao V338B: a queda de `eval_loss` tambem nao foi evidencia suficiente. O treino V338B caiu de `0.9057` para melhor `0.8996`, mas o weak eval dos checkpoints 2 e 4 ficou em `190/315`, `equation_transform=56/155`, `bit_manipulation=134/160`. Isso confirma que loss menor pode melhorar imitacao/formato sem mover as regras discretas que decidem ACC por familia.

## Metas

Meta minima para novo candidato:

- Weak: `>=193/315`.
- `equation_transform >=60/155`.
- `bit_manipulation >=136/160`.
- Truncation `0` ou nao regressiva.

Meta para submit novo:

- Full official-like `>823/947`.
- Sem queda nas familias criticas.
- Package permitido pelas regras do desafio.
- Manifest com diff por familia.
- Se o candidato passar todos os gates, submeter sem atraso especulativo: pelas regras, empate e decidido pela submissao enviada primeiro.

## Evidencias aceitas no roadmap ativo

| Evidencia | Impacto real | Uso permitido |
|---|---|---|
| V274/V275 numeric postprocessor | `+4` equation, `0` perdas, `196/315` | regra/verifier e teacher |
| V329 symbolic cryptarithm | `+1` equation adicional projetado, `0` perdas | regra/verifier e teacher |
| Tong Hui Kang bit solver | `1364/1602` train, mas `+1/-1` no weak | teacher/taxonomia, nao override direto |
| `equation-solver-swap-v1` | classes uteis, mas overlap com gates | taxonomia/fixture sintetico verificado |
| Discussions `690307`, `688461` | bit-pair/bitsum/stride e boolean gate taxonomy | implementar CPU bit gate |
| Discussions `689877`, `698293` | operadores ausentes e estrutura latente de equation | abstain/conflict count em DSL |
| Discussions `693260`, `697491` | synthetic accuracy alta pode piorar LB | traces curtos e kill-switch |

## Roadmap ativo em ordem de maior chance de ganho

### 1. V336A - CPU integrated no-loss solver gate

Objetivo: consolidar os ganhos ja medidos e procurar novo ganho sem perdas antes de qualquer GPU.

Entrada:

- Predicoes do melhor adapter-only atual: `192/315`, equation `56`, bit `136`.
- Regras V274/V275 numeric operator.
- Regra V329 symbolic cryptarithm.
- Taxonomia `equation-solver-swap-v1`: `concat`, `swap_concat`, `add`, `sub`, `abs_sub`, `mul`, `rev_both_add_rev`, `rev_both_mul_rev`, `rev_both_abs_sub_rev`.
- Taxonomia bit: Tong stride/bitsum, constantes, identity, NOT, gates 2-input, negacoes assimetricas, majority/choice/parity.

Saida obrigatoria:

- Manifest por candidato com `id`, `family`, `rule_class`, `old_prediction`, `new_prediction`, `candidate_count`, `conflict_count`, `accepted/rejected`, `reason`.
- Reproducao de V274/V275: `196/315`.
- Reproducao de V324+V329: `197/315` confirmado no V336A.
- Novo ganho so vale se `losses=0`, `bit>=136` e `equation>=61`.

Bloqueio:

- Se V336A nao demonstrar ganho no-loss, nao rodar HF GPU.

Status 2026-05-13:

- Implementado em `scripts/run_v336_integrated_no_loss_solver_gate.py`.
- Artefato: `artifacts/v336_integrated_no_loss_solver_gate/20260513T_cpu_gate/v336a_integrated_no_loss_solver_gate_manifest.json`.
- Resultado weak integrado: `197/315`, `equation_transform=61/155`, `bit_manipulation=136/160`, `5` ganhos, `0` perdas.
- Decisao: V336A passou. Proximo passo obrigatorio e V336B. HF GPU continua bloqueado ate o gate de permissao/package.

### 2. V336B - Gate de permissao/package do solver/verifier

Objetivo: decidir se o ganho CPU pode virar submissao permitida ou se precisa ser absorvido por LoRA.

Checagens:

- Regras Kaggle atuais do desafio.
- Conteudo permitido no `submission.zip`.
- Se codigo/verifier/postprocessor pode acompanhar o adapter.
- Se o pacote precisa ser adapter-only.

Decisao:

- Se solver/verifier fosse permitido, seria a rota de maior chance porque ja existe ganho CPU medido.
- Como V336B bloqueou a rota direta, seguir para V337D/V338 e tentar absorcao LoRA minima.

Bloqueio:

- Nenhum submit com solver/verifier sem esse gate.

Status 2026-05-13:

- Implementado em `scripts/run_v336b_package_permission_gate.py`.
- Artefato: `artifacts/v336b_package_permission_gate/20260513T_cpu_gate/v336b_package_permission_gate_manifest.json`.
- Resultado: rota solver/verifier direta bloqueada. A evidencia oficial/local confirma que a submissao deve ser `submission.zip` com LoRA adapter rank `<=32`, contendo `adapter_config.json` e pesos; o pacote local rejeita `prediction_postprocessor`.
- Decisao: seguir para V337D. Nao submeter package solver/verifier. Nao rodar HF ate existir dataset minimo transferivel e gateado.
- Hardening aplicado: a decisao agora tambem exige package hard-lock ativo e rank `<=32` no pacote de referencia; se algum desses sinais cair, o gate bloqueia.

### 3. V337D - Dataset minimo de transferencia

Objetivo: criar dataset pequeno que ensine exatamente os casos em que o solver acerta e o adapter erra.

Permitido:

- Hard positives: baseline erra, solver/verifier acerta, `losses=0`.
- Hard negatives: casos parecidos em que a regra deve abstain.
- Replay minimo para proteger `bit>=136` e familias ja saturadas.
- Dois formatos no maximo:
  - `answer_only_verified`;
  - `compact_trace_verified`.

Proibido:

- Dataset publico bruto com overlap.
- Traces longos sem necessidade.
- Mistura ampla V335-like.
- SFT generico de mais epochs.

Gate:

- `id_overlap=0`.
- `prompt_sha256_overlap=0`.
- verifier correctness em todas as linhas.
- tokenization/offset-mask gate.
- manifest de source/family counts.

Status 2026-05-13:

- Implementado em `scripts/build_v337d_minimal_transfer_dataset.py`.
- Artefato: `artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/v337d_minimal_transfer_manifest.json`.
- Dataset: `1440` treino, `340` validacao; treino `720` bit + `720` equation; validacao `160` bit + `180` equation.
- Anti-leakage contra referencias: `id_overlap=0`, `prompt_overlap=0`.
- Gate real V286 com tokenizer Nemotron passou: `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `offset_masks=1440/340`, `train_token_max=349`, `val_token_max=341`.
- Upload HF concluido em `felipesp1983/kg1-nemotron-training`, caminho `data/v337d_minimal_transfer/20260513T_cpu_gate`.
- Hard negatives existem nos arquivos de preferencia, mas o V338B SFT atual ainda nao os usa. Eles so entram em uma proxima rota se o smoke provar sinal ou se for criado treino de preferencia especifico.
- Hardening aplicado: referencias anti-leakage agora sao obrigatorias; hashes dos componentes V325/V330 agora precisam bater com os manifests antes de montar o dataset.

### 4. V338 - Tiny LoRA absorption smoke

Objetivo: testar absorcao real com gasto minimo.

Configuracao:

- Seed: melhor adapter-only atual.
- Dataset: apenas V337D.
- Checkpoint cedo.
- Weak eval imediato no primeiro checkpoint.
- A100 preferencial por FinOps; H200 so se A100 nao suportar runtime.

Continuar somente se:

- `total>192`.
- `equation>56`.
- `bit>=136`.
- truncation `0`.

Cancelar por FinOps se:

- `bit<136`;
- `total<192`;
- `equation=56`;
- OOM, erro runtime, upload travado ou perda de contrato.

Status 2026-05-13:

- Launcher criado em `artifacts/v338_hf_nemo_a100_minimal_transfer_launch/launch_v338_hf_nemo_a100_minimal_transfer.py`.
- Debug local passou: hardware `a100-large`, custo `0.041667`, dataset HF com hashes corretos, adapter inicial `checkpoint-6` presente, snippets antigos V331/V335 ausentes.
- Primeiro launch V338 foi cancelado por FinOps antes do treino: o log mostrou share efetivo de bit muito baixo (`~1.23%`), incompatível com o guardrail `bit>=136`.
- V338B corrige a rota: pesos balanceados para preservar bit (`v337d_v217_bit_replay=8.0`, `bit_manipulation=3.0`, `unknown=3.0`) e reduzir equation para um smoke responsavel.
- V338B treinou no HF em `felipesp1983/6a04d4a1e48bea4538b9bf6f`.
- Logs confirmaram: modelo carregou na A100, memoria ficou estavel em torno de `62-63 GiB`, tokenization sem truncation, checkpoints 2/4/6/8/10/12/14 e `final` foram enviados ao repo `felipesp1983/kg1-nemotron-lora-v338b-nemo-a100-minimal-transfer-balanced-v290ckpt6`.
- Loss: baseline `0.9057`, melhor eval_loss `0.8996` no checkpoint 8, final `0.9014`.
- Weak eval H200 `felipesp1983/6a04df073308d79117b8f267` foi cancelado por FinOps apos dois checkpoints:
  - checkpoint-2: `190/315`, `equation=56/155`, `bit=134/160`, `truncated=1`;
  - checkpoint-4: `190/315`, `equation=56/155`, `bit=134/160`, `truncated=0`.
- Decisao: V338B falhou. Nao promover para full, package ou submit. Nao avaliar checkpoints restantes sem uma nova hipotese de selector ou subset que tenha evidencia independente.
- O launcher agora exige upload manifest existente e tokenization gate V286 local real antes de novo launch.
- O preflight HF agora valida tambem `KG1_REQUIRED_VAL_SUBCATEGORIES`, evitando validacao sem cobertura de subtipo na validacao.
- O launcher `launch_v338b_hf_weak_eval.py` foi criado em modo seguro: por padrao faz debug e manifest local; so cria job H200 com `--launch`.
- Promocao nao pode ser decidida por `eval_loss`; precisa weak eval dos checkpoints.

### 5. V339 - Full eval, package e Kaggle submit adapter-only

Executar apenas se algum candidato futuro passar weak gate.

Passos:

1. Weak eval de candidato adapter-only que tenha sinal previo concreto.
2. Promover apenas checkpoint com `total>192`, `equation>56`, `bit>=136`, truncation nao regressiva.
3. Full eval official-like.
4. Comparar contra V291 `823/947`.
5. Gerar package somente se full `>823/947`.
6. Validar estrutura do pacote.
7. Submeter ao Kaggle somente com ganho medido.
8. Se o pacote estiver valido e houver cota diaria, submeter imediatamente; nao esperar nova rodada de treino sem evidencia, porque o desempate favorece envio mais cedo.

Bloqueio:

- Nao submeter adapter que mantem `equation=56` por expectativa.
- Nao submeter candidato que reduz bit contra V291/V290.

### 6. V340 - Se LoRA continuar falhando

Como V338B falhou:

- Parar SFT curto/misto.
- Nao gastar HF com variacao de LR, epochs ou pesos.
- Voltar apenas para CPU gate ou nova fonte concreta:
  - nova regra/verifier no-loss demonstrada em CPU;
  - fonte realmente nova de traces com acesso liberado e triagem anti-leakage;
  - dataset/teacher que aumente coverage de V337D sem repetir a mistura V335-like.
- Sem uma dessas evidencias, parar GPU por FinOps.

Proxima rota permitida:

1. Construir CPU gate de transferencia com hard negatives reais: para cada miss que o solver acerta e o adapter erra, adicionar contraexemplos parecidos onde o solver deve abster.
2. Testar selector/abstain em CPU antes de qualquer LoRA: ganho so vale com `losses=0`, `equation>61` ou novo coverage comprovado, e `bit>=136`.
3. So voltar ao HF se existir manifest CPU mostrando que o dataset novo contem informacao que V337D nao continha.

## Itens removidos do roadmap ativo

Os itens abaixo ficam apenas no arquivo historico. Eles nao fazem parte do plano ativo.

| Item/rota | Motivo da remocao do plano ativo |
|---|---|
| Adapter soups | testado; nao moveu equation |
| Prompt `no suffix` / thinking variants amplas | regressao severa |
| Public external adapters `gfinin`, `etencore`, outros | muito abaixo do baseline |
| ReasoningGym direto / Alice-style SFT | drift severo no weak |
| GGUF Space | nao e LoRA adapter/packageable |
| V303 bit fullbyte distill | nao transferiu verifier para LoRA |
| V326 equation+bit replay | equation ficou `56`, bit caiu |
| V331 symbolic mix | equation ficou `56`, bit caiu |
| V335 mixed trace replay | `190/315`, bit `134`; cancelado |
| V337S package solver/verifier direto | V336B confirmou package adapter-only; rota direta nao sera usada |
| Raw `kienngx` / `konbu17` / `furkankesen` datasets | overlap ou flags incorretas; usar so taxonomia |
| Generic distillation papers/datasets | P2 metodologico, sem acao direta |
| OpenRouter/destilacao como item proprio | conclusao metodologica ja incorporada nas regras; nao e acao executavel |
| Mais epochs/LR sem novo CPU gate | gasto sem fundamento |

## Regras permanentes

- Roadmap ativo so aceita item com impacto medido ou acao concreta.
- Toda evidencia externa deve ser classificada como `ganho medido`, `taxonomia/teacher`, `bloqueado por overlap`, `rejeitado por gate` ou `P2 metodologico`.
- Enquanto job HF estiver rodando, verificar logs a cada aproximadamente `40s`.
- Se o job nao puder mais bater o gate, cancelar. A decisao FinOps correta e cancelar e nao gastar.
- `eval_loss` menor nao promove candidato.
- Nenhum notebook alterado pode ser entregue sem `scripts/notebook_release_gate.py`.
- Nenhum Kaggle submit sem weak/full gain medido.

## Proxima acao unica

Implementar o CPU gate de hard negatives/abstain para V340 antes de qualquer novo job HF.

Entrada obrigatoria: misses V336A/V337D, predicoes adapter-only baseline, regras solver/verifier aceitas e contraexemplos por classe. Saida obrigatoria: manifest com `accepted/rejected`, `conflict_count`, `losses`, `equation_delta`, `bit_delta` e hashes anti-leakage. HF GPU continua bloqueado ate esse manifest mostrar novo sinal verificavel.
