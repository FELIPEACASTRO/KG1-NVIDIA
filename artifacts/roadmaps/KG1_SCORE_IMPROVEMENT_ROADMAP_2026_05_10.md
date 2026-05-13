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
| V343 solver/verifier expandido sobre baseline V290 | `199/315` confirmado | `63/155` | `136/160` | ganho real CPU, `7` ganhos, `0` perdas; ainda nao LoRA puro |
| V344 dataset transfer + V340 hard-negative gate | assets validos | dataset cobre `7` regras | replay bit preservado | GPU bloqueada ate existir launcher preference/abstain real |
| V306/V302 full verifier local | `838/947` potencial | `60/155` | `146/160` | depende de verifier/postprocessor |
| V335 LoRA mixed trace replay | `190/315` | `56/155` | `134/160` | falhou; cancelado por FinOps |
| V338B LoRA minimal transfer weak eval | `190/315` | `56/155` | `134/160` | checkpoints 2 e 4 falharam; cancelado por FinOps |
| V341 clean preference checkpoint-2 | `190/315` | `56/155` | `134/160` | falhou; preferencia interna saturada, cancelado por FinOps |
| V342 ACC-first diagnostic | V341 nao ganha; V336A preserva `197/315` | V341 `56`, V336A `61` | V341 `134`, V336A `136` | GPU preference bloqueado; voltar para CPU DSL/verifier |

Conclusao: a busca/documentacao gerou conhecimento util e ganho tecnico real. O erro foi assumir que SFT curto/misto transferiria automaticamente essas regras para LoRA. V303, V326, V331 e V335 falsificaram essa hipotese.

Atualizacao V338B: a queda de `eval_loss` tambem nao foi evidencia suficiente. O treino V338B caiu de `0.9057` para melhor `0.8996`, mas o weak eval dos checkpoints 2 e 4 ficou em `190/315`, `equation_transform=56/155`, `bit_manipulation=134/160`. Isso confirma que loss menor pode melhorar imitacao/formato sem mover as regras discretas que decidem ACC por familia.

Atualizacao V343/V344: o ganho tecnico real subiu de `197/315` para `199/315`, com `equation_transform=63/155`, `bit_manipulation=136/160`, `7` ganhos e `0` perdas. Isso veio de regras CPU verificadas, nao de loss. O proximo HF so pode rodar se o script consumir explicitamente as preferencias/hard negatives e aplicar kill-switch por ACC no primeiro checkpoint.

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
| V343 numeric/symbolic expansion | `+7` equation contra V290, `199/315`, `0` perdas | fonte primaria para transfer dataset V344 |
| V344/V340 hard-negative gate | dados validos, mas GPU bloqueada sem preference/abstain launcher | liberar apenas treino que use ACC gate, nao SFT comum |
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
- V343 reexecutou a trilha sobre o baseline exato V290 checkpoint-6 e expandiu a DSL:
  - `colon_absdiff_restore_trailing_zero`: `+1` equation;
  - `minus_direct_negative_restore_sign`: `+2` equation;
  - regras anteriores preservadas: add direct, minus signed opposite sign, symbolic cryptarithm.
- Resultado V343 integrado: `199/315`, `equation_transform=63/155`, `bit_manipulation=136/160`, `7` ganhos, `0` perdas.
- Artefato: `artifacts/v343_equation_residual_solver_audit/20260513T_integrated_on_v290_v3/v336a_integrated_no_loss_solver_gate_manifest.json`.

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
- V344 reconstruiu o dataset de transferencia usando o ganho V343:
  - treino: `1760` linhas, sendo `720` bit replay e `1040` equation;
  - validacao: `420` linhas, sendo `160` bit replay e `260` equation;
  - hashes: train `cab6b8370f2208c3e3fa954527967683be06639d7c556ae7697077d1d2bf8e03`, val `a2df22315cbd837d6b15c9ff646d76fb7b8d8e3930485ac0b02677b9ed9c87cc`;
  - anti-leakage: `id_overlap=0`, `prompt_overlap=0`;
  - V286 tokenization gate passou com `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `fallback_masks=0`.
- V340 hard-negative gate sobre V344 passou nos assets, mas bloqueou GPU porque ainda falta launcher de preference/abstain real.

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

### 6. V340/V341 - Preferencia limpa antes de novo HF

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

Status 2026-05-13:

- Implementado `scripts/run_v340_hard_negative_abstain_gate.py`.
- O primeiro V340 contra V337D bruto bloqueou HF com evidencia concreta: `37/720` hard negatives de treino e `5/180` de validacao tinham `rejected` com a mesma resposta em `\boxed{}` que o `chosen`. Esses pares eram contraditorios e podiam ensinar preferencia errada.
- Implementado `scripts/build_v341_clean_preference_transfer_dataset.py`.
- V341 removeu apenas esses pares invalidos:
  - treino: `2880 -> 2843` preference rows; hard negatives validos `683`;
  - validacao: `720 -> 715` preference rows; hard negatives validos `175`.
- Upload HF concluido para `felipesp1983/kg1-nemotron-training`, path `data/v341_clean_preference_transfer/20260513T_cpu_gate`.
- V340 reexecutado com V341 limpo passou:
  - `assets_valid=True`;
  - `preference_training_allowed=True`;
  - trainer existente: `scripts/hf_job_train_v315_preference.py`;
  - proximo smoke permitido apenas em A100, curto, com kill-switch no primeiro checkpoint.
- Launcher criado em `artifacts/v341_hf_a100_clean_preference_launch/launch_v341_hf_a100_clean_preference.py`.
- Debug local do launcher passou em `a100-large`, custo `0.041667`, com `MAX_STEPS=8`, checkpoints a cada `2` steps e gate de promocao `total>192`, `equation>56`, `bit>=136`.
- Primeiro launch HF V341 `felipesp1983/6a04e8e2e48bea4538b9c040` foi cancelado por FinOps: o job passou pelos gates de GPU/dados/adapter, mas ficou preso antes de treino na compilacao source de `causal-conv1d`. Nao houve checkpoint e nao houve ganho medido.
- Segundo launch HF V341 `felipesp1983/6a04eb833308d79117b8f29e` falhou rapido e barato no load do modelo: Nemotron exige `mamba_ssm`. O job confirmou que nao existe caminho de treino desse modelo sem Mamba.
- Probe CPU barato da imagem `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` passou e confirmou dependencias prebuilt:
  - `torch=2.9.0a0+50eac811a6.nv25.09`;
  - `transformers=4.57.6`;
  - `causal_conv1d=1.5.3`;
  - `mamba_ssm=2.2.6.post3`;
  - imports `mamba_ssm.ops.triton.layernorm_gated` e `mamba_ssm.ops.selective_scan_interface` OK.
- Launcher V341 ajustado para usar a imagem NeMo/Nemotron com Mamba prebuilt e sem compilacao source em A100.
- Primeiro launch com imagem NeMo/Nemotron `felipesp1983/6a04ed3be48bea4538b9c05f` falhou antes do treino por ordem de instalacao: `HF_HUB_ENABLE_HF_TRANSFER=1` estava ativo antes de instalar `hf_transfer`. Corrigir launcher instalando `hf_transfer` no bloco inicial antes do artifact gate.
- Launch corrigido `felipesp1983/6a04eda83308d79117b8f2aa` passou todos os gates, carregou adapter V290 checkpoint-6 e gerou `checkpoint-2`; o treino foi cancelado por FinOps assim que o primeiro checkpoint ficou disponivel.
- Sinal negativo importante: antes do treino, o baseline ja fazia `96/96` na metrica interna de preferencia (`preference_accuracy=1.0`, `chosen_mean_nll=1.6799`, `rejected_mean_nll=3.1616`). Portanto, a preferencia V341 nao media uma lacuna real do adapter atual.
- Weak eval H200 `felipesp1983/6a04f2b43308d79117b8f2c7` avaliou apenas `checkpoint-2`:
  - `190/315`;
  - `equation_transform=56/155`;
  - `bit_manipulation=134/160`;
  - `truncated=1`;
  - commit HF do resultado: `e891636bc215a2e8e3af7de72a3f38b6258470ef`.
- Implementado `scripts/analyze_v342_acc_first_diagnostic.py`.
- V342 comparou linha a linha o baseline adapter-only V290 checkpoint-6, V341 checkpoint-2 e V336A:
  - baseline adapter-only: `192/315`, `equation=56/155`, `bit=136/160`, `truncated=0`;
  - V341 checkpoint-2: `190/315`, `equation=56/155`, `bit=134/160`, `truncated=1`;
  - V336A solver/verifier: `197/315`, `equation=61/155`, `bit=136/160`, `truncated=0`;
  - V341 teve `0` ganhos contra baseline e `2` perdas em `bit_manipulation`;
  - V336A tem `5` ganhos de regra verificada que V341 nao aprendeu, todos em `equation_transform`;
  - ha `1` caso de `bit_manipulation` correto na referencia V336A sem `rule_class`/trace aceito; isso nao e ganho de solver e nao pode virar override sem nova regra label-free;
  - artefato: `artifacts/v342_acc_first_diagnostic/20260513T_cpu_gate/v342_acc_first_diagnostic_manifest.json`.
- Decisao: V341 falhou. Nao promover para full/package/submit. Nao rodar mais preference/CE nessa familia de dataset sem um gate CPU que prove uma preferencia nao saturada e ganho de ACC esperado.

Decisao:

- A correcao dos hard negatives foi necessaria, mas insuficiente para ganho adapter-only.
- O criterio de promocao volta a ser exclusivamente ACC por familia no weak/full gate.
- V343 produziu nova evidencia CPU (`199/315`), mas V344/V340 bloqueou GPU ate existir trainer que consuma preferencia/abstain de verdade.
- Proximo HF GPU fica bloqueado para SFT comum. Ele so e permitido para um launcher V344 especifico com preference/abstain e kill-switch de ACC no primeiro checkpoint.

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

Implementar V344 preference/abstain launcher, sem SFT comum:

1. Usar somente o dataset V344 que ja passou anti-leakage, tokenization e V340 hard-negative gate.
2. O launcher precisa consumir `preferences_train_jsonl` e `preferences_val_jsonl`; se consumir apenas SFT, esta bloqueado.
3. O primeiro checkpoint deve ser avaliado por weak ACC, nao por `eval_loss`.
4. Continuar somente se:
   - `total > 192`;
   - `equation > 56`;
   - `bit >= 136`;
   - `truncated=0` ou nao regressivo.
5. Cancelar por FinOps se o primeiro checkpoint repetir V341/V338B: `190/315`, `equation=56`, `bit=134`, ou se a metrica interna estiver saturada sem ganho de ACC.
6. Se o launcher nao puder implementar preference/abstain real, voltar para CPU DSL/verifier e nao gastar HF.
