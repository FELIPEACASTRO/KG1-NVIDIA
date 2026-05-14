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
| V344 dataset transfer + V340 hard-negative gate | assets validos | dataset cobre `7` regras | replay bit preservado | launcher preference/abstain criado; primeiro launch cancelado por FinOps antes de checkpoint |
| V344 preference/abstain checkpoint-2 | `192/315` | `56/155` | `136/160` | sem ganho; mudou `7` predicoes mas `0` ganhos e `0` perdas contra baseline |
| V345 failure audit sobre V344 | diagnostico concluido | `7` ganhos V343 nao transferidos | bit preservado | dataset cobria classes de regra, mas treino nao alterou os IDs-alvo; bloquear repeticao do objetivo |
| V346 answer exact-match checkpoint-2 | `191/315` | `56/155` | `135/160` | falhou; `0` ganhos, `1` perda em bit; bloquear mais H200 nessa variante |
| V348 residual CPU audit | mapa residual pronto | `92` misses restantes | `24` misses restantes | HF bloqueado; proximo passo e regra label-free no-loss |
| V349 Kaggle Discussion 140 double check | `140/140` topicos cobertos | reforca ambiguidade/DSL | reforca bit-pair/full-byte/3-input | nao gera submit; atualiza CPU gate |
| V306/V302 full verifier local | `838/947` potencial | `60/155` | `146/160` | depende de verifier/postprocessor |
| V335 LoRA mixed trace replay | `190/315` | `56/155` | `134/160` | falhou; cancelado por FinOps |
| V338B LoRA minimal transfer weak eval | `190/315` | `56/155` | `134/160` | checkpoints 2 e 4 falharam; cancelado por FinOps |
| V341 clean preference checkpoint-2 | `190/315` | `56/155` | `134/160` | falhou; preferencia interna saturada, cancelado por FinOps |
| V342 ACC-first diagnostic | V341 nao ganha; V336A preserva `197/315` | V341 `56`, V336A `61` | V341 `134`, V336A `136` | GPU preference bloqueado; voltar para CPU DSL/verifier |

Conclusao: a busca/documentacao gerou conhecimento util e ganho tecnico real. O erro foi assumir que SFT curto/misto transferiria automaticamente essas regras para LoRA. V303, V326, V331 e V335 falsificaram essa hipotese.

Atualizacao V338B: a queda de `eval_loss` tambem nao foi evidencia suficiente. O treino V338B caiu de `0.9057` para melhor `0.8996`, mas o weak eval dos checkpoints 2 e 4 ficou em `190/315`, `equation_transform=56/155`, `bit_manipulation=134/160`. Isso confirma que loss menor pode melhorar imitacao/formato sem mover as regras discretas que decidem ACC por familia.

Atualizacao V343/V344: o ganho tecnico real subiu de `197/315` para `199/315`, com `equation_transform=63/155`, `bit_manipulation=136/160`, `7` ganhos e `0` perdas. Isso veio de regras CPU verificadas, nao de loss. O proximo HF so pode rodar se o script consumir explicitamente as preferencias/hard negatives e aplicar kill-switch por ACC no primeiro checkpoint.

Atualizacao V344 HF FinOps: o primeiro launch A100 `felipesp1983/6a04fe603308d79117b8f2fb` foi cancelado antes de checkpoint. Nao houve erro de dados, adapter ou ambiente; o problema foi custo/tempo excessivo no `baseline_preference_eval_start max_examples=128`, sem progresso interno. Correcao obrigatoria: reduzir `EVAL_MAX_EXAMPLES` do smoke, imprimir `preference_eval_progress` durante a avaliacao e manter a regra de promocao somente por weak ACC.

Atualizacao V344 HF resultado: o relaunch A100 `felipesp1983/6a0501f03308d79117b8f310` treinou `MAX_STEPS=2` com `lr=1e-09`, gerou `checkpoint-2` e passou para weak eval H200 `felipesp1983/6a0504c43308d79117b8f31f`. Resultado weak: `192/315`, `equation_transform=56/155`, `bit_manipulation=136/160`, `truncated=0`. Comparacao linha a linha contra o baseline V290 checkpoint-6: `7` predicoes mudaram (`6` equation, `1` bit), com `0` ganhos e `0` perdas. Decisao: nao promover, nao rodar full eval, nao submeter e nao gastar mais H200 nesse checkpoint.

Atualizacao V345: a auditoria ACC-first confirmou que os `7` ganhos CPU V343 tinham cobertura por classe de regra no dataset V344 (`1000` linhas para cada classe numerica relevante e `1500` para symbolic cryptarithm), mas `0` overlap direto por `id` ou `prompt` com o weak, por desenho anti-leakage. O V344 checkpoint-2 nao mudou a predicao de nenhum dos `7` IDs que o V343 resolveria; as `7` mudancas ocorreram em outros IDs e continuaram incorretas. Diagnostico: repetir o mesmo objetivo de preferencia nao deve ser feito.

Atualizacao trainer preference: foi encontrado um bug objetivo no schedule de LR em `scripts/hf_job_train_v315_preference.py`. O loop incrementava `global_step` antes de calcular `base.get_lr`; com `MAX_STEPS=2`, o primeiro update ja usava `FINAL_LEARNING_RATE=1e-09`. Isso explica por que o log do V344 mostrou `lr=1.000e-09` em todos os steps. Correcao aplicada: calcular LR com o step anterior ao update e so depois marcar o step como concluido.

Atualizacao V346 HF resultado: o treino answer exact-match A100 `felipesp1983/6a050b53e48bea4538b9c1e3` corrigiu o schedule e realmente usou LR alto no inicio (`8.00e-08` no step 1 ate `2.00e-08` no step 6). Mesmo assim, o weak eval H200 checkpoint-2 `felipesp1983/6a050efe3308d79117b8f350` ficou em `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`, `truncated=0`. Auditoria V347 contra o baseline V290 e o solver V343: `7` ganhos V343 continuaram nao transferidos, `6` predicoes mudaram, `0` ganhos e `1` perda em bit (`8740ed31`). Decisao: V346 nao promove, nao full eval, nao package, nao submit e nao avaliar checkpoints 4/6 sem novo sinal independente.

Atualizacao V348 CPU residual: implementado `scripts/analyze_v348_residual_no_loss_expansion.py`. O mapa residual apos V343 ficou em `92` equation misses e `24` bit misses. Em equation, `82` sao `equation_symbolic_punct` e `10` sao `equation_numeric_operator`; os shapes sao `PPPPP=81`, `DDPDD=10`, `PPP=1`. Em bit, `14` misses estao a distancia de Hamming `1`, `8` a distancia `2`, `1` a distancia `3` e `1` a distancia `5`. Isso mostra oportunidade, mas ainda nao existe regra no-loss aceita; portanto HF continua bloqueado.

Atualizacao V349 Kaggle discussions: os arquivos `NVIDIA Nemotron Model Reasoning Challenge - Discussion Topic IDs.md` e `NVIDIA Nemotron Model Reasoning Challenge - Discussion Topics URLs.md` foram reconciliados contra cache local V328/V332 e cobrem `140/140` topicos. O novo coletor `scripts/analyze_v349_kaggle_discussions.py` usa cache primeiro porque o endpoint live do Kaggle rate-limita. Achados acionaveis: `690756` reforca testar duas interpretacoes de bit (`full-byte unary transform` e `per-output-bit bit-pair/bitsum/stride`) e so adicionar fallback 3-input sob no-loss; `684192` e `694556` reforcam que equation/symbolic precisa contar ambiguidade e fazer abstain quando operador/candidato nao e unico; `685886` reforca traces bit com delta/plausibility/verification. Nenhum topico gerou novo adapter-only submit-ready.

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
| V344/V340 hard-negative gate | dados validos; launcher preference/abstain existe; primeiro launch cancelado por eval interna cara | liberar apenas smoke barato com progresso e ACC gate, nao SFT comum |
| Tong Hui Kang bit solver | `1364/1602` train, mas `+1/-1` no weak | teacher/taxonomia, nao override direto |
| `equation-solver-swap-v1` | classes uteis, mas overlap com gates | taxonomia/fixture sintetico verificado |
| Discussions `690307`, `688461` | bit-pair/bitsum/stride e boolean gate taxonomy | implementar CPU bit gate |
| Discussions `689877`, `698293` | operadores ausentes e estrutura latente de equation | abstain/conflict count em DSL |
| Discussions `693260`, `697491` | synthetic accuracy alta pode piorar LB | traces curtos e kill-switch |
| V349 discussions `690756`, `684192`, `694556`, `685886` | bit full-byte vs bit-pair, limite 3-input, operador ausente, multiplos candidatos simbolicos, trace delta/plausibility | atualizar CPU gate; nao libera HF sozinho |

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
- V340 hard-negative gate sobre V344 passou nos assets. A liberacao GPU exige launcher de preference/abstain real e smoke curto; SFT comum segue bloqueado.
- Upload HF V344 concluido para `felipesp1983/kg1-nemotron-training`, path `data/v344_v343_minimal_transfer/20260513T_minimal_transfer_v343`, commit `6df9a5c7f997f4b0da61fa9a1eb7871449a77d7e`.
- Launcher V344 criado em `artifacts/v344_hf_a100_preference_abstain_launch/launch_v344_hf_a100_preference_abstain.py`.
- V340 reexecutado com o launcher V344 passou e liberou apenas smoke curto:
  - `assets_valid=True`;
  - `preference_training_allowed=True`;
  - `hf_gpu_allowed=True`;
  - limite: `MAX_STEPS=2`, checkpoint unico no step 2, kill-switch por weak ACC.
- Debug local do launcher passou em `a100-large`, custo `0.041667/min`, imagem `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`, dados/hashes V344 corretos.
- Primeiro launch HF A100 V344 `felipesp1983/6a04fe603308d79117b8f2fb` foi cancelado por FinOps antes de checkpoint: ficou sem progresso apos `baseline_preference_eval_start max_examples=128`. Correcao aplicada: `EVAL_MAX_EXAMPLES=8` para smoke e logs `preference_eval_progress` durante a avaliacao.
- Relaunch HF A100 V344 `felipesp1983/6a0501f03308d79117b8f310` passou pelos gates, treinou `2` steps e fez upload de `checkpoint-2` para `felipesp1983/kg1-nemotron-lora-v344-pref-abstain-a100-v290ckpt6`.
- Weak eval H200 V344 `felipesp1983/6a0504c43308d79117b8f31f` concluiu:
  - `192/315`;
  - `equation_transform=56/155`;
  - `bit_manipulation=136/160`;
  - `truncated=0`;
  - output HF: `evals/v344-h200-v221contract-pref-abstain-checkpoint2-20260513T230852Z`;
  - commit HF do resultado: `e2c51ca14e4d1ffd5ab3b56a214c8eb1a96973fb`.
- Diagnostico contra V290 baseline: `7` predicoes mudaram, mas todas continuaram incorretas; portanto a preferencia/abstain V344 nao transferiu nenhum dos `7` ganhos CPU V343 para LoRA.
- V345 audit implementado em `scripts/analyze_v345_v344_failure_audit.py`.
- Artefato: `artifacts/v345_v344_failure_audit/20260513T_acc_first/v345_v344_failure_audit_manifest.json`.
- Resultado V345:
  - baseline V290: `192/315`, equation `56`, bit `136`;
  - V343 CPU solver/verifier: `199/315`, equation `63`, bit `136`;
  - V344 checkpoint-2: `192/315`, equation `56`, bit `136`;
  - `v343_gain_not_transferred=7`;
  - `v344_changed_no_accuracy_delta=7`;
  - nos `7` ganhos V343: `direct_id_count=0`, `prompt_overlap_count=0`, `rule_count>0`, `v344_changed_prediction=false`.
- Decisao V345: bloquear repeticao de V344 preference objective. Proxima tentativa GPU so pode ser um V346 com sinal de answer exact-match/hard-positive ou LR/schedule mais agressivo e kill-switch no primeiro checkpoint.
- Bug de LR no trainer de preferencia corrigido em `scripts/hf_job_train_v315_preference.py`: o primeiro update agora usa `LEARNING_RATE`, nao `FINAL_LEARNING_RATE`. Isso permite V346 ser mais ousado sem repetir o erro do V344.
- V346 answer-exact-match dataset implementado em `scripts/build_v346_answer_exact_match_dataset.py`.
- Artefatos V346 CPU:
  - dataset: `artifacts/v346_answer_exact_match_dataset/20260513T_cpu_gate/v346_answer_exact_match_manifest.json`;
  - treino `1760` linhas, validation `420` linhas;
  - familias: treino `720` bit + `1040` equation; validation `160` bit + `260` equation;
  - hashes: train `cb2e244c04b88e4aa81e726a8a89740aa6ab554c07eb8778f6f2d2aa57cb1d34`, val `d9f8f7b7c2f3106f7e2f6bf88a531f0fe895bd7a8b16ea84501c3d2c21897087`;
  - V286 tokenization gate passou com `boxed_exact`, `prompt_truncation_rate=0.0`, `completion_tokens_dropped=0`, `fallback_masks=0`.
- Upload HF V346 concluido para `felipesp1983/kg1-nemotron-training`, path `data/v346_answer_exact_match/20260513T_cpu_gate`, commit `9ecf0f758bfb4fd8abf3d2d2f4df235947d30e98`.
- Launcher V346 criado em `artifacts/v346_hf_a100_answer_exact_match_launch/launch_v346_hf_a100_answer_exact_match.py`.
- Debug local V346 passou em `a100-large`, imagem `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`, custo `0.041667/min`, adapter inicial V290 checkpoint-6 presente, dados HF com hash correto. Receita: `MAX_STEPS=6`, `LEARNING_RATE=8e-8`, `FINAL_LEARNING_RATE=2e-8`, answer-span loss `24.0`, checkpoint a cada `2` steps.
- HF train V346 `felipesp1983/6a050b53e48bea4538b9c1e3` concluiu e confirmou que o bug de LR foi corrigido:
  - step 1 `lr=8.00e-08`;
  - step 2 `lr=6.80e-08`;
  - step 3 `lr=5.60e-08`;
  - step 4 `lr=4.40e-08`;
  - step 5 `lr=3.20e-08`;
  - step 6 `lr=2.00e-08`.
- Weak eval H200 V346 checkpoint-2 `felipesp1983/6a050efe3308d79117b8f350` concluiu:
  - `191/315`;
  - `equation_transform=56/155`;
  - `bit_manipulation=135/160`;
  - `truncated=0`;
  - output HF: `evals/v346-h200-v221contract-answer-exact-match-checkpoint2-20260513T235230Z`;
  - commit HF do resultado: `32b4240f3b7ace4a50b5bf5f960d5704257c7cee`.
- V347 audit sobre V346, usando o script ACC-first de V345 contra baseline V290 e solver V343:
  - artefato: `artifacts/v347_v346_failure_audit/20260513T_acc_first/v345_v344_failure_audit_manifest.json`;
  - `changed_prediction_count=6`;
  - `v343_gain_not_transferred=7`;
  - `v344_changed_no_accuracy_delta=5` (nome legado do script; neste run significa V346);
  - `v344_loss_vs_baseline=1` em `bit_manipulation`;
  - `preference_rows=0`, pois V346 era SFT answer-only, nao preference.
- Decisao: V346 falhou mesmo com LR efetivo. Encerrar a rota "mais LR + answer-only SFT sintetico" ate existir novo sinal CPU. Nao gastar H200 avaliando checkpoints 4/6 por expectativa.
- V348 residual no-loss audit implementado em `scripts/analyze_v348_residual_no_loss_expansion.py`.
- Artefato: `artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_residual_no_loss_expansion_manifest.json`.
- Resultado V348:
  - V343 summary preservado: `199/315`, `equation=63/155`, `bit=136/160`;
  - equation residual: `92` rows (`82` symbolic/punct, `10` numeric/operator);
  - equation shape residual: `PPPPP=81`, `DDPDD=10`, `PPP=1`;
  - bit residual: `24` rows; Hamming `1=14`, `2=8`, `3=1`, `5=1`;
  - `22/24` bit misses sao low-Hamming, mas sem seletor no-loss ainda.
- Decisao V348: mapa residual pronto, mas nenhum novo ganho aceito. HF GPU segue bloqueado ate V349 adicionar regra label-free com `losses=0`.

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
- V344 consumiu preference/abstain de verdade, mas com `lr=1e-09` e `2` steps nao moveu ACC. Proximo HF GPU fica bloqueado ate um diagnostico ACC-first provar uma mudanca concreta no desenho do treino; repetir o mesmo launcher, mais epochs ou `eval_loss` menor nao e permitido.
- V345 provou que o problema nao foi falta bruta de classes sinteticas, e sim transferencia fraca para answer exact-match. A proxima rota deve otimizar diretamente resposta final curta e medir ACC imediatamente.

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

Implementar V349 CPU candidate rules sobre os residuos V348 antes de qualquer novo HF:

1. Usar somente os CSVs residuais V348 como fila de trabalho:
   - `artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_equation_residuals.csv`;
   - `artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_bit_residuals.csv`.
2. Adicionar somente regras label-free com `candidate_count` baixo, `conflict_count=0` e explicacao deterministica:
   - equation symbolic/punctuation residual;
   - equation numeric residual que nao esteja coberto por V343;
   - bit Tong bit-pair/bitsum/stride apenas se gerar nova predicao no-loss;
   - bit full-byte unary transform e bounded 3-input fallback apenas para os `24` misses residuais e somente sob no-loss;
   - equation abstain obrigatorio quando o operador alvo nao aparece nos exemplos, quando houver multiplos programas simbolicos compativeis ou quando `candidate_count`/`conflict_count` nao for conclusivo.
3. Gate CPU obrigatorio:
   - novo ganho aceito apenas com `losses=0`;
   - `equation>63` ou `bit>136` no weak diagnostic;
   - sem usar weak/full rows como treino direto.
4. HF GPU continua bloqueado. So liberar novo job se V349 produzir um novo teacher com ganho no-loss e um dataset cujo sinal seja diferente de V344/V346. Repetir mais epochs, checkpoints 4/6 do V346 ou outra variacao de LR sem novo CPU gate esta removido do plano ativo.
