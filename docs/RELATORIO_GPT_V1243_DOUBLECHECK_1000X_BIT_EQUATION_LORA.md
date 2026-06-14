# Relatorio GPT V1243 Double Check 1000x: Bit/Equation -> LoRA

Gerado em: 2026-06-14

Objetivo operacional: preparar a solucao KG1/NVIDIA para buscar `>=0.89` com o menor risco possivel de falso positivo, GPU desperdicada, notebook errado ou loss bom sem direcao real de score.

Notebook recomendado para execucao:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/master/notebooks/KG1_V1243_COLAB_FINAL_SPRINT_MICRO_CONSOLIDATION.ipynb`

URL operacional mais segura depois da publicacao deste relatorio: usar a mesma URL pinada pelo commit do release final, para evitar drift de `master`.

## Veredito Curto

A solucao correta para transformar solver de `bit_manipulation` e `equation_transform` em LoRA nao e executar o solver no submit. A solucao coerente e projetar o conhecimento simbolico verificado em exemplos supervisionados curtos, score-facing, e treinar um delta LoRA pequeno para aproximar essa funcao dentro do subespaco low-rank permitido pela competicao.

Esse metodo existe no projeto como `GRAFT`: Gate-verified Replay Answer-Focused Transfer.

O que esta validado:

- Datasets V1243 foram reconstruidos e auditados.
- `micro_consolidation` contem bit + equation + replay protegido.
- Train/val nao tem overlap.
- Train/full947 nao tem overlap.
- Targets sao curtos e terminam em um unico `\boxed{answer}`.
- Tokenize dry-run passou para bit, equation e micro.
- Offset masks existem; fallback mask ficou zero.
- Prompt truncation ficou zero.
- Loss nao e usado como prova de score.
- Score trajectory exige bit e equation individualmente melhores.
- Pack do Colab tem SHA fixo e notebooks recusam pack divergente.
- Gates centrais passaram apos as correcoes desta rodada.

O que ainda nao e prova:

- Nao existe garantia matematica de `>=0.89` sem adapter treinado, geracao real `raw_output` e gate `full947_089`.
- Tokenize dry-run nao prova model-load real nem trainable ratio em GPU.
- Score proxy e teacher-forced; ele ajuda a nao gastar GPU em direcao errada, mas nao substitui geracao real.

## Contexto Externo Validado

A pagina publica do Kaggle para `NVIDIA Nemotron Model Reasoning Challenge` informa final deadline em 15/06/2026 e descreve o desafio como melhoria de raciocinio sobre benchmark de puzzles. Fonte publica: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`.

Isso justifica o foco em uma execucao final objetiva, sem rodar notebooks diagnosticos antigos.

## Modelo Mental Correto

### O erro que estava nos prendendo

Treinos anteriores podiam mostrar loss bom sem isso significar score. Isso acontece porque:

- Loss teacher-forced pode cair copiando formato local sem melhorar geracao livre.
- Uma familia pode melhorar e outra regredir.
- A media pode mascarar regressao por familia.
- Public extractor pode aceitar resposta que strict boxed rejeita.
- Prompt/token truncation pode manter loss baixo em subconjunto errado.
- Adapter pode carregar parcialmente e perder tensores LoRA.
- Notebook pode executar dry-run e parecer "OK", sem treinar.

### O que o GRAFT faz

O solver simbolico resolve exemplos de bit/equation fora do submit. Depois, cada solucao vira dado supervisionado:

Entrada:

```text
prompt oficial do puzzle
```

Target:

```text
</think>
\boxed{answer}
```

O treino nao aprende uma cadeia longa de raciocinio. Ele aprende a mapear o prompt oficial para a superficie que o score mede: resposta final fechada em `\boxed{}`.

### Formulacao matematica

Para cada exemplo `i`:

- `x_i`: prompt oficial.
- `a_i`: answer verificado por solver/gate.
- `y_i`: target curto `</think>\n\boxed{a_i}`.
- `M_i(t)`: mascara de completion.
- `B_i(t)`: mascara dos tokens dentro do payload boxed.
- `w_i`: peso de sampling da linha/familia.
- `theta_0`: base Nemotron + adapter 086 congelado.
- `Delta_phi`: delta LoRA treinavel.

Objetivo pratico:

```text
L(phi) =
  mean_i w_i *
  sum_t M_i(t) * (1 + lambda_box * B_i(t)) *
  CE(p_{theta_0 + Delta_phi}(token_t | x_i, y_<t), token_t)
  /
  sum_t M_i(t) * (1 + lambda_box * B_i(t))
```

Interpretacao geometrica:

- O solver define pontos/constraints no espaco de funcoes.
- O LoRA define um subespaco low-rank de atualizacoes possiveis.
- O treino projeta as constraints simbolicas para esse subespaco.
- Como o alvo e curto e score-facing, o gradiente fica concentrado no answer manifold, nao em texto explicativo.
- Replay protegido adiciona constraints de nao-regressao para familias ja fortes.

Essa e a forma correta de "transformar solver em LoRA": nao existe conversao algebraica direta segura de uma regra bit/equation para pesos LoRA sem resolver um problema inverso do modelo. O caminho robusto e destilacao supervisionada gateada.

## Artefatos Validados

Worktree auditada:

`artifacts/v1243_final_publish_worktree`

Artefato central:

`artifacts/v1243_solver_to_lora_graft`

Arquivos principais:

- `v1243_bit_specialist_train.jsonl`
- `v1243_equation_specialist_train.jsonl`
- `v1243_protected_replay_train.jsonl`
- `v1243_micro_consolidation_train.jsonl`
- `v1243_val170.jsonl`
- `v1243_hf_env_preview.json`
- `kg1_v1243_solver_to_lora_graft_manifest.json`

Contagens validadas:

- Bit specialist: `724` rows = `540` bit + `184` replay protegido.
- Equation specialist: `544` rows = `360` equation + `184` replay protegido.
- Protected replay: `184` rows.
- Micro consolidation: `1084` rows = `540` bit + `360` equation + `184` replay protegido.
- Val170: `170` rows = `90` bit + `60` equation + `20` protegidas.

Hashes apos correcao de metadata:

- Bit train: `b78a7b92c22ec0f4ba6f86be3cfe47f52a5a2d85ce9a9fa9cbe2649e67ac066c`
- Equation train: `53f971575cd34733f68d4b1848e9da3caacab6cc43ec51fe640230c01706e62b`
- Micro train: `8105632d90cd11d3fe3db293c666115d7da046ed7b04317056a5776c2185fbe5`
- Val170: `28e26f6f64812fdd6bc962550bd1c0ce1bf4de24806016d3ccc7ec963b9cb3eb`

Validacoes independentes executadas:

- Zero contrato ruim nos JSONL.
- Zero prompt duplicado dentro dos splits.
- Zero overlap entre train e val170.
- Zero overlap com full947 judge.
- Todos os assistant targets iniciam com `</think>\n`.
- Todos os assistant targets tem exatamente um `\boxed{}` fechado.
- Todos os payloads boxed verificam contra `answer`.
- Todos os rows tem pesos de answer/boxed payload.
- Todos os pesos top-level e metadata foram sincronizados.

## Bugs/Gaps Encontrados Nesta Rodada

### 1. Metadata de peso inconsistente

Achado:

O top-level `row_loss_weight` estava correto, mas `metadata.row_loss_weight` podia carregar peso antigo. Exemplo antes da correcao:

- micro bit top-level `1.05`
- metadata antigo `1.35`

Impacto:

O trainer atual usa top-level primeiro, entao o sampling efetivo estava correto. Mas isso era um bug silencioso de reprodutibilidade: qualquer audit, ferramenta futura ou script externo que lesse metadata poderia aplicar peso errado.

Correcao:

- `scripts/kg1_v1243_solver_to_lora_graft_builder.py` agora grava:
  - `metadata.v1243_sampling_weight`
  - `metadata.row_loss_weight`
  - `metadata.loss_weight`
  iguais ao peso efetivo.

Gates endurecidos:

- `scripts/kg1_v1243_dataset_logic_audit.py` falha se pesos top-level e metadata divergirem.
- `scripts/kg1_v1243_graft_trainer_contract_gate.py` falha se pesos top-level e metadata divergirem.
- `scripts/kg1_score_path_operational_audit.py` falha se pesos top-level e metadata divergirem.

Resultado validado:

`weight_mismatches = 0` para bit, equation, micro e protected replay.

### 2. `REQUIRE_SCORE_TRAJECTORY_PASS` podia aceitar WATCH

Achado:

Antes, se `REQUIRE_SCORE_TRAJECTORY_PASS=1` mas `REQUIRE_SCORE_TRAJECTORY_FINAL_ONLY=0`, o trainer falhava apenas para `RISK/STOP`; `WATCH` podia continuar.

Impacto:

Isso criava ambiguidade semantica: "require pass" deveria significar "status OK obrigatorio".

Correcao:

Em `scripts/hf_job_train_v90.py`, agora:

```python
trajectory_failed = trajectory_status != "OK"
```

Gates endurecidos:

- `kg1_score_path_operational_audit.py` exige esse snippet.
- `kg1_v1243_graft_trainer_contract_gate.py` exige esse snippet.

### 3. Pack do Colab nao incluia protected replay standalone nem algoritmo

Achado:

O treino micro ja continha replay protegido embutido, mas o pack nao incluia `v1243_protected_replay_train.jsonl` standalone nem `KG1_V1243_GRAFT_ALGORITHM.md`.

Impacto:

Nao quebrava treino, mas dificultava auditoria dentro do Colab.

Correcao:

`scripts/kg1_build_v1243_colab_launch_pack.py` agora inclui:

- `artifacts/v1243_solver_to_lora_graft/KG1_V1243_GRAFT_ALGORITHM.md`
- `artifacts/v1243_solver_to_lora_graft/v1243_protected_replay_train.jsonl`

Pack novo:

- arquivos: `17`
- SHA256: `ed074c74d106b9ca969ff27cb59af5e0d0bdf0010490c2041b64c70013361589`

### 4. Dry-run reports ficaram stale apos mudar JSONL

Achado:

Ao corrigir os JSONL, os gates falharam corretamente por hash antigo nos dry-run reports.

Correcao:

Foram regenerados os tokenize dry-runs para:

- `bit`
- `equation`
- `micro_consolidation`

Depois disso:

- `kg1_score_path_operational_audit.py`: PASS.
- `kg1_active_gate_registry_audit.py`: PASS.

## Gates Executados e Status

Comandos/gates executados apos as correcoes:

```powershell
python -m py_compile scripts\hf_job_train_v90.py scripts\kg1_v1243_solver_to_lora_graft_builder.py scripts\kg1_v1243_dataset_logic_audit.py scripts\kg1_v1243_graft_trainer_contract_gate.py scripts\kg1_score_path_operational_audit.py scripts\kg1_build_v1243_colab_launch_pack.py scripts\notebook_release_gate.py
python scripts\kg1_v1243_dataset_logic_audit.py --artifact-dir artifacts\v1243_solver_to_lora_graft --phase all
python scripts\kg1_v1243_graft_trainer_contract_gate.py
python scripts\kg1_score_path_operational_audit.py
python scripts\kg1_active_gate_registry_audit.py
python scripts\notebook_release_gate.py notebooks\KG1_V1243_COLAB_REALTIME_LAUNCHER.ipynb notebooks\KG1_V1243_COLAB_REALTIME_SAFE_LAUNCHER.ipynb notebooks\KG1_V1243_COLAB_MODEL_DRYRUN_LAUNCHER.ipynb notebooks\KG1_V1243_COLAB_REALTRAIN_SMOKE_ASSERTIVE.ipynb notebooks\KG1_V1243_COLAB_FINAL_SPRINT_MICRO_CONSOLIDATION.ipynb
```

Resultado:

- `py_compile`: PASS.
- Dataset logic audit: PASS.
- Trainer contract gate: PASS.
- Score-path operational audit: PASS.
- Active gate registry audit: PASS.
- Notebook release gate dos 5 notebooks V1243: PASS.

## Como Ler o Job em Tempo Real

Durante o job final, nao aceitar apenas `loss` ou uma mensagem generica de `OK`.

Sinais obrigatorios no log:

- `PACK_DOWNLOAD pack_sha256=ed074c74...`
- `final_sprint_micro_consolidation_hard_lock = true`
- `KG1_V1243_RUN_TRAIN = 1`
- `KG1_V1243_REQUIRE_REAL_TRAIN = 1`
- `RUN_TRAIN=1`
- `REQUIRE_REAL_TRAIN=1`
- `REAL_TRAIN OK`
- `WRAPPER_END real_train_executed=true`
- `final_adapter_created=true`
- upload para `felipesp1983/kg1-v1243-final-sprint-candidate`

Sinais de saude alem de loss:

- `KG1_SCORE_CONTRACT_STATUS=OK`
- `KG1_SCORE_PROXY_STATUS=...`
- `KG1_SCORE_TRAJECTORY_STATUS=final status=OK`
- `bit_exact_delta > 0`
- `equation_exact_delta > 0`
- `protected_exact_delta >= 0`
- `overall_exact_delta >= 0`
- `boxed_loss_delta <= 0`

Se aparecer `SCORE_TRAJECTORY=WATCH`, `RISK` ou `STOP` com `REQUIRE_SCORE_TRAJECTORY_PASS=1`, o job deve falhar.

## Criterio Real Para Claim >=0.89

O criterio final nao e o loss e nao e o proxy.

Para claim `>=0.89`, precisa:

- Gerar `raw_output` real com o adapter candidato.
- Comparar contra baseline 086 answer-only strict-clean.
- Rodar gate V1241 `full947_089`.
- Exigir `>=843/947`.
- Exigir ganho total `+20` sobre baseline `823/947`.
- Exigir `+1` bit.
- Exigir `+1` equation.
- Exigir zero regressao nas familias protegidas.
- Exigir zero truncation.
- Exigir strict boxed sem falha de formato.

O gate relevante e:

```powershell
python scripts\kg1_v1241_bit_equation_transfer_gate.py --profile full947_089 --solution-csv <full947_solution.csv> --baseline-predictions <086_answer_only_raw_output.csv> --candidate-predictions <candidate_raw_output.csv>
```

## Plano Perfeito de Execucao

### Etapa 1: Nao usar a raiz suja

Nao executar notebooks antigos da raiz principal. A worktree raiz esta desatualizada e tem artefatos antigos.

Usar somente o notebook final sprint publicado.

### Etapa 2: Model dry-run antes de confiar em GPU

O tokenize dry-run ja passou. Ainda e necessario o model-load dry-run em GPU para provar:

- adapter inicial 086 baixado no revision certo;
- SHA de config/weights;
- LoRA modules carregados;
- trainable ratio dentro do limite;
- `mamba_ssm` e `causal_conv1d` reais;
- device map e BF16 OK.

### Etapa 3: Treino micro final

Executar `micro_consolidation` com:

- `MAX_STEPS=20`
- `LEARNING_RATE=0.00000075`
- `FINAL_LEARNING_RATE=0.00000020`
- `EVAL_EVERY_STEPS=2`
- `SAVE_EVERY_STEPS=2`
- `EVAL_MAX_EXAMPLES=170`
- `SCORE_PROXY_EVAL_MAX_EXAMPLES=170`
- `REQUIRE_SCORE_TRAJECTORY_PASS=1`
- `REQUIRE_SCORE_TRAJECTORY_FINAL_ONLY=1`

### Etapa 4: Parar cedo se a direcao estiver errada

Parar se:

- adapter nao carregar;
- trainable modules divergirem;
- prompt truncation > 0;
- fallback masks > 0;
- score trajectory final nao for `OK`;
- bit ou equation nao melhorarem individualmente;
- protected regredir;
- final adapter nao for criado.

### Etapa 5: Validar raw-output antes de qualquer submit

Depois do treino:

1. Gerar raw outputs no mesmo contrato de prompt.
2. Rodar V1241 tiny.
3. Rodar V1241 val170.
4. Se passar, rodar full947_089.
5. So considerar submit se full947_089 passar.

## Limites Honestamente Declarados

Este relatorio nao prova que o leaderboard dara `>=0.89`.

Ele prova que:

- a transformacao solver -> LoRA esta operacionalmente coerente;
- os dados estao limpos e gateados;
- o notebook final nao e dry-run por engano;
- o pack/notebook tem hash travado;
- os gates barram varios falsos positivos;
- bugs silenciosos encontrados nesta rodada foram corrigidos.

O salto final depende de treino real, geracao real e gate full947.

## Resposta Curta Para o GPT

A solucao nao e tentar embutir o solver no runtime. A solucao e usar o solver para gerar constraints supervisionadas score-facing e projetar essas constraints no subespaco LoRA via treino curto, com replay protegido e gates por raw-output. Corrigimos inconsistencias de peso no metadata, endurecemos `REQUIRE_SCORE_TRAJECTORY_PASS` para falhar qualquer status nao-OK, regeneramos JSONL/dry-runs/pack/notebooks e validamos a cadeia com gates. O proximo passo real e executar o notebook final sprint pinado, aceitar somente logs que provem `real_train_executed=true` e promover candidato apenas se V1241 `full947_089` passar com `>=843/947`, ganho `+20`, bit/equation positivos e zero regressao protegida.
