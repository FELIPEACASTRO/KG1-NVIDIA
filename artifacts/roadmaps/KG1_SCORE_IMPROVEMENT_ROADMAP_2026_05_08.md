# KG1 NVIDIA - Roadmap de melhoria por familia

Gerado em: 2026-05-08

Objetivo: transformar os beneficios encontrados nas auditorias anteriores em um plano operacional para melhorar score sem gastar GPU em tentativas cegas.

## Estado atual do desafio

- Leaderboard publico Kaggle baixado em 2026-05-08T17:48:04: 2.815 linhas.
- Nosso time `Felipe Angelo`: rank 19, score publico 0.86, 59 submissoes.
- Topo publico: 3 times em 0.87; 580 times em 0.86; 297 times em 0.85.
- Conclusao: a margem para subir existe, mas o risco de regressao e alto. A solucao deve ser no-loss/family-gated, nao um novo treino cego.

## Beneficios ja encontrados

### 1. Adapters publicos e candidatos fortes

| Ativo | Onde apareceu | Beneficio pratico | Risco | Uso agora |
|---|---|---|---|---|
| `Naribow/nemotron-sft-lora` | HF validado no triple check | Adapter publico completo, nao gated, com `adapter_config.json` e `adapter_model.safetensors` | Peso grande; precisa weak gate | Candidato V220/V221 para weak eval identica |
| `dgxchen/trained-adapter` | Kaggle validado | Adapter completo com safetensors ~4.26GB | Baixar/avaliar custa tempo; pode nao superar baseline | Entrar no registry e rodar weak only |
| `konbu17/exp026-s012-lora` | Kaggle validado | Adapter completo no mesmo tamanho esperado | Mesma familia de risco de adapter externo | Rodar weak only |
| `konbu17/nemotron-sft-lora-cot-selection` | Kaggle validado | Adapter + `train_split_with_cot.csv` | Dados podem conter ruido; nao misturar cegamente | Avaliar adapter; inspecionar COT apenas com filtro |
| `kienngx/nemotron-nano-30b-trained` | Kaggle validado | Inclui instancia `Triton/tinker-adapter/1`; competidor forte | Packaging/paths Kaggle podem exigir conversao | Prioridade alta para triagem |
| `huikang/nemotron-adapter` via Tinker | Notebooks Tinker locais | Mostra conversao de adapter e tensor surgery Mamba | Pode ser dependente de Kaggle input | Reaproveitar tecnica de conversao e logprob gate |

### 2. Dados e traces uteis

| Ativo | Tamanho/forma | Beneficio pratico | Uso correto |
|---|---:|---|---|
| `v60_selective_train_boxed_dedup.jsonl` | 5.770 rows, 100% boxed | Pool antigo forte em text/equation/bit | Reusar somente com dedupe, conflito e gate por familia |
| V87 `nemotron_087_train.final.jsonl` | 13.527 rows | Mix historico com Kienngx/Kishan/Huikang/Gojay | Fonte de dados e validacoes; nao garante checkpoint |
| V87 `train_plus_solver` | 20.314 rows | Inclui `solver_guided` | Bom para familias faceis; cuidado com equation_transform baixa cobertura |
| V87 validation clean | 720 rows, 120 por familia | Slice equilibrado para gates | Usar como weak/family sanity |
| `andy279/nemotron-reasoning-challenge` | HF gated/manual | SFT train/val grande reportado | Usar se acesso autorizado; filtrar respostas corretas |
| `andy279/...raw-traces` | HF gated/manual | Traces de solver bit/transformation | Usar para solver/verifier e treinamento filtrado |
| `dgxchen/nemotron-cot-tong` | Kaggle validado | COT e IDs matched | Inspecionar como fonte auxiliar, nao treino direto |
| `kienngx/...cot-labels` | Kaggle validado | COT labels externos | Usar com controle de qualidade e family split |

### 3. Gates e engenharia que reduzem custo

- `scripts/notebook_release_gate.py` ja passa em V218, V219 e V220.
- Gates antigos V87 reaproveitaveis:
  - `kg1_submission_gate.py`: ZIP, adapter files, root shape, blocked entries, adapter key contract.
  - `kg1_checkpoint_gate.py`: checkpoint/repo/loss/manifest.
  - `kg1_dataset_gate.py`: total minimo, boxed rate, familias, duplicados.
  - `kg1_training_data_gate.py`: overlap train/test, segredos, mensagens, boxed, duplicidade.
  - `kg1_boxed_parser_suite.py`: robustez do parser de `\\boxed{}`.
  - `kg1_local_metric_gate.py`: metricas por familia e erro CSV.
- Licao de runtime:
  - Treino Transformers/PEFT/Mamba deve ser isolado de vLLM.
  - vLLM 0.20.1 pode trocar stack Torch/CUDA; nao instalar vLLM antes de compilar/importar dependencias de treino.
  - Avaliacao vLLM roda sem forcar build manual de `causal-conv1d`/`mamba_ssm` em alguns logs; nao adicionar build caro se nao for necessario.

### 4. Tinker: aprendizado tecnico reaproveitavel

- Os notebooks Tinker apontam para `huikang/nemotron-adapter`.
- Eles copiam uma referencia `huikang/nvidia-nemotron-all-linear`.
- Reescrevem `target_modules` para:
  - `k_proj`, `o_proj`, `in_proj`, `q_proj`, `up_proj`, `v_proj`, `down_proj`, `out_proj`, `lm_head`.
- Fazem tensor surgery: juntam pares Mamba `gate_proj` + `x_proj` em `in_proj`.
- A variante Tinker v3 adiciona `logprobs=1` e calcula `minlogprob`.
- Beneficio direto: usar `minlogprob` como sinal de confianca para selecao no-loss e para bloquear respostas incertas.

## Diagnostico por familia

### unit_conversion

Estado: familia mais apropriada para solver deterministico.

Plano agora:
1. Extrair prompts e respostas do train/weak.
2. Implementar/validar parser de unidades e fatores.
3. Gerar resposta canonica com tolerancia numerica.
4. Gate: aceitar override apenas se o solver consegue revalidar todos os exemplos do prompt.

Beneficio esperado: alto ganho com baixo risco e baixo custo GPU.

### gravity_constant

Estado: tambem apropriada para solver deterministico.

Plano agora:
1. Usar parser numerico robusto.
2. Fixar formato de saida canonico.
3. Validar contra train e weak por familia.
4. Aceitar override apenas quando todas as variaveis do prompt foram extraidas sem ambiguidade.

Beneficio esperado: estabiliza score e reduz dependencia do modelo.

### numeral_system

Estado: alta chance de solucao simbolica.

Plano agora:
1. Implementar conversores base/roman/bin/hex conforme prompts reais.
2. Criar suite de testes por padrao de prompt.
3. Aplicar no-loss selector: solver vence modelo quando parser tem confianca total.

Beneficio esperado: baixo risco, melhora ou preserva familia.

### text_encryption

Estado: problematica, mas boa candidata a PBE/DSL.

Plano agora:
1. Construir DSL minima de transformacoes: cifra de Caesar, substituicao, reversao, rotacao, case, split/join, pattern mapping.
2. Usar exemplos dentro do prompt para inferir regra.
3. Validar regra nos pares de treino do proprio prompt.
4. Gerar answer apenas se regra explicar todos os pares.
5. Alimentar com v60 e raw traces filtrados.

Beneficio esperado: ganho seletivo alto; risco baixo se houver abstention.

### bit_manipulation

Estado: familia dificil, mas com boa rota por bitvector/superoptimization.

Plano agora:
1. Criar parser de exemplos input/output.
2. Definir DSL de operacoes: xor, and, or, not, shifts, rotates, mask, reverse bits, swap nibbles, add/sub modulo, popcount quando aplicavel.
3. Buscar expressao que explica todos os exemplos.
4. Usar ranking por simplicidade.
5. Aceitar override apenas quando a expressao passa todos os exemplos do prompt.
6. Usar Tinker `minlogprob` para detectar casos onde modelo esta inseguro.

Beneficio esperado: maior potencial de subida, mas precisa gate rigoroso.

### equation_transform

Estado: maior gargalo historico. V216/V217 falharam principalmente aqui.

Plano agora:
1. Separar subtipos: numeric deduce, numeric guess, symbolic transform, cryptarithm.
2. Para numeric deduce: gerar hipoteses de formula e validar em exemplos.
3. Para symbolic: usar SymPy quando possivel.
4. Para cryptarithm: usar constraint solver/DFS com pruning.
5. Para casos sem prova: abstain, nao sobrescrever modelo.
6. Usar traces `andy279` e dados Kienngx/DGXChen apenas para criar candidatos/verificadores, nao para treino cego.

Beneficio esperado: principal rota para sair de 0.86, mas tambem maior risco. Deve ser no-loss.

## Roadmap operacional

### Atualizacao implementada - V221 candidate registry

Arquivos criados/alterados para transformar o roadmap em execucao weak-only:

- `notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb`
- `scripts/build_v221_candidate_registry_weak_ab_colab.py`
- `scripts/evaluate_lora_adapters_batch.py`
- `scripts/notebook_release_gate.py`

Colab URL planejada:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v221-candidate-registry/notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb`

Importante: a URL acima so funciona depois que o notebook for enviado para a branch `v221-candidate-registry`.

Beneficio pratico do V221:

- avalia V194, V217 e adapters publicos candidatos no mesmo weak split;
- baixa candidatos HF/Kaggle somente quando credenciais estao disponiveis;
- valida cada adapter por `adapter_config.json` + pesos antes de gastar vLLM;
- usa `scripts/evaluate_lora_adapters_batch.py` para carregar o modelo base uma vez e iterar LoRAs com `LoRARequest`;
- mantem thinking ligado por padrao;
- bloqueia treino, full eval automatico e Kaggle submit;
- escreve `batch_candidate_summary.csv/json`, relatorio por candidato e manifesto final.

Gates adicionados ao arquivo central `scripts/notebook_release_gate.py`:

- V221 agora exige os arquivos de suporte, hashes/linhas dos datasets V217, registry embutido, candidatos HF/Kaggle, download/resolucao de adapters, `v221_ready_candidates.json`, `batch_candidate_summary.json`, bloqueio de `RUN_TRAIN`, bloqueio de `RUN_FULL_IF_GATE` por padrao e ausencia de comando de submit.
- O novo batch evaluator tambem e validado no gate: CLI obrigatoria, `LLM(**llm_kwargs)`, `LoRARequest`, `render_prompts`, `validate_adapter_dir` e saidas agregadas.

Validacao local realizada:

- `python -m py_compile scripts/notebook_release_gate.py scripts/evaluate_lora_adapters_batch.py scripts/build_v221_candidate_registry_weak_ab_colab.py`
- `python scripts/evaluate_lora_adapters_batch.py --help`
- `python scripts/notebook_release_gate.py notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb --output-json artifacts/notebook_release_gate/v221_candidate_registry_report.json`
- `python scripts/notebook_release_gate.py notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb --output-json artifacts/notebook_release_gate/v218_v221_combined_report.json`
- varredura local sem imprimir segredos nos arquivos novos: 0 tokens HF/GitHub/Kaggle hardcoded encontrados.

### P0 - Hoje, sem GPU

1. Criar `candidate_registry.json` com todos os candidatos:
   - origem;
   - tipo: adapter, dataset, solver, notebook;
   - familia alvo;
   - caminho local/remoto;
   - hash/tamanho quando disponivel;
   - status: pending, downloaded, weak_eval_done, rejected, accepted.
2. Consolidar V87 gates no `scripts/notebook_release_gate.py`.
3. Criar uma tabela baseline por familia com:
   - V194/V216/V217/V218 weak score;
   - truncation;
   - custo de execucao;
   - motivo de rejeicao.
4. Criar `family_override_policy.json`:
   - quais familias aceitam solver;
   - criterio de confianca;
   - criterio de abstention;
   - limite maximo de regressao permitido: zero.

### P1 - Proxima execucao barata

1. Rodar weak-only adapter A/B, sem full eval:
   - Naribow;
   - dgxchen/trained-adapter;
   - konbu17/exp026-s012-lora;
   - konbu17/nemotron-sft-lora-cot-selection;
   - Kienngx/Tinker se o caminho Kaggle estiver acessivel.
2. Usar mesma configuracao para todos:
   - thinking ligado;
   - max_tokens suficiente;
   - parser identico;
   - mesmo weak CSV.
3. Rejeitar candidato se:
   - weak_total nao melhora baseline;
   - qualquer familia critica piora;
   - truncation aumenta;
   - adapter contract falha.

### P2 - Solvers/verifiers por familia

1. Implementar solver deterministico para unit/gravity/numeral.
2. Implementar bitvector DSL para bit.
3. Implementar equation hypothesis generator + SymPy/constraint checks.
4. Implementar PBE/DSL para text_encryption.
5. Rodar contra weak, gerar `override_candidates.csv`.
6. Aceitar override apenas se:
   - todos os exemplos do prompt sao satisfeitos;
   - answer format passa parser;
   - nao conflita com baseline quando baseline esta correto.

### P3 - Ensemble no-loss

1. Entrada: predicoes do melhor adapter + predicoes solver + logprob/confidence.
2. Saida: uma resposta por ID com justificativa de selecao.
3. Politica:
   - solver verificado vence modelo;
   - adapter externo vence baseline apenas na familia onde weak melhora;
   - caso incerto: manter baseline.
4. Gerar relatorio por familia:
   - ganhos;
   - perdas;
   - abstentions;
   - exemplos conflitantes.

### P4 - Treino apenas depois dos gates

Treino novo so deve acontecer se P1/P2 mostrarem que ha dados verificados capazes de melhorar familia especifica.

Regras:
1. Nada de `max_steps`/LR novo sem dataset manifest.
2. Nada de vLLM antes do stack de treino.
3. Dataset precisa passar:
   - SHA fixo;
   - no prompt truncation;
   - boxed rate;
   - offset masks;
   - family balance;
   - duplicate/conflict check.
4. Treino deve ser small delta e reversivel.
5. Resultado so avanca se weak family gate passar.

### P5 - Full eval e submissao

Full eval apenas se:
1. weak_total >= gate atual;
2. weak_eq >= gate atual;
3. weak_bit >= gate atual;
4. truncation <= gate atual;
5. no-loss selector mostra zero regressao no weak.

Submissao apenas se:
1. `notebook_release_gate.py` passa;
2. `submission_gate` passa;
3. manifest contem hashes;
4. ZIP esta no formato raiz correto;
5. nao ha arquivos de credencial;
6. existe relatorio com familia, score e custo.

## Proximo notebook recomendado

Nome sugerido: `KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb`.

Objetivo:
- Nao treinar.
- Baixar/validar candidatos.
- Rodar weak eval identico.
- Gerar ranking por familia.
- Atualizar registry.
- Bloquear full eval por default.

Celulas obrigatorias:
1. Setup repo + compile gate.
2. Candidate registry build/validation.
3. Dependency split: eval-only vLLM.
4. Adapter contract check.
5. Weak eval loop por candidato.
6. Family report.
7. No-loss candidate recommendation.
8. Full eval hard block.

## Definicao de sucesso

Curto prazo:
- reduzir custo por tentativa;
- eliminar notebook que roda sem saber se candidato e plausivel;
- identificar melhor adapter externo por familia;
- colocar solvers deterministas em gate.

Medio prazo:
- melhorar `equation_transform` e `bit_manipulation` sem derrubar familias faceis;
- trocar estrategia de "mais treino" para "selecao comprovada".

Alvo:
- sair do plateau publico 0.86 com ganho pequeno, verificavel e sem regressao.
