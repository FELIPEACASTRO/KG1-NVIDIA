# V466 Component Readiness Audit

Gerado em 2026-05-15 para auditar as pecas que afetam loss, ACC e score das
familias `bit_manipulation` e `equation_transform`.

## Estado Resumido

| Componente | Status | Evidencia | Acao |
|---|---|---|---|
| Dados V464 | pronto | `v464_v463_numeric_multirule_dataset_manifest.json`: 558 train, 138 val, hashes fixos, sem issues | nenhum bloqueio |
| Tokenizacao/loss mask | pronto | `v286_generic_tokenization_gate_manifest.json`: 0 truncation, 0 fallback masks, offset masks em 558/138 linhas | nenhum bloqueio |
| Loss de treino | executado, nao conclusivo | V465 final: baseline `1.8248`, best `1.8234`, final `1.8239` | nao promover por loss; precisa weak ACC |
| Checkpoints V465 | pronto para weak eval | HF repo contem `checkpoint-4/8/12/16` e `final` com `adapter_config.json` e `adapter_model.safetensors` | avaliar sweep |
| ACC/metric | pronto | V449 `metric_path_ok`; promocao usa `src.competition_utils.verify_answer` | manter estrito |
| Weak eval runner | pronto para launch | `launch_v466_hf_weak_eval_v465.py` debug passou e encontrou todos adapters | commitar/push, depois launch |
| Full eval/package/submit | bloqueado | Gate exige weak `total>192`, `equation>56`, `bit>=136`, `truncated=0` | so liberar se V466 passar |
| FinOps | configurado | H200 `0.083333` USD/min, timeout 3600s, max unit cost `0.09` | cancelar se falha ou sem valor |

## Auditoria por Peca

### 1. Dados

O dataset V464 esta criado e consistente:

- train SHA256: `16abd61d2e14b0c0911d32a41e23e2f58b62424cac23e1b0e9147ef6358634ad`
- val SHA256: `7379295c39459f531b1fd0dc97e9d5c6441976a961079b6cef2b3a2e0c82d228`
- train: `512` bit replay + `46` equation
- val: `128` bit replay + `10` equation
- hard negatives reais: `22` train + `4` val
- classes equation cobertas: add direct, minus direct negative, minus opposite sign, colon absdiff restore trailing zero

Risco: dataset pequeno e altamente direcionado. Ele e adequado para smoke
controlado, nao para afirmar ganho sem weak eval.

### 2. Tokenizacao e Loss

A tokenizacao passou:

- max length `8192`
- train token max `327`
- val token max `356`
- prompt truncation `0`
- completion dropped `0`
- fallback masks `0`
- offset masks em todas as linhas
- answer-span weighting ligado com peso `5.0`

Conclusao: a infraestrutura de loss esta pronta. O problema restante nao e
erro de mascara ou truncation detectado; e transferencia de comportamento para
ACC.

### 3. Treino V465

V465 treinou no H200 com:

- imagem `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`
- init adapter `v290 checkpoint-6`
- LoRA modules `q_proj,k_proj,v_proj,o_proj,lm_head`
- max steps `16`
- learning rate `2.8e-8`
- final learning rate `8.0e-9`
- checkpoints a cada `4` steps

Resultado de loss:

- baseline eval loss: `1.8248`
- best eval loss: `1.8234`
- final eval loss: `1.8239`

Conclusao: o treino esta tecnicamente saudavel, mas a queda de loss e pequena.
Isso nao prova ganho de ACC e historicamente loss baixo nao tem melhorado as
duas familias sem weak eval positivo.

### 4. Checkpoints

O repo HF `felipesp1983/kg1-nemotron-lora-v465-v464-numeric-multirule-v290ckpt6`
tem todos os checkpoints esperados:

- `checkpoint-4`
- `checkpoint-8`
- `checkpoint-12`
- `checkpoint-16`
- `final`

Cada um contem config e pesos LoRA. Portanto a proxima etapa pode avaliar todos
no mesmo weak job.

### 5. ACC, Parser e Score

A metrica esta pronta e auditada:

- parser: `extract_final_answer`
- scoring: `verify_answer`
- bit-like answers usam match exato
- respostas numericas nao-binarias usam tolerancia historica de `1%`
- `answers_equivalent` permanece apenas diagnostico

O V449 confirmou `metric_path_ok`. Isso evita overcount de respostas binarias
como `101` vs `101.0`.

### 6. Gates

Gate de promocao atual:

- weak total deve ser `>192`
- `equation_transform` deve ser `>56`
- `bit_manipulation` deve ser `>=136`
- `truncated` deve ser `0`
- full/package/submit ficam bloqueados ate esse gate passar

Isso esta correto para evitar regressao e gasto desnecessario.

## Decisao

As pecas de dados, tokenizacao, treino, checkpoints, metrica e gate estao
criadas e configuradas. A unica peca ainda pendente para saber se houve melhora
real e executar o V466 weak eval nos checkpoints V465.

Se V466 reprovar, a rota V465 deve ser rejeitada por FinOps e nao deve ir para
full eval, package ou Kaggle submit.

