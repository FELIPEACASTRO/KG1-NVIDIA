# KG1 V386 ACC Measurement Double Check

Data: 2026-05-14

Objetivo: fazer um segundo passe rigoroso sobre o caminho que mede ACC, validando scripts, datasets, contratos, prompt, scorer, truncation, origem Kaggle oficial e sujeira local.

## Conclusao Executiva

O caminho atual de medicao de ACC weak/full esta tecnicamente correto para comparar candidatos adapter-only, desde que o resultado seja sempre acompanhado do contrato de prompt usado.

O V384 confirmou que o V383 nao falhou por bug de medicao: com o prompt historico V221 e thinking habilitado, o melhor checkpoint V382 chegou a `193/315`, `bit=137`, `equation=56`, mas teve `truncated=1`. Isso nao passa o gate, porque `equation` nao subiu e a melhora veio com truncamento.

Nao foi encontrado dataset sujo entrando no score. O `weak315` e o `full947` batem com o `train.csv` oficial do Kaggle por `id`, `prompt` e `answer`.

## Scripts Auditados

- `src/competition_utils.py`
- `scripts/evaluate_lora_adapter.py`
- `scripts/evaluate_lora_adapters_batch.py`
- `scripts/hf_job_weak_eval_v245.py`
- `scripts/hf_job_official_like_eval_gate_v284.py`
- `scripts/analyze_v230_v226_complementarity.py`
- `scripts/run_v336_integrated_no_loss_solver_gate.py`
- `scripts/run_v336b_package_permission_gate.py`

Validacao executada:

- `python -m py_compile` nos scripts principais de scoring/eval.
- Testes sinteticos de `extract_final_answer`, `verify_answer`, `normalize_questions` e `prepare_merged_predictions`.
- Re-score local dos arquivos V384 baixados do HF para confirmar que `candidate_summary` bate com `predictions.csv`.
- Download temporario do `train.csv` oficial via Kaggle CLI; arquivo removido ao final.

## Dataset Weak 315

Fonte HF:

- Repo: `felipesp1983/kg1-nemotron-training`
- Path: `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv`
- SHA: `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`
- Row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`

Checks:

- Rows: `315`
- `bit_manipulation`: `160`
- `equation_transform`: `155`
- IDs duplicados: `0`
- Nulos em `id`, `prompt`, `answer`: `0`
- Mismatch `family` vs `classify_puzzle(prompt)`: `0`
- IDs ausentes no `train.csv` oficial Kaggle: `0`
- Prompt mismatch vs `train.csv` oficial Kaggle: `0`
- Answer mismatch vs `train.csv` oficial Kaggle: `0`

## Dataset Full 947

Fonte HF:

- Repo: `felipesp1983/kg1-nemotron-training`
- Path: `runtime_artifacts/v276_full_eval_bridge/v276-full947-bridge-20260511T1245Z/official_train_seed42_stratified10_val.csv`
- SHA: `84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935`
- Row contract: `5441932fc270eb9621a32b4d7e85ff444c45aa31d75e2bb7aea0de96cd638f21`

Checks:

- Rows: `947`
- Family counts: `bit=160`, `equation=155`, `gravity=159`, `numeral=157`, `text_encryption=157`, `unit_conversion=159`
- IDs duplicados: `0`
- Nulos em `id`, `prompt`, `answer`: `0`
- Mismatch `family` vs `classify_puzzle(prompt)`: `0`
- IDs ausentes no `train.csv` oficial Kaggle: `0`
- Prompt mismatch vs `train.csv` oficial Kaggle: `0`
- Answer mismatch vs `train.csv` oficial Kaggle: `0`

## Scorer

O caminho de scoring e:

1. Gerar `raw_output` com vLLM.
2. Extrair `prediction` com `extract_final_answer`.
3. Fazer merge `solution.merge(pred, on="id", validate="one_to_one")`.
4. Usar `answer` apenas depois da geracao.
5. Calcular `correct` com `verify_answer(answer, prediction)`.
6. Calcular truncamento por `finish_reason == "length"`.
7. Agregar por `type/family` e no total.

Pontos confirmados:

- `answer` nao entra no prompt renderizado; `render_prompts` usa apenas `row.prompt + prompt_suffix`.
- O merge e `one_to_one` por `id`; duplicata quebraria a execucao.
- `bit_manipulation` e avaliado por igualdade binaria exata.
- Respostas numericas fora de `[01]+` usam tolerancia numerica `rel_tol=1e-2`, `abs_tol=1e-5`.
- Respostas esperadas compostas apenas de `0` e `1` usam igualdade exata. Existem 2 casos de `equation_transform` assim no weak/full (`101` e `100`), mas V384 respondeu exatamente esses valores; portanto nao houve distorcao no resultado atual.
- `truncated` nao e inferido por tamanho do texto; vem do `finish_reason` do vLLM.
- V383/V384 weak eval nao usa solver nem postprocessor.

## V384 Re-score

Job:

- `https://huggingface.co/jobs/felipesp1983/6a0602f5e48bea4538b9d117`
- Status: `COMPLETED`
- Image: `vllm/vllm-openai:v0.20.1`
- Flavor: `h200`
- Prompt suffix: V221 historico
- Thinking: habilitado
- Postprocessor: `none`

Resultados baixados do HF e reescore local:

| Candidato | Total | Equation | Bit | Truncated | Decisao |
|---|---:|---:|---:|---:|---|
| `v382_ckpt4_v221prompt` | `193/315` | `56/155` | `137/160` | `1` | rejeitar; falha por truncation e equation nao sobe |
| `v382_ckpt6_v221prompt` | `190/315` | `55/155` | `135/160` | `1` | rejeitar; regressao |

O re-score local dos CSVs de predicao reproduziu exatamente os numeros do `batch_candidate_summary.json`.

## Sujeira e Artefatos

- Nenhum arquivo maior que `500MB` foi encontrado dentro do worktree auditado.
- Downloads temporarios do Kaggle/HF usados nesta auditoria foram removidos.
- Permanecem artefatos pequenos nao rastreados de execucoes antigas em `artifacts/v322...`, `artifacts/v323...` e `artifacts/v326...`; eles nao entram no score, mas devem ser arquivados/removidos em uma limpeza separada para nao misturar historico operacional com candidatos atuais.

## Decisao Tecnica

- A medicao atual e confiavel para gates weak/full.
- Nao ha evidencia de que o ACC baixo seja bug de scorer.
- O problema principal continua sendo modelo/candidato, nao medicao.
- Encerrar a linha V381/V382/V384 para promocao/submissao.
- Proximo trabalho deve voltar para CPU gate com novo sinal real: solver/trace que aumente `equation>56` com `bit>=136`, `truncated=0`, antes de gastar novo HF training.
