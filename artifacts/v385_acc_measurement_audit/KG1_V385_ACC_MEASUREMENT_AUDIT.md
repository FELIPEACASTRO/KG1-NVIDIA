# KG1 V385 ACC Measurement Audit

Data: 2026-05-14

Objetivo: auditar se o job/script que mede ACC weak esta correto, se o dataset usado no score esta sob contrato fixo, e se ha sujeira ou informacao desnecessaria entrando na medicao.

## Conclusao

O caminho atual de medicao weak ACC esta correto para comparar candidatos adapter-only, desde que o prompt contract seja declarado junto com o resultado.

O unico problema operacional encontrado nao e no scorer nem no dataset: o V383 foi um diagnostico valido com sufixo curto, mas nao era comparavel 1:1 com o contrato historico V221. O V384 corrige isso ao avaliar os checkpoints V382 com o sufixo V221 e thinking habilitado.

## Scripts auditados

- `scripts/hf_job_weak_eval_v245.py`
- `scripts/evaluate_lora_adapter.py`
- `scripts/evaluate_lora_adapters_batch.py`
- `src/competition_utils.py`
- `scripts/hf_job_official_like_eval_gate_v284.py`

Validacao local executada:

- `python -m py_compile src/competition_utils.py scripts/evaluate_lora_adapter.py scripts/evaluate_lora_adapters_batch.py scripts/hf_job_weak_eval_v245.py scripts/hf_job_official_like_eval_gate_v284.py`
- Testes sinteticos de `extract_final_answer`, `verify_answer` e `classify_puzzle`.
- Download pequeno do CSV weak via HF cache e validacao pelo mesmo `validate_weak_csv` usado no job HF.

## Dataset weak auditado

Arquivo:

- HF dataset repo: `felipesp1983/kg1-nemotron-training`
- Path: `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv`

Resultado da validacao:

- `rows=315`
- `family_counts={"bit_manipulation": 160, "equation_transform": 155}`
- `sha256=85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`
- `observed_shared_row_contract_sha256=bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`
- IDs duplicados: `0`

O contrato inclui `id`, `family`, `answer` e `prompt_sha256`. Isso impede comparar resultados contra CSV alterado, rows reordenadas com conteudo divergente, familia incorreta ou labels trocados.

## Como o ACC e calculado

O fluxo do weak job e:

1. Baixa o weak CSV e o manifest do HF dataset.
2. Valida SHA do CSV, row count, family counts, IDs unicos e shared row contract antes de carregar modelo.
3. Baixa os adapters candidatos.
4. Valida `adapter_config.json`, `adapter_model.safetensors`, `r`, `lora_alpha` e tamanho do adapter.
5. Renderiza prompt via chat template do Nemotron.
6. Gera com vLLM usando temperatura `0.0`, `top_p=1.0`, `max_tokens=7680`, `max_model_len=8192`.
7. Extrai a resposta final com `extract_final_answer`.
8. Faz merge `one_to_one` por `id`.
9. Usa a familia do CSV de solucao quando disponivel.
10. Calcula `correct` por `verify_answer`.
11. Marca truncation quando `finish_reason == "length"`.
12. Emite `candidate_summary` e `candidate_per_task`.

## Regras do scorer relevantes para nossas familias

- `bit_manipulation`: comparacao binaria exata.
- `equation_transform`: usa a resposta extraida; quando a resposta esperada e numerica fora de `[01]+`, o verificador aplica tolerancia numerica do projeto. Quando a resposta esperada contem apenas `0/1`, o verificador atual aplica igualdade exata, mesmo se a familia for `equation_transform`.
- `extract_final_answer`: prefere a ultima ocorrencia de `\boxed{...}`, depois frases de final answer, depois ultimo numero, depois ultima linha nao vazia.

Para bit e equation, isso e consistente com o uso atual do weak gate. Nao foi encontrado uso de solver, postprocessor ou label leak no V383/V384 weak eval.

## Prompt contract

V383:

- Sufixo curto: `Return only one line: \boxed{answer}. No reasoning. No explanation.`
- `disable_thinking=false`
- Resultado: valido como diagnostico, mas nao comparavel diretamente ao historico V221.

V384:

- Sufixo V221: `Please put your final answer inside \boxed{}. For example: \boxed{your answer}`
- `disable_thinking=false`
- Resultado: comparavel ao contrato historico V221.

## Sujeira local

Busca por arquivos locais maiores que `500MB` dentro do worktree auditado: nenhum encontrado.

Ha artefatos pequenos nao rastreados de execucoes antigas em:

- `artifacts/v322_hf_nemo_a100_v51_filtered_hybrid_launch/...`
- `artifacts/v323_hf_nemo_a100_equation_narrow_o_lmhead_launch/...`
- `artifacts/v326_hf_nemo_a100_equation_bit_replay_launch/...`

Eles nao entram no score atual e nao foram apagados porque podem ser historico operacional. Se forem removidos depois, remover apenas arquivos gerados e nao scripts/manifests uteis.

## Decisao

- Manter o scorer atual; a auditoria V386 confirmou que a excecao `[01]+` nao alterou o resultado V384 porque os dois casos de equation afetados foram respondidos exatamente.
- Nao promover resultados que nao declarem prompt suffix e thinking mode.
- V384 e o teste correto para responder se o problema do V383 era o sufixo curto.
- Se V384 nao bater `total>192`, `equation>56`, `bit>=136`, `truncated=0`, encerrar a linha V382/V381 teacher-transfer.
- Antes de qualquer novo HF training, exigir novo CPU gate com ganho verificavel por ACC, nao por `eval_loss`.
