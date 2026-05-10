# KG1 NVIDIA - Roadmap de melhoria por familia

Gerado em: 2026-05-10

Objetivo: consolidar os achados Kaggle/Hugging Face, resultados V221/V226/V229/V230 e o plano operacional para melhorar as duas familias criticas sem treino cego.

## Resumo executivo

- Baseline protegido atual: `v226__v226_best_checkpoint1_observed_191`.
- Weak score observado: `191/315`.
- Gate para liberar full eval: total `>=193`, `equation_transform>=60`, `bit_manipulation>=133`, truncation `<=3`.
- Resultado do baseline: total `191`, equation `55`, bit `136`, truncation `0`.
- Gargalo: `equation_transform`, com gap `5`.
- `bit_manipulation` ja passa o gate, mas com margem pequena de `+3`.
- Oracle V230 por linha melhora para `197/315`, equation `57`, bit `140`, truncation `0`, mas ainda falha o gate de equation por `3`.
- Conclusao: roteamento entre os adapters ja avaliados nao resolve. O proximo ganho precisa vir de mineracao dos miss-packs, solver/verifier deterministico ou dados/traces filtrados.

## Evidencias consolidadas

### V230 Colab executado

- Notebook: `notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb`.
- Branch: `v230-v226-complementarity`.
- Commit clonado no Colab: `e916eb2111fe5590d6df6aee6186d5d9ea325897`.
- Run ID: `20260510T070126Z`.
- Output root: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity`.
- Analysis out: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity/analysis_v230_v226_complementarity/20260510T070126Z`.
- Shared row contract observado: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Manifest SHA256: `8d88ee47359a9d3bcd5cf1aeb589dc8084279f1dbaac0491cf6251dd00fd6ab4`.

### Artefatos V230 para proxima analise

- `v230_v226_complementarity_equation_miss_pack.csv`
- `v230_v226_complementarity_bit_miss_pack.csv`
- `v230_v226_complementarity_baseline_miss_hits.csv`
- `v230_v226_complementarity_pairwise_detail.csv`
- `v230_v226_complementarity_pairwise_summary.csv`
- `v230_v226_complementarity_router_simulation.csv`
- `v230_v226_complementarity_manifest.json`

## Status dos candidatos avaliados

| Candidato | Origem | Total | Equation | Bit | Trunc | Status | Decisao |
|---|---|---:|---:|---:|---:|---|---|
| `v226_best_checkpoint1_observed_191` | V226 | 191 | 55 | 136 | 0 | baseline protegido | Manter |
| `v217_final_existing` | V221 | 190 | 55 | 135 | 0 | nao supera baseline | Rejeitar como substituto |
| `v226_checkpoint_3_observed_190` | V226 | 190 | 55 | 135 | 1 | pior que baseline | Rejeitar |
| `v226_checkpoint_2_observed_189` | V226 | 189 | 55 | 134 | 1 | pior que baseline | Rejeitar |
| `v194_protected_baseline` | V221 | 190 | 54 | 136 | 0 | equation pior | Rejeitar como substituto |
| `kienngx_tinker_adapter` | Kaggle/V221 | 183 | 55 | 128 | 3 | bit abaixo do gate | Rejeitar como deployable |
| `konbu17_exp026_s012_lora` | Kaggle/V221 | 179 | 51 | 128 | 3 | equation e bit piores | Rejeitar como deployable |
| `dgxchen_trained_adapter` | Kaggle/V221 | 176 | 55 | 121 | 0 | bit pior | Rejeitar como deployable |
| `konbu17_sft_lora_cot_selection` | Kaggle/V221 | 58 | 25 | 33 | 161 | truncation alto | Rejeitar como adapter; manter COT apenas para inspecao filtrada |
| `naribow_hf_nemotron_sft_lora` | HF/V221 | 30 | 20 | 10 | 267 | truncation alto | Rejeitar como adapter weak; manter apenas como evidencia externa |
| `v227_final_adapter` | V229 | 16 | 9 | 7 | 0 | regressao severa | Nao usar |

## Achados Kaggle/Hugging Face adicionados

### Modelo base HF

- Repositorio: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Fonte: `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Observacao tecnica: modelo Nemotron-H com uso documentado via Transformers, vLLM e SGLang.
- Implicacao: manter `trust_remote_code=True`, revisar compatibilidade vLLM/torch por notebook, e nao misturar instalacao vLLM com stack de treino.

### Dataset HF SFT `andy279/nemotron-reasoning-challenge`

- Fonte: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`.
- Acesso: gated/manual; exige aceitar condicoes.
- Conteudo reportado: train `49,290` exemplos, `7,200` puzzles unicos; validation `1,165` exemplos, `1,123` puzzles unicos.
- Ponto critico: validation inclui `399` puzzles de transformation marcados como unsolved.
- Distribuicao train reportada:
  - bit_manipulation: `17,285`
  - cipher: `6,722`
  - gravity: `3,294`
  - numeral: `3,282`
  - transformation: `10,741`
  - unit_conversion: `7,966`
- Uso correto: nao treinar direto. Primeiro aceitar acesso, baixar com hash, filtrar por correctness, dedupe, conflito de resposta, familia, boxed/extractor e contrato do Kaggle verify.

### Dataset HF raw traces `andy279/nemotron-reasoning-challenge-raw-traces`

- Fonte: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`.
- Acesso: gated/manual.
- Arquivos relevantes reportados:
  - `ds_traces.jsonl`: `6,028` puzzles, DeepSeek V3.2 non-thinking, 4 attempts/puzzle.
  - `ds_traces_thinking.jsonl`: `6,028` puzzles, thinking mode.
  - `solver_transformation_traces_merged.jsonl`: `1,101` solver-guided transformation traces.
  - `solver_bit_manipulation_traces_merged.jsonl`: `1,602` solver-guided bit manipulation traces.
  - `solver_transformation_traces_gpt54.jsonl`: `85` hard transformation traces.
- Uso correto: fonte P0 para minerar regras/verifiers de `equation_transform` e `bit_manipulation`; nao usar como SFT bruto.

### Kaggle model publico `ashok205/nvidia-nemotron-3-nano-30b`

- Fonte: `https://www.kaggle.com/models/ashok205/nvidia-nemotron-3-nano-30b`.
- Status: achado externo publico ainda nao avaliado localmente.
- Decisao: pendente; so entra se metadata, download, adapter/model contract e weak eval identico passarem.

### Busca HF por adapters do mesmo base model

- Fonte: `https://huggingface.co/models?other=base_model%3Aadapter%3Anvidia%2FNVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Observacao: ha outros adapters HF recentes para o base model.
- Decisao: nao adicionar ao pipeline automaticamente. Cada candidato novo precisa entrar no registry com origem, hash, tamanho, licenca, target_modules, weak eval e no-regression gates.

## Diagnostico por familia apos V230

### equation_transform

Estado: principal gargalo.

Evidencia:

- Baseline V226: `55/155`.
- Gate: `60/155`.
- Oracle V230: `57/155`.
- Mesmo com escolha por linha entre candidatos atuais, ainda faltam `3` acertos de equation.

Plano:

1. Abrir `v230_v226_complementarity_equation_miss_pack.csv`.
2. Separar casos em:
   - baseline errou e algum candidato acertou;
   - todos os candidatos erraram;
   - resposta certa presente mas extractor/formato falhou;
   - casos simbolicos;
   - casos numericos;
   - casos tipo cryptarithm/constraint.
3. Para cada subtipo, implementar verifier/solver com abstention.
4. Aceitar override somente quando o solver prova todos os exemplos do prompt.
5. Meta minima: `+5` equation sem perder bit.
6. Meta segura: `+6` ou `+7` equation para criar margem contra ruido.

### bit_manipulation

Estado: passa o gate, mas deve ser protegido.

Evidencia:

- Baseline V226: `136/160`.
- Gate: `133/160`.
- Margem: `+3`.
- Oracle V230: `140/160`.

Plano:

1. Abrir `v230_v226_complementarity_bit_miss_pack.csv`.
2. Minerar somente regras provaveis por DSL bitvector.
3. Nao aceitar adapter externo que reduza bit abaixo de `136` sem compensacao verificada e sem passar gate total.
4. Se solver bit for usado, deve ser no-loss: quando incerto, manter V226.

## Roadmap operacional atualizado

### Atualizacao implementada - V231 miss-pack mining

Arquivos criados/alterados para executar o primeiro passo pos-V230:

- `scripts/analyze_v231_miss_packs.py`
- `scripts/build_v231_miss_pack_mining_colab.py`
- `notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb`
- `scripts/notebook_release_gate.py`

Colab URL planejada:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb`

Importante: a URL acima so funciona depois que o notebook for enviado para a branch `v230-v226-complementarity`.

O V231:

- le o manifest V230 mais recente ou o path explicito em `KG1_V231_V230_ANALYSIS_MANIFEST_JSON`;
- exige o row-contract `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff` por padrao;
- valida existencia, linhas, bytes e SHA256 dos CSVs V230 obrigatorios;
- classifica `equation_miss_pack` por rota de solver;
- classifica `bit_miss_pack` como guardrail;
- gera `equation_miss_taxonomy_csv`;
- gera `equation_solver_candidate_rules_json`;
- gera `bit_guardrail_candidates_json`;
- bloqueia treino, full scoring, package e Kaggle submit.

Validacoes locais realizadas:

- `python -m py_compile scripts/notebook_release_gate.py scripts/analyze_v231_miss_packs.py scripts/build_v231_miss_pack_mining_colab.py`
- `python scripts/analyze_v231_miss_packs.py --self-test --v230-analysis-manifest-json dummy --output-dir dummy`
- `python scripts/notebook_release_gate.py --self-test`
- `python scripts/notebook_release_gate.py notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb --output-json artifacts/notebook_release_gate/v231_miss_pack_mining_report.json`
- `python scripts/notebook_release_gate.py notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb notebooks/KG1_V219_WEAK_DECODE_AB_COLAB.ipynb notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb --output-json artifacts/notebook_release_gate/release_notebooks_v231_report.json`
- `python scripts/scan_repo_secrets.py`
- `git diff --check`

### P0 - Congelar baseline e evidencias

1. Manter V226 checkpoint-1 como baseline protegido.
2. Registrar `shared_row_contract_sha256=bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff` em qualquer rerun nao diagnostico.
3. Para rerun reprodutivel do V230, fixar:
   - `KG1_V230_EXPECTED_REPO_COMMIT=e916eb2111fe5590d6df6aee6186d5d9ea325897`
   - `KG1_V230_EXPECTED_SHARED_ROW_CONTRACT_SHA256=bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`
4. Nao liberar full eval/submission com `V230_DIAGNOSTIC_MODE=True`.

### P1 - Minerar miss-packs

1. Fazer auditoria manual/programatica do `equation_miss_pack`.
2. Gerar uma tabela de subtipo por row id.
3. Identificar os `+2` acertos do oracle em equation e os `+4` acertos do oracle em bit.
4. Separar linhas onde adapter externo acerta e V226 erra, sem usar label para deploy.
5. Transformar padroes em regras verificaveis.

### P2 - Solver/verifier para equation

1. Implementar parser de exemplos input/output.
2. Implementar rotas:
   - numeric formula search;
   - symbolic transform com SymPy;
   - integer/constraint solver;
   - cryptarithm/DFS com pruning;
   - pattern transform com DSL pequena.
3. Cada regra precisa produzir:
   - answer;
   - confidence/proof;
   - motivo de abstention quando falha.
4. Gate: nenhum override sem prova local.

### P3 - Bit guardrail

1. Implementar ou reaproveitar DSL bitvector.
2. Aceitar apenas expressao que explica todos os exemplos do prompt.
3. Manter V226 como fallback.
4. Exigir que qualquer candidato final mantenha `bit_manipulation>=136` em weak local, ou no minimo `>=133` com ganho total/equation demonstrado.

### P4 - Ingestao controlada dos dados HF/Kaggle

1. Gated HF: baixar somente apos autorizacao.
2. Criar manifest com:
   - URL/ref;
   - revision;
   - hashes;
   - row counts;
   - family counts;
   - schema;
   - correctness filter;
   - overlap/duplicate/conflict report.
3. Usar `andy279` primeiro para verifiers e solvers, nao para SFT.
4. COT Kaggle/Kienngx/DGXChen so entra apos limpeza de resposta e validacao contra metric/extractor.

### P5 - Treino so depois de prova

Treino novo so e permitido se P1/P2/P4 demonstrarem um pool de dados que:

- ataca `equation_transform`;
- nao degrada `bit_manipulation`;
- tem hash fixo e manifest;
- passa tokenization dry-run;
- passa offset-mask/truncation gates;
- tem weak eval A/B antes de qualquer full eval.

### P6 - Full eval/submission

Full eval so pode rodar se:

- total weak `>=193`;
- equation weak `>=60`;
- bit weak `>=133`;
- truncation `<=3`;
- release gate passa;
- manifest final registra decisao;
- Kaggle submit continua bloqueado ate aprovacao humana.

## Regras de rejeicao

- Nao usar V227: regressao confirmada no V229.
- Nao usar Naribow como adapter weak: truncation `267/315` e score `30/315`.
- Nao usar Konbu COT selection como adapter weak: truncation `161/315` e score `58/315`.
- Nao trocar V226 por Kienngx/DGXChen/Konbu sem novo criterio, pois todos pioram total ou bit/equation.
- Nao treinar com datasets externos sem manifest, hash, dedupe, conflict check e prova de relevancia por familia.

## Proxima tarefa recomendada

Criar um notebook/script V231 CPU-only para minerar:

- `v230_v226_complementarity_equation_miss_pack.csv`;
- `v230_v226_complementarity_baseline_miss_hits.csv`;
- `v230_v226_complementarity_pairwise_detail.csv`.

Saida esperada do V231:

- `equation_miss_taxonomy.csv`;
- `equation_solver_candidate_rules.json`;
- `bit_guardrail_candidates.json`;
- relatorio indicando se ha caminho para recuperar pelo menos `+5` equation sem GPU.
