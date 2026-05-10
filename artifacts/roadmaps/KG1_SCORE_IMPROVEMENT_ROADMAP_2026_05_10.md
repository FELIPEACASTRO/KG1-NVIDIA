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

### Auditoria OpenRouter anexada - 2026-05-10

Arquivos analisados:

- `C:\Users\davis\Downloads\OpenRouter Chat Sun May 10 2026 (1).json`
  - SHA256: `4F87904D23F7988F2CA3F2E2917B7F3355C9F6027FF910C91DDF7D6A50E823BE`
  - Estrutura: JSON valido com `messages`, `items`, `artifacts`, `artifactFiles`, `artifactVersions`, `artifactFileContents`.
  - URLs tecnicas extraidas: `69` Hugging Face, `23` GitHub.
- `C:\Users\davis\Downloads\OpenRouter Chat Sun May 10 2026.json`
  - SHA256: `B8AAB44AF917338956F12C2513AE75F309025BBC56D01EF15D972EF33CA3825E`
  - Estrutura: JSON valido com `messages`, `items`, `artifacts`, `artifactFiles`, `artifactVersions`, `artifactFileContents`.
  - URLs tecnicas extraidas: `146` Kaggle, `204` Hugging Face, `54` GitHub, `35` NVIDIA docs/blogs.

Conclusao para ACC:

- Os anexos reforcam que o alvo principal de ganho e `equation_transform`, nao troca cega de adapter.
- O dado operacional mais importante registrado nos chats e a quebra interna de equation:
  - `equation_transform` weak observado: `55/155`.
  - subfamilia numerica citada: `47/62`.
  - subfamilia simbolica/mista citada: `8/93`.
  - leitura: o gargalo real e simbolico/misto; treinar mais exemplos numericos tem baixa prioridade.
- `bit_manipulation` esta perto do teto local e ja passa gate:
  - weak observado: `135-136/160`, conforme candidato.
  - regra de negocio: qualquer melhoria em equation nao pode reduzir bit abaixo de `136` sem nova evidencia de gate total.
- Nao foi encontrado nos JSONs um peso/adapter pronto com evidencia suficiente para uso direto. Todos os novos modelos/adapters citados entram apenas como candidatos a triagem, nunca como substitutos do V226 sem weak eval.

Fontes externas dos anexos com maior valor potencial para ACC:

| Fonte | Tipo | Valor esperado | Acao |
|---|---|---|---|
| `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces` | HF raw traces | Solver-guided traces de transformation e bit | P0 para V234 ingest/audit |
| `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge` | HF SFT dataset | Distribuicao por familias do desafio | Usar so apos hash, dedupe e conflict check |
| `https://github.com/open-thought/reasoning-gym` | Gerador/benchmark | Geracao controlada de tarefas simbolicas/bit | Usar como fonte de probes, nao treino direto |
| `https://huggingface.co/datasets/Shalyt/ASyMOB-Algebraic_Symbolic_Mathematical_Operations_Benchmark` | HF symbolic math | Casos de operacoes simbolicas | Extrair somente se schema ajudar equation symbolic/mixed |
| `https://huggingface.co/datasets/SAIRfoundation/equational-theories-benchmark` | HF equational reasoning | Leis/equivalencias simbolicas | Usar para verifier/DSL, nao SFT bruto |
| `https://huggingface.co/datasets/nvidia/OpenMathReasoning` | HF math reasoning | Dados gerais de matematica | Baixa prioridade ate provar overlap com equation_transform |
| `https://github.com/TheAlgorithms/Python/tree/master/bit_manipulation` | Algoritmos bit | DSL/guardrail bitvector | Usar para testes e no-loss guardrail |
| `https://huggingface.co/datasets/ftajwar/training_bitwise_arithmetic-4` | HF bit arithmetic | Casos bitwise externos | Apenas triagem; risco de nao bater formato KG1 |
| `https://huggingface.co/datasets/ftajwar/evaluation_bitwise_arithmetic-4` | HF bit arithmetic eval | Probes bitwise | Apenas triagem; nao substituir weak local |

Fontes dos anexos que devem ser tratadas como ruido ou baixo valor para ACC:

- Qualquer resultado de `Huggies`, `faces`, app familiar, imagem, produto ou dataset visual. Isso veio de erro semantico em "Hugging Face" e nao tem relacao com KG1.
- Metadados OpenRouter/model-provider sem artefato reprodutivel local.
- Adapters HF/Kaggle sem `adapter_config.json`, hashes, target modules, tamanho, licenca e weak eval identico.
- Datasets matematicos genericos sem mapeamento para `equation_transform` simbolico/misto.
- COT bruto longo que aumenta truncation; o historico V221 mostrou Naribow e Konbu COT com truncation severo.

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
- Atualizacao OpenRouter 2026-05-10: esta e a fonte externa mais importante para ACC nos anexos, porque contem traces solver-guided alinhados exatamente com as familias problematicas.
- Prioridade V234:
  - baixar com revision fixo;
  - registrar hashes por arquivo;
  - auditar schema;
  - separar `solver_transformation_traces_merged.jsonl` e `solver_bit_manipulation_traces_merged.jsonl`;
  - extrair regras/probes, nao respostas para treino cego;
  - bloquear uso se houver conflito de answer, leakage ou formato nao verificavel.

### Kaggle notebooks e discussoes citados nos anexos

Entram como inteligencia externa, nao como evidencia de score ate reproducao local.

URLs/ids relevantes extraidos dos anexos:

- Competicao oficial: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`.
- Discussao citada como possivel reverse-engineering de familias: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461`.
- Outros ids citados nos chats: `685462`, `689915`.
- Notebooks publicos citados:
  - `https://www.kaggle.com/code/adilshamim8/nvidia-nemotron-model-reasoning-challenge-101`
  - `https://www.kaggle.com/code/adriano313/nemotron-lora-train-v1`
  - `https://www.kaggle.com/code/afidhadra/nemotron-optimized-training`
  - `https://www.kaggle.com/code/damndeepesh/nemotron-sft-lora-with-cot-v2-prep-now-plz-wait`
  - `https://www.kaggle.com/code/dennisfong/nvidia-nemotron-sfttrainer-training`
  - `https://www.kaggle.com/code/emanuellcs/nvidia-nemotron-sft`
  - `https://www.kaggle.com/code/franksunp/nemotron-train-v1`
  - `https://www.kaggle.com/code/halitta/aimo3-nemotron-3-solver-critique-pipeline`
  - `https://www.kaggle.com/code/jek1wantaufik/nvidia-nemotron-model-reasoning-0-68`
  - `https://www.kaggle.com/code/kienngx/nemotron-sft-reasoning-trajectories-dataset`
  - `https://www.kaggle.com/code/kienngx/nvidia-nemotron-trained-models-submission`
  - `https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-copy-run-instantly`
  - `https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot`
  - `https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot-selected-data`

Uso correto:

- Baixar/copiar metadados so se publico e permitido.
- Registrar URL, versao, data, hash e arquivos de saida.
- Procurar apenas:
  - parser de familias;
  - solver/verifier;
  - geracao de traces;
  - calibragem de prompt que reduza truncation;
  - exemplos de failure analysis.
- Nao copiar treino/submission sem auditoria de leakage, licenca, path e weak gate local.

### Repositorios GitHub citados nos anexos

Prioridade de leitura:

- `https://github.com/tonghuikang/nemotron`
- `https://github.com/Ayman-Sabek/NVIDIA_Kaggle_Nemotron`
- `https://github.com/Jerry2003826/nivida`
- `https://github.com/NVIDIA-NeMo/Nemotron`
- `https://github.com/NVIDIA/NeMo-Skills/tree/b37fe403e6dc6e2f9700a64231247a0d1b33d8a2`
- `https://github.com/NVIDIA-NeMo/RL`
- `https://github.com/NVIDIA-NeMo/Evaluator`
- `https://github.com/huggingface/trl`

Uso correto:

- Tratar `tonghuikang`, `Ayman-Sabek` e `Jerry2003826` como candidatos de engenharia reversa/estrategia; nenhum score entra sem reproducao local.
- Tratar NeMo/NVIDIA/TRL como referencia de metodo de SFT/RL/GRPO, nao como caminho imediato de ACC.
- Para o objetivo atual, preferir solver/verifier antes de novo treino RL/SFT.

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
5. Prioridade de ingestao apos auditoria OpenRouter:
   - P0: `andy279/nemotron-reasoning-challenge-raw-traces`.
   - P1: discussoes/notebooks Kaggle com parser/solver/verifier reproduzivel.
   - P2: `reasoning-gym` e benchmarks simbolicos para probes externos.
   - P3: adapters/modelos HF/Kaggle apenas como triagem, nunca como baseline.
6. Para cada fonte externa, registrar `source_url`, `retrieved_at_utc`, `revision_or_version`, `license_or_access_status`, `sha256`, `row_count`, `family_counts`, `schema`, `duplicate_count`, `conflict_count`, `leakage_check`, `extractor_check` e decisao.

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
- Nao usar achado de chat/OpenRouter como evidencia de score sem baixar o artefato original e reproduzir localmente.
- Nao usar dados "Huggies/faces" ou qualquer resultado visual/familiar; isso e ruido da busca por Hugging Face.
- Nao usar COT bruto longo se aumentar truncation; primeiro converter para resposta curta/verificada.
- Nao aceitar ganho em `equation_transform` se `bit_manipulation` cair abaixo do baseline protegido `136/160`, salvo se um weak gate completo provar total `>=193`, equation `>=60`, bit `>=133` e truncation `<=3`.

## Atualizacao executada - V231

V231 foi executado no Colab e terminou com `returncode=0`.

Evidencia da execucao:

- Notebook: `notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb`.
- Manifest V230 resolvido: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity/analysis_v230_v226_complementarity/20260510T070126Z/v230_v226_complementarity_manifest.json`.
- Row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Manifest V231: `/content/drive/MyDrive/KG1_NVIDIA_V231/output_v231_v230_miss_pack_mining/analysis_v231_miss_pack_mining/20260510T074735Z/v231_v230_miss_pack_mining_manifest.json`.
- Manifest SHA256: `ffa21f0f9a0e1845bbe4f55143aed7733d2d3933162775a5ab69d32a783f83b3`.

Resultados V231:

- `baseline_misses=124`.
- `equation_misses=100`.
- `equation_rows_with_correct_alternative=2`.
- `bit_misses=24`.
- `bit_rows_with_correct_alternative=4`.

Decisao V231:

- `mine_equation_solvers_before_training`.
- Proxima acao: construir candidatos de solver/verifier de equation antes de qualquer treino ou full scoring.

## Atualizacao implementada - V232 verified solver workbench

Arquivos criados/alterados para o proximo passo pos-V231:

- `scripts/analyze_v232_verified_solver_workbench.py`
- `scripts/build_v232_verified_solver_workbench_colab.py`
- `notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb`
- `scripts/notebook_release_gate.py`
- `artifacts/notebook_release_gate/v232_verified_solver_workbench_report.json`

URL Colab:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb`

O V232:

- le o manifest V231 mais recente ou o path explicito em `KG1_V232_V231_ANALYSIS_MANIFEST_JSON`;
- valida contrato de rows e artefatos V231 obrigatorios;
- reabre o manifest V230 original para recuperar os prompts completos dos miss-packs;
- gera `equation_solver_workitems_jsonl`;
- gera `bit_guardrail_workitems_jsonl`;
- gera `acceptance_matrix_csv`;
- gera `solver_contracts_json`;
- bloqueia treino, full scoring, package e Kaggle submit.

## Atualizacao executada - V232

V232 foi executado no Colab e terminou com `returncode=0`.

Evidencia da execucao:

- Notebook: `notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb`.
- Manifest V231 resolvido: `/content/drive/MyDrive/KG1_NVIDIA_V231/output_v231_v230_miss_pack_mining/analysis_v231_miss_pack_mining/20260510T074735Z/v231_v230_miss_pack_mining_manifest.json`.
- Row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Manifest V232: `/content/drive/MyDrive/KG1_NVIDIA_V232/output_v232_verified_solver_workbench/analysis_v232_verified_solver_workbench/20260510T080950Z/v232_verified_solver_workbench_manifest.json`.
- Manifest SHA256: `6415efbad28577c675f8847f6b84eb5a2d63709b6b9e5ae42fdba5a002c9b7bf`.

Resultados V232:

- `equation_solver_workitems=100`.
- `equation_rows_with_correct_alternative=2`.
- `bit_guardrail_workitems=24`.
- `bit_rows_with_correct_alternative=4`.

Decisao V232:

- `build_v233_verified_equation_solver_probes`.
- Proxima acao: usar os workitems V232 para implementar probes/verificadores deterministas antes de qualquer treino ou full scoring.

## Atualizacao implementada - V233 verified equation solver probes

Arquivos criados/alterados para o proximo passo pos-V232:

- `scripts/analyze_v233_verified_equation_solver_probes.py`
- `scripts/build_v233_verified_equation_solver_probes_colab.py`
- `notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb`
- `scripts/notebook_release_gate.py`
- `artifacts/notebook_release_gate/v233_verified_equation_solver_probes_report.json`

URL Colab:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V233_VERIFIED_EQUATION_SOLVER_PROBES_COLAB.ipynb`

O V233:

- le o manifest V232 mais recente ou o path explicito em `KG1_V233_V232_ANALYSIS_MANIFEST_JSON`;
- valida artefatos V232 obrigatorios;
- roda probe deployable conservador `sympy_single_equation_probe`;
- registra evidencia nao-deployable separada via `oracle_alternative_candidate_probe`;
- gera `equation_probe_results_jsonl`;
- gera `equation_probe_summary_csv`;
- gera `equation_verified_overrides_csv`;
- gera `equation_oracle_evidence_csv`;
- bloqueia treino, full scoring, package e Kaggle submit.

Decisao operacional agora:

- executar V233;
- se `deployable_verified_equation_overrides >= 5`, preparar notebook separado de weak eval com overrides verificados;
- se ficar abaixo de 5, revisar abstentions e ampliar parsers antes de qualquer weak/full eval;
- manter treino bloqueado ate existir solver/verifier com prova local.

## Atualizacao de prioridade - achados OpenRouter/HF/Kaggle para ACC

Esta atualizacao consolida as ultimas interacoes e altera a prioridade do roadmap:

1. O objetivo de curto prazo nao e novo LoRA. E obter `+5` acertos em `equation_transform` preservando `bit_manipulation`.
2. O caminho mais promissor e uma etapa V234 de ingestao/auditoria de traces externos, com foco em `andy279/nemotron-reasoning-challenge-raw-traces`.
3. O V234 deve produzir artefatos verificaveis, nao treino:
   - manifest de fontes externas;
   - auditoria de schema/hash;
   - tabela de traces por familia;
   - candidatos de regra/solver para symbolic/mixed equation;
   - candidatos de guardrail bitvector;
   - lista de itens rejeitados por ruido, leakage, conflito ou formato.
4. O V234 so pode desbloquear treino se houver evidencia concreta de:
   - ganho potencial em `equation_transform` simbolico/misto;
   - ausencia de degradacao de `bit_manipulation`;
   - compatibilidade com resposta curta `\boxed{answer}`;
   - zero dependencia de label/row-id no deploy.

### Especificacao proposta - V234 external intel ingest

Notebook proposto:

- `notebooks/KG1_V234_EXTERNAL_INTEL_INGEST_AND_SOLVER_TRACE_AUDIT_COLAB.ipynb`

Scripts propostos:

- `scripts/analyze_v234_external_intel_ingest.py`
- `scripts/build_v234_external_intel_ingest_colab.py`

Inputs P0:

- V230 manifest: `/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity/analysis_v230_v226_complementarity/20260510T070126Z/v230_v226_complementarity_manifest.json`.
- V232/V233 manifests se existirem, para cruzar workitems com traces.
- HF gated dataset: `andy279/nemotron-reasoning-challenge-raw-traces`, quando autorizado manualmente.

Outputs obrigatorios:

- `external_source_manifest.json`.
- `raw_trace_file_audit.csv`.
- `equation_symbolic_trace_candidates.jsonl`.
- `bit_guardrail_trace_candidates.jsonl`.
- `trace_to_weak_miss_alignment.csv`.
- `rejected_external_items.csv`.
- `v234_external_intel_ingest_manifest.json`.

Gates obrigatorios:

- falhar se qualquer fonte externa nao tiver hash;
- falhar se row/schema esperado estiver ausente;
- falhar se houver conflito de answer nao resolvido;
- falhar se a fonte exigir acesso manual e nao estiver disponivel;
- falhar se a fonte tentar substituir baseline sem weak eval;
- bloquear train/full/package/submit por padrao.

Decisao esperada do V234:

- `build_solver_from_external_traces` se houver regras/probes concretos para `equation_transform`;
- `external_intel_not_actionable` se os traces forem ruidosos, conflitantes ou sem mapeamento para weak misses;
- `manual_access_required` se o dataset HF gated nao estiver autorizado no ambiente.

## Atualizacao rigorosa - segunda auditoria OpenRouter/Kaggle/HF - 2026-05-10

Escopo executado:

- Reprocessados integralmente os anexos:
  - `OpenRouter Chat Sun May 10 2026 (1).json`: `9,708` linhas, SHA256 `4F87904D23F7988F2CA3F2E2917B7F3355C9F6027FF910C91DDF7D6A50E823BE`.
  - `OpenRouter Chat Sun May 10 2026.json`: `19,091` linhas, SHA256 `B8AAB44AF917338956F12C2513AE75F309025BBC56D01EF15D972EF33CA3825E`.
- JSON parse passou para os dois anexos.
- URLs brutas analisadas: `2,830`.
- Chaves normalizadas unicas: `576`.
- Categorias brutas extraidas:
  - Kaggle: `908` URLs brutas.
  - Hugging Face/HF: `927` URLs brutas.
  - GitHub: `302` URLs brutas.
  - NVIDIA docs/blogs: `123` URLs brutas.
- OpenRouter live API: nao executada porque `OPENROUTER_API_KEY` nao estava presente no ambiente local. As evidencias OpenRouter usadas vieram dos JSONs anexados.
- Kaggle CLI publico: executado sem chave para listagem e pull de notebooks publicos selecionados.
- Hugging Face plugin/API: executado para validar datasets/modelos citados.
- Web search: executado para verificar paginas publicas e paginas indexadas.

Conclusao adicional:

- A primeira atualizacao do roadmap nao estava errada, mas estava incompleta. Faltavam itens publicos do Kaggle CLI e um gap concreto de metric/extractor.
- O achado mais importante para ACC continua sendo solver/verifier, nao novo LoRA.
- Existe um risco real de subcontagem em `equation_transform` se o parser local de `\boxed{}` nao aceitar respostas com `}` literal ou braces aninhados.

### Gap corrigido no codigo local - extractor boxed

Fonte do achado:

- Kaggle kernel: `metric/nvidia-nemotron-metric`.
- O metric notebook trata cada `\boxed{` pegando o conteudo ate o ultimo `}` antes do proximo `\boxed{` ou fim do texto.
- Isso cobre respostas como `\boxed{}52}` para answer `}52` e casos com LaTeX aninhado como `\boxed{\frac{1}{2}}`.

Problema local encontrado:

- `src/competition_utils.py` usava regex `\\boxed\{([^}]*)(?:\}|$)`.
- Esse regex para no primeiro `}` e pode extrair payload errado quando a resposta correta contem brace literal, caso plausivel em `equation_transform` simbolico.

Correcao aplicada:

- `extract_boxed_answers` foi atualizado para seguir o comportamento do metric notebook: varrer todos os starts `\boxed{`, delimitar pelo proximo boxed/fim, e cortar no ultimo `}` do segmento.

Smoke test executado:

- `python -m py_compile src\competition_utils.py`
- Casos validados:
  - `\boxed{42}` -> `42`
  - `\boxed{1} ... \boxed{2}` -> `2`
  - `\boxed{\frac{1}{2}}` -> `\frac{1}{2}`
  - `\boxed{}52}` -> `}52`
  - `\boxed{abc` -> `abc`

Impacto esperado:

- Nao melhora o modelo diretamente, mas reduz risco de avaliacao local divergente da metric publica.
- Deve ser incorporado a qualquer V234/V235 antes de medir ganho em `equation_transform`.

### Novos achados Kaggle CLI que ainda faltavam no roadmap

Kernels publicos mais relevantes por evidencia de titulo, votos, metadata e pull local selecionado:

| Ref | Evidencia | Valor potencial | Decisao |
|---|---:|---|---|
| `huikang/end-to-end-finetuning-for-lb-0-85` | Kaggle CLI; pull local ok | Receita Progress Prize/LB 0.85, mask loss, LoRA, corpus token masks | P0 para leitura metodologica, nao copiar treino |
| `huikang/tinker-submission-notebook` | Kaggle CLI; pull local ok | Submission com `huikang/nemotron-adapter` versions 20/26 e extractor metric-like | P0 para adapter registry/metric parity |
| `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers` | Kaggle CLI; pull local ok | Classifica 6 familias; solvers rule-based para familias faceis; bit/symbol marcados como dificeis | P0 para V234 taxonomy/verifier |
| `optiminist/equation-eda-operator-operation-84-solve-rate` | Kaggle CLI; pull local ok | EDA de equation numeric; hipotese Pre-Op/Mid-Op/Post-Op; 84%/99% em subconjunto reportado | P0 para equation numeric verifier |
| `konbu17/bit-manipulation-solver-cot-generator` | Kaggle CLI; pull local ok | Solver bit por funcao booleana por bit; inclui INHIB/IMPL ausentes em solvers simples | P0 para bit guardrail |
| `johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster` | Kaggle CLI; pull local ok | Pipeline SFT -> GRPO com solvers como verificadores | P2; usar so depois de solver/verifier local |
| `kalyankkr/all-6-puzzle-types-decoded-sft-training-data` | Kaggle CLI; pull local ok | Classificacao, formatos de resposta, dados SFT | P2; nao treinar sem dedupe/leakage |
| `dgxchen/training-with-unsloth-to-achieve-0-85-lb` | Kaggle CLI; pull local ok | Receita Unsloth/LB 0.84-0.85, remove `lm_head`, microbatch/accum | P2; nosso V221 ja mostrou adapter pior em weak local |
| `metric/nvidia-nemotron-metric` | Kaggle CLI; pull local ok | Extractor/verify publico; paridade de metric | P0; ja gerou correcao local |
| `hammadfarooq470/think-twice-self-correcting-reasoning` | Kaggle CLI; pull local ok | Self-correction com adapter Huikang | P3; baixo valor ate provar ganho local |
| `anhtuan299/blackboard-expert-agent-assembly-solving-technique` | Kaggle CLI; pull local ok | Multi-agent/TIR de AIMO3, nao KG1 direto | P3; inspiracao, nao pipeline imediato |

Kernels publicos relevantes ainda nao baixados/analisados profundamente, mas devem entrar na triagem V234:

- `ryanholbrook/nvidia-nemotron-submission-demo`
- `dennisfong/nvidia-nemotron-sfttrainer-training`
- `kienngx/nvidia-nemotron-training-cot-labels`
- `kienngx/nvidia-nemotron-trained-models-submission`
- `asalhi/tinker-adapter-to-ready-to-submit-adapter`
- `huikang/adapter-validation-notebook`
- `kienngx/nvidia-nemotron-training-copy-run-instantly`
- `mayukh18/unsloth-sft-full-data-training`
- `llkh0a/nemotron-unsloth-sft-training-3-30-2`
- `newduck/nvidia-nemotron-soft-balanced-sampling-sft`
- `konbu17/nemotron-tong-style-cot-sft-updated-v2`
- `pearpn25/bit-cot-85-1364-sample`
- `kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset`
- `mohamedamr992/easy-loading-of-nemotron-3`
- `bloodymonday/eda-problem-families`
- `vickymaan/alice-puzzle-solver`

Regra:

- Todo Kaggle kernel entra como inteligencia externa. Nenhum notebook externo vira codigo de producao sem diff review, licenca, hash, teste unitario e weak gate.

### Novos datasets Kaggle/HF que faltavam como candidatos de triagem

Kaggle datasets listados pela API publica:

| Ref | Evidencia | Valor potencial | Decisao |
|---|---:|---|---|
| `kishanvavdara/nemotron-reasoning-traj` | `40.8 MB`, `349` downloads, `25` votes | Reasoning trajectories KG1 | P1 triagem; baixar com hash |
| `kienngx/nemotron-30b-competition-trainingdata-cot-labels` | `3.9 MB`, `1235` downloads, `47` votes | COT + labels de competicao | P1 triagem; alto risco de leakage/overfit |
| `konbu17/bit-manipulation-cot-dataset` | `625 KB`, `70` downloads | Bit CoT | P1 para bit guardrail, nao SFT bruto |
| `konbu17/bit-manipulation-synthetic-cot` | `895 KB`, `58` downloads | Bit synthetic CoT | P1 para solver tests |
| `nctuan/nvidia-nemotron-reasoning-challenge` | `643 KB`, usability `0.94` | Mirror/dataset KG1 | P2, comparar com jasonkung/sebmontreal |
| `mohammedtanvir/nemotron-reasoning-traces` | `12.8 MB`, `26` downloads | Traces | P2 triagem |
| `kevpan096/nemotron-reasoning-competition` | `7.5 MB`, `23` downloads | Competition data | P3; verificar origem |
| `sebmontreal/nvidia-nemotron-model-reasoning-challenge` | `643 KB` | Mirror Kaggle | Baixa prioridade; provavel mirror |
| `harshmali0403/nvidia-nemotron-model-reasoning-challenge` | `643 KB` | Mirror Kaggle | Baixa prioridade; provavel mirror |
| `vsnihal/nvidia-nemotron-model-reasoning-challenge-01` | `643 KB` | Mirror Kaggle | Baixa prioridade; provavel mirror |

Hugging Face datasets/modelos validados pela API:

- `andy279/nemotron-reasoning-challenge`: gated, Apache-2.0, `49,290` train, `1,165` validation, relevante.
- `andy279/nemotron-reasoning-challenge-raw-traces`: gated, Apache-2.0, raw teacher traces, relevante.
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: CSV, `9.5k` train rows + `3` test rows, Apache-2.0, pode servir como mirror/audit de prompt/answer.
- `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2`: grande e generico; baixa prioridade para o gap atual, mas util como contexto de post-training.
- `GaryNENE/nemotron-nano-8b-reasoning-lora`: modelo LoRA 8B, nao compativel diretamente com o base 30B; usar apenas como referencia de receita/dados.
- `AdaptKey/AdaptKey-Nemotron-30b`: gated, telecom LoRA; nao relevante para KG1 ACC.
- `Taurine511/nvidia-nemotron-model-reasoning-challenge`: apareceu na busca web, mas a API HF retornou `not found`; tratar como instavel/deletado ate verificacao manual.
- `justus27/reasoning-gym-bitwise-arithmetic`: rebaixado em 2026-05-10; busca local Kaggle kernels/datasets nao encontrou slug valido. Nao usar ate haver URL verificavel.

### Novos modelos Kaggle a triagem

Kaggle model list publico trouxe candidatos ainda nao suficientemente refletidos no roadmap:

- `kienngx/nemotron-nano-30b-trained`: familia de variacoes treinadas com pipeline Kienngx.
- `atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80`: titulo declara LoRA 0.80; precisa validacao, porque titulo nao e evidencia de weak score local.
- `charancherrychowdary/nemotron-lora-adapter-v1`: adapter LoRA; baixa evidencia.
- `sluitel/nemotron-70b-reasoning-lora`: outro base/modelo; nao compativel direto.
- `nathangaskell/llama-3-1-nemotron-nano-8b`: 8B math-focused; nao compativel direto.
- `metric/nemotron-3-nano-30b-a3b-bf16`: base metric model source ja usado em notebooks Kaggle; manter como referencia de path/metric, nao como novo candidato.

Regra:

- Todo modelo/adaptador novo exige: `adapter_config.json`, tensor count, weight bytes, target_modules, base_model_name_or_path, licenca, hash, weak eval 315 rows, truncation check e no-regression por familia.

### Atualizacao de prioridade V234

O V234 deve ser ampliado alem de `andy279`:

1. Fonte metric/paridade:
   - `metric/nvidia-nemotron-metric`
   - objetivo: garantir extractor/verify equivalentes ao Kaggle.
2. Fonte equation numeric:
   - `optiminist/equation-eda-operator-operation-84-solve-rate`
   - objetivo: implementar/verificar Pre-Op/Mid-Op/Post-Op em weak miss-pack.
3. Fonte bit solver:
   - `konbu17/bit-manipulation-solver-cot-generator`
   - objetivo: testar boolean functions por output bit, incluindo INHIB/Rev-INHIB/IMPL/Rev-IMPL.
4. Fonte taxonomy:
   - `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`
   - objetivo: comparar classificacao de familias e solvers faceis contra nosso classificador.
5. Fonte adapter/corpus de alta evidencia:
   - `huikang/end-to-end-finetuning-for-lb-0-85`
   - `huikang/tinker-submission-notebook`
   - objetivo: extrair criterios de mask loss/corpus/adapter registry, nao copiar treino.
6. Fonte traces/datasets:
   - `andy279/nemotron-reasoning-challenge-raw-traces`
   - `kishanvavdara/nemotron-reasoning-traj`
   - `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
   - objetivo: baixar apenas com hash e usar para regra/verifier, nao SFT bruto.

Nova saida esperada do V234:

- `external_metric_parity_report.json`
- `kaggle_kernel_triage.csv`
- `kaggle_dataset_triage.csv`
- `kaggle_model_triage.csv`
- `equation_numeric_operator_probe_results.csv`
- `bit_boolean_function_probe_results.csv`
- `external_adapter_registry_candidates.csv`

Nova decisao possivel:

- `metric_gap_fixed_continue_to_solver`
- `equation_numeric_probe_promising`
- `bit_guardrail_probe_promising`
- `external_adapter_requires_weak_eval`
- `external_sources_no_actionable_gain`

## Double check OpenRouter e matriz de destino dos achados - 2026-05-10

Registro de auditoria e correcao de reproducibilidade:

- Correcao 2026-05-10: uma versao anterior desta secao afirmava que `OPENROUTER_API_KEY` tinha sido validada localmente e que o endpoint `/api/v1/models` respondeu HTTP 200. O double check atual nao reproduziu essa chamada e outras secoes deste roadmap registram `OPENROUTER_API_KEY` ausente no ambiente local. Portanto, essa afirmacao fica rebaixada para registro historico nao reprodutivel, nao evidencia operacional.
- As evidencias OpenRouter aceitas neste roadmap sao apenas os JSONs anexados pelo usuario e as fontes primarias verificadas separadamente por web/HF/Kaggle CLI.
- Qualquer uso futuro de OpenRouter como evidencia operacional deve salvar em manifest: `prompt_sha256`, `response_sha256`, `model`, `request_id` quando disponivel, `http_status`, `created_at_utc`, `has_api_key=true` e caminho do JSON bruto.
- Ajuste exigido pelo double check: alguns itens estavam no roadmap, mas com caminho de uso implicito demais. A matriz abaixo torna explicito se cada achado ja foi implementado, sera usado no V234/V236, sera triado futuramente, ou fica como baixa prioridade/nao acionavel.

### Matriz implementado agora

| Achado | Uso | Status |
|---|---|---|
| `metric/nvidia-nemotron-metric` | Paridade de extractor/metric publica | Usado agora |
| `extract_boxed_answers` | Corrigir extracao local de `\boxed{}` com braces aninhados/literal `}` | Implementado agora em `src/competition_utils.py` |

Regra:

- Nenhuma medicao nova de `equation_transform` deve ser aceita sem esse extractor corrigido.
- Se um notebook futuro recalcular score usando extractor antigo, o gate deve rejeitar ou registrar `metric_parity_failed`.

### Matriz V234 obrigatoria

Esses itens devem ser usados diretamente no V234, com hash, logs e saidas auditaveis:

| Achado | Uso no V234 | Saida esperada |
|---|---|---|
| `metric/nvidia-nemotron-metric` | Verificar paridade de metric/extractor | `external_metric_parity_report.json` |
| `optiminist/equation-eda-operator-operation-84-solve-rate` | Probar solver numerico de equation por Pre-Op/Mid-Op/Post-Op | `equation_numeric_operator_probe_results.csv` |
| `konbu17/bit-manipulation-solver-cot-generator` | Probar solver bit por funcoes booleanas por output bit | `bit_boolean_function_probe_results.csv` |
| `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers` | Validar taxonomy e solvers por familia | `kaggle_kernel_triage.csv` |
| `huikang/end-to-end-finetuning-for-lb-0-85` | Extrair criterios de treino/mask loss/corpus, sem copiar treino | `kaggle_kernel_triage.csv` |
| `huikang/tinker-submission-notebook` | Extrair adapter registry e metric parity | `external_adapter_registry_candidates.csv` |
| `andy279/nemotron-reasoning-challenge` | Comparar dataset oficial/gated com weak miss-pack quando autorizado | `kaggle_dataset_triage.csv` ou `hf_dataset_triage.csv` |
| `andy279/nemotron-reasoning-challenge-raw-traces` | Minerar traces para regras/verifiers, nao SFT bruto | `kaggle_dataset_triage.csv` ou `hf_dataset_triage.csv` |
| `kishanvavdara/nemotron-reasoning-traj` | Triar trajectories externas contra misses V230 | `kaggle_dataset_triage.csv` |
| `kienngx/nemotron-30b-competition-trainingdata-cot-labels` | Triar CoT/labels com dedupe e leakage guard | `kaggle_dataset_triage.csv` |
| `konbu17/bit-manipulation-cot-dataset` | Gerar probes/fixtures de bit solver | `bit_boolean_function_probe_results.csv` |
| `konbu17/bit-manipulation-synthetic-cot` | Gerar probes/fixtures de bit solver | `bit_boolean_function_probe_results.csv` |
| `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` | Mirror/audit de prompt/answer | `hf_dataset_triage.csv` |

Rebaixado apos double check:

- `justus27/reasoning-gym-bitwise-arithmetic`: nao encontrado via `kaggle kernels list` nem `kaggle datasets list` em busca exata/local. Removido da matriz obrigatoria ate existir URL/slug verificavel.

Gates V234:

- `missing_external_source_hash` se qualquer fonte baixada nao tiver hash registrado.
- `external_source_license_unknown` se licenca nao estiver clara.
- `weak_miss_pack_overlap_missing` se o achado nao mapear para linhas do miss-pack.
- `solver_probe_no_gain` se nao houver ganho mensuravel por familia.
- `bit_guardrail_regression` se bit cair abaixo de `136`.
- `equation_target_not_met` se equation nao ganhar pelo menos `+5` no pacote alvo.

### Matriz de triagem futura obrigatoria

Esses itens nao sao implementacao imediata, mas devem ser catalogados no V234 e decididos por evidencia:

| Achado | Caminho futuro | Criterio de uso |
|---|---|---|
| `johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster` | P2 apos solver/verifier local | Usar somente se verifiers locais ja existirem |
| `kalyankkr/all-6-puzzle-types-decoded-sft-training-data` | P2 taxonomy/SFT format audit | Usar para formato/taxonomy, nao treino bruto |
| `dgxchen/training-with-unsloth-to-achieve-0-85-lb` | P2 receita Unsloth | Nao usar adapter sem novo weak eval, pois V221 local foi pior |
| `hammadfarooq470/think-twice-self-correcting-reasoning` | P3 self-correction | Usar somente se houver ganho local comprovado |
| `anhtuan299/blackboard-expert-agent-assembly-solving-technique` | P3 inspiracao TIR/multi-agent | Nao entra em pipeline KG1 sem prova local |
| `ryanholbrook/nvidia-nemotron-submission-demo` | Catalogar baseline de submissao | Usar so como sanity check de formato |
| `dennisfong/nvidia-nemotron-sfttrainer-training` | Catalogar receita SFT | Exigir hash, licenca e dedupe |
| `kienngx/nvidia-nemotron-training-cot-labels` | Cruzar com dataset Kienngx | Exigir leakage guard |
| `kienngx/nvidia-nemotron-trained-models-submission` | Adapter/model triage | Exigir weak 315 rows local |
| `asalhi/tinker-adapter-to-ready-to-submit-adapter` | Converter/formato adapter | Usar so para compatibilidade, nao score |
| `huikang/adapter-validation-notebook` | Validacao adapter | Incorporar checks uteis ao gate se concretos |
| `kienngx/nvidia-nemotron-training-copy-run-instantly` | Receita replicavel | Baixa prioridade, risco de duplicacao |
| `mayukh18/unsloth-sft-full-data-training` | Receita Unsloth | Exigir dedupe/leakage before use |
| `llkh0a/nemotron-unsloth-sft-training-3-30-2` | Receita Unsloth | Exigir dedupe/leakage before use |
| `newduck/nvidia-nemotron-soft-balanced-sampling-sft` | Balanced sampling | Avaliar se resolve `equation_transform` sem perder bit |
| `konbu17/nemotron-tong-style-cot-sft-updated-v2` | CoT style | Usar so como estilo/verifier, nao treino bruto |
| `pearpn25/bit-cot-85-1364-sample` | Bit CoT sample | Cruzar com bit solver fixtures |
| `kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset` | Solver-verified CoT idea | Usar conceito de verificacao, nao familia direta |
| `mohamedamr992/easy-loading-of-nemotron-3` | Loading reference | Baixo impacto, usar se loader quebrar |
| `bloodymonday/eda-problem-families` | Family EDA | Comparar taxonomy se houver divergencia |
| `vickymaan/alice-puzzle-solver` | Solver reference | Fora das duas familias alvo, baixa prioridade |
| `nctuan/nvidia-nemotron-reasoning-challenge` | Mirror audit | Comparar hash/linhas com mirrors |
| `mohammedtanvir/nemotron-reasoning-traces` | Trace triage | Usar somente se mapear para weak misses |
| `kevpan096/nemotron-reasoning-competition` | Dataset mirror/triage | Baixa prioridade ate origem ser clara |
| `sebmontreal/nvidia-nemotron-model-reasoning-challenge` | Mirror audit | Baixa prioridade, provavel mirror |
| `harshmali0403/nvidia-nemotron-model-reasoning-challenge` | Mirror audit | Baixa prioridade, provavel mirror |
| `vsnihal/nvidia-nemotron-model-reasoning-challenge-01` | Mirror audit | Baixa prioridade, provavel mirror |
| `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` | Contexto generic KD | Nao usar para V234 salvo evidencias KG1 |
| `GaryNENE/nemotron-nano-8b-reasoning-lora` | Receita 8B | Nao compativel direto com 30B |
| `AdaptKey/AdaptKey-Nemotron-30b` | Model/adaptador telecom | Nao relevante para KG1 atual |
| `Taurine511/nvidia-nemotron-model-reasoning-challenge` | Verificar existencia manual | HF API retornou not found; nao usar ate confirmar |

### Matriz de modelos/adapters externos

Nenhum desses entra direto em submissao. Todos exigem weak eval local completo:

| Modelo/adaptador | Caminho futuro | Gate minimo |
|---|---|---|
| `kienngx/nemotron-nano-30b-trained` | Adapter candidate registry | Weak 315 rows + no-regression por familia |
| `atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80` | Adapter candidate registry | Validar se titulo 0.80 reproduz localmente |
| `charancherrychowdary/nemotron-lora-adapter-v1` | Adapter candidate registry | Validar config/tensores/base model |
| `sluitel/nemotron-70b-reasoning-lora` | Nao compativel direto | Usar somente como referencia metodologica |
| `nathangaskell/llama-3-1-nemotron-nano-8b` | Nao compativel direto | Usar somente como referencia metodologica |
| `metric/nemotron-3-nano-30b-a3b-bf16` | Referencia de base/metric path | Nao e novo candidato de score |

### Artefatos obrigatorios para provar uso futuro

O proximo notebook/script que consumir esse roadmap deve produzir:

- `external_metric_parity_report.json`: prova de paridade de extractor.
- `kaggle_kernel_triage.csv`: uma linha por kernel, com status `used_now`, `future_triage`, `rejected`, ou `not_actionable`.
- `kaggle_dataset_triage.csv`: uma linha por dataset Kaggle, com hash, licenca, linhas e decisao.
- `hf_dataset_triage.csv`: uma linha por dataset HF, com gated status, licenca, linhas e decisao.
- `kaggle_model_triage.csv`: uma linha por modelo/adaptador, com base model, config e gate.
- `equation_numeric_operator_probe_results.csv`: resultado dos probes `equation_transform`.
- `bit_boolean_function_probe_results.csv`: resultado dos probes `bit_manipulation`.
- `external_adapter_registry_candidates.csv`: adapters que merecem weak eval.

Decisao de negocio:

- Prioridade maxima: aumentar `equation_transform` de `55` para pelo menos `60` sem reduzir `bit_manipulation` abaixo de `136`.
- O caminho mais promissor e solver/verifier, nao treino bruto.
- Treino novo so deve acontecer depois que V234 provar que os novos dados/regras atacam misses reais, com hash, dedupe e leakage guard.

## V234 implementado - external intel triage executavel

Status:

- Notebook criado: `notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb`.
- Script criado: `scripts/analyze_v234_external_intel_triage.py`.
- Builder criado: `scripts/build_v234_external_intel_triage_colab.py`.
- Gate atualizado: `scripts/notebook_release_gate.py` agora valida o contrato especifico V234.

URL Colab:

- `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb`

O que o V234 faz:

- Confirma que todos os achados do roadmap tem destino explicito.
- Revalida paridade local do extractor `\boxed{}` com os casos criticos da metric publica.
- Materializa os CSVs/JSONs obrigatorios:
  - `external_metric_parity_report.json`
  - `kaggle_kernel_triage.csv`
  - `kaggle_dataset_triage.csv`
  - `hf_dataset_triage.csv`
  - `kaggle_model_triage.csv`
  - `equation_numeric_operator_probe_results.csv`
  - `bit_boolean_function_probe_results.csv`
  - `external_adapter_registry_candidates.csv`
- Bloqueia treino, geracao, scoring, pacote e Kaggle submit.

Resultado do dry run local:

- `coverage.missing_refs=[]`
- `coverage.refs_without_action_path=[]`
- `metric_parity.passed=true`
- Decisao: `external_intel_triage_ready_for_source_download`

Proximo passo depois de executar no Colab:

- Criar o notebook/script de download controlado das fontes com hash, licenca, linha por linha e mapping para miss-pack antes de implementar qualquer solver novo ou avaliar candidatos externos.

## V235 implementado - source access, hash e license triage

Status:

- Notebook criado: `notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb`.
- Script criado: `scripts/analyze_v235_source_access_triage.py`.
- Builder criado: `scripts/build_v235_source_access_triage_colab.py`.
- Gate atualizado: `scripts/notebook_release_gate.py` agora valida o contrato especifico V235.

URL Colab:

- `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb`

O que o V235 faz:

- Consome o manifest V234 executado.
- Valida que V234 passou `coverage` e `metric_parity`.
- Valida todos os CSVs/JSONs V234 obrigatorios.
- Audita acesso a Kaggle/HF sem imprimir segredos.
- Opcionalmente consulta metadata publica Hugging Face para datasets/modelos.
- Materializa:
  - `source_access_inventory.csv`
  - `hf_metadata_audit.csv`
  - `kaggle_access_audit.csv`
  - `source_download_plan.csv`
  - `license_gate_report.json`
- Bloqueia download de payload, treino, geracao, scoring, pacote e Kaggle submit.

Contrato de seguranca:

- Nenhuma fonte externa pode ser ingerida em treino/solver sem `license_status` conhecido e `hash_status` registrado.
- Fontes Kaggle seguem bloqueadas para uso direto ate metadata/licenca/hash serem resolvidos.
- Fontes HF gated exigem token/metadata antes de qualquer payload.

Resultado da execucao no Colab:

- Notebook executado: `notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb`.
- Run ID: `20260510T145150Z`.
- Commit clonado no Colab: `65be3e3b5992cfd841c7a075242f5418950932ec`.
- Manifest V234 consumido: `/content/drive/MyDrive/KG1_NVIDIA_V234/output_v234_external_intel_triage/analysis_v234_external_intel_triage/20260510T143802Z/v234_external_intel_triage_manifest.json`.
- Manifest V235 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V235/output_v235_source_access_triage/analysis_v235_source_access_triage/20260510T145150Z/v235_source_access_triage_manifest.json`.
- Manifest V235 SHA256: `d0bf0eb30bf236da0c08e09805b02c808fc5e793170489cd44a8d3b25c60eaf3`.
- `source_access_inventory.csv`: `51` linhas, SHA256 `eb71efb47fee7bc61a90df2b0ce34371b17ef600896ae95aa92f9c9c26a66396`.
- `hf_metadata_audit.csv`: `8` linhas, SHA256 `b27bcbedbf0e7ee1175b0764891031e3973ff621187b085ebe5fa9205efb0abd`.
- `kaggle_access_audit.csv`: `43` linhas, SHA256 `b9075abb3a0231402ffb6fa5a7a8543d0e51e80054d4c22735e1174a54283544`.
- `source_download_plan.csv`: `46` linhas, SHA256 `858760a2daebc194e6b1d71635463c950d222282525dc436cb18b35e0f2e6732`.
- `license_gate_report.json`: SHA256 `e7f3b187583b0ba000443871820f112e2ffcfba52856761849271ef0b855180c`.
- Resumo por tipo de fonte:
  - `kaggle_kernel`: `27`
  - `kaggle_dataset`: `10`
  - `kaggle_model`: `6`
  - `hf_dataset`: `5`
  - `hf_model`: `3`
- Status das fontes:
  - `v234_required`: `13`
  - `used_now`: `1`
  - `future_triage`: `31`
  - `reference_only`: `4`
  - `manual_verify`: `1`
  - `not_actionable`: `1`
- Metadata HF:
  - HTTP `200`: `7`
  - HTTP `401`: `1`
- Download permitido pelo gate:
  - `true`: `5`
  - `false`: `46`
- Credenciais no runtime:
  - `kaggle_cli_path=/usr/local/bin/kaggle`
  - `kaggle_json_exists=false`
  - `kaggle_username_present=false`
  - `kaggle_key_present=false`
  - `hf_token_present=false`
  - `openrouter_key_present=false`
- Decisao: `manual_source_access_or_license_required_before_download`.
- Motivo: fontes obrigatorias ainda precisam de credenciais, metadata de licenca ou hash de download antes de qualquer payload.

Fontes obrigatorias ainda bloqueadas:

- `metric/nvidia-nemotron-metric`
- `huikang/end-to-end-finetuning-for-lb-0-85`
- `huikang/tinker-submission-notebook`
- `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`
- `optiminist/equation-eda-operator-operation-84-solve-rate`
- `konbu17/bit-manipulation-solver-cot-generator`
- `kishanvavdara/nemotron-reasoning-traj`
- `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
- `konbu17/bit-manipulation-cot-dataset`
- `konbu17/bit-manipulation-synthetic-cot`

Proximo passo depois de executar no Colab:

- Se o V235 decidir `manual_source_access_or_license_required_before_download`, resolver credenciais/licencas primeiro.
- Se decidir `source_access_plan_ready_needs_controlled_download`, criar o downloader V236 que baixa apenas fontes permitidas, registra hash, licenca, row counts e mapping para miss-pack.

## Deep dive de literatura para `bit_manipulation` e `equation_transform` - 2026-05-10

Familias da imagem/weak gate:

| Familia | Linhas avaliadas | Corretas | ACC | Status |
|---|---:|---:|---:|---|
| `bit_manipulation` | 160 | 136 | 85.00% | medido |
| `equation_transform` | 155 | 55 | 35.48% | medido |
| `OVERALL weak` | 315 | 191 | 60.63% | medido |

Leitura de negocio:

- `bit_manipulation` ja passa o gate minimo de `133/160`, mas a margem real e pequena. Deve virar guardrail, nao foco principal de treino.
- `equation_transform` e o gargalo: precisa sair de `55/155` para pelo menos `60/155`.
- Regra operacional: qualquer melhoria em `equation_transform` deve preservar `bit_manipulation>=136/160`, salvo se o weak completo provar `total>=193`, `equation>=60`, `bit>=133`, `trunc<=3`.
- Hipotese mais forte ja registrada: `equation_transform` numerico esta muito melhor que symbolic/mixed; o ganho de ACC deve mirar symbolic/mixed e operadores customizados, nao apenas aritmetica comum.

### Fontes publicas e literatura tecnica consultadas

Bit-vectors e bit manipulation:

- Z3 Bitvectors: `https://microsoft.github.io/z3guide/docs/theories/Bitvectors/`
  - Evidencia: Z3 modela semantica precisa de bit-vectors de tamanho fixo, com aritmetica signed/unsigned, literais binarios/hex e operacoes bitwise.
  - Uso no KG1: representar cada entrada de 8 bits como `BitVec(8)` e validar regras candidatas sobre todos os exemplos do prompt.
- Z3 BitVector API: `https://z3prover.github.io/api/html/ml/Z3.BitVector.html`
  - Evidencia: API tem AND, OR, XOR, NOT, shifts, rotates, concat, extract, carry e xor3.
  - Uso no KG1: cobrir exatamente o vocabulario do enunciado `shift`, `rotate`, `XOR`, `AND`, `OR`, `NOT`, majority/choice via boolean formulas.
- Component-based synthesis applied to bitvector circuits: `https://www.microsoft.com/en-us/research/publication/component-based-synthesis-applied-to-bitvector-circuits/`
  - Evidencia: sintese de programas bitvector por biblioteca de componentes + constraints + SMT/CEGIS e adequada para composicoes nao intuitivas de operacoes bitvector.
  - Uso no KG1: trocar tentativa por LLM por enumerador/CEGIS de DSL pequena, com ranking por simplicidade e verificacao total dos exemplos.
- SyGuS: `https://www.microsoft.com/en-us/research/publication/syntax-guided-synthesis-2/`
  - Evidencia: problema combina especificacao semantica, gramatica de candidatos e CEGIS.
  - Uso no KG1: transformar exemplos `input -> output` em problema PBE-BV; gramatica limitada evita overfit e explosao.
- Reasoning Gym: `https://github.com/open-thought/reasoning-gym` e `https://arxiv.org/abs/2505.24760`
  - Evidencia: geradores procedurais e verificadores com reward verificavel; inclui scoring em cascata e dominios de algebra/arithmetic/computation/logic.
  - Uso no KG1: fonte de fixtures/probes e verifiers, nao treino bruto; especialmente `bitwise_arithmetic` para guardrails.

Equation transform, simbolico e regra por operador:

- SymPy solving: `https://docs.sympy.org/latest/guides/solving/index.html`
  - Evidencia: SymPy resolve equacoes simbolicas, sistemas, inequacoes, diofantinas e tambem numericamente.
  - Uso no KG1: apenas para subclasse com equacao algebraica clara e variavel unica; deve abstain em prompt ambiguidade.
- SymPy simplify: `https://docs.sympy.org/latest/modules/simplify/simplify.html`
  - Evidencia: `simplify()` nao e uma operacao bem definida; docs recomendam usar funcoes especificas quando o algoritmo depende de uma transformacao concreta.
  - Uso no KG1: nao usar `simplify()` generico como solver final; usar `factor`, `expand`, `cancel`, `together`, `solve`, `solveset`, `nsimplify` com contratos especificos.
- egg / equality saturation: `https://arxiv.org/abs/2004.03082`
  - Evidencia: e-graphs representam muitas expressoes equivalentes e sao usados em otimizacao, reescrita e program synthesis.
  - Uso no KG1: para symbolic/mixed, inferir e validar transformacoes por regras, sem comprometer cedo com uma sequencia de reescritas.
- Rewrite Rule Inference Using Equality Saturation: `https://arxiv.org/abs/2108.10436`
  - Evidencia: e-graphs podem ajudar a inferir regras menores e mais gerais a partir de enumeracao de termos.
  - Uso no KG1: minerar regras de `equation_transform` a partir dos exemplos do prompt e dos traces, depois aceitar somente se todos os exemplos forem satisfeitos.

Fontes KG1/Kaggle/HF relevantes localizadas:

- HF `andy279/nemotron-reasoning-challenge`
  - Gated, Apache-2.0, SFT KG1; `49,290` train, `1,165` validation, inclui `399` transformation unsolved.
  - Uso: P0 apos aceite manual/licenca/hash; nao SFT bruto antes de dedupe/conflict/leakage.
- HF `andy279/nemotron-reasoning-challenge-raw-traces`
  - Gated, Apache-2.0; inclui solver-guided transformation e bit traces.
  - Uso: P0 para extrair regras/probes e nao respostas cegas.
- Kaggle `kishanvavdara/nemotron-reasoning-traj`
  - CLI listou `40764029` bytes, `349` downloads, `25` votos.
  - Busca web indicou `9,500` rows, `bit_manipulation=1,602`, `equation_symbolic=823`, `equation_numeric=732`.
  - Uso: error analysis e candidate traces apos hash/licenca.
- Kaggle `optiminist/equation-eda-operator-operation-84-solve-rate`
  - CLI confirmou kernel com `25` votos.
  - Uso: P0 para decompor equation por operador/operacao; nao assumir `84%` sem reproduzir localmente.
- Kaggle `konbu17/bit-manipulation-solver-cot-generator`
  - CLI confirmou kernel com `39` votos.
  - Uso: P0 para DSL bitvector/boolean por bit; nao usar COT longo como saida final.
- Kaggle `konbu17/bit-manipulation-cot-dataset`
  - CLI listou `624941` bytes, `70` downloads, `5` votos.
  - Uso: fixtures e testes de bit guardrail, nao treino direto.
- Kaggle `konbu17/bit-manipulation-synthetic-cot`
  - CLI listou `895101` bytes, `58` downloads, `4` votos.
  - Uso: probes de solver bit, nao treino direto.
- Kaggle Huikang:
  - `huikang/end-to-end-finetuning-for-lb-0-85`: `257` votos.
  - `huikang/tinker-submission-notebook`: `454` votos.
  - `huikang/adapter-validation-notebook`: `272` votos.
  - Uso: metodo/validacao/adapter registry; nenhum score entra sem weak eval local.

### Plano tecnico para melhorar `bit_manipulation`

Objetivo: manter `136/160` ou melhorar sem tocar no baseline quando o solver nao tiver prova.

Implementacao recomendada:

1. Parser robusto de prompt:
   - Extrair todos os pares `8-bit -> 8-bit`.
   - Confirmar largura fixa `8`.
   - Extrair target.
   - Abort se houver qualquer par malformado, largura diferente ou resposta fora de `[01]{8}`.
2. DSL pequena e verificavel:
   - Termos atomicos: `x`, `~x`, `shl(x,k)`, `lshr(x,k)`, `rotl(x,k)`, `rotr(x,k)`, constantes/masks para `k in 0..7`.
   - Combinadores binarios: `AND`, `OR`, `XOR`, `NAND`, `NOR`, `XNOR`.
   - Combinadores ternarios: `majority(a,b,c)`, `choice(a,b,c)`, `xor3(a,b,c)`.
   - Pos-processamento permitido: identidade, NOT, XOR mask, reverse bits, optional zero/sign handling se aparecer no prompt.
3. Busca:
   - Primeiro enumeracao direta ate profundidade baixa.
   - Depois CEGIS/SMT se houver multiplas hipoteses ou regra mais profunda.
   - Ranking por menor AST, menor numero de constantes, menor numero de operacoes exoticas.
4. Aceitacao:
   - Uma regra so pode sobrescrever baseline se acertar 100% dos exemplos do prompt.
   - Se houver duas regras com outputs diferentes para o target e mesmo score nos exemplos, abstain.
   - Se o baseline ja estava correto, nao sobrescrever sem prova unica e output igual.
5. Uso esperado:
   - `bit_manipulation` e guardrail: usar solver para detectar regressao e recuperar poucos misses, nao como roteador agressivo.

### Plano tecnico para melhorar `equation_transform`

Objetivo: ganhar pelo menos `+5` em `equation_transform`, com foco em symbolic/mixed.

Classes a separar antes de qualquer treino:

1. Numeric operator transform:
   - Exemplos do tipo `AA op BB = OUT`.
   - Candidatos: aritmetica direta, aritmetica com reversao de operandos, reversao de resultado, concatenacao, zero padding, sinal customizado, modulo/base, soma/subtracao por digito, produto por digito.
   - Verificar por operador: regras de `+`, `-`, `*`, `/`, `%`, `?`, `@`, `&` podem ser independentes.
2. Symbolic/mixed token transform:
   - Operandos e saidas podem conter simbolos, aspas, pipes e caracteres nao alfanumericos.
   - Candidatos: permutacao de tokens, reversao por lado, concatenacao esquerda/direita, substituicao por tabela, shift em alfabeto observado, operador como selector.
   - Nao tentar SymPy nesse subtipo.
3. Algebraic equation:
   - Usar SymPy somente quando houver uma equacao algebrica unica e variavel unica.
   - Exigir substituicao verificada e resposta normalizada.
4. Constraint/cryptarithm:
   - Usar DFS/Z3 sobre digitos ou simbolos com unicidade se o prompt indicar mapeamento.
   - Aceitar apenas solucao unica.

Aceitacao de override:

- A regra deve explicar todos os exemplos do prompt.
- A regra deve gerar target unico.
- O output deve passar extractor `\boxed{}` corrigido, preservando zeros a esquerda e simbolos literais.
- Se a resposta envolve braces, pipes, aspas ou caracteres especiais, usar extractor balanceado, nao regex simples.
- Se os exemplos por operador forem poucos e houver multiplas regras, abstain.

### Prompt OpenRouter recomendado se a chave for disponibilizada

OpenRouter nao foi chamado neste runtime porque `OPENROUTER_API_KEY` estava ausente. Se a chave estiver disponivel em ambiente seguro, usar o prompt abaixo em modelos fortes e pedir saida JSON, nunca decisao livre:

```text
You are auditing a Kaggle NVIDIA Nemotron KG1 solver plan. Do not speculate.
Use only evidence from the provided weak rows, prompt examples, known public sources,
and formal methods literature. We need improve equation_transform from 55/155 to >=60
while preserving bit_manipulation >=136/160.

Tasks:
1. Classify each equation_transform miss into numeric operator, symbolic/mixed token transform,
   algebraic equation, cryptarithm/constraint, extractor issue, or unknown.
2. For each class, propose only deterministic solvers with acceptance predicates.
3. For bit_manipulation, propose a finite 8-bit DSL and ambiguity/abstention checks.
4. Return JSON with: class, evidence, proposed_solver, required_inputs, acceptance_tests,
   regression_risks, expected_gain_upper_bound, and reasons_to_abstain.
5. Do not claim any ACC gain unless it is computable from supplied rows.
```

### Roadmap executavel apos esta revisao

P0 sem novas credenciais:

1. Evoluir V233 para `V236_LOCAL_SOLVER_DSL_PROBES` usando apenas miss-packs e dados locais ja versionados.
2. Implementar `bitvector_dsl_probe` como guardrail com abstention.
3. Implementar `equation_operator_dsl_probe` com regras por operador e verificacao total dos exemplos.
4. Implementar `symbolic_token_transform_probe` para permutation/concat/reverse/substitution em `equation_transform`.
5. Rodar contra `v230_v226_complementarity_equation_miss_pack.csv` e `bit_miss_pack.csv`.
6. Promover para eval somente se houver `>=5` overrides deployable em equation e zero regressao bit.

P0 com acao humana:

1. Configurar credenciais Kaggle/HF no Colab sem imprimir segredo.
2. Resolver licenca/metadata/hash das fontes bloqueadas no V235.
3. Criar V236 downloader apenas se `license_gate.direct_ingestion_allowed=true` para as fontes requeridas.
4. Baixar primeiro:
   - `metric/nvidia-nemotron-metric`
   - `optiminist/equation-eda-operator-operation-84-solve-rate`
   - `konbu17/bit-manipulation-solver-cot-generator`
   - `andy279/nemotron-reasoning-challenge-raw-traces`
   - `kishanvavdara/nemotron-reasoning-traj`
5. Gerar hash, row count, schema, family counts, dedupe, conflict check e leakage guard antes de qualquer ingestao.

Nao fazer:

- Nao trocar adapter pelo titulo `LB 0.85` sem weak eval identico.
- Nao treinar em COT bruto longo que historicamente elevou truncation.
- Nao usar dados externos sem licenca/hash.
- Nao aceitar `equation_transform` medido por extractor regex simples.
- Nao reduzir `bit_manipulation` abaixo de `136/160` em nome de ganho hipotetico.

## Auditoria do anexo OpenRouter `Sun May 10 2026 (2)` - 2026-05-10

Arquivo auditado:

- `C:\Users\davis\Downloads\OpenRouter Chat Sun May 10 2026 (2).json`
- SHA256: `F705A612BB848A8588F99826CF7DC4781822DAB0460EEF04B50A9FEE75D8DFC7`
- Tamanho: `721917` bytes.

Leitura de confiabilidade:

- O anexo contem respostas de multiplos modelos. Algumas respostas declaram acesso real a busca; outras declaram explicitamente ausencia de internet. Portanto, nada do anexo deve ser tratado como fato ate ser verificado em fonte primaria.
- Validacao externa feita nesta revisao confirmou que os achados mais fortes sao fontes ja acionaveis para engenharia, nao prova de ganho direto de ACC.
- Nao ha no anexo um adapter pronto que possa ser promovido com seguranca sem weak eval identico, licenca, hash e auditoria de leakage.

Achados confirmados e uteis para ACC:

1. `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`
   - URL: `https://huggingface.co/datasets/jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`
   - Evidencia verificada: dataset HF em CSV, `apache-2.0`, cerca de `9.5k` linhas; viewer mostra prompts reais com assinatura de `bit_manipulation` e `equation_transform`.
   - Uso correto: usar como referencia de schema/prompt signatures e gerador de testes de parser. Nao usar para treino ou avaliacao ate resolver risco de overlap/leakage com weak/test.
   - Sinal tecnico: `bit_manipulation` aparece como transformacao de numeros binarios de 8 bits com shifts, rotations, XOR, AND, OR, NOT, majority e choice. `equation_transform` aparece como regras sobre simbolos e caracteres especiais, reforcando que SymPy nao e a rota principal para symbolic/mixed.

2. `andy279/nemotron-reasoning-challenge`
   - URL: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`
   - Evidencia verificada: dataset HF gated, `apache-2.0`, SFT data com `train=49290` e `validation=1165`; a propria card declara traces de teacher models e solvers.
   - Sinais relevantes: `Solver-guided transformation=1101`, `Solver-guided bit manipulation=1602`, `GPT-5.4 transformation=85`.
   - Uso correto: P0/P1 para minerar regras e acceptance predicates; nao usar COT bruto como SFT sem auditoria, porque o risco de truncation/formato/leakage e alto.

3. `andy279/nemotron-reasoning-challenge-raw-traces`
   - URL: `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`
   - Evidencia verificada: dataset HF gated, `apache-2.0`, com arquivos:
     - `solver_transformation_traces_merged.jsonl`
     - `solver_bit_manipulation_traces_merged.jsonl`
     - `solver_transformation_traces_gpt54.jsonl`
   - Uso correto: fonte mais forte para extrair DSLs, regras por subtipo e criterios de abstencao. Deve passar por downloader com hash, row count, schema, family count, dedupe, conflito e leakage guard.

4. `tonghuikang/nemotron`
   - URL: `https://github.com/tonghuikang/nemotron`
   - Evidencia verificada: repositorio do Progress Prize para NVIDIA Nemotron Model Reasoning Challenge; README aponta writeup/notebook Kaggle e contem pastas como `reasoners`, `problems`, `investigators`, `training/sft`, `corpus`, `metrics`.
   - Uso correto: estudar engenharia de corpus, min-logprob, reasoners e tabulacoes de problemas. Nao usar adapter, treino ou codigo sem licenca, hash e weak eval isolado.

Consenso tecnico util do anexo:

- `equation_transform` deve ser dividido antes de qualquer novo treino:
  - numeric operator transform;
  - symbolic/mixed token rewrite;
  - sequence/token transform;
  - algebraic equation;
  - cryptarithm/constraint-like.
- O ganho de curto prazo mais plausivel continua sendo `+5` por solver/router conservador em `equation_transform`, nao por troca cega de LoRA.
- O solver so deve sobrescrever o modelo se:
  - parseou todos os exemplos do prompt;
  - encontrou uma regra unica;
  - a regra explica todos os exemplos;
  - preserva zeros a esquerda e caracteres especiais;
  - o extractor balanceado consegue serializar `\boxed{...}` sem cortar braces, pipes, aspas ou barras.
- `bit_manipulation` deve ficar como guardrail. Qualquer DSL/Z3/CEGIS deve atuar primeiro como validador e detector de regressao. So pode virar override se provar zero regressao contra os `160` exemplos weak e manter pelo menos `136/160`.

Descartes/ajustes feitos a partir do anexo:

- Descartar afirmacoes de ganho garantido como `+3`, `+5` ou `90% probabilidade` sem medicao local.
- Descartar rollback com limite `bit<134`; o guardrail operacional correto e baseline protegido `bit>=136/160`, salvo weak gate completo provando `total>=193`, `equation>=60`, `bit>=133`, `trunc<=3`.
- Nao usar SymPy para symbolic/mixed. SymPy fica restrito a algebra clara, variavel unica, solucao unica e substituicao verificada.
- Nao tratar `cryptarithm` como subtipo dominante sem evidencia local. Implementar apenas se o classificador achar prompts com restricoes explicitas de mapeamento/aritmetica.
- Nao aceitar adapters HF/Kaggle por popularidade, LB title ou progress-prize label sem reproduzir em weak identico.

Atualizacao P0 para V236:

1. Criar `V236_LOCAL_SOLVER_DSL_PROBES` com tres saidas obrigatorias:
   - `equation_subtype_audit.csv`;
   - `equation_solver_probe_results.csv`;
   - `bit_guardrail_probe_results.csv`.
2. O subtipo `symbolic/mixed token rewrite` deve ser P0, porque o anexo e o HF viewer reforcam que ha muitos caracteres nao algebricos em `equation_transform`.
3. O subtipo `numeric operator transform` deve testar no minimo:
   - operacao direta;
   - reversao de operandos;
   - reversao de resultado;
   - concatenacao;
   - aritmetica por digito;
   - base/modulo;
   - operador remapeado por simbolo.
4. O `bitvector_dsl_probe` deve incluir exatamente as operacoes do prompt publico: shifts, rotations, XOR, AND, OR, NOT, majority e choice.
5. Promocao bloqueada ate haver evidencias locais:
   - `equation_transform >= 60/155`;
   - `bit_manipulation >= 136/160` preferencialmente, ou `>=133/160` apenas se o gate total completo passar;
   - `truncated <= 3`;
   - nenhum ganho medido com extractor regex simples.

## V236 implementado - local solver DSL probes

Implementacao adicionada em 2026-05-10:

- Script: `scripts/analyze_v236_local_solver_dsl_probes.py`.
- Builder: `scripts/build_v236_local_solver_dsl_probes_colab.py`.
- Notebook: `notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`.

Escopo:

- CPU-only.
- Consome o manifest V232 e os workitems:
  - `equation_solver_workitems_jsonl`;
  - `bit_guardrail_workitems_jsonl`;
  - `acceptance_matrix_csv`;
  - `solver_contracts_json`.
- Nao treina, nao gera modelo, nao roda scoring completo, nao empacota, nao baixa payload externo e nao submete ao Kaggle.

Saidas obrigatorias:

- `v236_local_solver_dsl_probes_equation_subtype_audit.csv`;
- `v236_local_solver_dsl_probes_equation_solver_probe_results.csv`;
- `v236_local_solver_dsl_probes_bit_guardrail_probe_results.csv`;
- `v236_local_solver_dsl_probes_equation_probe_summary.csv`;
- `v236_local_solver_dsl_probes_manifest.json`.

Probes iniciais:

- `symbolic_char_map_probe`: deployable apenas quando todos os exemplos tem mesmo comprimento, mapeamento caractere-a-caractere consistente, query totalmente coberta e predicao unica.
- `reverse_token_probe`: deployable apenas quando todos os exemplos provam reversao simples.
- `numeric_operator_dsl_probe`: testa somente regras conservadoras `direct_arithmetic`, `reverse_result_arithmetic`, `reverse_operands_arithmetic`, `digitwise_add_mod10`; abstain em ambiguidade.
- `bitvector_prompt_signature_guardrail`: nao sobrescreve resposta; apenas confirma assinatura de prompt bitvector e escopo de operadores permitidos para o guardrail.

Validacoes ja executadas localmente:

- `python -m py_compile scripts/analyze_v236_local_solver_dsl_probes.py`
- `python scripts/analyze_v236_local_solver_dsl_probes.py --self-test`
- `python -m py_compile scripts/build_v236_local_solver_dsl_probes_colab.py`
- `python scripts/build_v236_local_solver_dsl_probes_colab.py`
- `python scripts/notebook_release_gate.py notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`

Resultado do gate:

- `ok=true`
- notebook SHA256: `be80f4ca59097b6aa964e734cfc8186dc1b922df99d1062640451c5ea13731ee`

Proximo passo:

- Executar o V236 no Colab.
- Se `deployable_verified_equation_overrides >= 5` e `bit_guardrail_signature_verified_rows` cobrir os workitems sem override incorreto, criar apenas entao um notebook separado de rescue measurement.
- Se V236 decidir `continue_local_solver_development`, abrir `equation_subtype_audit.csv` e expandir somente os subtipos com parser exato. Nenhuma avaliacao ou pacote deve ser feito antes disso.
