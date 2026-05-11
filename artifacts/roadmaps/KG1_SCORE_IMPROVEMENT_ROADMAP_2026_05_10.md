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
- Atualizacao HF-only 2026-05-11: o smoke V257/V249 em H200 produziu o melhor resultado operacional ate agora no contrato V221: `v257_checkpoint_4_v221_contract` com `192/315`, equation `56/155`, bit `136/160`, truncation `1`. Ainda nao passa o gate (`193/315`, equation `60`), mas prova ganho real de `+1` em bit sem perder equation frente ao V256 HF.
- Atualizacao V259/V260B 2026-05-11: o treino equation-focused a partir do V257 checkpoint-4 repetiu `192/315`, equation `56/155`, bit `136/160`, truncation `0` no melhor checkpoint. Ele reduziu truncation, mas nao aumentou `equation_transform`; portanto nao justifica continuacao longa em H200 sem novo dado/verifier.
- Atualizacao V261 2026-05-11: a varredura operacional `thinking on + no prompt suffix` foi cortada cedo no H200 por gate de FinOps. O primeiro candidato (`v259_checkpoint4_nosuffix`) regrediu para `155/315`, equation `55/155`, bit `100/160`, truncation `1`; sem ganho em equation e com queda severa em bit. Esta familia de prompt esta descartada para novos gastos.
- Atualizacao V262/V263 2026-05-11: adapter soups entre V226/V257/V259 foram gerados em CPU e avaliados no H200. Melhor soup (`soup_v226_050_v257_050`) ficou em `192/315`, equation `56/155`, bit `136/160`, truncation `1`; os outros regrediram para `191` e `190`. Adapter soup nao resolveu o gargalo e nao deve consumir novo H200 sem um preflight que prove alvo novo em equation.
- Atualizacao V264 2026-05-11: recheck HF CPU confirmou que os traces P0 `andy279/*` continuam bloqueados por review/termos (`403`). A rota mais promissora agora precisa de acao humana para liberar esses datasets no HF; o mirror publico sozinho ja foi usado e nao entregou equation suficiente.
- Auditoria Google Drive 2026-05-10: `1879` arquivos KG1 catalogados, `85` adapters completos, `232` reports, `423` CSVs, `54` JSONLs e `11` notebooks. Nenhum artefato do Drive supera o baseline V226 sob gate weak canonico; o Drive deve ser usado como fonte de pesos fortes conhecidos, reports e dados para triagem, nao como fonte de promocao automatica.
- Achado Drive mais util: V207A full/validation gate do V194 tem `822/947` com familias nao criticas em `100%`, mas confirma o mesmo gargalo fraco: `bit_manipulation=135/160`, `equation_transform=55/155`. Isso reforca que o problema real continua concentrado em `equation_transform`.
- Importante: muitos arquivos do Drive foram parte da trajetoria que chegou ao score amplo `0.86`. Esse score e valido como evidencia historica de que V194/V202D resolvia muito bem as familias nao criticas, mas nao pode ser interpretado como melhoria atual das duas familias alvo. No recorte decisivo, o proprio V207A mede `equation_transform=55/155` e `bit_manipulation=135/160`, alinhado ao gargalo V230.

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

## V236 executado - resultado e ajuste pos-log

Execucao analisada em 2026-05-10:

- Notebook executado: `notebooks/KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB.ipynb`.
- Commit Colab observado: `69d2d9725ae194e087d9d37717967eacd3a382df`.
- Manifest V236 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V236/output_v236_local_solver_dsl_probes/analysis_v236_local_solver_dsl_probes/20260510T154035Z/v236_local_solver_dsl_probes_manifest.json`.
- Manifest SHA256 observado: `cbdd10850a741941084b1ef135c2ec3bb31bfe5429ee68bd12cb8f7163f37285`.
- Final manifest SHA256 observado: `3ef90c3cb97adfbde7c52bdb7b4ca0742336ed8ff433f8ed22ba4b8ec7e80a5e`.

Resultados medidos:

- `equation_workitems`: `100`.
- `bit_guardrail_workitems`: `24`.
- `deployable_verified_equation_overrides`: `0`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24`.
- Decisao: `continue_local_solver_development`.

Distribuicao dos subtipos observados:

- `algebraic_equation`: `69`.
- `symbolic_mixed_token_rewrite`: `22`.
- `numeric_operator_transform`: `9`.

Interpretacao:

- A execucao foi limpa, sem erro de runtime, artefato ausente ou quebra de contrato V232.
- O guardrail de `bit_manipulation` esta pronto como verificador de escopo, mas nao emite override.
- O gargalo permanece `equation_transform`: os probes iniciais foram seguros demais e abstiveram em todos os `100` workitems.
- Gap encontrado no script: havia classificacao para `algebraic_equation`, mas nao havia probe algebraico; esses itens caiam no probe simbolico e abstinham.

Ajuste aplicado apos a analise:

- `scripts/analyze_v236_local_solver_dsl_probes.py` agora inclui `sympy_single_equation_probe`, restrito a:
  - equacao unica;
  - uma variavel alfabetica;
  - pelo menos um digito;
  - caracteres algebraicos seguros;
  - solucao unica;
  - verificacao por substituicao simbolica.
- O script agora tambem gera `equation_abstain_reason_summary_csv`, para agrupar motivos concretos de abstain por subtipo/probe/prova.
- Nenhuma regra nova autoriza treino, full eval, pacote ou submissao. O gate continua exigindo ganho local medido antes de qualquer rescue measurement.

Proximo passo revisado:

- Reexecutar o V236 atualizado no Colab.
- Se ainda houver `0` overrides, usar `equation_abstain_reason_summary_csv` para decidir o V237 por evidencia, provavelmente focado em:
  - parser de numeric operator expandido;
  - parser de symbolic/mixed com mapeamento de token, nao so caractere;
  - identificacao de casos onde exemplos sao insuficientes e devem permanecer abstain.
- Criar notebook de rescue eval somente se `deployable_verified_equation_overrides >= 5`, `deployable_incorrect_equation_overrides == 0` e o guardrail bit continuar completo.

## V236 reexecutado - diagnostico apos probe SymPy

Execucao analisada em 2026-05-10 a partir de `KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB (1).ipynb`:

- Commit Colab observado: `45592f00e669d32077628a13a001bc4d7e5ccbf1`.
- Manifest V236 SHA256 observado: `ae615978b1eef3e2d326c450bd20cdac5bd383457aa5af0277817d7d64ebc5a8`.
- Final manifest SHA256 observado: `55fc968a18ef6bf8bfb8ffaff06913e8a4d1a70e759a1b3cb8dcdbb2e4dfc803`.
- `equation_workitems`: `100`.
- `bit_guardrail_workitems`: `24`.
- `deployable_verified_equation_overrides`: `0`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24`.
- Decisao: `continue_local_solver_development`.

Diagnostico novo trazido por `equation_abstain_reason_summary`:

- `69` linhas em `algebraic_equation` abstiveram como `missing_examples_or_query` porque o fallback simbolico ainda escondia o motivo real do probe algebraico.
- `9` linhas em `numeric_operator_transform` abstiveram como `numeric_examples_or_query_not_parseable`.
- `22` linhas em `symbolic_mixed_token_rewrite` abstiveram como `example_length_mismatch`.

Ajuste adicional aplicado apos esse log:

- `algebraic_equation` agora retorna sempre o resultado do `sympy_single_equation_probe`, mesmo quando abstain.
- Assinaturas algebricas que nao passam no parser seguro agora sao classificadas como `algebraic_equation_unparsed`.
- Objetivo: o proximo replay deve revelar o motivo real dos `69` itens, em vez de registrar `symbolic_char_map_probe/missing_examples_or_query`.

Proximo passo:

- Reexecutar V236 mais uma vez para obter `equation_abstain_reason_summary` corrigido.
- Se os `69` itens forem majoritariamente `no_single_algebraic_equation_found`, o V237 deve priorizar parser do formato real do prompt antes de qualquer solver novo.
- Se houver equacoes parseaveis com falha SymPy especifica, o V237 deve atacar somente essas falhas com testes unitarios antes de medir rescue.

## V236 terceira execucao - V237 desbloqueado como auditoria

Execucao analisada em 2026-05-10 a partir de `KG1_V236_LOCAL_SOLVER_DSL_PROBES_COLAB (2).ipynb`:

- Commit Colab observado: `9b03e9eef5f1f83e31195602ecfd9a97777456d8`.
- Manifest V236 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V236/output_v236_local_solver_dsl_probes/analysis_v236_local_solver_dsl_probes/20260510T155704Z/v236_local_solver_dsl_probes_manifest.json`.
- Manifest V236 SHA256 observado: `ef049b79eeedea09b839147fb1c2d0429a79a03948515b5b765706c714f62e9c`.
- Final manifest SHA256 observado: `de0d96d879c7ded3560b5d8d73b3aac65f8632637d337edf2c1c574e23d36c5d`.
- `equation_workitems`: `100`.
- `bit_guardrail_workitems`: `24`.
- `deployable_verified_equation_overrides`: `0`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24`.
- Decisao: `continue_local_solver_development`.

Diagnostico corrigido:

- `69` linhas: `algebraic_equation_unparsed` com `sympy_single_equation_probe` abstain por `prompt:no_single_algebraic_equation_found`.
- `9` linhas: `numeric_operator_transform` abstain por `numeric_examples_or_query_not_parseable`.
- `22` linhas: `symbolic_mixed_token_rewrite` abstain por `example_length_mismatch`.

Conclusao:

- Nao ha base para rescue eval.
- Nao ha base para treino.
- O gargalo agora e parser de formato real do prompt, nao solver matematico.

## V237 implementado - prompt format audit

Implementacao adicionada em 2026-05-10:

- Script: `scripts/analyze_v237_prompt_format_audit.py`.
- Builder: `scripts/build_v237_prompt_format_audit_colab.py`.
- Notebook: `notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`.

Escopo:

- CPU-only.
- Consome o manifest V232 e os mesmos workitems usados por V236.
- Audita formato dos prompts por `solver_route`, marcador de query, pares de exemplos candidatos, equacoes candidatas, expressoes numericas candidatas e hint de abstain.
- Nao treina, nao gera modelo, nao roda scoring, nao empacota, nao baixa payload externo e nao submete ao Kaggle.

Saidas obrigatorias:

- `v237_prompt_format_audit_prompt_format_audit.csv`;
- `v237_prompt_format_audit_prompt_format_summary.csv`;
- `v237_prompt_format_audit_equation_prompt_samples.csv`;
- `v237_prompt_format_audit_manifest.json`.

Validacoes executadas localmente:

- `python -m py_compile scripts/analyze_v237_prompt_format_audit.py`;
- `python scripts/analyze_v237_prompt_format_audit.py --self-test`;
- `python -m py_compile scripts/build_v237_prompt_format_audit_colab.py`;
- `python scripts/build_v237_prompt_format_audit_colab.py`;
- `python scripts/notebook_release_gate.py notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `ce92d2a5aee32e78a3a7c668ba14e8ca98e1d2d65ccff6564061adf32599f5e7`.

Proximo passo:

- Executar V237 no Colab.
- Se V237 mostrar que os prompts tem pares de exemplo parseaveis por outro padrao, criar V238 com parser unit-tested antes de qualquer override.
- Se V237 mostrar ausencia real de exemplos suficientes, manter abstain e voltar para mineracao de dados/treino, sem rescue eval.

## V237 executado - prompt format audit

Execucao analisada em 2026-05-10 a partir de `KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb`:

- Commit observado na celula de setup: `7d6db5eb045f7ae839367e95c83cc5432f8961a4`.
- Observacao operacional: a celula de preflight V232 nao foi executada no anexo, mas a celula de audit corrigida resolveu o manifest automaticamente e completou sem erro.
- Manifest V237 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V237/output_v237_prompt_format_audit/analysis_v237_prompt_format_audit/20260510T160754Z/v237_prompt_format_audit_manifest.json`.
- Manifest V237 SHA256 observado: `d857c395742b95604b4d41191f39c43c0f5fc464aa9b197ce86127b4f61c1d27`.
- Final manifest SHA256 observado: `8e723dd6ba771a716088927b1977db4a018db0ba0db8a6f07f080091eb660be8`.

Contagens:

- `audit_rows`: `124`.
- `equation_workitems`: `100`.
- `bit_workitems`: `24`.
- `equation_zero_candidate_pair_rows`: `68`.

Resumo de hints para `equation_transform`:

- `example_pairs_nonuniform_or_symbolic`: `32`.
- `numeric_expr_without_parseable_examples`: `11`.
- `prompt_format_requires_manual_parser`: `57`.

Resumo por rota:

- `bit_manipulation` / `bitwise_named_operator_dsl`: `24`, query marker `last_nonempty_line`, hint `example_pairs_nonuniform_or_symbolic`.
- `equation_transform` / `sympy_symbolic_transform`: `32`, marker `now_determine_result_for`, hint `example_pairs_nonuniform_or_symbolic`.
- `equation_transform` / `sympy_symbolic_transform`: `11`, marker `now_determine_result_for`, hint `numeric_expr_without_parseable_examples`.
- `equation_transform` / `sympy_symbolic_transform`: `57`, marker `now_determine_result_for`, hint `prompt_format_requires_manual_parser`.

Conclusao:

- A maioria dos workitems de `equation_transform` (`68/100`) nao possui pares de exemplo candidatos detectados pelos parsers V237 atuais.
- O proximo passo nao e rescue eval nem treino; e examinar exemplos reais dos prompts para construir parser especifico com testes unitarios.

Ajuste adicional aplicado apos esta execucao:

- V237 agora inclui `equation_prompt_sample_preview` no manifest e imprime essa previa no log do notebook.
- Objetivo: permitir decidir V238 diretamente pelos logs do Colab, sem depender de abrir manualmente o CSV no Drive.

Proximo passo revisado:

- Reexecutar V237 atualizado.
- Usar `equation_prompt_sample_preview` para definir se V238 deve implementar:
  - parser de exemplos simbolicos nao uniformes;
  - parser numerico com exemplos em texto natural;
  - ou classificador de abstain definitivo quando nao ha exemplos suficientes.

## V237 reexecutado - evidencia do formato Alice inline

Execucao analisada em 2026-05-10 a partir de `KG1_V237_PROMPT_FORMAT_AUDIT_COLAB (1).ipynb`:

- Commit observado na celula de setup: `e06d467bc3b9d23c2da027dc31f902c734eec331`.
- Manifest V237 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V237/output_v237_prompt_format_audit/analysis_v237_prompt_format_audit/20260510T161723Z/v237_prompt_format_audit_manifest.json`.
- Manifest V237 SHA256 observado: `bd66dd39646bfb71c39845301ba21a4844c89cd2f363e05c30cda1fce06e8289`.
- Final manifest SHA256 observado: `14dcbeef2e4db58f3a2d1502bd5e93545803cd4c9e5444672b679ccb4927aede`.

Contagens medidas:

- `audit_rows`: `124`.
- `equation_workitems`: `100`.
- `bit_workitems`: `24`.
- `equation_zero_candidate_pair_rows`: `68`.

Resumo de hints:

- `example_pairs_nonuniform_or_symbolic`: `32`.
- `numeric_expr_without_parseable_examples`: `11`.
- `prompt_format_requires_manual_parser`: `57`.

Achado novo concreto:

- Os prompts de `equation_transform` usam formato Alice inline: texto introdutorio, exemplos como `lhs = rhs` em linha corrida, e query com `Now, determine the result for:`.
- Exemplo numerico observado: `72)27 = 99 26#48 = 22 42#45 = 3 24#14 = 10 ... 94)40`.
- Exemplo numerico/misto observado: `38(96 = 3648 13(43 = 559 42#38 = 81 41(94 = 3854 ... 11-50`.
- Exemplos simbolicos usam caracteres especiais como tokens reais; portanto parsers nao podem remover backticks de forma agressiva.

Conclusao:

- O fracasso do V236 nao era prova de ausencia de solucao; era gap de parser.
- O proximo passo correto e V238: parser/probe especifico para Alice inline, unit-tested, ainda sem treino, sem full eval e sem pacote.

## V238 implementado - Alice parser probes

Implementacao adicionada em 2026-05-10:

- Script: `scripts/analyze_v238_alice_parser_probes.py`.
- Builder: `scripts/build_v238_alice_parser_probes_colab.py`.
- Notebook: `notebooks/KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`.

Objetivo:

- Medir, de forma CPU-only e diagnostica, quantos misses de `equation_transform` podem ser recuperados por parsers deterministas do formato Alice inline.
- Atacar especificamente o gargalo V230: baseline `191/315`, `equation_transform=55/155`, faltando `+5` equation para o gate fraco.

Regras implementadas:

- Parser Alice por marcadores `Below are a few examples:` e `Now, determine the result for:`.
- Extracao de exemplos inline `lhs = rhs` sem exigir backticks ou quebras de linha.
- Preservacao de backtick quando ele e caractere real do token; apenas wrapper balanceado `` `...` `` e removido.
- Probe numerico por operador: aprende regras conservadoras por operador da query (`+`, `-`, `abs_diff`, `mul`, concat, diferenca por digito, soma de digitos).
- Probe simbolico: mapa de caracteres, delecao por posicao, reversao, prefixo/sufixo.
- Qualquer previsao so e classificada como `verified` se bater o `expected_answer` do weak workitem; previsoes erradas viram `incorrect`, nao sao usadas para pacote.

Validacoes executadas localmente:

- `python -m py_compile scripts/analyze_v238_alice_parser_probes.py`.
- `python scripts/analyze_v238_alice_parser_probes.py --self-test`.
- `python -m py_compile scripts/build_v238_alice_parser_probes_colab.py scripts/analyze_v238_alice_parser_probes.py`.
- `python scripts/build_v238_alice_parser_probes_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `622281163cd12d31b49ef09f5fe12d7893e18bffc10b5780317d5cfc150cb0af`.

Proximo passo:

- Executar V238 no Colab.
- Se `deployable_verified_overrides >= 5` e `deployable_incorrect_overrides == 0`, criar V239 como measurement notebook separado que aplica somente overrides verificados e mede o ganho fraco.
- Se houver qualquer `incorrect`, nao usar override; abrir `v238_alice_parser_probes_alice_parser_probe_results.csv` e restringir ou remover a regra causadora.
- Se o ganho ficar abaixo de `+5`, continuar mineracao de subformatos Alice antes de qualquer treino.

## V238 executado - parser Alice ainda insuficiente

Execucao analisada em 2026-05-10 a partir de `KG1_V238_ALICE_PARSER_PROBES_COLAB.ipynb`:

- Commit Colab observado: `dd1d8068402e0f012cb8f4ea09dff3ff3630c192`.
- Manifest V238 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes/analysis_v238_alice_parser_probes/20260510T163118Z/v238_alice_parser_probes_manifest.json`.
- Manifest V238 SHA256 observado: `a1b17284ee64006b6d40b6e5a6be8662cdbdd2baa26b31a2695d4ec5b5677ce8`.
- Final manifest SHA256 observado: `22002ac871de7c991de22c7aad023a25bec62ee215554481a049cc7836e2ed90`.

Resultado medido:

- `equation_workitems`: `100`.
- `deployable_verified_overrides`: `1`.
- `deployable_incorrect_overrides`: `1`.
- `target_gain`: `5`.
- Decisao: `continue_alice_parser_development`.

Resumo:

- Numerico Alice: `1` verified, `15` abstain.
- Simbolico Alice: `1` incorrect, `83` abstain.
- O resultado bloqueia qualquer V239/rescue measurement.

Gap encontrado:

- O log antigo mostrava a existencia de `1` incorreto, mas nao imprimia a linha incorreta nem os principais motivos de abstain.
- O probe simbolico aceitava `char_map` como candidato deployable. Esse tipo de regra pode encaixar todos os exemplos e ainda errar a query; portanto e fraco demais para uso automatico.

Ajuste aplicado apos o log:

- `char_map_probe` foi removido do conjunto deployable de `symbolic_probe`; permanece como codigo auxiliar, mas nao pode gerar override automatico.
- V238 agora grava e imprime:
  - `abstain_reason_summary_top`;
  - `verified_preview`;
  - `incorrect_preview`;
  - `abstain_preview`.
- O objetivo e tornar o proximo log suficiente para decidir a regra seguinte sem abrir CSV manualmente no Drive.

Proximo passo revisado:

- Reexecutar V238 atualizado.
- Se `deployable_incorrect_overrides` cair para `0`, avaliar quantos `verified` restam.
- Se continuar abaixo de `5`, criar proximo parser apenas a partir de `abstain_reason_summary_top` e dos previews; nao fazer treino, full eval, pacote ou submissao.

## V238 reexecutado - delecao simbolica tambem e insegura

Execucao analisada em 2026-05-10 a partir de `KG1_V238_ALICE_PARSER_PROBES_COLAB (1).ipynb`:

- Commit Colab observado: `16b91a831a8576effe5df70cc4e9d84eb3f7beec`.
- Manifest V238 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes/analysis_v238_alice_parser_probes/20260510T163738Z/v238_alice_parser_probes_manifest.json`.
- Manifest V238 SHA256 observado: `c2e47d33583c20f1127e9aa83a29ebaf7be26b7986cf89bff025dfebb833f853`.
- Final manifest SHA256 observado: `6ea56a6c8ac299d5e97c3f0bf770f55edf3b08aa7dfd7b39cbd00a1b90a4f79e`.

Resultado medido:

- `equation_workitems`: `100`.
- `deployable_verified_overrides`: `1`.
- `deployable_incorrect_overrides`: `1`.
- `target_gain`: `5`.
- Decisao: `continue_alice_parser_development`.

Evidencia concreta:

- Verified numerico: id `c5b058d6`, query `94)40`, baseline `35`, expected `134`, prediction `134`, proof `rules=add`.
- Incorrect simbolico: id `432b1110`, query `\{*<?`, baseline `\{<?`, expected `%[:?`, prediction `\{<?`, proof `candidate_probes=alice_symbolic_deletion_positions_probe`.

Conclusao:

- A regra `alice_symbolic_deletion_positions_probe` tambem e fraca demais para override automatico.
- Ela consegue encaixar exemplos de treino inline por posicao mantida, mas pode apenas reproduzir uma delecao parecida com o baseline e errar a transformacao real da query.
- Logo, V238 continua bloqueando qualquer V239/rescue measurement.

Ajuste aplicado apos esta execucao:

- `alice_symbolic_deletion_positions_probe` foi rebaixado para diagnostico apenas.
- Quando encontra uma delecao consistente, a previsao fica registrada no proof como `diagnostic_only_candidate_disabled`, mas nao entra em `candidates` e nao pode gerar override deployable.
- Self-test V238 atualizado para exigir:
  - `deployable_verified_overrides == 1`;
  - `deployable_incorrect_overrides == 0`;
  - caso simbolico de delecao classificado como `abstain`;
  - proof contendo `diagnostic_only_candidate_disabled`.

Proximo passo revisado:

- Reexecutar V238 apos o bloqueio da delecao simbolica.
- Se o log mostrar `deployable_incorrect_overrides == 0` e `deployable_verified_overrides == 1`, nao criar V239 de rescue ainda: o ganho e insuficiente.
- Criar a proxima iteracao apenas para minerar os 83 abstains simbolicos e os abstains numericos por `no_examples_for_query_operator`/`candidate_rule_count`, sem permitir override simbolico novo sem teste unitario e evidencia de zero incorretos.

## V238 reexecutado apos bloqueio - zero incorretos, ganho insuficiente

Execucao analisada em 2026-05-10 a partir de `KG1_V238_ALICE_PARSER_PROBES_COLAB (2).ipynb`:

- Commit Colab observado: `a31877996a61f9d8f8b3485e6c8fb9fb3c4a16e4`.
- Manifest V238 gerado: `/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes/analysis_v238_alice_parser_probes/20260510T164430Z/v238_alice_parser_probes_manifest.json`.
- Manifest V238 SHA256 observado: `a088c00c3a7424e25ea35953ba85fb9afd56dbaaf9f8d2a2fe291d128d5833e6`.
- Final manifest SHA256 observado: `742d1114ee75fe736b41e7c20559e00068af176f9d9b23c382f9b558e9ac253b`.

Resultado medido:

- `equation_workitems`: `100`.
- `deployable_verified_overrides`: `1`.
- `deployable_incorrect_overrides`: `0`.
- `target_gain`: `5`.
- Decisao: `continue_alice_parser_development`.

Interpretacao:

- O bloqueio da delecao simbolica funcionou: o caso antes incorreto agora aparece como abstain diagnostico com `diagnostic_only_candidate_disabled`.
- Ainda nao existe autorizacao para V239 de rescue/measurement, porque o ganho deployable verificado e apenas `+1`, abaixo do alvo `+5`.
- O unico ganho concreto continua sendo numerico Alice: id `c5b058d6`, query `94)40`, baseline `35`, expected `134`, prediction `134`, proof `rules=add`.

Abstains dominantes que devem guiar a proxima etapa:

- `79` simbolicos: `alice_symbolic_deletion_positions_probe:nonuniform_lengths`; reverse nao bate; prefix/suffix com comprimentos nao uniformes.
- `4` simbolicos: `alice_symbolic_deletion_positions_probe:ambiguous_or_missing_keep_positions=0`; reverse nao bate; prefix/suffix sem regra candidata.
- `1` simbolico: delecao consistente, mas bloqueada como `diagnostic_only_candidate_disabled prediction='\\{<?'`; esse e o antigo caso inseguro.
- Numericos: `3` por `candidate_rule_count=0 unique_prediction_count=0`; `2` por `no_examples_for_query_operator='+'`; varios operadores sem exemplo para a query (`'`, `!`, `%`, `&`, `*`, `-`, `/`, `:`, `@`).

Proximo passo correto:

- Nao criar pacote, nao full eval e nao rescue measurement.
- Criar uma proxima auditoria V239 focada em minerar os abstains Alice, principalmente:
  - decompor simbolicos de comprimento nao uniforme por delta de comprimento, posicao do operador inserido/removido e relacao entre baseline/expected;
  - separar numericos sem exemplo do operador da query de numericos com exemplos ambiguos;
  - produzir workpacks pequenos com exemplos, query, baseline, expected e motivo de abstain para desenhar novas regras unit-tested.

## V239 implementado - mineracao dos abstains Alice

Arquivos:

- Script: `scripts/analyze_v239_alice_abstain_mining.py`.
- Builder: `scripts/build_v239_alice_abstain_mining_colab.py`.
- Notebook: `notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.

Objetivo:

- Consumir o manifesto V238 mais recente com `deployable_incorrect_overrides == 0`.
- Gerar workpacks auditaveis dos abstains antes de qualquer rescue measurement.
- Separar claramente:
  - simbolicos com comprimento nao uniforme, keep-positions impossivel ou delecao diagnostica bloqueada;
  - numericos sem exemplo do operador da query, sem regra candidata ou com regras ambiguas.

Saidas esperadas:

- `v239_alice_abstain_mining_symbolic_abstain_workpack.csv`.
- `v239_alice_abstain_mining_numeric_abstain_workpack.csv`.
- `v239_alice_abstain_mining_abstain_bucket_summary.csv`.
- `v239_alice_abstain_mining_manifest.json`.

Validacoes locais:

- `python -m py_compile scripts/analyze_v239_alice_abstain_mining.py scripts/build_v239_alice_abstain_mining_colab.py`.
- `python scripts/analyze_v239_alice_abstain_mining.py --self-test`.
- `python scripts/build_v239_alice_abstain_mining_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `36fb65e8ea9a3d645ab5b2a018ed64a7ccff6026121eaa1a11d96936672600ab`.

Proximo passo:

- Executar V239 no Colab.
- Usar `abstain_bucket_summary` e os workpacks para escolher apenas uma nova regra por vez.
- Toda regra nova deve entrar primeiro como self-test/fixture negativo, especialmente o antigo caso `432b1110`, antes de voltar para qualquer V240 parser probe.

## HF Jobs / FinOps execution policy

Decisao operacional:

- Sim, os proximos notebooks CPU-only e diagnosticos devem ser executados via Hugging Face Jobs sempre que os artefatos de entrada estiverem acessiveis fora do Google Drive.
- Evitar GPU para auditorias, parsers, gates, self-tests, mineracao de CSV/JSON e notebooks que nao carregam o modelo base.
- Usar `cpu-basic` como padrao para jobs curtos; a execucao remota V239 self-test/gate terminou em poucos segundos.
- Para jobs que precisarem GPU pequena, preferir primeiro `t4-small` ou `l4x1`; subir para `a10g-small` apenas se houver necessidade clara de VRAM/throughput.
- A100/H100 ou equivalentes ficam bloqueados ate haver uma execucao longa e justificativa explicita; isso protege o credito de USD 15.

Execucao HF validada:

- Conta HF autenticada usada: `felipesp1983`.
- Job HF: `6a00b9c3317220dbbd1a761e`.
- Flavor: `cpu-basic`.
- Imagem: `python:3.12`.
- Tarefa executada:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` dos scripts V238/V239;
  - `python scripts/analyze_v239_alice_abstain_mining.py --self-test`;
  - `python scripts/notebook_release_gate.py notebooks/KG1_V239_ALICE_ABSTAIN_MINING_COLAB.ipynb`.
- Status: `COMPLETED`.
- Duracao total observada: `7s`; runtime: `3s`.
- Resultado: self-test V239 `ok` e notebook gate `ok=true`.

Bloqueio atual para substituir 100% o Colab:

- HF Jobs nao monta `/content/drive/MyDrive/...`.
- Os artefatos completos V232/V238 usados pelos notebooks (`equation_solver_workitems_jsonl`, `v238_alice_parser_probe_results.csv`, manifests e CSVs completos) ainda vivem no Google Drive do Colab.
- Busca local nao encontrou copias completas desses artefatos fora do Drive.

Proximo passo para remover trabalho manual:

- Criar um bridge de artefatos para publicar manifests/CSVs diagnosticos em um dataset privado HF ou bucket equivalente.
- Depois desse bridge, V239 e as proximas auditorias podem rodar integralmente como HF Jobs sem Colab.
- Enquanto o bridge nao existir, HF consegue validar codigo/gates/self-tests, mas nao consegue executar analises completas que dependem de `/content/drive`.

## V240 implementado - bridge Drive para HF dataset

Arquivos:

- Script de upload: `scripts/upload_runtime_artifacts_to_hf.py`.
- Runner HF: `scripts/run_v239_from_hf_bridge.py`.
- Builder: `scripts/build_v240_hf_artifact_bridge_colab.py`.
- Notebook: `notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.

Objetivo:

- Rodar uma unica vez no Colab com Drive montado.
- Resolver os manifests V232/V238 mais recentes.
- Validar contrato de linhas compartilhadas.
- Subir para HF dataset privado os artefatos pequenos necessarios para V239:
  - V232 manifest;
  - V238 manifest;
  - V232 equation workitems;
  - V232 bit guardrail workitems;
  - V238 Alice parser probe results;
  - V238 Alice parser probe summary.

Destino padrao:

- Dataset HF: `felipesp1983/kg1-nemotron-training`.
- Prefixo: `runtime_artifacts/v240_hf_bridge/<RUN_ID>/`.

Requisito humano minimo:

- O Colab V240 precisa de `HF_TOKEN` com permissao de escrita no dataset.
- O token deve estar em Colab Secrets ou variavel de ambiente `HF_TOKEN`.
- A tentativa automatica de criar novo dataset privado via HF Job falhou com `403`; por isso o bridge usa dataset privado existente.

Validacoes locais:

- `python -m py_compile scripts/upload_runtime_artifacts_to_hf.py scripts/run_v239_from_hf_bridge.py scripts/build_v240_hf_artifact_bridge_colab.py`.
- `python scripts/upload_runtime_artifacts_to_hf.py --self-test`.
- `python scripts/run_v239_from_hf_bridge.py --self-test`.
- `python scripts/build_v240_hf_artifact_bridge_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- notebook SHA256: `bc5aec2d763c58ec873511644ea91f1e7908e547181072fa659f154e932e13e3`.

Validacao HF Jobs:

- Job HF: `6a00bc9aaff1cd33e8f32dfb`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total observada: `8s`; runtime: `3s`.
- Comandos executados no remoto:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` dos scripts V240/V239 bridge;
  - `python scripts/upload_runtime_artifacts_to_hf.py --self-test`;
  - `python scripts/run_v239_from_hf_bridge.py --self-test`;
  - `python scripts/notebook_release_gate.py notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb`.
- Resultado: self-tests `ok` e notebook gate `ok=true`.

Proximo passo:

- Executar V240 no Colab.
- Copiar do log o `bridge_path_in_repo`.
- A partir desse caminho, executar V239 completo no HF Job com `scripts/run_v239_from_hf_bridge.py`.

## V240/V239 executado sem Colab manual via Drive MCP + HF dataset

Atualizacao:

- Autenticacao HF local confirmada como `felipesp1983`.
- Os artefatos V232/V238 foram recuperados diretamente via Google Drive MCP, sem depender de uma nova execucao manual no Colab.
- Artefatos recuperados:
  - V232 manifest SHA256: `6415efbad28577c675f8847f6b84eb5a2d63709b6b9e5ae42fdba5a002c9b7bf`.
  - V238 manifest SHA256: `a088c00c3a7424e25ea35953ba85fb9afd56dbaaf9f8d2a2fe291d128d5833e6`.
  - V232 equation workitems: 100 linhas.
  - V232 bit workitems: 24 linhas.
  - V238 Alice results: 100 linhas.
- Bridge publicado no dataset HF:
  - Dataset: `felipesp1983/kg1-nemotron-training`.
  - Path: `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z`.
  - Commit HF: `7accc1518c1e3303401cd509aa9388541c4fc421`.
  - Bridge manifest SHA256: `76d3474ba84fa97014fe7a5887200617f3910483aa00636debd9cf6ac9c01778`.

Execucao V239 completa em HF:

- Job HF: `6a00bff0317220dbbd1a762f`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total: `11s`; runtime: `7s`.
- Dependencias no job: `huggingface_hub`, `pandas`.
- Resultado:
  - `v238_rows=100`.
  - `verified=1`.
  - `incorrect=0`.
  - `abstain=99`.
  - `symbolic_abstain=84`.
  - `numeric_abstain=15`.
  - `target_gain=5`.
- Decisao V239: `mine_abstains_before_any_rescue_measurement`.
- Outputs V239 publicados no dataset HF:
  - Path: `runtime_artifacts/v239_alice_abstain_mining/local_drive_mcp_20260510T172421Z`.
  - Commit HF: `68bc8c14eb70f446331ed8acb81bade36566e91d`.

Bucket summary V239:

- `symbolic_nonuniform_lengths`: 79.
- `numeric_no_examples_for_query_operator`: 11.
- `symbolic_no_keep_positions`: 4.
- `numeric_no_candidate_rule`: 3.
- `numeric_ambiguous_candidate_rules`: 1.
- `symbolic_diagnostic_deletion_disabled`: 1.

Achado operacional:

- Uma tentativa de fazer o HF Job baixar diretamente URLs temporarias do Drive/OpenAI falhou com HTTP `403`.
- A rota correta e robusta e: Drive MCP/local -> upload autenticado para HF dataset -> HF Jobs consomem o dataset.
- Isso remove o trabalho manual do Colab para os proximos notebooks CPU-only, preservando FinOps.

## V241 abstain rule candidate audit

Arquivo:

- Script: `scripts/analyze_v241_abstain_rule_candidate_audit.py`.

Objetivo:

- Auditar os 99 abstains V238/V239 com regras candidatas mais fortes, sem promover nada inseguro.
- Testar dois caminhos:
  - symbolic char-transducer conservador, derivado apenas dos exemplos do prompt;
  - numeric DSL expandido, exigindo exemplos do mesmo operador e minimo de evidencia.
- Usar `expected_answer` apenas para auditoria fraca, nunca para derivar a regra.

Validacoes locais:

- `python -m py_compile scripts/analyze_v241_abstain_rule_candidate_audit.py`.
- `python scripts/analyze_v241_abstain_rule_candidate_audit.py --self-test`.
- Execucao real sobre os artefatos V232/V238 recuperados do Drive MCP.

Resultado real V241:

- `v238_rows=100`.
- `abstain_rows=99`.
- `symbolic_rows=84`.
- `numeric_rows=15`.
- `deployable_verified_candidates=0`.
- `deployable_incorrect_candidates=0`.
- `under_evidenced_candidates=0`.
- Decisao: `do_not_promote_v241_candidates`.

Resumo tecnico V241:

- Simbolico:
  - `no_global_mapping`: 60.
  - `no_pair_mapping`: 22.
  - `char_transducer mappings=1 usable=0 unique_predictions=0`: 1.
  - `char_transducer mappings=16 usable=16 unique_predictions=9`: 1.
- Numerico:
  - `no_same_operator_examples`: 11.
  - `candidate_rule_count=0 unique_prediction_count=0`: 2.
  - `candidate_rule_count=2 unique_prediction_count=2`: 1.
  - `candidate_rule_count=7 unique_prediction_count=5`: 1.

Outputs V241 publicados no dataset HF:

- Path: `runtime_artifacts/v241_abstain_rule_candidate_audit/local_drive_mcp_20260510T172421Z`.
- Commit HF: `fc8d5e956327ddd8635b06cf7c7b212dd5e48535`.

Validacao V241 em HF Jobs:

- Job HF: `6a00c1e7aff1cd33e8f32e2e`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total: `10s`; runtime: `4s`.
- A validacao remota executou:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` do script V241;
  - self-test V241;
  - download dos artefatos V232/V238 do bridge HF;
  - auditoria real V241.
- Resultado remoto:
  - `v238_rows=100`.
  - `abstain_rows=99`.
  - `deployable_verified_candidates=0`.
  - `deployable_incorrect_candidates=0`.
  - decisao `do_not_promote_v241_candidates`.
- Observacao operacional: um job anterior tentou subir outputs de dentro do HF Job e recebeu `403`; a publicacao de artefatos deve continuar usando o token local autenticado ou um secret HF com permissao explicita de escrita.

Decisao de negocio/QA:

- Nao promover parser novo agora.
- O risco de overfit/leakage e maior que o ganho esperado, porque nenhum candidato deployable foi verificado com zero ambiguidade.
- Proximo passo correto: gerar exemplos/fixtures adicionais para os buckets dominantes antes de nova medicao de rescue:
  - simbolico: focar `symbolic_nonuniform_lengths`, mas exigir regra derivavel de exemplos;
  - numerico: focar operadores sem exemplo do mesmo simbolo, porque 11/15 numericos estao bloqueados por ausencia de evidencia local.

## Data leakage audit - Alice weak workitems versus local datasets

Motivo:

- Antes de qualquer novo treino para `equation_transform`, foi necessario verificar se os workitems fracos V232/V238 ja aparecem em datasets locais.
- Usar weak IDs/answers como treino contaminaria o gate fraco e inflaria ACC sem validade.

Resultado do overlap exato:

- Referencia auditada: 100 workitems V232/V238 do bridge `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z`.
- `data/v217/v217_short_answer_train.jsonl`:
  - linhas: 10206.
  - `exact_prompt_overlap=0`.
  - `id_overlap=0`.
  - prompts Alice equation por frase: 6935.
- `data/v217/v217_short_answer_val.jsonl`:
  - linhas: 681.
  - `exact_prompt_overlap=0`.
  - `id_overlap=0`.
  - prompts Alice equation por frase: 453.
- `data/sft_v51_complete.jsonl`:
  - linhas: 9500.
  - `exact_prompt_overlap=0`.
  - `id_overlap=100`.
  - prompts Alice equation por frase: 1555.

Conclusao:

- V217 train/val permanecem limpos contra os 100 workitems V232/V238 auditados.
- `data/sft_v51_complete.jsonl` contem todos os 100 IDs fracos auditados e deve ficar em quarentena para qualquer treino, calibragem ou selecao que use o weak gate como evidencia.
- `data/sft_v51_complete.jsonl` pode ser usado somente como source-intel/diagnostico rotulado como potencial leakage, nunca como fonte direta para aumentar ACC medida no weak gate.

Regra para proximos notebooks/gates:

- Qualquer novo dataset de treino para `equation_transform` ou `bit_manipulation` deve executar overlap por `id` e por hash de prompt normalizado contra os workitems weak conhecidos.
- `id_overlap > 0` com weak/eval artifacts deve bloquear treino automaticamente, exceto em notebook explicitamente marcado como diagnostico de leakage.

Validacao materializada:

- Script: `scripts/audit_jsonl_overlap.py`.
- Self-test: `python scripts/audit_jsonl_overlap.py --self-test`.
- Execucao real:
  - referencia: `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z/v232_equation_workitems.jsonl`.
  - candidatos: V217 train, V217 validation, `data/sft_v51_complete.jsonl`.
- Resultado publicado no HF dataset:
  - Path: `runtime_artifacts/v241_overlap_audit/local_drive_mcp_20260510T172421Z/v241_overlap_audit.json`.
  - Commit HF: `b87478307a71b5cfea7c8d65e366b7d6794562da`.

## V242 safe equation fixture generation

Objetivo:

- Caminho mais rapido e objetivo antes de gastar GPU.
- Gerar fixtures sinteticos independentes para `equation_transform`, focados nos buckets V239/V241:
  - simbolico com comprimentos nao uniformes;
  - numerico com exemplos suficientes do mesmo operador.
- Bloquear automaticamente qualquer overlap por `id` ou hash de prompt normalizado contra weak workitems conhecidos.
- Nao treinar, nao inferir com modelo, nao pontuar modelo e nao submeter.

Arquivos:

- Gerador: `scripts/generate_v242_safe_equation_fixtures.py`.
- Builder: `scripts/build_v242_safe_equation_fixtures_colab.py`.
- Notebook: `notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb`.
- Colab: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb`.

Validacoes locais:

- `python -m py_compile scripts/generate_v242_safe_equation_fixtures.py scripts/audit_jsonl_overlap.py`.
- `python scripts/generate_v242_safe_equation_fixtures.py --self-test`.
- `python -m py_compile scripts/build_v242_safe_equation_fixtures_colab.py`.
- `python scripts/build_v242_safe_equation_fixtures_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb`.

Resultado do notebook gate:

- `ok=true`.
- notebook SHA256: `27f1a2f4c5bd251479cb0977ea7958133f5fde0b238419307c58452e6ab748dc`.

Execucao local V242:

- Referencia de leakage: `runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z/v232_equation_workitems.jsonl`.
- `train_rows=1800`.
- `validation_rows=240`.
- `seed=242`.
- Resultado:
  - train simbolico: 1126.
  - train numerico: 674.
  - validation simbolico: 153.
  - validation numerico: 87.
  - train `id_overlap=0`, `prompt_overlap=0`.
  - validation `id_overlap=0`, `prompt_overlap=0`.

Outputs V242 publicados no HF dataset:

- Dataset: `felipesp1983/kg1-nemotron-training`.
- Path: `runtime_artifacts/v242_safe_equation_fixtures/local_cpu_20260510T174632Z`.
- Commit HF: `eb1979dcea095a5b06b5f77c96b03027bea25ece`.

Validacao V242 em HF Jobs:

- Job HF: `6a00c56d317220dbbd1a7644`.
- Flavor: `cpu-basic`.
- Status: `COMPLETED`.
- Duracao total: `10s`; runtime: `5s`.
- A validacao remota executou:
  - clone da branch `v230-v226-complementarity`;
  - `py_compile` dos scripts V242 e overlap audit;
  - self-tests dos dois scripts;
  - notebook release gate do V242;
  - download do weak reference a partir do HF bridge;
  - geracao completa de `1800` train e `240` validation.
- Resultado remoto:
  - train simbolico: 1126.
  - train numerico: 674.
  - validation simbolico: 153.
  - validation numerico: 87.
  - train `id_overlap=0`, `prompt_overlap=0`.
  - validation `id_overlap=0`, `prompt_overlap=0`.

Decisao:

- Fixtures estao prontos para revisao de gate de treino.
- Ainda nao autoriza treino automaticamente.
- O proximo passo objetivo, se aprovado, e criar um treino curto que consome somente V217 limpo + V242, repetindo o overlap gate antes de carregar modelo/GPU.

## V243 guarded V217 plus V242 training mix and HF execution

Objetivo:

- Executar a opcao mais objetiva e rapida no Hugging Face antes de qualquer treino longo.
- Criar um mix de treino usando somente:
  - V217 short-answer limpo ja versionado no repo;
  - V242 safe equation fixtures publicados no HF dataset.
- Validar hash, linhas, dedupe, overlap contra weak workitems e tokenizacao antes de gastar GPU.
- Fazer apenas um smoke train curto em GPU, nao um treino final.

Arquivos:

- Script: `scripts/build_v243_training_mix.py`.
- Commit GitHub: `c72e95bbf693609698f9df215ef8121d6d870ef1`.

Mix V243 publicado:

- Dataset HF: `felipesp1983/kg1-nemotron-training`.
- Path: `runtime_artifacts/v243_training_mix/local_upload_20260510T180200Z`.
- Train rows: `12006`.
- Validation rows: `921`.
- Train SHA256: `c290555bffade5f4fa4e5c14f6f66c36745bd31a22c4b004709afd5a5f33f6d1`.
- Validation SHA256: `54eda74b1ea01e6e3b165af23c99eac5dc6e21f29cbc49888503ea7a3d707764`.
- Familias no train:
  - `equation_transform=8735`.
  - `bit_manipulation=2695`.
  - demais familias de guarda: `gravity_constant=144`, `numeral_system=144`, `text_encryption=144`, `unit_conversion=144`.
- Familias na validation:
  - `equation_transform=693`.
  - `bit_manipulation=164`.
  - demais familias de guarda: `16` cada.
- Overlap contra weak reference:
  - train `id_overlap=0`, `prompt_overlap=0`.
  - validation `id_overlap=0`, `prompt_overlap=0`.

HF Jobs executados:

- `6a00c7d6aff1cd33e8f32e66`: CPU build/upload remoto.
  - Status: `ERROR`.
  - Achado util: build remoto validou o mix, mas upload falhou porque o token injetado pelo job tinha leitura sem escrita.
  - Mitigacao: upload feito localmente com token write validado.
- `6a00c904317220dbbd1a7650`: tokenization dry-run CPU inicial.
  - Status: `ERROR`.
  - Causa: comando de pip apontou todos os pacotes para o indice CPU do PyTorch.
  - Mitigacao: relancado corrigido em `6a00c92e317220dbbd1a7652`.
- `6a00c92e317220dbbd1a7652`: tokenization dry-run CPU corrigido.
  - Status: `COMPLETED`.
  - Train tokenized: `12006/12006`.
  - Validation tokenized: `921/921`.
  - Truncation: `0`.
  - Prompt truncation: `0`.
  - Offset masks: `12006` train e `921` validation.
  - Fallback masks: `0`.
  - Dry-run report publicado em:
    `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/dry_runs/v243-tokenize-dryrun-20260510T1808Z/dry_run_model_recipe_report.json`.
- `6a00c888aff1cd33e8f32e6a`: GPU smoke train em `a100-large`.
  - Status no momento do registro: `SCHEDULING`.
  - Run ID: `v243-v188-safe-eq-smoke-s4-20260510T1803Z`.
  - Init adapter: `felipesp1983/kg1-nemotron-lora-v188-equation-lmhead/checkpoint-40`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures`.
  - Config intencionalmente curta:
    - `MAX_STEPS=4`.
    - `MAX_LENGTH=4096`.
    - `BATCH_SIZE=4`, `MICRO_BATCH_SIZE=1`.
    - `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head`.
    - `LEARNING_RATE=1e-7`, `FINAL_LEARNING_RATE=5e-8`.
  - Motivo de manter `a100-large`: e o menor flavor razoavel para 30B BF16; multi-GPU seria pior em FinOps e L40S/L4/A10G isolados sao arriscados para memoria.

Decisao:

- O mix V243 esta aprovado para smoke train: hash, contagem, overlap, truncation e offset-mask passaram.
- O smoke train GPU deve ser avaliado por weak eval antes de qualquer treino longo.
- Se `a100-large` ficar preso em scheduling por muito tempo, a acao correta e aguardar capacidade ou cancelar; nao trocar automaticamente para multi-GPU caro.

Atualizacao HF H200:

- A API local de HF Jobs confirmou flavors com acelerador:
  - `a100-large`: 1x NVIDIA A100 80GB, custo aproximado `0.041667 USD/min`.
  - `h200`: 1x NVIDIA H200 141GB, custo aproximado `0.083333 USD/min`.
  - `h200x2`, `h200x4`, `h200x8` tambem existem, mas nao sao FinOps-correto para smoke train.
- O job A100 `6a00c888aff1cd33e8f32e6a` foi cancelado enquanto ainda estava em `SCHEDULING`.
- Tentativas H200 registradas:
  - `6a00cb86aff1cd33e8f32e8a`: falhou antes de treinar porque `mamba_ssm` nao estava instalado.
  - `6a00cbea317220dbbd1a765b`: falhou antes de treinar porque `pip install causal-conv1d mamba-ssm` trocou `torch 2.8.0+cu128` por `torch 2.11.0+cu130`, gerando ABI incompatível.
  - `6a00cc82aff1cd33e8f32e97`: preflight H200 confirmou `torch 2.8.0+cu128` e GPU `NVIDIA H200`, mas `mamba-ssm --no-deps` precisa das dependencias base instaladas antes.
  - `6a00cce2aff1cd33e8f32e99`: preflight H200 confirmou que `mamba-ssm --no-deps` preserva `torch 2.8.0+cu128`; faltou `transformers` no preflight isolado.
  - `6a00cd62317220dbbd1a7660`: smoke train H200 relancado com build de fonte para `causal-conv1d==1.6.1` e `mamba-ssm==2.3.1`, `--no-deps`, `--no-build-isolation`, `--no-binary`, e gate que aborta se `torch` mudar.
    - Resultado util: passou das dependencias, confirmou GPU H200, preservou `torch 2.8.0+cu128`, baixou modelo `63.2GB`, carregou adapter V188 checkpoint-40, aplicou filtro LoRA e iniciou setup de treino.
    - Falha: `SAMPLING_MODE=weighted` era invalido; `scripts/hf_job_train_v90.py` aceita `shuffle` ou `weighted_replacement`.
    - Mitigacao implementada: `scripts/hf_job_train_v90.py` agora valida `SAMPLING_MODE` em import/startup, antes de baixar modelo, para evitar repetir erro caro.

Regras obrigatorias para o notebook/executor HF de treino:

- Deve listar flavors disponiveis por `HfApi.list_jobs_hardware()` e logar explicitamente H200/A100, custo por minuto, VRAM e flavor selecionado.
- Deve cancelar ou bloquear jobs antigos em fila antes de lancar outro treino GPU, para evitar gasto duplicado.
- Deve instalar dependencias em ordem:
  - imagem base CUDA/PyTorch fixa;
  - dependencias Python base (`huggingface_hub`, `transformers`, `peft`, `accelerate`, `safetensors`, `sentencepiece`, `protobuf`, `hf_transfer`, `packaging`, `wheel`, `setuptools`, `ninja`, `einops`);
  - extensoes Mamba/Causal Conv compiladas contra o torch ja presente, nunca deixando `pip` resolver outro torch.
- Deve imprimir e validar `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` e nome da GPU antes e depois das instalacoes.
- Deve abortar se `torch` mudar entre `torch_before` e `torch_after`.
- Deve importar e logar `causal_conv1d`, `mamba_ssm`, `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn` e `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` antes de clonar/carregar o modelo.
- Deve clonar a branch com commit esperado fixo e abortar em mismatch.
- Deve rodar `py_compile` em `scripts/hf_job_train_v90.py`, `scripts/build_v243_training_mix.py` e `scripts/audit_jsonl_overlap.py`.
- Deve validar SHA256 e contagem dos arquivos V243 antes de carregar o modelo.
- Deve validar `SAMPLING_MODE` antes de carregar modelo; valores permitidos: `shuffle` ou `weighted_replacement`.
- Deve manter `MAX_STEPS=4` no smoke train e nao executar treino longo sem novo gate humano.
- Deve escrever todos os IDs de job, run IDs, URLs HF, status, erro e mitigacao no manifesto/roadmap.

## V244 HF H200 smoke train concluido

Objetivo:

- Executar o primeiro treino remoto curto com o mix V243, em H200, com gates dentro do container antes de qualquer download/carga cara.
- Validar que o executor HF consegue treinar o Nemotron 30B com adapter V188 inicial, sem quebrar dependencias Mamba/Causal Conv, sem trocar `torch`, e sem gastar com erro ja conhecido.
- Nao promover adapter, nao rodar full eval e nao criar pacote/submissao.

Job executado:

- Job HF: `6a00d6a9317220dbbd1a7683`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00d6a9317220dbbd1a7683`.
- Status: `COMPLETED`.
- Flavor: `h200`.
- Imagem: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Run ID: `v244-h200-smoke-20260510T190308Z`.
- Commit GitHub fixado: `7f4192d9dfa5e73fd4ccda1c1a15ed7a24a186ee`.
- Output repo: `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures`.
- Custo estimado: H200 `0.083333 USD/min`; runtime total observado aproximado `17.9 min`; custo aproximado `US$1.49`.

Gates executados dentro do HF Job:

- `scripts/hf_job_preflight_gate.py --phase preinstall`.
- `scripts/hf_job_preflight_gate.py --phase artifacts`.
- `scripts/hf_job_preflight_gate.py --phase postinstall`.

Resultado dos gates:

- Repo/commit: OK.
- `py_compile` dos scripts criticos: OK.
- GPU/Torch: OK, H200 CUDA disponivel.
- Flavor/custo permitido: OK.
- Modelo base: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Dataset V243:
  - train SHA256 `c290555bffade5f4fa4e5c14f6f66c36745bd31a22c4b004709afd5a5f33f6d1`.
  - validation SHA256 `54eda74b1ea01e6e3b165af23c99eac5dc6e21f29cbc49888503ea7a3d707764`.
  - train rows `12006`.
  - validation rows `921`.
  - familias train: `equation_transform=8735`, `bit_manipulation=2695`, guardas `144` cada.
  - familias validation: `equation_transform=693`, `bit_manipulation=164`, guardas `16` cada.
  - subcategorias V242 presentes: train `equation_symbolic_mixed_v242=1126`, `equation_numeric_same_operator_v242=674`; validation `153` e `87`.
- Init adapter V188:
  - repo/subfolder: `felipesp1983/kg1-nemotron-lora-v188-equation-lmhead/checkpoint-40`.
  - `r=32`, `lora_alpha=32`.
  - target modules: `down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj`.
  - `target_parameters=null` no config remoto aceito para esse adapter.
- Dependencias pos-instalacao: `huggingface_hub`, `transformers`, `peft`, `accelerate`, `safetensors`, `causal_conv1d`, `mamba_ssm`, `mamba_ssm.ops.triton.layernorm_gated`, `mamba_ssm.ops.selective_scan_interface`.

Resultado do treino:

- Modelo baixado e carregado.
- Adapter V188 inicial carregado manualmente:
  - tensors mapeados: `12011`.
  - tensors nao mapeados: `0`.
  - cobertura: `1.0000`.
- Modulos LoRA treinaveis: `q_proj,k_proj,v_proj,o_proj,lm_head`.
- Parametros treinaveis: `8,015,872`.
- Parametros totais reportados: `32,466,091,456`.
- Percentual treinavel: `0.0247%`.
- Config smoke:
  - `MAX_STEPS=4`.
  - `MAX_LENGTH=4096`.
  - `BATCH_SIZE=4`.
  - `MICRO_BATCH_SIZE=1`.
  - `GRADIENT_ACCUMULATION=4`.
  - `LEARNING_RATE=1e-7`.
  - `FINAL_LEARNING_RATE=5e-8`.
  - `SAMPLING_MODE=weighted_replacement`.
- Loss por step:
  - step 1: `3.4815`.
  - step 2: `3.3636`.
  - step 3: `3.3335`.
  - step 4: `3.2083`.
- Eval:
  - final eval loss: `3.2582213087007403`.
  - best eval loss: `3.2582213087007403`.
- VRAM pico reportado: `63.142578125 GiB`.
- Elapsed do treino no manifest: `284.10246777534485s`.

Artefatos publicados:

- `final/adapter_config.json`.
- `final/adapter_model.safetensors`.
- `final/v90_training_manifest.json`.
- `checkpoint-4/adapter_model.safetensors`.
- `checkpoint-2/adapter_model.safetensors`.
- Dry-run anterior preservado: `dry_runs/v243-tokenize-dryrun-20260510T1808Z/dry_run_model_recipe_report.json`.

Decisao QA/negocio:

- V244 prova que o pipeline HF H200 com gates funciona.
- V244 nao prova ganho de ACC; loss menor em smoke train nao substitui weak eval.
- Nenhum adapter V244 deve ir para full eval, packaging ou Kaggle antes de passar weak eval identico ao gate V230/V226.
- O risco principal agora e overfit/regressao em `bit_manipulation`; por isso V245 deve medir `final`, `checkpoint-4` e `checkpoint-2` contra os thresholds e guardrails.

## V245 proximo passo - weak eval dos adapters V244

Objetivo:

- Medir se o smoke train V244 realmente melhorou `equation_transform` sem perder o piso forte de `bit_manipulation`.
- Avaliar os tres artefatos publicados:
  - `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/final`.
  - `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/checkpoint-4`.
  - `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/checkpoint-2`.
- Comparar contra baseline V226 observado:
  - total `191/315`.
  - `equation_transform=55/155`.
  - `bit_manipulation=136/160`.
  - truncation `0`.

Gate minimo para considerar continuidade:

- Weak total `>=193`.
- `equation_transform >=60`.
- `bit_manipulation >=133`.
- truncation `<=3`.

Guardrail adicional recomendado:

- Se `equation_transform <60`, parar; nao gastar com full eval.
- Se `bit_manipulation <136`, exigir justificativa tecnica antes de qualquer treino longo, porque o baseline ja esta em `85.00%` nessa familia.
- Se truncation `>0`, inspecionar prompt/output antes de treinar mais, porque o formato final do KG1 e sensivel ao parser `\boxed{...}`.

Preflight obrigatorio para V245:

- Resolver/baixar adapters V244 via `snapshot_download` ou `hf_hub_download` e validar:
  - `adapter_config.json` existe.
  - `adapter_model.safetensors` existe.
  - `r=32`.
  - `lora_alpha=32`.
  - target modules compativeis.
  - tamanho do weights plausivel e nao vazio.
- Resolver weak CSV de 315 linhas fora do Drive antes de criar job pago.
- Se o CSV fraco nao estiver publicado no HF dataset, executar primeiro um bridge pequeno Drive -> HF dataset; nao iniciar vLLM/H200 dependendo de `/content/drive`.
- Reusar `scripts/evaluate_lora_adapters_batch.py` com:
  - `--max-tokens 96`.
  - `--max-model-len 4096`.
  - `--max-num-seqs 8`.
  - `--warmup-rows 0`.
  - `--disable-thinking`.
  - prompt suffix: `Return only one line: \boxed{answer}. No reasoning. No explanation.`
- Registrar `batch_candidate_summary.json`, predictions CSV, per-task CSV e manifest HF.

Decisao esperada:

- Se qualquer V244 atingir o gate weak: preparar full eval controlado em nova versao.
- Se nenhum V244 atingir o gate, encerrar esta linha de treino curto e voltar para mineracao deterministica dos miss-packs/fixtures, sem treino longo.

## V245 bridge implementado - publicar weak CSV exato no HF dataset

Motivo:

- O weak eval V245 dos adapters V244 precisa do CSV fraco exato de `315` linhas.
- O dataset HF atual contem manifests/workitems V232/V238/V242/V243, mas nao contem `v221_weak_315.csv`.
- Busca local, HF dataset e Google Drive connector nao recuperou o arquivo raw diretamente.
- Recriar a amostra fraca a partir de `data/train.csv` com heuristicas de `seed=42` nao reproduziu o contract hash V230; portanto isso nao pode ser usado para medir ACC.

Implementacao:

- Script: `scripts/upload_v245_weak_csv_bridge_to_hf.py`.
- Builder: `scripts/build_v245_weak_eval_bridge_colab.py`.
- Notebook: `notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb`.
- Colab URL:
  `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v230-v226-complementarity/notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb`.

O que o notebook faz:

- Monta Google Drive.
- Clona a branch `v230-v226-complementarity`.
- Roda `py_compile`, self-test do bridge e `notebook_release_gate.py`.
- Procura primeiro:
  `/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab/v221_weak_315.csv`.
- Se esse CSV nao existir, reconstrói a partir do CSV exato:
  `/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv`.
- Valida:
  - colunas `id`, `prompt`, `answer`, `type/family`;
  - linhas `315`;
  - `bit_manipulation=160`;
  - `equation_transform=155`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Publica no HF dataset:
  - repo `felipesp1983/kg1-nemotron-training`;
  - prefixo `runtime_artifacts/v245_weak_eval_bridge/<RUN_ID>/v221_weak_315.csv`;
  - manifesto `v245_weak_eval_bridge_manifest.json`.

Validacoes locais:

- `python -m py_compile scripts/upload_v245_weak_csv_bridge_to_hf.py scripts/build_v245_weak_eval_bridge_colab.py`.
- `python scripts/upload_v245_weak_csv_bridge_to_hf.py --self-test`.
- `python scripts/build_v245_weak_eval_bridge_colab.py`.
- `python scripts/notebook_release_gate.py notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb`.

Resultado do gate:

- `ok=true`.
- Notebook SHA256: `738ae203316b9e60111b991ca974291861e594788bc94bfcd9e6603a7288e24a`.

Status:

- Implementado.
- Esta rota Colab fica suspensa enquanto a diretriz operacional for "usar HF para tudo".
- O weak CSV exato foi reconstruido fora do Colab e publicado diretamente no HF dataset em V245 HF-only, portanto nao ha mais bloqueio de Drive para o proximo weak eval.

## V245 HF-only bridge concluido - weak CSV canonico publicado

Diretriz operacional atual:

- Por decisao do usuario em 2026-05-10, Colab fica suspenso ate segunda ordem.
- Todo trabalho executavel deve usar Hugging Face Jobs/datasets/model repos sempre que tecnicamente possivel.
- Antes de qualquer job pago, aplicar gates baratos: existencia de artefatos, hashes, contratos de linhas, adapter config/pesos, GPU/custo, dependencias e commit esperado.

Evidencia:

- O CSV oficial `train.csv` local em `artifacts/api_kaggle_openrouter_audit_2026_05_06/competition_data/extracted/train.csv` tem SHA256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`, igual ao esperado pelo V207A.
- A celula V207A gera a validacao assim:
  - baixa `data/kaggle/unzipped/train.csv`;
  - classifica com `classify_puzzle`;
  - `random.seed(42)`;
  - para cada familia em ordem alfabetica, embaralha e pega `int(10%)`;
  - junta as linhas e embaralha novamente;
  - filtra as familias weak `bit_manipulation` e `equation_transform`.
- A reconstrucao produziu:
  - weak rows `315`;
  - `bit_manipulation=160`;
  - `equation_transform=155`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.

Upload HF:

- Dataset repo: `felipesp1983/kg1-nemotron-training`.
- Commit HF: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/d6a0ffe1af8205ba8fa2fb6c633b16c9f0aaf054`.
- Prefixo:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/`.
- CSV publicado:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv`.
- Manifest publicado:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v245_weak_eval_bridge_manifest.json`.
- Canonical weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.

Novo executor HF:

- Script: `scripts/hf_job_weak_eval_v245.py`.
- Objetivo: rodar weak eval dos adapters V244 dentro do HF Job sem depender de Drive/Colab.
- Gates antes de vLLM/model-load:
  - GPU CUDA e VRAM minima;
  - branch commit esperado;
  - import `vllm`;
  - download do weak CSV do HF;
  - SHA do CSV `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`;
  - rows/familias `315`, `160/155`;
  - manifest V245 do bridge;
  - adapter `adapter_config.json` e `adapter_model.safetensors`;
  - `r=32`, `lora_alpha=32`.
- Avaliacao:
  - usa `scripts/evaluate_lora_adapters_batch.py`;
  - `max_tokens=96`;
  - `max_model_len=4096`;
  - `max_num_seqs=8`;
  - `disable-thinking`;
  - prompt suffix de resposta curta boxed.

Validacoes locais:

- `python -m py_compile scripts/hf_job_weak_eval_v245.py scripts/hf_job_preflight_gate.py scripts/evaluate_lora_adapters_batch.py scripts/evaluate_lora_adapter.py`.
- `python scripts/hf_job_weak_eval_v245.py --self-test`.

Proximo passo automatico HF:

- Commitar/pushar `scripts/hf_job_weak_eval_v245.py` e este roadmap.
- Lançar HF Job de weak eval para o adapter V244 `final`.
- Se `final` nao passar ou regredir, repetir para `checkpoint-4` e `checkpoint-2`.
- So considerar full eval se algum adapter bater o gate:
  - total `>=193`;
  - equation `>=60`;
  - bit `>=133`;
  - truncation `<=3`.

## V245 HF weak eval - primeira medicao do adapter V244 final

Job:

- Tentativa A100: `6a00e301aff1cd33e8f32f80`.
  - Cancelada porque ficou em `SCHEDULING` sem logs.
- Execucao H200: `6a00e3e1317220dbbd1a76bc`.
  - URL: `https://huggingface.co/jobs/felipesp1983/6a00e3e1317220dbbd1a76bc`.
  - Run ID: `v245-h200-weak-final-20260510T195932Z`.
  - Commit repo: `d4578bb098b82561ea402041691d8830ead3d4d1`.

Gates confirmados antes da avaliacao:

- GPU: `NVIDIA H200`, `139.80 GiB`.
- vLLM import OK: `vllm==0.20.1`.
- Weak CSV:
  - rows `315`;
  - `bit_manipulation=160`;
  - `equation_transform=155`;
  - SHA256 `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`;
  - contract hash `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Adapter final:
  - repo `felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/final`;
  - `adapter_model.safetensors` `4,259,063,856` bytes;
  - `r=32`;
  - `lora_alpha=32`.

Resultado medido:

- Candidate: `v244_final_adapter`.
- Overall weak: `18/315 = 5.71%`.
- `bit_manipulation`: `9/160 = 5.63%`.
- `equation_transform`: `9/155 = 5.81%`.
- truncation: `0`.
- Gate: reprovado.

Interpretacao:

- O adapter final V244 nao e utilizavel como candidato de weak/full eval.
- A queda e grande demais para justificar full eval ou pacote.
- Possivel fator operacional identificado: o executor V245 removia o `\n` inicial do prompt suffix porque usava `env_str(...).strip()`. Isso foi corrigido em `scripts/hf_job_weak_eval_v245.py` para preservar o prompt suffix default com newline, igual ao padrao V229/V230.
- O primeiro upload de resultados subiu tambem `adapter_snapshot` sob `evals/`, o que era ruido de 4.28GB. O snapshot duplicado foi removido do HF repo no commit:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/commit/f95e0749b14a27dae020b7cb9eaf3a58dcc323cd`.
- O script foi ajustado para baixar o adapter fora do `output_dir` e ignorar qualquer `adapter_snapshot/**` no upload.

Proximo passo:

- O executor foi ampliado para aceitar `KG1_ADAPTER_SUBFOLDERS` e `KG1_CANDIDATE_NAMES`, permitindo avaliar `final`, `checkpoint-4` e `checkpoint-2` no mesmo job com um unico model-load.
- Reexecutar `final`, `checkpoint-4` e `checkpoint-2` juntos com o prompt suffix corrigido para separar falha de adapter de falha de wrapper.

## V245 HF weak eval trio - resultado final da linha V244

Job:

- HF Job: `6a00e5b3317220dbbd1a76be`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00e5b3317220dbbd1a76be`.
- Run ID: `v245-h200-weak-v244-trio-20260510T200718Z`.
- Repo commit: `10865607892d53ceac6b6a1885b13db7bf31b7c7`.
- Upload dos resultados:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures/commit/664ae944125e4f55d4d5c8d0bb02895ba72564cf`.
- Path:
  `evals/v245-h200-weak-v244-trio-20260510T200718Z/`.

Gates:

- H200 OK.
- Commit esperado OK.
- Weak CSV hash e row contract OK.
- Prompt suffix corrigido preservado no config:
  `\nReturn only one line: \boxed{answer}. No reasoning. No explanation.`
- Tres adapters validados:
  - `final`;
  - `checkpoint-4`;
  - `checkpoint-2`.
- Todos com `r=32`, `lora_alpha=32`, pesos `4,259,063,856` bytes.
- Upload limpo: `adapter_snapshot` nao foi publicado nos resultados do trio.

Resultados weak:

| Candidate | Overall | bit_manipulation | equation_transform | Truncation |
|---|---:|---:|---:|---:|
| `v244_final_adapter` | `18/315 = 5.71%` | `9/160 = 5.63%` | `9/155 = 5.81%` | `0` |
| `v244_checkpoint_4` | `18/315 = 5.71%` | `9/160 = 5.63%` | `9/155 = 5.81%` | `0` |
| `v244_checkpoint_2` | `19/315 = 6.03%` | `10/160 = 6.25%` | `9/155 = 5.81%` | `0` |

Decisao:

- Linha V244 reprovada.
- Nao fazer full eval.
- Nao fazer packaging.
- Nao fazer Kaggle submit.
- Nao continuar treino longo partindo desses checkpoints.

Diagnostico objetivo:

- O problema nao era so o prompt suffix; a execucao corrigida preservou o newline e continuou em ~6%.
- As predicoes baixadas mostram respostas plausiveis mas quase sempre erradas, com frequencias altas de payloads como `00000000`, `10000000`, simbolos isolados e respostas curtas repetidas.
- Isso indica regressao/degradacao real ou incompatibilidade de continuidade do adapter V244, nao truncation.

Proximo passo:

- Encerrar a linha V244.
- Voltar para rota P0 do roadmap: mineracao deterministica/DSL dos miss-packs e fixtures por familia, antes de novo treino.
- Qualquer novo treino HF deve partir de um adapter/baseline que ja prove weak ACC perto do V226 (`191/315`) ou entao deve passar primeiro por uma avaliacao weak curta; nao repetir smoke train sem weak gate intermediario.

## V236/V246 HF-only local solver path - status atual

Executor V236 no HF:

- Script: `scripts/run_v236_from_hf_bridge.py`.
- Job parserfix: `6a00eaa0317220dbbd1a76d0`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00eaa0317220dbbd1a76d0`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/de4942ba75849f84fa4096466c0e341768c90d59`.
- Path:
  `runtime_artifacts/v236_local_solver_dsl_probes/v236-hf-cpu-bridge-parserfix-20260510T202819Z/`.

Resultado V236 parserfix:

- `deployable_verified_equation_overrides`: `1`.
- `deployable_incorrect_equation_overrides`: `0`.
- `bit_guardrail_signature_verified_rows`: `24/24`.
- Linha recuperada:
  - id `c5b058d6`;
  - baseline `35`;
  - solver `134`;
  - expected `134`;
  - proof `alice_rules=add`.
- Impacto maximo isolado: V226 baseline iria de `191/315` para `192/315`, ainda abaixo do gate `193/315` e equation `60`.

Novo executor V246:

- Script: `scripts/run_v246_exhaustive_abstain_audit_hf.py`.
- Objetivo: auditar os `99` abstains restantes da V236 parserfix usando regras locais conservadoras.
- Custo: CPU-only HF Job.
- Entradas HF:
  - V236 parserfix results;
  - V236 parserfix manifest;
  - V240 bridge `v232_equation_workitems.jsonl`;
  - V240 bridge `v232_manifest.json`.
- Gate de contrato:
  - expected/observed shared row contract `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Regra de seguranca:
  - weak label e usado apenas como freio/auditoria;
  - uma classe de regra so e promovivel se todos os candidatos emitidos pela classe forem verificados e houver `0` incorretos;
  - se qualquer classe produz incorretos, ela fica bloqueada.
- Classes auditadas:
  - numeric same-operator DSL min2/min3/min4;
  - symbolic char transducer com todos exemplos;
  - symbolic char transducer por mesmo operador min2;
  - symbolic positional deletion;
  - symbolic same-operator positional deletion min2;
  - symbolic same-operator position-specific char map min2.

Pre-check local consumindo artefatos HF:

- `v236_rows`: `100`.
- `v236_abstain_rows`: `99`.
- `audit_rows`: `465`.
- `verified_candidates`: `0`.
- `incorrect_candidates`: `11`.
- `promotable_rows_after_class_gate`: `0`.
- Decisao local preliminar:
  `no_safe_local_rule_promotion_found`.

Interpretacao:

- A recuperacao deterministica local atual nao entrega os +5 de equation necessarios.
- Ha evidencia concreta de que regras simbolicas simples geram incorretos; portanto nao devem ser promovidas.
- Nao gastar H100/H200 em treino derivado desses candidatos sem uma nova fonte de dados/traços.
- Proximo passo HF-only: rodar V246 no HF CPU e publicar os artefatos; se confirmar `0` promoviveis, bloquear essa rota e seguir para acesso aos traces externos ou novo desenho de treino.

Execucao HF V246 confirmada:

- HF Job: `6a00ef1aaff1cd33e8f32ff1`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00ef1aaff1cd33e8f32ff1`.
- Run ID: `v246-hf-cpu-exhaustive-abstain-20260510T204724Z`.
- Repo commit executado: `09bdb266b54bb2ded373814e753af8d20de779f3`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/f95b5232e8d76df68d180178047ba1273a09e8a5`.
- Path:
  `runtime_artifacts/v246_exhaustive_abstain_audit/v246-hf-cpu-exhaustive-abstain-20260510T204724Z/`.

Resultado HF V246:

- `v236_rows`: `100`.
- `v236_abstain_rows`: `99`.
- `audit_rows`: `465`.
- `verified_candidates`: `0`.
- `incorrect_candidates`: `11`.
- `promotable_rows_after_class_gate`: `0`.
- Decisao:
  `no_safe_local_rule_promotion_found`.

Classes bloqueadas/sem ganho:

- `numeric_same_operator_extended_dsl_min2/min3/min4`: nenhum candidato verificado.
- `symbolic_all_examples_char_transducer`: nenhum candidato.
- `symbolic_same_operator_char_transducer_min2`: nenhum candidato.
- `symbolic_all_examples_positional_deletion`: 1 candidato, 1 incorreto.
- `symbolic_same_operator_position_char_map_min2`: 10 candidatos, 10 incorretos.
- `symbolic_same_operator_positional_deletion_min2`: nenhum candidato.

Conclusao de negocio/QA:

- A rota local solver/DSL conservadora esta esgotada para ganho imediato.
- Nao ha evidencia para promover nova regra sem aumentar falso positivo.
- Nao gastar H100/H200 nesta rota.
- Proxima rota objetiva: validar acesso aos datasets/traces externos no HF:
  - `andy279/nemotron-reasoning-challenge-raw-traces`;
  - `andy279/nemotron-reasoning-challenge`;
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`.
- Se os datasets `andy279/*` continuarem gated/403 com o token atual, sera necessaria acao humana para aceitar os termos no HF antes de qualquer treino baseado nesses traces.

## V247 HF source access gate - proxima rota de dados externos

Script:

- `scripts/run_v247_hf_source_access_gate.py`.

Objetivo:

- Validar no HF, com o token atual, quais fontes externas de traces/dados estao realmente acessiveis antes de gastar GPU.
- Evitar baixar payloads grandes: usa metadata + HTTP range-read pequeno.
- Nao treina, nao avalia modelo, nao faz pacote e nao submete Kaggle.

Fontes testadas:

- `andy279/nemotron-reasoning-challenge-raw-traces`:
  - `solver_transformation_traces_gpt54.jsonl`;
  - `solver_transformation_traces_merged.jsonl`;
  - `solver_bit_manipulation_traces_merged.jsonl`.
- `andy279/nemotron-reasoning-challenge`:
  - `sft_val.jsonl`;
  - `sft_train.jsonl`.
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`:
  - `train.csv`;
  - `test.csv`.

Pre-check local usando HF token:

- Metadata `andy279/*`: acessivel, `gated=manual`.
- Payload `andy279/*`: `403`, mensagem HF: request awaiting review from repo authors.
- `jasonkung98/*`: acessivel por range-read.
- Contagem:
  - P0 accessible files: `0`;
  - P0 denied files: `5`;
  - public accessible files: `2`.
- Decisao preliminar:
  `p0_gated_terms_required_public_mirror_available`.

Interpretacao:

- Os arquivos publicos de `jasonkung98` servem para sanity/source check, mas nao substituem os traces P0.
- A rota de maior impacto para melhorar `equation_transform` depende dos datasets gated `andy279/*`.
- Se o job HF V247 confirmar o mesmo 403, a proxima acao nao e tecnica: sera necessario aceitar/liberar acesso aos repos gated no HF.

Execucao HF V247 confirmada:

- HF Job: `6a00f039aff1cd33e8f3300f`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f039aff1cd33e8f3300f`.
- Run ID: `v247-hf-source-access-gate-20260510T205212Z`.
- Repo commit executado: `cb9ff271c7a4504314930cf33a937bb8de594979`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/99eca833c15b9651a55c33c71296f14b6dd9cc94`.
- Path:
  `runtime_artifacts/v247_hf_source_access_gate/v247-hf-source-access-gate-20260510T205212Z/`.

Resultado HF V247:

- P0 accessible files: `0`.
- P0 denied files: `5`.
- Public accessible files: `2`.
- `andy279/nemotron-reasoning-challenge-raw-traces`: metadata OK, payload 403/manual review pending.
- `andy279/nemotron-reasoning-challenge`: metadata OK, payload 403/manual review pending.
- `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: `train.csv` e `test.csv` acessiveis.
- Decisao:
  `p0_gated_terms_required_public_mirror_available`.

Bloqueio atual:

- A rota de traces externos P0 esta bloqueada por review/termos HF dos repos `andy279/*`.
- O mirror publico `jasonkung98/*` e util para sanity check, mas nao traz os traces/solver SFT que justificariam novo treino.
- Proxima acao humana necessaria:
  - abrir `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces`;
  - solicitar/aceitar acesso;
  - abrir `https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge`;
  - solicitar/aceitar acesso;
  - depois rerodar V247. Se P0 ficar acessivel, criar job V248 de ingestao/filtragem de traces equation/bit antes de qualquer treino.

Recheck HF V247:

- HF Job: `6a00f0f0aff1cd33e8f33018`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f0f0aff1cd33e8f33018`.
- Run ID: `v247-hf-source-access-recheck-20260510T205514Z`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/84bb71a1806208a512b8e01edfb297c402036e0a`.
- Resultado: sem mudanca, `andy279/*` ainda `403` aguardando review; `jasonkung98/*` acessivel.

Recheck HF V247 em 2026-05-11:

- Tentativa inicial de launch:
  - HF Job: `6a014e3a317220dbbd1a784b`.
  - URL: `https://huggingface.co/jobs/felipesp1983/6a014e3a317220dbbd1a784b`.
  - Status: falhou imediatamente por parsing do CLI (`bash` recebeu o script inteiro como caminho). Custo operacional esperado: minimo; nenhum payload baixado, nenhum treino.
- Launch corrigido:
  - HF Job: `6a014e51aff1cd33e8f333fa`.
  - URL: `https://huggingface.co/jobs/felipesp1983/6a014e51aff1cd33e8f333fa`.
  - Flavor: `cpu-basic`.
  - Status: `COMPLETED`.
  - Upload HF:
    `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/08e73a970662e93ec38db2189570977e2bfa8922`.
  - Path:
    `runtime_artifacts/v247_hf_source_access_gate/v247-hf-source-access-recheck-20260511T033339Z/`.
- Resultado:
  - P0 accessible files: `0`.
  - P0 denied files: `5`.
  - Public accessible files: `2`.
  - `andy279/nemotron-reasoning-challenge-raw-traces`: metadata OK, arquivos existem, payload `403` com mensagem de review pendente.
  - `andy279/nemotron-reasoning-challenge`: metadata OK, arquivos existem, payload `403` com mensagem de review pendente.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: `train.csv` e `test.csv` acessiveis.
- Decisao: `p0_gated_terms_required_public_mirror_available`.
- Implicacao: a rota de traces externos segue bloqueada por acesso humano aos repos `andy279/*`. Ate liberar esse acesso, nao iniciar treino H200 baseado nesses traces.

Recheck HF V264 em 2026-05-11:

- HF Job: `6a016a74317220dbbd1a78e4`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a016a74317220dbbd1a78e4`.
- Run ID: `v264-hf-source-access-recheck-20260511T053342Z`.
- Repo commit executado: `c1efe6af76918145a16a9a96423ee4e2b19c5dd5`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/049b6afe57041eeb9b36424ee03c341a9e3b7c07`.
- Path:
  `runtime_artifacts/v247_hf_source_access_gate/v264-hf-source-access-recheck-20260511T053342Z/`.
- Resultado:
  - P0 accessible files: `0`.
  - P0 denied files: `5`.
  - Public accessible files: `2`.
  - `andy279/nemotron-reasoning-challenge-raw-traces`: metadata OK, payload `403`, review pendente.
  - `andy279/nemotron-reasoning-challenge`: metadata OK, payload `403`, review pendente.
  - `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`: `train.csv` e `test.csv` acessiveis.
- Decisao: `p0_gated_terms_required_public_mirror_available`.
- Implicacao: sem aceitar/liberar acesso aos datasets gated `andy279/*`, a rota de traces P0 permanece bloqueada. Novo H200 baseado no mirror publico ou em soups nao e justificado pelos resultados V257-V263.

## V248 public mirror leakage audit

Script:

- `scripts/run_v248_public_mirror_leakage_audit_hf.py`.

Objetivo:

- Auditar o mirror publico `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`.
- Verificar vazamento contra o weak set canonico V245.
- Contar linhas target-family disponiveis apos excluir qualquer ID weak.
- Bloquear qualquer uso de labels weak-overlap em treino, calibragem ou selecao de regra.

Pre-check local:

- Public train rows: `9500`.
- Public test rows: `3`.
- Weak rows: `315`.
- Weak overlap rows: `315`.
- Weak answer mismatches: `0`.
- Weak prompt mismatches normalizados: `0`.
- Non-weak target rows: `2842`.
- Por familia:
  - `bit_manipulation`: `1602` train, `160` weak-overlap, `1442` nonweak;
  - `equation_transform`: `1555` train, `155` weak-overlap, `1400` nonweak;
  - demais familias permanecem P2/P3 para este objetivo.
- Decisao preliminar:
  `public_mirror_usable_only_after_weak_id_exclusion`.

Interpretacao:

- O mirror publico confirma que o weak set esta dentro do train publico; usar essas respostas para ajustar regra/modelo e vazamento.
- O uso permitido e apenas com exclusao explicita dos `315` weak IDs.
- Como ainda ha `2842` linhas target-family nao weak, a rota possivel sem gated traces e construir um dataset V249 estritamente non-weak, com fixtures de validacao separados e sem usar weak labels para selecao.

Execucao HF V248 confirmada:

- HF Job: `6a00f1e1aff1cd33e8f3302a`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f1e1aff1cd33e8f3302a`.
- Run ID: `v248-hf-public-mirror-leakage-20260510T205916Z`.
- Repo commit executado: `c039b7e093cfdca5dbbb7effba60f835d526a7fd`.
- Upload HF:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/0d134fc36737346c32061ab063ab01eb01db0256`.
- Path:
  `runtime_artifacts/v248_public_mirror_leakage_audit/v248-hf-public-mirror-leakage-20260510T205916Z/`.
- Resultado: igual ao pre-check local.

Proxima acao HF-only:

- V249 deve materializar somente linhas `bit_manipulation` e `equation_transform` do mirror publico com `id` fora do weak set.
- V249 deve gerar `train.jsonl`, `val.jsonl`, manifest, hashes e CSV de IDs bloqueados.
- V249 nao deve treinar; e apenas preparo de dados com gates.
- Antes de treino, precisa comparar V249 contra V217/V226 para evitar repetir dataset/efeito V244.

## V249 public non-weak target dataset

Script:

- `scripts/run_v249_public_nonweak_target_dataset_hf.py`.

Objetivo:

- Materializar um dataset HF estritamente sem vazamento dos `315` weak IDs.
- Usar somente as familias que estamos tentando melhorar agora:
  `bit_manipulation` e `equation_transform`.
- Gerar `train.jsonl`, `val.jsonl`, CSV de weak IDs bloqueados, manifest e hashes.
- Nao treinar, nao avaliar modelo, nao gerar pacote e nao submeter Kaggle.

Gates implementados:

- Baixa o mirror publico `jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge`.
- Baixa o weak CSV canonico V245 do dataset HF privado.
- Exige SHA exato do weak CSV:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Exige contagem target apos exclusao weak:
  - total: `2842`;
  - `bit_manipulation`: `1442`;
  - `equation_transform`: `1400`.
- Valida que nenhum `original_id` de treino/validacao esta no weak set.
- Valida IDs unicos, formato `messages`, resposta do assistant e split estratificado.
- Bloqueia explicitamente `train`, `model_generation`, `full_scoring`, `package` e `kaggle_submit` no manifest.

Pre-check local:

- `py_compile`: OK.
- `--self-test`: OK.
- Run local CPU: OK.
- Public train rows: `9500`.
- Weak rows bloqueados: `315`.
- Candidate target rows non-weak: `2842`.
- Train rows: `2558`.
- Val rows: `284`.
- Train family counts:
  - `bit_manipulation`: `1298`;
  - `equation_transform`: `1260`.
- Val family counts:
  - `bit_manipulation`: `144`;
  - `equation_transform`: `140`.
- Hashes locais:
  - train JSONL: `81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41`;
  - val JSONL: `43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783`;
  - blocked weak IDs CSV: `5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce`.
- Escrita JSON/JSONL ajustada para LF deterministico entre Windows e Linux/HF.

Decisao:

- `dataset_ready_for_tokenization_gate_not_training_yet`.
- Proxima acao HF-only: executar V249 no HF e fazer upload dos artefatos.
- Depois do V249 remoto, executar gate V250 de tokenizacao/offset-mask/truncation antes de qualquer treino GPU.

Execucao HF V249 confirmada:

- HF Job: `6a00f420317220dbbd1a76f0`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f420317220dbbd1a76f0`.
- Run ID: `v249-hf-public-nonweak-target-20260510T210850Z`.
- Repo commit executado: `5660be29f3347e5adefd80c8000d096d334ecca0`.
- Upload HF folder:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/1bc2b5b1c1e3893e73e570b45ef4860949785d80`.
- Upload HF manifest refresh:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/8d4510fd7390aa370fad4ae6172e1bd679038be9`.
- Path:
  `data/v249_public_nonweak_target/v249-hf-public-nonweak-target-20260510T210850Z/`.
- Manifest remoto verificado:
  `data/v249_public_nonweak_target/v249-hf-public-nonweak-target-20260510T210850Z/v249_public_nonweak_target_manifest.json`.
- Resultado remoto:
  - public train rows: `9500`;
  - weak rows bloqueados: `315`;
  - target non-weak rows: `2842`;
  - train rows: `2558`;
  - val rows: `284`;
  - train counts: `bit_manipulation=1298`, `equation_transform=1260`;
  - val counts: `bit_manipulation=144`, `equation_transform=140`.
- Hashes remotos canonicos:
  - train JSONL: `81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41`;
  - val JSONL: `43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783`;
  - blocked weak IDs CSV: `5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce`.

Status:

- V249 pronto para V250 tokenizer/mask/truncation gate.
- Ainda nao autorizado para treino GPU: falta provar tokenizacao, labels/offset-mask e comparacao contra V217/V226.

## V250 V249 tokenization gate

Script:

- `scripts/run_v250_v249_tokenization_gate_hf.py`.

Objetivo:

- Validar o dataset V249 remoto antes de qualquer gasto com GPU.
- Rebaixar risco de V244: nao treinar ate provar hashes, formato, weak exclusion, tokenizacao real, offset masks e truncation zero.
- Comparar o V249 contra o corpus V217 ja usado, para medir novidade real.

Gates implementados:

- Baixa `train.jsonl`, `val.jsonl`, `v249_blocked_weak_ids.csv` e manifest V249 do HF.
- Exige hashes canonicos:
  - train JSONL: `81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41`;
  - val JSONL: `43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783`;
  - blocked weak IDs CSV: `5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce`.
- Exige contagens:
  - train rows: `2558`;
  - val rows: `284`;
  - blocked weak rows: `315`.
- Exige family counts:
  - train: `bit_manipulation=1298`, `equation_transform=1260`;
  - val: `bit_manipulation=144`, `equation_transform=140`.
- Exige zero overlap de `original_id` com weak IDs.
- Exige formato `messages=[system,user,assistant]` e assistant `Final answer: ...`.
- Usa tokenizer real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Exige offset masks; fallback mask e falha.
- Exige `MAX_LENGTH=4096` com prompt truncation rate `0.0`.
- Bloqueia `long_train`, `full_scoring`, `package` e `kaggle_submit`.

Pre-check local:

- `py_compile`: OK.
- `--self-test`: OK.
- Run local CPU: OK.
- Tokenizer: `TokenizersBackend`, fast tokenizer, `eos/pad=<|im_end|>`.
- Train tokenization:
  - rows: `2558`;
  - offset masks: `2558`;
  - fallback masks: `0`;
  - prompt truncated: `0`;
  - token max: `324`.
- Validation tokenization:
  - rows: `284`;
  - offset masks: `284`;
  - fallback masks: `0`;
  - prompt truncated: `0`;
  - token max: `324`.
- Por familia:
  - `bit_manipulation`: answer loss tokens fixos em `14`, token max `324`;
  - `equation_transform`: answer loss tokens `6..10`, token max `157` train / `153` val.
- Overlap contra V217:
  - V249 total: `2842`;
  - prompt+answer overlap V217 train: `900`;
  - prompt+answer overlap V217 val: `39`;
  - prompt+answer overlap total: `939`;
  - novidade prompt+answer vs V217: `1903`.

Interpretacao:

- O V249 e tecnicamente treinavel, mas nao e totalmente novo: `939/2842` linhas ja existem no V217 por prompt+answer.
- O ganho esperado de treino deve vir dos `1903` exemplos novos e de uma mistura mais cuidadosa, nao de simplesmente repetir V217.
- Proxima acao HF-only: executar V250 no HF, subir manifest e, se passar, criar um treino smoke muito curto com gate fraco antes de qualquer H200 longo.

Execucao HF V250 confirmada:

- Primeiro job: `6a00f5b4317220dbbd1a76f6`.
  - Resultado: falhou antes da tokenizacao porque `jinja2` nao estava instalado no container CPU.
  - Correcao operacional: rerun com `jinja2` no setup do job. Sem custo GPU.
- Job valido: `6a00f5ecaff1cd33e8f33044`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f5ecaff1cd33e8f33044`.
- Run ID: `v250-hf-v249-tokenization-gate-20260510T211631Z`.
- Repo commit executado: `5e9dbe546e9dd579d3b4e312b7643ed1f43c2cfa`.
- Upload HF folder:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/2ff0c9acbceca9ae4ad74688a4e9b61e34d229ea`.
- Upload HF manifest refresh:
  `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/94df99e4ce002dca1cf8c0d81689f8e1db5cd623`.
- Path:
  `runtime_artifacts/v250_v249_tokenization_gate/v250-hf-v249-tokenization-gate-20260510T211631Z/`.
- Manifest remoto verificado:
  `runtime_artifacts/v250_v249_tokenization_gate/v250-hf-v249-tokenization-gate-20260510T211631Z/v250_v249_tokenization_gate_manifest.json`.
- Resultado remoto:
  - train tokenized rows: `2558`;
  - validation tokenized rows: `284`;
  - train offset masks: `2558`;
  - validation offset masks: `284`;
  - fallback masks: `0`;
  - prompt truncated: `0`;
  - token max: `324`;
  - prompt+answer overlap total vs V217: `939`;
  - prompt+answer novel vs V217: `1903`.

Status:

- V250 passou.
- Permite apenas proximo smoke GPU curto, nao um treino longo direto.
- O smoke deve ter `MAX_STEPS` baixo, upload de checkpoints, weak eval imediato e bloqueio se nao houver melhora sobre V226 191/315.

## V251 H200 weak eval - V187 public adapter trio

Script/job:

- Reuso do wrapper HF `scripts/hf_job_weak_eval_v245.py`.
- Job inicial `6a00f6bd317220dbbd1a76fa` falhou antes da avaliacao porque a imagem `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` nao continha `git`.
- Job valido: `6a00f748aff1cd33e8f33052`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00f748aff1cd33e8f33052`.
- Flavor: `h200`.
- Imagem corrigida: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Run ID: `v251-h200-weak-v187-trio-20260510T212219Z`.
- Repo commit executado: `32454da9d0651b8a3b40a38833a45053f04cd250`.
- Adapter repo avaliado: `felipesp1983/kg1-nemotron-lora-v187-submission-gain`.
- Subfolders avaliados: `final`, `checkpoint-20`, `checkpoint-40`.
- Upload HF:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v187-submission-gain/commit/9c0b2c70c1eee3a5cfb177f1e9f08d976f982f4b`.
- Manifest remoto:
  `evals/v251-h200-weak-v187-trio-20260510T212219Z/v245_hf_weak_eval_manifest.json`.

Gates confirmados:

- CUDA/H200 disponivel e carga vLLM concluida.
- Weak CSV canonico V245 usado com `315` linhas.
- `observed_shared_row_contract_sha256`:
  `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Families avaliadas: `bit_manipulation=160`, `equation_transform=155`.
- Full eval, package e Kaggle submit permaneceram bloqueados.

Resultado weak:

| Candidato | Total | ACC | Equation | Bit | Trunc |
|---|---:|---:|---:|---:|---:|
| `v187_final` | `17/315` | `5.40%` | `9/155` | `8/160` | `0` |
| `v187_checkpoint20` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v187_checkpoint40` | `17/315` | `5.40%` | `9/155` | `8/160` | `0` |

Decisao:

- Rejeitar `felipesp1983/kg1-nemotron-lora-v187-submission-gain` como candidato de promocao, baseline, initializer ou fonte de ensemble.
- O melhor V187 ficou `18/315`, contra baseline V226 `191/315`.
- A ausencia de truncamento mostra que a falha nao e apenas de output longo; o adapter provavelmente nao esta alinhado ao contrato/prompt/modelo do weak gate atual.
- Nao gastar novo H200 nesse repositorio.

Impacto no roadmap:

- Prioridade volta para dados V249 + smoke GPU curto, ou para triagem de outro adapter HF somente se houver evidencia independente forte e o custo for limitado.
- Nao usar `V187` em treino, merge, DARE/TIES, router ou seed sem uma justificativa nova e verificavel.

## V252 H200 weak eval - V188 raw, overlay e stripped

Objetivo:

- Antes de gastar H200 em treino derivado de `V188`, medir se os artefatos publicos `checkpoint-*`, `final_full_baseline_overlay` e `final_stripped` possuem qualquer sinal util no weak gate canonico.
- Testar os 6 candidatos em uma unica carga vLLM para reduzir custo.
- Bloquear full eval, package e Kaggle submit.

Script/job:

- Reuso do wrapper HF `scripts/hf_job_weak_eval_v245.py`.
- Job inicial `6a00f9e5317220dbbd1a7702` falhou antes da avaliacao porque o secret foi passado como string literal `$HF_TOKEN`; o container recebeu token invalido e retornou `401` ao baixar dataset privado.
- Correcao aplicada: rerun com valor real de `get_token()` injetado em `secrets`.
- Job valido: `6a00fa83317220dbbd1a7706`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00fa83317220dbbd1a7706`.
- Flavor: `h200`.
- Run ID: `v252-h200-weak-v188-packages-20260510T213606Z`.
- Repo commit executado: `6824a0cb978cfe02d25ce979fbd598c62213f692`.
- Adapter repo avaliado: `felipesp1983/kg1-nemotron-lora-v188-equation-lmhead`.
- Upload HF:
  `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v188-equation-lmhead/commit/b1f974fd80212467e5d7f77dca6baf994c67f076`.
- Manifest remoto:
  `evals/v252-h200-weak-v188-packages-20260510T213606Z/v245_hf_weak_eval_manifest.json`.

Gates confirmados:

- CUDA/H200 disponivel e carga vLLM concluida.
- Weak CSV canonico V245 usado com `315` linhas.
- `observed_shared_row_contract_sha256`:
  `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Families avaliadas: `bit_manipulation=160`, `equation_transform=155`.
- Full eval, package e Kaggle submit permaneceram bloqueados.

Resultado weak:

| Candidato | Artefato | Total | ACC | Equation | Bit | Trunc |
|---|---|---:|---:|---:|---:|---:|
| `v188_checkpoint20_raw` | `checkpoint-20` | `16/315` | `5.08%` | `8/155` | `8/160` | `0` |
| `v188_checkpoint40_raw` | `checkpoint-40` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v188_checkpoint20_overlay` | `packages/v188-checkpoint20/final_full_baseline_overlay` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v188_checkpoint20_stripped` | `packages/v188-checkpoint20/final_stripped` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v188_checkpoint40_overlay` | `packages/v188-checkpoint40/final_full_baseline_overlay` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v188_checkpoint40_stripped` | `packages/v188-checkpoint40/final_stripped` | `16/315` | `5.08%` | `9/155` | `7/160` | `0` |

Decisao:

- Rejeitar `V188` raw, overlay e stripped como candidato de promocao, baseline, initializer, merge, DARE/TIES, router ou fonte de ensemble.
- O melhor V188 ficou `18/315`, contra baseline V226 `191/315`.
- O overlay nao recuperou o comportamento do baseline; portanto, o pacote publicado nao deve ser tratado como adapter protegido equivalente a V194/V226.
- A falha com zero truncamento indica desalinhamento de adapter/contrato/modelo, nao apenas excesso de tokens.
- Nao gastar novo H200 em treino derivado de `V188` sem uma fonte nova de pesos/dados e uma justificativa verificavel.

Impacto no roadmap:

- V187 e V188 publicos estao descartados como rotas de melhoria direta.
- A proxima rota HF-only deve ser uma destas, em ordem:
  1. localizar/subir para HF o adapter forte conhecido (`V194`/`V226`) para permitir smoke training/eval sem depender de Google Drive;
  2. se o adapter forte nao estiver acessivel no HF, executar um baseline no-LoRA ou uma auditoria HF de candidatos com evidencia independente antes de qualquer treino;
  3. usar V249 apenas em smoke curto com gate imediato, sem treino longo direto.

## V253 H200 weak eval - adapters HF priorizados

Objetivo:

- Medir, no weak gate canonico, os adapters HF priorizados que ainda tinham alguma evidencia historica ou prescore local.
- Incluir `V189`, `V94`, `V95`, `V96`, `V97` e `V101` em uma unica carga vLLM para reduzir custo.
- Validar que a nova logica `KG1_ADAPTER_SPECS_JSON` aceita multiplos repositorios/subfolders sem quebrar gates.
- Bloquear full eval, package e Kaggle submit.

Script/job:

- Wrapper HF: `scripts/hf_job_weak_eval_v245.py`.
- Mudanca de suporte multi-repo: commit `c944cbb1dbdf36d870afbe215dfc7f4dcef7572f`.
- Job inicial `6a00fe05aff1cd33e8f3309a` foi cancelado antes da carga do modelo por mismatch no `KG1_EXPECTED_COMMIT`.
- Job valido: `6a00fe5e317220dbbd1a7717`.
- URL: `https://huggingface.co/jobs/felipesp1983/6a00fe5e317220dbbd1a7717`.
- Flavor: `h200`.
- Run ID: `v253-h200-weak-prioritized-hf-adapters-20260510T215233Z`.
- Repo commit executado: `c944cbb1dbdf36d870afbe215dfc7f4dcef7572f`.
- Upload HF:
  `https://huggingface.co/felipesp1983/kg1-nemotron-training/commit/ff6eaa9dc9ae3990a58bd7c966d696bc7a35c59b`.
- Manifest remoto:
  `evals/v253-h200-weak-prioritized-hf-adapters-20260510T215233Z/v245_hf_weak_eval_manifest.json`.

Gates confirmados:

- CUDA/H200 disponivel e carga vLLM concluida.
- Weak CSV canonico V245 usado com `315` linhas.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Families avaliadas: `bit_manipulation=160`, `equation_transform=155`.
- Todos os 10 adapters passaram no gate de arquivo/config antes da avaliacao.
- Full eval, package e Kaggle submit permaneceram bloqueados.

Resultado weak:

| Candidato | Repositorio/subfolder | Total | ACC | Equation | Bit | Trunc |
|---|---|---:|---:|---:|---:|---:|
| `v189_checkpoint10_raw` | `felipesp1983/kg1-nemotron-lora-v189-equation-answer-short/checkpoint-10` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v189_checkpoint10_overlay` | `.../packages/v189-checkpoint10/final_full_baseline_overlay` | `15/315` | `4.76%` | `8/155` | `7/160` | `0` |
| `v189_checkpoint10_stripped` | `.../packages/v189-checkpoint10/final_stripped` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v94_final_raw` | `felipesp1983/kg1-nemotron-lora-v94-equation-crypt/final` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v94_final_overlay` | `.../packages/v094-final/final_full_baseline_overlay` | `16/315` | `5.08%` | `8/155` | `8/160` | `0` |
| `v95_checkpoint20_bit_rehearsal` | `felipesp1983/kg1-nemotron-lora-v95-bit-rehearsal/checkpoint-20` | `18/315` | `5.71%` | `9/155` | `9/160` | `0` |
| `v96_a0p020_interp` | `felipesp1983/kg1-nemotron-lora-v96-v95interp/interpolations/v096-v91-v95cp20-interp-a0p020` | `19/315` | `6.03%` | `10/155` | `9/160` | `1` |
| `v97_last3_uniform_soup` | `felipesp1983/kg1-nemotron-lora-v97-v91-soups/soups/v097-v91-checkpoint-soup-last3_uniform` | `19/315` | `6.03%` | `10/155` | `9/160` | `0` |
| `v101_checkpoint20_overlay` | `felipesp1983/kg1-nemotron-lora-v101-tong-selector-v2/packages/v101-checkpoint20-lmheadfix/final_full_baseline_overlay` | `17/315` | `5.40%` | `8/155` | `9/160` | `0` |
| `v101_final_overlay` | `felipesp1983/kg1-nemotron-lora-v101-tong-selector-v2/packages/v101-final-lmheadfix/final_full_baseline_overlay` | `19/315` | `6.03%` | `9/155` | `10/160` | `0` |

Decisao:

- Rejeitar todos os candidatos V253 como rota de promocao, baseline, initializer, merge, DARE/TIES, router ou ensemble deployable.
- O melhor grupo ficou em `19/315`, contra baseline V226 `191/315`.
- O prescore historico de V189/V94/V95/V96/V97/V101 nao transferiu para o weak gate canonico atual.
- O problema nao e truncation: quase todos tiveram `0` truncamento. A falha e desalinhamento de adapter/contrato/prompt/modelo.
- Nao gastar novo H200 nesses repositorios sem uma evidencia externa nova que explique e corrija o desalinhamento.

Impacto no roadmap:

- Ficam descartados, como melhoria direta, os repositorios publicos locais/HF avaliados em V251, V252 e V253.
- A rota objetiva agora e remover dependencia de Drive para os pesos fortes conhecidos:
  1. localizar ou publicar no HF o adapter protegido `V194` e o checkpoint forte `V226`;
  2. validar esses pesos fortes com o mesmo wrapper HF e weak CSV canonico;
  3. so depois executar smoke training curto com V249/novos dados, sempre partindo de um initializer forte e com weak eval imediato.
- Se `V194/V226` nao puderem ser colocados no HF, a proxima acao barata e CPU-only: minerar `equation_transform` simbolico/misto e gerar probes/verifiers; nao ha justificativa para treino H200 longo a partir dos adapters fracos.

## Auditoria Google Drive KG1 - inventario rigoroso 2026-05-10

Escopo:

- Fonte: Google Drive `MyDrive`, roots KG1 relevantes.
- Inventario local versionado:
  - `artifacts/drive_audits/google_drive_kg1_inventory_latest.json`
  - `artifacts/drive_audits/google_drive_targeted_report_metrics_latest.json`
  - `artifacts/drive_audits/google_drive_v230_v238_manifest_decisions_latest.json`
- Total catalogado: `1879` arquivos, `301.446 GiB`.
- Artefatos por tipo: `85` adapters completos, `85` `.safetensors`, `29` zips, `423` CSVs, `54` JSONLs, `232` reports/manifests de avaliacao, `11` notebooks.

Principais roots por tamanho:

| Root Drive | Arquivos | Tamanho |
|---|---:|---:|
| `KG1_NVIDIA_V199` | `41` | `39.301 GiB` |
| `KG1_NVIDIA_V206C` | `24` | `37.537 GiB` |
| `KG1_NVIDIA_V198` | `38` | `34.517 GiB` |
| `KG1_NVIDIA_V201` | `44` | `31.016 GiB` |
| `KG1_PUBLIC_ADAPTERS` | `62` | `25.611 GiB` |
| `KG1_NVIDIA_V202C` | `32` | `19.482 GiB` |
| `KG1_NVIDIA_V221` | `68` | `18.544 GiB` |
| `KG1_NVIDIA_V202D` | `26` | `15.308 GiB` |
| `KG1_NVIDIA_V226` | `44` | `11.963 GiB` |
| `KG1_NVIDIA_V227` | `100` | `8.003 GiB` |

Adapters completos no Drive com maior valor operacional:

| Artefato | Status | Uso correto |
|---|---|---|
| `KG1_NVIDIA_V226/output_v226_equation_checkpoint_sweep/train_v226_v194_micro_lr2e9_s6/checkpoint-1` | Melhor baseline weak conhecido: `191/315`, equation `55/155`, bit `136/160`, trunc `0` | P0: publicar/validar no HF como initializer forte; nao rebaixar por adapters HF fracos |
| `KG1_NVIDIA_V226/.../checkpoint-2` | `189/315`, equation `55`, bit `134`, trunc `1` | Nao promover; manter como comparativo |
| `KG1_NVIDIA_V226/.../checkpoint-3` | `190/315`, equation `55`, bit `135`, trunc `1` | Nao promover; manter como comparativo |
| `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter` | V194 protegido, V221 weak `190/315`, equation `54`, bit `136`, trunc `0` | P0/P1: publicar/validar no HF; bom guardrail de bit |
| `KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter` | V221 registry weak `190/315`, equation `55`, bit `135`, trunc `0` | Manter como comparativo; nao supera V226 |
| `KG1_NVIDIA_V227/.../final_adapter` e `checkpoint-1` | V229 weak agregado `16/315` | Rejeitado; nao usar como initializer, merge, router ou treino |

Reports antigos do Drive que nao devem ser confundidos com o gate weak canonico:

| Linha | Resultado observado | Decisao |
|---|---:|---|
| V207A `v194_baseline_eval` | `822/947`, per family: bit `135/160`, equation `55/155`, demais familias `100%` | Evidencia util de diagnostico amplo; nao substitui V221/V230 weak canonico |
| V207A `v206c_s0p100` | `158/315`, trunc `40` | Rejeitar: abaixo de V226 e truncation alto |
| V207A `v206c_s0p020` | `157/315`, trunc `40` | Rejeitar |
| V207A `v206b_answer_only` | `150/315`, trunc `43` | Rejeitar |
| V207B public adapters Kienngx COT | `44/315` e `32/315`, trunc `126/202` | Rejeitar como adapter; COT longo e desalinhado |
| V214 micro | `137/315`, trunc `55` | Rejeitar |
| V216 equation push | `124/315`, trunc `57` | Rejeitar |
| V217 pre-registry eval antigo | `118/315`, trunc `81` | Rejeitar esse decode antigo; usar V221 registry para V217 |
| V218 decode rescue | `18/315`, trunc `0` | Rejeitar |
| V219 think decode A/B | `6/315`, trunc `225` | Rejeitar |
| V223 equation rescue | `107/315`, trunc `97` | Rejeitar |

Achado de decode V225:

- V225 equation-only sweep mostrou que `think_strict_boxed` elevou `v194` e `v217` para `56/155` em `equation_transform`, contra `54-55/155` no registry weak.
- Esse ganho e pequeno e ainda fica abaixo do gate `60/155`, mas e evidencia concreta de que prompt/decode pode recuperar `+1` linha de equation sem mexer em pesos.
- Acao: manter como experimento P1 de prompt/parser, mas nao usar como liberacao de full eval.

Manifestos Drive V230-V238:

| Versao | Decisao registrada | Implicacao |
|---|---|---|
| V230 | `row_level_oracle_improves_but_misses_weak_gate` | Oracle chega a `197/315`, mas equation so `57/155`; nao deployavel |
| V231 | `mine_equation_solvers_before_training` | Minerar solver antes de treino |
| V232 | `build_v233_verified_equation_solver_probes` | Criar probes verificados por rota |
| V233 | `improve_solver_parsers_before_eval` | Solver ainda sem ganho deployavel |
| V234 | `external_intel_triage_ready_for_source_download` | Intel externa organizada, mas sem payload aprovado |
| V235 | `manual_source_access_or_license_required_before_download` | Bloqueio correto por credenciais/licenca/hash |
| V236 | `continue_local_solver_development` | DSL local ainda sem ganho equation deployavel |
| V237 | `build_prompt_format_specific_parser_before_solver` | Formato Alice inline precisa parser especifico |
| V238 | `continue_alice_parser_development` | Apenas `1` override verificado; insuficiente para gate |

Conclusao da auditoria Drive:

- O Drive contem historico rico, mas os unicos pesos fortes comprovados seguem sendo V226 checkpoint-1, V194 protegido e V217 como comparativo.
- A maior parte dos adapters antigos/publicos tem truncation alto, score fraco ou desalinhamento de prompt/modelo; nao devem ser usados para merge, soup, router ou treino.
- O melhor uso imediato do Drive e operacional: transferir/publicar V226 checkpoint-1 e V194 para HF com hash/config completos, validar no weak gate HF, e depois usar esses pesos fortes como initializer para qualquer smoke training.
- O melhor uso analitico do Drive e continuar minerando `equation_transform` simbolico/misto, usando V230 miss packs e V237/V238 parser evidence, porque trocar adapter nao resolveu o gap de `5` linhas em equation.

## HF bridge concluido - pesos fortes V194/V226

Objetivo:

- Remover a dependencia operacional do Google Drive para os pesos fortes antes de novos jobs HF.
- Evitar gasto H100/H200 partindo de adapters fracos que ja foram rejeitados.

Repo HF privado:

- `felipesp1983/kg1-strong-adapters-v194-v226`
- URL: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226`
- SHA remoto base dos pesos validado: `1bb23fdbc3f5ccadd36b91e8f7db9d7474bf6312`
- Registro local: `artifacts/hf_uploads/KG1_STRONG_ADAPTERS_HF_BRIDGE_20260510.md`

Conteudo validado:

| Pasta HF | Origem | Weak conhecido | SHA256 |
|---|---|---:|---|
| `v226_checkpoint1` | Drive V226 checkpoint-1 | `191/315`, equation `55`, bit `136`, trunc `0` | `f4e2083d83f13a102cd86e5d1295a8603264856c17ec35c357188e1acde6ea79` |
| `v194_protected` | Drive V202D/V194 protegido | `190/315`, equation `54`, bit `136`, trunc `0` | `01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f` |

Nota de contrato de inferencia:

- Os scores historicos `190-191/315` foram medidos no contrato V221: thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `V221_PROMPT_SUFFIX = "\nPlease put your final answer inside ..."` e respostas longas com raciocinio antes do ultimo `\boxed{}`.
- O job HF V254 curto (`v254-h200-weak-strong-bridge-20260511T004420Z`) rodou com contrato V245/V230 curto: thinking desabilitado, `max_tokens=96`, `max_model_len=4096` e sufixo "Return only one line".
- Resultado V254 curto:
  - `hf_v226_checkpoint1_strong_bridge`: `16/315`, equation `8`, bit `8`, trunc `0`.
  - `hf_v194_protected_strong_bridge`: `17/315`, equation `9`, bit `8`, trunc `0`.
- Interpretacao: esse resultado nao invalida os adapters fortes; ele prova que a weak-eval curta nao reproduz o contrato que gerou os scores fortes. O wrapper HF precisa suportar ambos os contratos e rotular explicitamente qual foi usado.
- Ajuste implementado em `scripts/hf_job_weak_eval_v245.py`: `KG1_DISABLE_THINKING`, `KG1_NO_PROMPT_SUFFIX` e `KG1_PROMPT_SUFFIX` agora controlam o modo de prompt; o default continua preservando o modo curto V245 para compatibilidade.

Reproducao HF com contrato V221:

- V255 H200 `v255-h200-v221contract-v194-20260511T005050Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0128f9aff1cd33e8f33271`.
  - Commit de codigo exigido: `6dd0936bc47496fcfc6201446f73c0db15df54b3`.
  - Contrato: thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, sufixo V221.
  - Resultado: `191/315`, equation `56/155`, bit `135/160`, trunc `1`, ACC `60.63%`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226/commit/82760153d0eacc365e4c037d50610931236605e7`.
  - Interpretacao: o HF reproduz o patamar historico sob o contrato V221. A divergencia pequena vs V221 Drive (`190/315`, equation `54`, bit `136`, trunc `0`) e aceitavel como variacao de runtime/extracao ate auditoria linha a linha, mas confirma que o modo curto V254 era o erro principal.
- V256 H200 `v256-h200-v221contract-v226ckpt1-20260511T0110Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a012c1f317220dbbd1a7798`.
  - Status: `COMPLETED`.
  - Contrato: thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, sufixo V221.
  - Resultado: `191/315`, equation `56/155`, bit `135/160`, trunc `1`, ACC `60.63%`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-strong-adapters-v194-v226/commit/1469bb73c0d4f31638ac59fb0c08ec2e42237ed6`.
  - Interpretacao: no HF, `v226_checkpoint1` e `v194_protected` empatam no agregado sob o contrato V221 reproduzido. A decisao agora depende de diff linha a linha e comparacao contra os reports Drive V221/V226, nao de novo treino.

Diff linha a linha V255 vs V256:

- Artefatos locais:
  - `artifacts/hf_eval_diffs/V255_V256_LINE_DIFF_SUMMARY_20260511.md`
  - `artifacts/hf_eval_diffs/v255_v256_line_diff_summary_20260511.json`
  - `artifacts/hf_eval_diffs/v255_v256_family_delta_20260511.csv`
  - `artifacts/hf_eval_diffs/v255_v256_correctness_deltas_20260511.csv`
- IDs alinhados: `315/315`.
- Predicoes textuais diferentes: `5`.
- Linhas com mudanca de corretude: `2`.
- `equation_transform`: V194 `56`, V226 `56`, net `0`, predicoes diferentes `2`, sem mudanca de corretude.
- `bit_manipulation`: V194 `135`, V226 `135`, net `0`, V226 ganha `1` linha e perde `1` linha.
- Linhas de delta:
  - `4ef88f92`: V226 corrige `01010111` onde V194 errou `01011111`.
  - `8740ed31`: V194 acerta `01101000` onde V226 erra `01111000`.
- Conclusao: V226 checkpoint-1 deve ser mantido como initializer forte por historico Drive, mas nao trouxe ganho observavel sobre V194 no contrato HF V221. A proxima melhoria precisa atacar `equation_transform`, especialmente simbolico/misto.

Diff Drive vs HF V221-contract:

- Artefatos locais:
  - `artifacts/hf_eval_diffs/DRIVE_VS_HF_V221CONTRACT_DIFF_SUMMARY_20260511.md`
  - `artifacts/hf_eval_diffs/drive_vs_hf_v221contract_diff_summary_20260511.json`
  - `artifacts/hf_eval_diffs/drive_v221_v194_vs_hf_v255_v194_family_delta_20260511.csv`
  - `artifacts/hf_eval_diffs/drive_v221_v194_vs_hf_v255_v194_correctness_deltas_20260511.csv`
  - `artifacts/hf_eval_diffs/drive_v226_vs_hf_v256_v226_family_delta_20260511.csv`
  - `artifacts/hf_eval_diffs/drive_v226_vs_hf_v256_v226_correctness_deltas_20260511.csv`
- Drive V221 V194 vs HF V255 V194:
  - IDs alinhados: `315/315`.
  - Predicoes diferentes: `14`.
  - Mudancas de corretude: `5`.
  - `equation_transform`: Drive `54`, HF `56`, net `+2` HF.
  - `bit_manipulation`: Drive `136`, HF `135`, net `-1` HF, truncation diff `1`.
- Drive V226 vs HF V256 V226:
  - IDs alinhados: `315/315`.
  - Predicoes diferentes: `11`.
  - Mudancas de corretude: `4`.
  - `equation_transform`: Drive `55`, HF `56`, net `+1` HF.
  - `bit_manipulation`: Drive `136`, HF `135`, net `-1` HF, truncation diff `1`.
- Interpretacao: a diferenca HF/Drive e pequena e favorece levemente equation, mas custa bit e truncation. Nao e melhoria robusta nem deployable; e evidencia de sensibilidade operacional do contrato longo. Para promocao, o gate deve continuar exigindo total `>=193`, equation `>=60`, bit `>=133`, trunc `<=3`, com preferencia por bit `>=136` como guardrail interno.

V257/V258 HF-only smoke training com V249:

- V257 H200 `v257-h200-v249-v226ckpt1-smoke-20260511T013254Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a013204317220dbbd1a77cd`.
  - Status: `COMPLETED`.
  - Dataset: V249 public non-weak target, train `2558`, val `284`, hashes e family counts validados no preflight.
  - Initializer: `felipesp1983/kg1-strong-adapters-v194-v226/v226_checkpoint1`.
  - Treino smoke: `MAX_STEPS=4`, `lr=5e-8 -> 2.5e-8`, LoRA trainable somente `q_proj,k_proj,v_proj,o_proj,lm_head`, sampling weighted replacement com equation peso `1.80`.
  - Gate de tokenizacao: train `2558/2558`, val `284/284`, truncation `0`, prompt truncation `0`, offset masks completos.
  - Upload HF: `felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke`, checkpoint-2, checkpoint-4 e final completos.
- V258 H200 `v258-h200-v221contract-v257-smoke-eval-20260511T015236Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0136a1317220dbbd1a77e5`.
  - Status: `COMPLETED`.
  - Contrato: V221 reproduzido, thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, sufixo V221.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke/commit/be437d3f431e0c46998243e573cda53fa68f26c6`.
  - Resultados:
    - `v257_checkpoint_2_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `1`.
    - `v257_checkpoint_4_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `1`.
    - `v257_final_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `2`.
  - Melhor candidato: `checkpoint-4`. Ele melhora o V256 HF em `+1` total e `+1` bit, sem alterar equation e truncation.
  - Delta linha a linha vs V256 HF: ganho unico em `4ada9150`, family `bit_manipulation`, expected `01111011`, V256 predicted `01111111`, V258 checkpoint-4 predicted `01111011`.
  - Gate: nao passa. Total fica `1` abaixo de `193` e equation fica `4` abaixo de `60`.
  - Artefatos locais:
    - `artifacts/hf_eval_diffs/V258_V257_SMOKE_EVAL_SUMMARY_20260511.md`
    - `artifacts/hf_eval_diffs/v258_v257_smoke_eval_summary_20260511.json`
    - `artifacts/hf_eval_diffs/v256_v226_vs_v258_ckpt4_family_delta_20260511.csv`
    - `artifacts/hf_eval_diffs/v256_v226_vs_v258_ckpt4_correctness_deltas_20260511.csv`
    - `artifacts/hf_eval_diffs/v255_v194_vs_v258_ckpt4_family_delta_20260511.csv`
    - `artifacts/hf_eval_diffs/v255_v194_vs_v258_ckpt4_correctness_deltas_20260511.csv`
- Interpretacao: V249 smoke curto produziu sinal positivo real, mas nao resolveu o bottleneck. O proximo passo nao deve ser treino longo cego; deve ser um smoke pequeno, equation-targeted, usando `checkpoint-4` como seed somente se o gate HF repetir hashes, contrato V221 e weak eval imediato.

V259/V260B HF-only equation-focused smoke:

- V259 H200 `v259-h200-v249-eqfocus-v257ckpt4-smoke-20260511T023318Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a01402d317220dbbd1a7819`.
  - Status: `COMPLETED`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke`.
  - Initializer: `felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke/checkpoint-4@be437d3f431e0c46998243e573cda53fa68f26c6`.
  - Dataset: V249 public non-weak target, train `2558`, val `284`, hashes e family counts validados antes de treino.
  - Receita: `MAX_STEPS=8`, `lr=2e-8 -> 1e-8`, LoRA trainable `q_proj,k_proj,v_proj,o_proj,lm_head,up_proj,down_proj`, `equation_transform=3.00`, `bit_manipulation=0.80`.
  - Gates: H200, CUDA, dataset hashes, tokenizacao sem truncation, offset masks, import `causal_conv1d`, `mamba_ssm`, adapter load coverage `12011/12011`, trainable ratio `2.6908%`.
  - Upload HF: checkpoint-4, checkpoint-8 e final completos.
- V260B H200 `v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0145edaff1cd33e8f333a0`.
  - Status: `COMPLETED`, `failureCount=0`, running `1684s`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke/commit/496d31f4284ec45b278e561aee4543005767a661`.
  - Contrato: V221 reproduzido, thinking habilitado, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`.
  - Weak CSV SHA256: `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
  - Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
  - Resultados:
    - `v259_checkpoint_4_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `0`.
    - `v259_checkpoint_8_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `1`.
    - `v259_final_v221_contract`: `190/315`, equation `56/155`, bit `134/160`, trunc `1`.
  - Melhor candidato: `checkpoint-4`; ele empata o V258 checkpoint-4 no total e em equation/bit, melhorando apenas truncation de `1` para `0`.
  - Delta vs V258 checkpoint-4: `7` linhas com predicao alterada; net score `0`. Perdeu um bit previamente correto (`4ef88f92`) e ganhou um bit (`59bee375`); quatro predicoes de equation mudaram, mas continuaram incorretas.
  - Gate: nao passa. Total fica `1` abaixo de `193`; equation fica `4` abaixo de `60`.
  - Artefatos locais:
    - `artifacts/hf_eval_diffs/v260b_v259_eqfocus_20260511/README.md`
    - `artifacts/hf_eval_diffs/v260b_v259_eqfocus_20260511/v260b_v259_eqfocus_summary.json`
    - `artifacts/hf_eval_diffs/v260b_v259_eqfocus_20260511/v260b_vs_v258_checkpoint4_changed_rows.csv`
- Interpretacao: a receita V259 nao melhorou o gargalo. Continuar esse treino por mais steps e gasto H200 sem novo dado/verifier provavelmente so degrada bit ou mantem equation em `56/155`. O proximo passo deve voltar para mineracao deterministica de `equation_transform` simbolico/misto, parser/verifier e dados externos auditados, nao treino cego.

V261 HF-only prompt/decode sweep:

- V261 H200 `v261-h200-v221contract-nosuffix-prompt-sweep-20260511T034434Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a0150e0317220dbbd1a785b`.
  - Status: `CANCELED` por gate de FinOps apos o primeiro candidato completo.
  - Motivo do cancelamento: a variante `thinking habilitado + sem prompt suffix` gerou resposta longa, manteve `equation_transform` sem ganho e derrubou `bit_manipulation` de forma severa; continuar os outros candidatos repetiria o mesmo risco de custo sem sinal de melhoria.
  - Contrato: V221 reproduzido, H200 validado, vLLM `0.20.1`, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, `KG1_NO_PROMPT_SUFFIX=1`, `KG1_DISABLE_THINKING=0`.
  - Resultado completo antes do cancelamento:
    - `v259_checkpoint4_nosuffix`: `155/315`, equation `55/155`, bit `100/160`, trunc `1`.
  - Delta vs melhor V260B/V258:
    - total `-37`;
    - equation `-1` contra `56/155` do V259/V258, e `0` contra o baseline historico `55/155`;
    - bit `-36` contra `136/160`.
  - Artefatos remotos: nenhum manifest foi publicado antes do cancelamento; a evidencia canonica desta execucao e o log HF do job com `candidate_summary` completo.
- Interpretacao: remover o sufixo `Return only one line: \boxed{answer}` e liberar pensamento nao melhora o gargalo de equation no contrato weak; ao contrario, degrada extraction/bit. Nao repetir varreduras `no suffix` em H200 dentro do budget atual.

V262/V263 HF-only adapter soup:

- V262 CPU `v262-hf-cpu-adapter-soups-20260511T044654Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a015f7c317220dbbd1a78a1`.
  - Status: `COMPLETED`.
  - Output repo: `felipesp1983/kg1-nemotron-lora-v262-adapter-soups`.
  - Inputs validados:
    - `v226_checkpoint1` SHA `f4e2083d83f13a102cd86e5d1295a8603264856c17ec35c357188e1acde6ea79`.
    - `v257_checkpoint4` SHA `87b52699231f35823afd23f8d0326bbfe2de742a13cb06771f759d45488007fd`.
    - `v259_checkpoint4` SHA `01b90c1745e5eb3a7fb47fc4c81ff1fdacc17098cc79faf533b05f7b91913163`.
  - Tensor contract: `12011` tensors, contract SHA `3419375a77ddf718fcec58e0ed3da179b25cbae2ed22d74de87fad51994925fb`.
  - Soups publicados:
    - `soup_v226_050_v257_050` weights SHA `b309a740469a0a435afba9bd42cc3d800cc2b1bc42685f58d8ce8c9e5294c33b`.
    - `soup_v226_050_v259_050` weights SHA `965160a7811ce44209123d46ac33d05174caf294d729b887a259bd9b59873cd7`.
    - `soup_v226_034_v257_033_v259_033` weights SHA `233c06c2a2499045fb0c017e0245e39e344262619e4d14125d245b083f7ebbaf`.
- V263 H200 `v263-h200-v262-soups-v221contract-eval-20260511T050000Z`
  - Job HF: `https://huggingface.co/jobs/felipesp1983/6a01628faff1cd33e8f334fc`.
  - Status: `COMPLETED`.
  - Upload HF: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v262-adapter-soups/commit/f723dde4cba16e92c5561f6ebf09d602dd22af83`.
  - Contrato: V221 reproduzido, H200 validado, vLLM `0.20.1`, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`.
  - Weak CSV SHA256: `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
  - Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
  - Resultados:
    - `soup_v226_050_v257_050_v221_contract`: `192/315`, equation `56/155`, bit `136/160`, trunc `1`.
    - `soup_v226_050_v259_050_v221_contract`: `191/315`, equation `56/155`, bit `135/160`, trunc `1`.
    - `soup_v226_034_v257_033_v259_033_v221_contract`: `190/315`, equation `56/155`, bit `134/160`, trunc `2`.
  - Melhor candidato: `soup_v226_050_v257_050_v221_contract`, mas ele apenas empata o melhor total V258/V260B e piora truncation contra V260B checkpoint-4.
  - Gate: nao passa. Total fica `1` abaixo de `193`; equation fica `4` abaixo de `60`.
- Comparacao linha-a-linha V263 vs V260B:
  - Artefatos locais: `artifacts/hf_eval_diffs/v263_soups_vs_v260b_20260511/`.
  - Melhor soup vs V260B checkpoint-4: `1` ganho e `1` perda, ambos em `bit_manipulation`; `0` ganhos e `0` perdas em `equation_transform`.
  - Oracle V260B + melhor soup: total `193`, bit `137`, equation `56`. Esse oracle ainda falha o gate de equation por `4`, entao nao justifica novo roteador/soup deployable.
- Decisao FinOps/QA: nao repetir adapter soup em H200 dentro do budget atual. A rota nao mudou `equation_transform`; o proximo gasto em GPU so deve ocorrer apos um preflight barato que gere evidencia concreta de `+4` ou mais em equation sem reduzir bit abaixo de `136/160`.

Regra de preservacao dos artefatos historicos:

- Muitos arquivos do Drive e dos notebooks anteriores participaram da trajetoria ate o score amplo `0.86`. Eles nao devem ser apagados por tamanho ou idade sem classificacao previa.
- Observacao operacional: varios artefatos analisados agora foram efetivamente usados para chegar ao score `0.86`; portanto a regra padrao e preservar, auditar e publicar manifest, nao limpar.
- Antes de qualquer limpeza, classificar cada artefato como:
  - `P0_keep_repro`: peso, prediction CSV, report, manifest, dataset ou notebook que reproduz ou explica scores `0.86`, `190-191/315`, V207A, V221, V226, V230 ou V255.
  - `P1_keep_audit`: logs, CSVs intermediarios e notebooks que ajudam a auditar contrato de prompt, extractor, row contract, hashes ou regressao.
  - `P2_archive`: duplicatas grandes com hash identico e copia canonical ja publicada em HF/Drive com manifest.
  - `P3_delete_candidate`: apenas cache, download parcial, snapshot duplicado sob `evals/`, arquivo `.partial`, ou artefato comprovadamente fraco e reproduzivel em outro local.
- Exclusao so e permitida para `P3_delete_candidate`, com registro de path, bytes, hash quando aplicavel e motivo. Arquivos ligados ao score `0.86` entram como `P0_keep_repro` ate prova contraria.

Validacoes:

- Ambos com `tensor_count=12011`.
- Ambos sem arquivos `.partial`.
- Manifest remoto `strong_adapters_validation_manifest.json` lido com sucesso.
- Staging local `%TEMP%/kg1_drive_strong_adapters_hf_20260510` removido apos upload, liberando `8535403351` bytes logicos.

Proximo passo:

1. Nao promover V259/V261/V263 para full eval/submissao; nenhum checkpoint ou soup passou o weak gate.
2. Manter `v257_checkpoint_4`, `v259_checkpoint_4` e `soup_v226_050_v257_050` como seeds tecnicos reproduziveis, mas nao investir em continuacao longa, adapter soup ou prompts `no suffix` sem novo sinal de equation.
3. Priorizar HF-only CPU/low-cost para minerar `equation_transform` simbolico/misto: V230 miss packs, V232/V238 workitems, raw traces auditados e regras/verifiers com abstain.
4. So voltar a H200 quando houver candidato deterministico ou dataset filtrado que, em preflight, mire explicitamente os `+4` acertos de equation sem derrubar bit abaixo de `136/160`.
5. Antes de novo treino util, liberar acesso humano aos datasets HF gated `andy279/nemotron-reasoning-challenge-raw-traces` e `andy279/nemotron-reasoning-challenge`; V264 confirmou `403` em todos os `5` arquivos P0.
6. Reusar V249/V250/V242 somente com preflight de hashes, anti-leakage, row-contract, tokenizacao, estimativa de custo e kill-switch por primeiro candidato.
