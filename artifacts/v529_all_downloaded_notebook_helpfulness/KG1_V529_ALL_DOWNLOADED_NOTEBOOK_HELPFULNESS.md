# KG1 V529 - Analise Dos Notebooks Baixados Do Kaggle

Data: 2026-05-17

## Escopo

Foram analisados os kernels publicos baixados em
`artifacts/v527_kaggle_code_audit/pulled_code/`.

Artefatos da varredura:

- `all_downloaded_helpfulness.csv`;
- `challenge_relevant_helpfulness.csv`;
- `challenge_relevant_bit_candidates.csv`;
- `challenge_relevant_equation_candidates.csv`;
- `challenge_relevant_training_packaging_candidates.csv`;
- `v529_all_downloaded_helpfulness_manifest.json`;
- `v529_challenge_relevant_manifest.json`.

Cobertura:

| Item | Quantidade |
|---|---:|
| Kernels puxados do Kaggle | 704 |
| Diretórios parseados | 702 |
| Diretórios marcados como ligados ao desafio | 699 |
| Candidatos automaticos com sinal de bit | 190 |
| Candidatos automaticos com sinal de equation | 89 |
| Candidatos automaticos de treino/packaging | 197 |

Observacao: a varredura automatica foi usada apenas como triagem. Ela gera
falsos positivos quando o notebook usa palavras como `bit`, `train` ou
`adapter` fora do contexto util. A lista abaixo e a lista manualmente filtrada.

## Pode Ajudar Agora

| Prioridade | Notebook | Ajuda real | Acao KG1 |
|---|---|---|---|
| P0 | `pjt222/nemotron-cot-review` | Bit: cascata NOT/XOR/rotation/permutation/GF(2)/ANF. Equation: arithmetic, digit reversal, `Z_94`/mod-94 simbolico. | Portar para um gate CPU V530: solver primeiro, trace depois, sem GPU. |
| P0 | `pearpn25/bit-cot-85-1364-sample` | Bit solver-first com cobertura alta; final answer forçado pelo solver verificado. | Comparar contra nossos misses bit; gerar traces curtas somente para source rows verificadas. |
| P0 | `konbu17/bit-manipulation-solver-cot-generator` | Enumeracao bit-wise com operacoes assimetricas, shift/stride, priors e ambiguity scoring. | Completar o solver bit com INHIB/IMPL/CH/MAJ/XOR3/stride antes de novo treino. |
| P0 | `zzys0316/full-pipeline-nvidia-nemotron-3-reasoning` | DSL equation com concat, reverse concat, operands/result reversed, prefix/suffix operator encoding e operacoes raras. | Expandir DSL equation CPU e medir ganho em source/weak diagnostic antes de SFT. |
| P1 | `dgxchen/training-with-unsloth-to-achieve-0-85-lb` | Receita LoRA consistente: rank 32, alpha 32, max length alto, batching estratificado. | Usar como guardrail de configuracao, nao como novo dado. |
| P1 | `huikang/tinker-submission-notebook` e `huikang/adapter-validation-notebook` | Conversao/auditoria de adapter Tinker: target modules, experts, SVD para `in_proj`, namespace e validacao vLLM. | Reforcar gates de package/conversao; nao resolve familia sozinho. |
| P1 | `llkh0a/nemotron-unsloth-sft-training-3-30-2` | Baseline publico de SFT/CoT; mostra limites do SFT sem solver bit forte. | Usar como referencia negativa e para boxed-loss; nao repetir broad SFT. |
| P1 | `kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset` | Alerta importante: bit CoT estruturalmente errado derruba LB. | Bloquear traces bit que nao expliquem regra per-bit/stride. |
| P2 | `habanwer/nemotron-atlas` | Ideia de LoRA em modulos sempre ativos/shared experts, checkpoint isolation e loss masking. | Somente ablation futura, depois de CPU signal; nao e prioridade para hoje. |

## Ajuda Apenas Como Empacotamento

| Notebook | Uso permitido | Nao usar para |
|---|---|---|
| `afr1ste/nemotron-0-86-tinker-adapter-guide` | Verificar zip root, `adapter_config.json`, `adapter_model.safetensors`, provenance do Tinker adapter. | Inferir tecnica nova de bit/equation. |
| `safar1/lb-score-0-86` | Confirmar que score `0.86` veio de adapter artifact pronto. | Treinar, minerar regra ou substituir nosso adapter sem gate. |
| `mohamedamr992/0-86-adapter-packaging-workflow` | Conferir estrutura de pacote para submit. | Resolver familia. |
| `teasue05/tinker-submission-notebook` | Repeticao de fluxo Tinker/package. | Qualquer decisao de treino. |

## Nao Ajuda Ou Fica Bloqueado

| Tipo | Motivo |
|---|---|
| Broad SFT/GRPO sem solver novo | Ja testamos variantes; loss cai e ACC nao transfere. |
| Notebooks de score que so zipam adapter publico | Nao ensinam regra de familia. |
| Synthetic dataset generico sem solver/verifier deterministico | Risco alto de ruido e conflito de formato. |
| Notebooks falsos positivos por texto/metadata | Nao estao resolvendo o desafio apesar de aparecerem na triagem. |
| Public adapter como substituicao direta | Pode servir como referencia de schema/provenance, mas nao como ganho submit-safe nosso sem gate. |

## Decisao V529

O melhor uso dos arquivos baixados nao e rodar outro H200. O melhor uso e
transformar os kernels P0 em implementacao CPU verificavel:

1. Criar V530 como harness CPU de solver:
   - bit: per-bit/bitsum/stride, INHIB/IMPL, CH/CHO, MAJ3, XOR3, GF(2), ANF;
   - equation: concat/reverse concat, operand/result reversal, `+1/-1`,
     divisao/modulo, prefix/suffix operator encoding, `Z_94`/mod-94.
2. Medir somente comportamento label-free:
   - source train para cobertura permitida;
   - weak diagnostic apenas para medir, nunca como treino;
   - hard guard `8740ed31=01101000`;
   - promocao minima `total>=193`, `equation>=57`, `bit>=136`, `trunc=0`.
3. Gerar novo trace pack somente a partir de source rows resolvidas pelo V530.
4. Usar `LOSS_NORMALIZATION_MODE=example_mean` e traces curtas; bloquear
   `token_mean` para este pacote.
5. Liberar GPU apenas se o V530 mostrar novo sinal CPU acima do V523/V524.

## Resposta Direta

Sim, alguns baixados podem ajudar, mas nao como submissao pronta. Os que ajudam
sao os notebooks que contem solvers e geradores de CoT verificaveis:

- `pjt222/nemotron-cot-review`;
- `pearpn25/bit-cot-85-1364-sample`;
- `konbu17/bit-manipulation-solver-cot-generator`;
- `zzys0316/full-pipeline-nvidia-nemotron-3-reasoning`;
- `huikang/*` apenas para adapter conversion/package validation.

Os notebooks `0.86` ajudam a nao errar o pacote, mas nao explicam como subir
`bit_manipulation` ou `equation_transform`.
