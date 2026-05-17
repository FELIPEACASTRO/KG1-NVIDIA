# KG1 V530 - Auditoria Dos Anexos Bit CoT

Data: 2026-05-17

## Arquivos Analisados

| Arquivo | SHA256 | Conteudo |
|---|---|---|
| `C:\Users\davis\Downloads\archive.zip` | `157b80a2e8e4ae37ca6411ca8462ecdcb4ec79f6817116d45bcfd7bed7a2a340` | Dataset Kaggle `konbu17/bit-manipulation-cot-dataset` v2 |
| `C:\Users\davis\Downloads\archive (1).zip` | `e5ca20ea63c7709d6a9f0e9b309ffe0677f17966755e90265e528c524b97a4ca` | Dataset sintetico `synthetic_bit_manipulation.csv` |
| `C:\Users\davis\Downloads\bit-manipulation-cot-dataset-metadata.json` | `343c9a33db0c2e6f9d4ed920ac4f9341e76af468c8018c7ba1efae997981d367` | Metadados Croissant do dataset `konbu17/bit-manipulation-cot-dataset` |
| `C:\Users\davis\Downloads\archive (2).zip` | `dfc732f88cae6cf92b729f692a029d6292717405d8e5d400dde9e85560302ed3` | Dataset oficial `train.csv/test.csv`; duplicado de `competition_train.csv` |
| `C:\Users\davis\Downloads\archive (3).zip` | `8ed0320c8389a25bc14f80947879e413fc21356fdf3f5d85d2f3263d7696918b` | Dataset oficial `train.csv/test.csv`; mesmo conteudo interno de `archive (2).zip` |
| `C:\Users\davis\Downloads\archive (4).zip` | `e00dd68f2847e450bd55f14e99c1d0c53721e310fcb7c7f3444c366c5c86cd78` | Dataset oficial duplicado + `smart_train.csv` |
| `C:\Users\davis\Downloads\archive (5).zip` | `157b80a2e8e4ae37ca6411ca8462ecdcb4ec79f6817116d45bcfd7bed7a2a340` | Duplicata exata de `archive.zip` |
| `C:\Users\davis\Downloads\archive (6).zip` | `e5ca20ea63c7709d6a9f0e9b309ffe0677f17966755e90265e528c524b97a4ca` | Duplicata exata de `archive (1).zip` |

## Resultado Do `archive.zip`

Conteudo:

- `bit_manipulation_cot_success.csv`;
- `bit_manipulation_cot_failed.csv`;
- `confidence_analysis.png`;
- `error_analysis.png`.

Validação contra `C:\Users\davis\Downloads\competition_train.csv`:

| Conjunto | Linhas | Overlap com train | Prompt mismatch | Answer mismatch | Familia |
|---|---:|---:|---:|---:|---|
| success | 1134 | 1134 | 0 | 0 | bit |
| failed | 374 | 374 | 0 | 0 | bit |
| total coberto | 1508 | 1508 | 0 | 0 | bit |

Cobertura:

- `competition_train.csv` tem `1602` linhas bit.
- O anexo cobre `1508/1602` linhas bit (`94.13%`).
- O arquivo `success` cobre `1134/1602` linhas bit (`70.79%`).
- Ficam `94` linhas bit do train sem CoT no anexo.

Qualidade do `success`:

| Metrica | Valor |
|---|---:|
| Linhas | 1134 |
| Respostas 8-bit validas | 1134 |
| IDs duplicados | 0 |
| CoT com answer no tail | 1134 |
| CoT com `boxed` | 0 |
| Word count min/median/p95/max | `71 / 281 / 362 / 404` |
| Confidence high | 671 |
| Confidence low | 463 |

Distribuicao de metodos em `success`:

| Metodo | Linhas |
|---|---:|
| `bf` | 463 |
| `ctx` | 374 |
| `w_mix` | 186 |
| `w_rot` | 80 |
| `w_uni2` | 23 |
| `w_perm` | 8 |

O `failed` nao deve entrar em treino positivo:

- `374` linhas;
- `0` prompt mismatches e `0` answer mismatches no campo `answer`;
- somente `5/374` CoTs terminam com a resposta correta no tail;
- portanto o campo `generated_cot` de `failed` e material de diagnostico/hard
  negative, nao material SFT positivo.

## Resultado Do `archive (1).zip`

Conteudo:

- `synthetic_bit_manipulation.csv`;
- `3000` linhas sinteticas;
- colunas: `id`, `prompt`, `answer`, `generated_cot`, `confidence`,
  `method`, `n_ambig_bits`, `true_rule`, `solver_correct`.

Qualidade:

| Metrica | Valor |
|---|---:|
| Linhas | 3000 |
| Respostas 8-bit validas | 3000 |
| IDs duplicados | 0 |
| CoT com `boxed` | 0 |
| CoT com answer no tail | 1664 |
| Word count min/median/p95/max | `71 / 305 / 362 / 407` |
| `solver_correct=True` | 1659 |
| `solver_correct=False` | 1341 |

Distribuicao de operadores em `true_rule`:

| Operador | Ocorrencias |
|---|---:|
| `ID` | 9837 |
| `AND` | 3595 |
| `XOR` | 2021 |
| `XNOR` | 1824 |
| `OR` | 1759 |
| `NOR` | 1391 |
| `NOT` | 1234 |
| `NAND` | 874 |
| `IMPL` | 647 |
| `INHIB` | 623 |
| `MAJ` | 195 |

Decisao: nao usar bruto em SFT. O arquivo sintetico e util como fonte de
vocabulario/fixtures para solver bit, mas contem muitos `solver_correct=False`
e CoTs longas sem `boxed`.

## Resultado Dos Zips `archive (2)` A `archive (6)`

### `archive (2).zip` e `archive (3).zip`

Ambos contem apenas:

- `train.csv`;
- `test.csv`.

Os arquivos internos sao identicos:

| Arquivo interno | SHA256 | Linhas |
|---|---|---:|
| `train.csv` | `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa` | 9500 |
| `test.csv` | `c59d7eb0464b0a872a0c3f81e60cd6643fc1932a2dedaa05972bfd02cc638589` | 3 |

O `train.csv` e byte-identico a
`C:\Users\davis\Downloads\competition_train.csv`. Portanto nao traz dado novo,
mas confirma o contrato oficial usado nas auditorias.

### `archive (4).zip`

Contem:

- `nvidia-nemotron-model-reasoning-challenge/train.csv`;
- `nvidia-nemotron-model-reasoning-challenge/test.csv`;
- `train.csv`;
- `test.csv`;
- `smart_train.csv`.

Os quatro arquivos `train/test` sao duplicatas do dataset oficial. O unico
arquivo novo e `smart_train.csv`.

Auditoria de `smart_train.csv`:

| Metrica | Valor |
|---|---:|
| Linhas | 28 |
| Overlap com `competition_train.csv` | 28 |
| Prompt mismatch | 0 |
| Answer oficial contido no texto | 28 |
| Answer igual ao oficial | 0 |
| Linhas com `boxed` | 28 |
| Familia observada | numeral/Roman |

Decisao: `smart_train.csv` nao ajuda as duas familias problematicas. Ele e uma
colecao pequena de CoT para numeral/Roman em que a coluna `answer` contem a
explicacao inteira, nao apenas a resposta. Para nosso plano atual, e ruido.

### `archive (5).zip` e `archive (6).zip`

- `archive (5).zip` e duplicata exata de `archive.zip`;
- `archive (6).zip` e duplicata exata de `archive (1).zip`.

Nao adicionam cobertura nova.

## Impacto Para As Duas Familias Problematicas

### `bit_manipulation`

Ajuda concreta:

1. Temos `1134` CoTs de sucesso alinhadas com `competition_train.csv`, sem
   mismatch de prompt/resposta.
2. Temos `671` linhas high-confidence (`n_ambig_bits=0`) que podem virar
   pacote P0 de treino ou validação apos conversao para formato KG1:
   `Final answer: \boxed{...}`.
3. O dataset cobre metodos que faltavam no nosso pacote ativo:
   `ctx`, `w_mix`, `w_rot`, `w_uni2`, `w_perm`, alem de `bf`.
4. O arquivo sintetico confirma operadores que o V529 ja indicava:
   `INHIB`, `IMPL`, `MAJ`, `XNOR`, `NAND`, `NOR`.

Risco:

- CoTs brutas sao longas (`p50 281` palavras) e sem `boxed`.
- Se usadas sem compressao e `example_mean`, vao recriar o erro V524 de loss
  dominada por massa de tokens bit.
- O `failed.csv` nao pode entrar como positivo.

Acao recomendada:

1. Criar V531 CPU converter:
   - entrada: `bit_manipulation_cot_success.csv`;
   - primeiro bloco: somente `confidence=high`;
   - converter final para `Final answer: \boxed{answer}`;
   - encurtar CoT para regra + verificacao curta + resposta final;
   - manter `id` original em metadata, mas gerar `source_id` KG1 separado;
   - bloquear qualquer linha com mismatch de resposta, missing answer tail ou
     control chars.
2. Rodar tokenization/trace gates.
3. Comparar cobertura contra V523/V515.
4. Só depois considerar SFT `example_mean`.

### `equation_transform`

Nao ha ganho direto. Os anexos sao bit-only.

Acao para equation continua a do roadmap V529:

- portar DSL de `pjt222`/`zzys0316`;
- concat/reverse concat;
- operand/result reversal;
- `+1/-1`, divisao/modulo;
- prefix/suffix operator encoding;
- `Z_94`/mod-94.

## Decisao V530

Os anexos devem entrar no roadmap como nova fonte P0 para `bit_manipulation`,
mas nao liberam submit nem H200 imediatamente.

Promocao minima para qualquer uso em treino:

- nenhum uso de `failed.csv` como positivo;
- `success.csv` convertido para formato KG1 com `\boxed{}`;
- traces encurtadas;
- `LOSS_NORMALIZATION_MODE=example_mean`;
- hard guard `bit>=136`, `8740ed31=01101000`, `trunc=0`;
- weak diagnostic apenas para medir, nunca para treinar.
